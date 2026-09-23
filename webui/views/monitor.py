"""训练监控：由 WebUI 启动的任务的实时进度、曲线与日志。

刷新用 ``st.fragment(run_every=...)``，它只重跑这一块，不重跑整个脚本。
装饰器**必须**写在这个函数体内现定义的函数上：被 import 的模块只在首次导入时
执行一次，写在模块顶层的 ``@st.fragment`` 会被永久冻结在第一次的参数值，
之后改「刷新间隔」不会生效。
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from webui import data, jobs, live, theme
from webui.views import _shared

#: 已经结束的任务最多展开几个完整卡片。注册表是只增不减的流水账——跑完一个变体批次
#: 就是六百多条，全部渲染出来这一页要两分钟才能打开（每个卡片都读 progress.csv、
#: 贴 200 行日志），而那六百个卡片里没有一条是「现在正在跑的东西」。
_FINISHED_LIMIT = 5


def _session_jobs() -> dict[str, jobs.Job]:
    """当前会话持有的任务句柄（``Popen`` 对象不能放进缓存）。"""
    return st.session_state.setdefault("jobs", {})


def _all_jobs() -> list[jobs.Job]:
    """会话内的任务 + 磁盘注册表里的任务。

    服务重启后 ``session_state`` 没了，但磁盘注册表还在，靠它重新接管
    那些仍在运行的训练。
    """
    merged: dict[str, jobs.Job] = {}
    for job in jobs.load_registry():
        merged[job.job_id] = job
    for job_id, job in _session_jobs().items():
        # 会话里的那份带着活着的 Popen，优先。
        merged[job_id] = job
    return sorted(merged.values(), key=lambda item: item.started_at, reverse=True)


def _external_runs(known: list[jobs.Job]) -> list[data.RunInfo]:
    """磁盘上正在跑、但不在注册表里的运行——比如直接在命令行起的训练。

    WebUI 不该因为「不是自己启动的」就装作看不见：用户关心的是「这台机器现在在跑什么」，
    而不是「谁启动的」。
    """
    claimed = set()
    for job in known:
        if job.run_dir:
            claimed.add(Path(job.run_dir).name)
    return [
        info for info in _shared.load_runs()
        if info.status == "running" and info.name not in claimed
    ]


def _render_external(info: data.RunInfo) -> None:
    """外部启动的运行的卡片。只读观察，不接管（拿不到它的 ``Popen``）。"""
    @st.fragment(run_every=st.session_state.get("poll_interval", 2.0),
                 key=f"external-{info.name}")
    def _card() -> None:
        theme.section_head(f"{info.name} · 🧭 外部启动", "不是这个 WebUI 拉起来的")
        meta = st.columns(5)
        meta[0].metric("状态", data.STATUS_LABELS.get(info.status, info.status))
        meta[1].metric("计算设备", info.device or "CPU")
        meta[2].metric("算法", (info.algorithm or "-").replace("SB3-", "SB3 "))
        meta[3].metric("环境", (info.environment or "-").split("/")[-1][:18])
        meta[4].metric("种子", info.seed if info.seed is not None else "-")
        _render_progress_block(info.path, info.total_timesteps, label=info.name)

    _card()


def _render_manual() -> None:
    """挑一个已有运行看进度。是任务列表的补充，不是与它平行的入口。"""
    with st.expander("查看某个已有运行的进度", expanded=False):
        runs = _shared.load_runs()
        if not runs:
            st.caption("还没有发现任何运行。")
            return
        names = [info.name for info in runs]
        picked = st.selectbox("运行", names, key="monitor_manual_run")
        info = next(item for item in runs if item.name == picked)
        _render_progress_block(info.path, info.total_timesteps, label=info.name)


def _render_progress_block(
    run_dir: Path | None, total_timesteps: int | None, *, label: str
) -> None:
    """进度条 + 实时信号 + 日志尾。轮询路径，所有数据都是现读的。"""
    if run_dir is None:
        st.caption("这个任务没有关联的运行目录。")
        return

    progress = live.compute_progress(run_dir, total_timesteps)
    if progress is None:
        st.info("还没写出 `logs/progress.csv`——训练可能正在初始化环境。")
    else:
        left, right = st.columns([3, 1])
        left.progress(
            min(progress.fraction, 1.0),
            text=f"{progress.done:,} / {progress.total:,} 步"
            if progress.total
            else f"已训练 {progress.done:,} 步",
        )
        right.metric("预计剩余", live.format_eta(progress.eta_seconds))

        # 每个信号一张小图、各自缩放。它们的量纲差得极远（回报可能上百、KL 只有
        # 0.01 量级），叠在一根轴上会把小量纲的曲线压成贴着 x 轴的一条直线。
        frame = live.read_progress_csv(run_dir / "logs" / "progress.csv")
        signals = [key for key in live.available_signals(frame) if key != "steps"]
        preferred = [key for key in ("reward", "kl", "entropy", "clip") if key in signals]
        selected = preferred[:4] or signals[:2]
        if selected:
            _shared.signal_charts(run_dir, selected, height=190)


def render() -> None:
    theme.page_header(
        "训练监控",
        "这里列出这台机器上正在跑的训练——不管是这个 WebUI 启动的，还是命令行直接起的。",
        eyebrow="Reinforce / monitor",
    )

    all_jobs = _all_jobs()
    externals = _external_runs(all_jobs)

    if not all_jobs and not externals:
        theme.empty_state(
            "现在没有正在跑的任务",
            "到「发起训练」一次跑齐一个环境上的多个算法；已经在跑的运行会自动出现在这里。",
        )
        _render_manual()
        return

    interval = st.session_state.get("poll_interval", 2.0)
    if interval is None:
        st.caption("自动刷新已关闭（在侧边栏开启）。点下方按钮手动刷新。")
        if st.button("刷新一次"):
            st.rerun()

    if externals:
        theme.section_head("外部启动", f"{len(externals)} 个不在注册表里")
        for info in externals:
            _render_external(info)

    if not all_jobs:
        _render_manual()
        return

    # 注册表是只增不减的：跑完一个变体批次就有六百多条记录，其中绝大多数早就结束了。
    # 全部渲染成完整卡片要两分钟（每个都读 progress.csv、贴 200 行日志），而这页的
    # 目的是「现在在跑什么」。所以活着的全展开，已结束的只留最近几个，其余折叠成一行。
    running = [job for job in all_jobs if job.alive]
    finished = [job for job in all_jobs if not job.alive]
    shown = finished[:_FINISHED_LIMIT]
    hidden = len(finished) - len(shown)

    if running:
        theme.section_head("正在跑", f"{len(running)} 个")
        for job in running:
            _render_job(job, interval)
    if shown:
        theme.section_head(
            "刚结束" if running else "已结束",
            f"最近 {len(shown)} 个" + (f"，另有 {hidden} 个更早的没列出" if hidden else ""),
        )
        for job in shown:
            _render_job(job, None)
    if hidden:
        theme.caption(
            f"更早的 {hidden} 个任务已收起来（这一页只看正在跑的与刚结束的）。"
            f"要看某一次的完整输出，到「运行详情」页按运行名找——产物目录里有记录。"
        )
    _render_manual()


def _render_job(job: jobs.Job, interval: float | None) -> None:
    """渲染一个任务。定时刷新只重跑这个卡片。"""

    @st.fragment(run_every=interval, key=f"monitor-{job.job_id}")
    def _card() -> None:
        job.refresh()
        run_dir = Path(job.run_dir) if job.run_dir else None
        status = data.run_status(
            run_dir, exited_code=job.exit_code, pid=job.pid
        ) if run_dir else ("completed" if job.exit_code == 0 else "failed")

        dev_str = job.device if job.device else "CPU"
        theme.section_head(f"{job.label} · 📟 {dev_str}", "后台任务")
        meta = st.columns(5)
        meta[0].metric("状态", data.STATUS_LABELS.get(status, status))
        meta[1].metric("计算设备", dev_str)
        meta[2].metric("PID", job.pid if job.pid else "-")
        meta[3].metric("已运行", live.format_duration(time.time() - job.started_at))
        meta[4].metric("退出码", job.exit_code if job.exit_code is not None else "-")

        _render_progress_block(run_dir, job.total_timesteps, label=job.label)

        with st.expander("命令行", expanded=False):
            st.code(job.command, language="bash")
            st.caption(f"日志：`{job.log_path}`")

        st.caption("训练日志（尾部 200 行）")
        st.code(live.tail_text(Path(job.log_path), lines=200) or "（还没有输出）",
                language="text")

        if status == "completed":
            st.success("训练完成。到「运行详情」页看结果。")
        elif status == "failed":
            st.error(
                f"训练失败（退出码 {job.exit_code}）。"
                "常见原因：算法没有对应的 profile、动作空间不匹配、依赖缺失。"
            )

    _card()
