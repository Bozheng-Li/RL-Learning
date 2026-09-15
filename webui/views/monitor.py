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


def _render_manual() -> None:
    """没有 WebUI 启动的任务时，允许手动指定一个运行目录看进度。"""
    st.subheader("查看已有运行的进度")
    runs = data.list_runs()
    if not runs:
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
    theme.page_header("训练监控", "由 WebUI 启动的任务，实时刷新")

    all_jobs = _all_jobs()
    if not all_jobs:
        st.info(
            "还没有由 WebUI 启动的任务。到「发起训练」页启动一个，"
            "或在这里直接查看某个已有运行的进度（它不会自动刷新）。"
        )
        _render_manual()
        return

    interval = st.session_state.get("poll_interval", 2.0)
    if interval is None:
        st.caption("自动刷新已关闭（在侧边栏开启）。点下方按钮手动刷新。")
        if st.button("刷新一次"):
            st.rerun()

    for job in all_jobs:
        _render_job(job, interval)


def _render_job(job: jobs.Job, interval: float | None) -> None:
    """渲染一个任务。定时刷新只重跑这个卡片。"""

    @st.fragment(run_every=interval, key=f"monitor-{job.job_id}")
    def _card() -> None:
        job.refresh()
        run_dir = Path(job.run_dir) if job.run_dir else None
        status = data.run_status(
            run_dir, exited_code=job.exit_code, pid=job.pid
        ) if run_dir else ("completed" if job.exit_code == 0 else "failed")

        st.subheader(job.label)
        meta = st.columns(4)
        meta[0].metric("状态", data.STATUS_LABELS.get(status, status))
        meta[1].metric("PID", job.pid if job.pid else "-")
        meta[2].metric("已运行", live.format_duration(time.time() - job.started_at))
        meta[3].metric("退出码", job.exit_code if job.exit_code is not None else "-")

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
