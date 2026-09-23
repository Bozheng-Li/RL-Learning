"""页面之间共享的小工具。

只放「多个页面都要用、且与具体排版无关」的东西，避免每个 view 各抄一份。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from webui import data, paths, theme


def outputs_root() -> Path:
    """取侧边栏设置的 outputs 根目录。"""
    raw = st.session_state.get("outputs_root") or str(paths.OUTPUTS_ROOT)
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (paths.REPO_ROOT / candidate)


def selected_run_name() -> str | None:
    """当前选中的运行名（由总览页写进 session_state）。"""
    return st.session_state.get("selected_run")


def load_runs(pattern: str | None = None) -> list[data.RunInfo]:
    """扫描运行；对共享的 outputs 根做一次轻量缓存。

    缓存的意义在 Streamlit 的交互模型上：点一下多选框、切一次下拉，整个脚本会重跑，
    而一次全量扫描是 1.1 秒（639 个运行，要读 639 份 ``resolved_config.yaml`` 与
    639 个 ``progress.csv`` 的尾部）。用户在筛选器上连点几下就白等好几秒。

    ``ttl`` 给 8 秒：训练在跑时进度条仍然每几秒前进一次，而同一轮交互里的多次
    调用共享同一次扫描。要立刻看到新结束的运行，用侧边栏的「清空缓存并刷新」。

    ``st.cache_data`` 每次返回的是**反序列化的副本**，所以调用方可以放心改自己
    拿到的 ``RunInfo``（例如把 ``latest_timestep`` 置空），不会污染别人的那一份。
    """
    return _scan_runs(str(outputs_root()), pattern)


@st.cache_data(ttl=8.0, show_spinner=False)
def _scan_runs(root: str, pattern: str | None) -> list[data.RunInfo]:
    candidate = Path(root)
    return data.list_runs(candidate, pattern=pattern)


def signal_charts(
    run_dir: Path,
    keys: list[str],
    *,
    height: int = 220,
    columns: int = 2,
) -> None:
    """每个信号一张小图，各自独立缩放。

    刻意**不**把多个信号叠在同一根轴上：它们的量纲差得极远（回合回报可能到几百，
    近似 KL 只有 0.01 量级），画在一起时小量纲的曲线会被压成贴着 x 轴的一条直线，
    看起来像「没数据」。分工明确的独立缩放才读得出来。
    """
    import streamlit as st  # noqa: PLC0415  —— 本模块只在 Streamlit 里用

    from webui import live  # noqa: PLC0415

    frame = live.signal_frame(run_dir, keys=keys)
    if frame is None or frame.empty:
        st.info("这些信号还没有数据。")
        return

    available = [key for key in keys if key in frame.columns]
    if not available:
        st.info("这些信号还没有数据。")
        return

    rows = (len(available) + columns - 1) // columns
    for index in range(rows):
        chunks = available[index * columns : (index + 1) * columns]
        slots = st.columns(len(chunks))
        for slot, key in zip(slots, chunks):
            with slot:
                st.caption(live.SIGNAL_LABELS.get(key, key))
                st.line_chart(frame[[key]], height=height)


def gpu_panel(*, interval: float | None = None, key: str = "gpu-panel") -> None:
    """显卡状态面板：每张卡一行显存条 + 利用率 + 占用进程。

    ``interval`` 给定时用 ``st.fragment(run_every=...)`` 定时重跑——注意装饰器必须写在
    函数体内现定义的函数上（模块顶层的 ``@st.fragment`` 会被冻结在首次导入的参数值，
    见 ``views/monitor.py`` 开头的说明）。``interval=None`` 就只画一次。

    进程是否「由本 WebUI 启动」靠与任务注册表里的 pid 比对得出，这是判断「这张卡上的
    负载是不是我自己造的」的唯一可靠依据。
    """
    from webui import gpus  # noqa: PLC0415  —— 避免调用方为了画面板而顶层导入 torch
    from webui import jobs as jobs_module  # noqa: PLC0415

    def _paint() -> None:
        inventory = gpus.detect_devices(force=True, with_processes=True)
        if not inventory.devices:
            theme.empty_state(
                "没有可用的显卡",
                inventory.note or "没有检测到 NVIDIA 显卡，任务会跑在 CPU 上。",
            )
            return
        known_pids = {
            job.pid for job in jobs_module.load_registry() if job.pid is not None
        }
        by_uuid: dict[str, list[gpus.GpuProcess]] = {}
        for process in inventory.processes:
            by_uuid.setdefault(process.gpu_uuid, []).append(process)

        cards = []
        for device in inventory.devices:
            used = device.used_mb
            busy = device.is_busy
            lines = []
            for process in by_uuid.get(device.uuid or "", []):
                mine = " · 本 WebUI 启动" if process.pid in known_pids else ""
                amount = f" · {process.used_memory_mb / 1024:.1f} GB" if process.used_memory_mb else ""
                lines.append(f"PID {process.pid} · {process.name}{amount}{mine}")
            cards.append(
                theme.gpu_card(
                    device.short_label,
                    subtitle=f"{device.memory_free_mb / 1024:.1f} GB 空闲"
                    if device.memory_free_mb is not None else "显存未知",
                    used_mb=used,
                    total_mb=device.memory_total_mb,
                    utilization_pct=device.utilization_pct,
                    processes=lines,
                    tone="busy" if busy else "free",
                    note="" if lines else "没有进程占用",
                )
            )
        st.markdown(
            '<div class="rl-stats" style="grid-template-columns:repeat(auto-fit,minmax(230px,1fr))">'
            + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )
        if inventory.source == "torch":
            theme.caption("`nvidia-smi` 不可用，利用率与占用进程读不到（已回退到 torch）。")

    if interval is None:
        _paint()
    else:
        st.fragment(run_every=interval, key=key)(_paint)()


def run_selector(
    runs: list[data.RunInfo],
    *,
    key: str,
    label: str = "选择运行",
    default_first: bool = True,
) -> data.RunInfo | None:
    """渲染一个运行下拉框，返回选中的 ``RunInfo``。

    默认选中 ``selected_run_name()`` 指向的运行（用户可能从总览页的表格点过来的），
    否则选第一个。
    """
    if not runs:
        st.info("没有找到任何运行。先到「发起训练」页跑一个，或检查侧边栏的 outputs 目录。")
        return None

    names = [info.name for info in runs]
    preferred = selected_run_name()
    index = 0
    if preferred in names:
        index = names.index(preferred)
    elif not default_first:
        index = 0

    labels = {
        info.name: f"{info.name}　·　{data.STATUS_LABELS[info.status]}　·　"
        f"{info.algorithm or '?'}　·　{info.environment or '?'}"
        for info in runs
    }
    chosen = st.selectbox(
        label,
        options=names,
        index=index,
        key=key,
        format_func=lambda name: labels.get(name, name),
    )
    return next(info for info in runs if info.name == chosen)
