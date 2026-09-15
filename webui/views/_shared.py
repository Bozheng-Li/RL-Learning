"""页面之间共享的小工具。

只放「多个页面都要用、且与具体排版无关」的东西，避免每个 view 各抄一份。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from webui import data, paths


def outputs_root() -> Path:
    """取侧边栏设置的 outputs 根目录。"""
    raw = st.session_state.get("outputs_root") or str(paths.OUTPUTS_ROOT)
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (paths.REPO_ROOT / candidate)


def selected_run_name() -> str | None:
    """当前选中的运行名（由总览页写进 session_state）。"""
    return st.session_state.get("selected_run")


def load_runs(pattern: str | None = None) -> list[data.RunInfo]:
    """扫描运行；对共享的 outputs 根做一次轻量缓存。"""
    return data.list_runs(outputs_root(), pattern=pattern)


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
