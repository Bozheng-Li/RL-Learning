"""Reinforce 实验台 —— WebUI 入口。

启动::

    streamlit run webui/app.py

远程机器上先用 SSH 端口转发（服务只监听回环地址）::

    ssh -N -L 8501:127.0.0.1:8501 <user>@<host>

然后用浏览器打开 http://127.0.0.1:8501

这个文件是纯路由与全局控件。每个页面的实现都在 ``webui/views/`` 下。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit 会把脚本所在目录（webui/）放进 sys.path[0]，因此 `import webui`
# 需要仓库根也在路径里。这一步必须早于任何 webui 的 import。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from webui import nav, paths, theme  # noqa: E402


def _render_view(name: str) -> None:
    """Load only the active page so optional plotting dependencies cannot block startup."""
    from importlib import import_module

    import_module(f"webui.views.{name}").render()

PAGES = {
    "运行总览": lambda: _render_view("overview"),
    "发起训练": lambda: _render_view("train"),
    "训练监控": lambda: _render_view("monitor"),
    "跨运行对比": lambda: _render_view("compare"),
    "变体分析": lambda: _render_view("analysis"),
    "运行详情": lambda: _render_view("detail"),
}

#: 侧边栏可选的自动刷新间隔；None 表示关闭定时刷新。
POLL_OPTIONS: list[float | None] = [None, 1.0, 2.0, 5.0, 10.0, 30.0]

PAGE_HINTS = {
    "运行总览": "按环境分组，同环境内比算法",
    "发起训练": "一次跑齐一个环境上的多个算法",
    "训练监控": "同一批任务的进度、信号与日志",
    "跨运行对比": "先选环境，再看各算法排名",
    "变体分析": "同算法换配置：敏感度与噪声底线",
    "运行详情": "曲线、诊断、轨迹与产物",
}


def main() -> None:
    st.set_page_config(
        page_title="Reinforce 实验台",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="auto",
    )
    theme.inject()

    with st.sidebar:
        theme.sidebar_brand()

        # 先把别的页面登记的跳转落下来，再实例化导航控件。反过来做就晚了：
        # 控件一旦实例化，它的键就是只读的（见 webui/nav.py）。
        nav.consume(list(PAGES))

        st.markdown('<div class="rl-nav-label">实验</div>', unsafe_allow_html=True)
        choice = st.radio(
            "页面",
            list(PAGES),
            key="page",
            label_visibility="collapsed",
        )
        st.divider()

        theme.sidebar_note(
            "同环境，多算法",
            "一次训练只回答一个算法。要比较，就在同一个环境上把它们一起跑。",
        )

        if "outputs_root" not in st.session_state:
            st.session_state.outputs_root = str(paths.OUTPUTS_ROOT)
        st.text_input(
            "产物目录",
            key="outputs_root",
            help="扫描训练产物的根目录。默认是仓库下的 outputs/。",
        )

        if "poll_interval" not in st.session_state:
            st.session_state.poll_interval = 2.0
        st.selectbox(
            "监控刷新",
            options=POLL_OPTIONS,
            key="poll_interval",
            format_func=lambda value: "关闭" if value is None else f"{value:g} 秒",
            help="只影响「训练监控」页的定时刷新。",
        )

        if st.button("清空缓存并刷新", width="stretch"):
            st.cache_data.clear()
            from webui import data  # noqa: PLC0415

            data.clear_caches()
            st.rerun()

        st.divider()
        st.caption(PAGE_HINTS.get(choice, ""))
        st.caption("只读查看产物；训练以独立子进程运行。")

    PAGES[choice]()


main()
