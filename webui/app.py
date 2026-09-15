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

from webui import paths  # noqa: E402
from webui.views import compare as view_compare  # noqa: E402
from webui.views import detail as view_detail  # noqa: E402
from webui.views import monitor as view_monitor  # noqa: E402
from webui.views import overview as view_overview  # noqa: E402
from webui.views import train as view_train  # noqa: E402

PAGES = {
    "运行总览": view_overview.render,
    "运行详情": view_detail.render,
    "跨运行对比": view_compare.render,
    "发起训练": view_train.render,
    "训练监控": view_monitor.render,
}

#: 侧边栏可选的自动刷新间隔；None 表示关闭定时刷新。
POLL_OPTIONS: list[float | None] = [None, 1.0, 2.0, 5.0, 10.0, 30.0]


def main() -> None:
    st.set_page_config(
        page_title="Reinforce 实验台",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Reinforce 实验台")
        choice = st.radio("页面", list(PAGES), key="page")

        st.divider()

        if "outputs_root" not in st.session_state:
            st.session_state.outputs_root = str(paths.OUTPUTS_ROOT)
        st.text_input(
            "outputs 目录",
            key="outputs_root",
            help="扫描训练产物的根目录。默认是仓库下的 outputs/。",
        )

        if "poll_interval" not in st.session_state:
            st.session_state.poll_interval = 2.0
        st.selectbox(
            "自动刷新间隔",
            options=POLL_OPTIONS,
            key="poll_interval",
            format_func=lambda value: "关闭" if value is None else f"{value:g} 秒",
            help="只影响「训练监控」页的定时刷新。",
        )

        if st.button("清空缓存并刷新", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption("只读视图 + 以子进程方式发起训练，不改动已有产物。")

    PAGES[choice]()


main()
