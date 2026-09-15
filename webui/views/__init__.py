"""界面页面。

每个模块只暴露一个 ``render() -> None``，由 ``app.py`` 按侧边栏的选择分发。

目录名刻意用 ``views`` 而不是 ``pages``：Streamlit 会自动把 ``pages/`` 下的
每个脚本注册成一个页面，与本项目的显式路由冲突。
"""

from __future__ import annotations

__all__ = ["compare", "detail", "monitor", "overview", "train"]
