"""Reinforce 实验台的 WebUI。

纯新增的一层视图：读写 ``outputs/`` 下的产物、以子进程方式调用既有的
``train.py`` / ``visualize.py``，不改动 ``rl_common`` 与 ``algorithms`` 的任何行为。

启动::

    streamlit run webui/app.py
"""

from __future__ import annotations

__all__ = ["paths", "data", "live", "jobs", "plots"]
