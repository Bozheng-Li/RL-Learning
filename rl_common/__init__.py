"""配置驱动的强化学习实验框架。

模块划分：

- ``config``         配置加载、路径解析、运行目录
- ``experiment``     训练编排（把配置、环境、算法、回调串起来）
- ``training_log``   训练日志（episode 明细、动作分布诊断）
- ``visualization``  训练曲线与策略轨迹
"""

from .config import load_config, resolve_repo_path, run_directory
from .experiment import train_from_config
from .visualization import visualize_from_config

__all__ = [
    "load_config",
    "resolve_repo_path",
    "run_directory",
    "train_from_config",
    "visualize_from_config",
]
