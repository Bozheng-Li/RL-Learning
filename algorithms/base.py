"""所有强化学习算法的统一接口。

这个模块定义了 ``BaseAlgorithm``：本项目里每一个算法——不管是自己实现的
（``algorithms/reinforce.py``、``a2c.py``、``ppo.py``）还是外部的
（SB3 / sb3-contrib，通过 ``algorithms/sb3_wrapper.py`` 适配）——都必须实现
这一套方法。训练编排层只认这个接口，不关心背后是哪种实现。

接口刻意保持与 Stable-Baselines3 的 ``BaseAlgorithm`` 一致，这一点很关键：
框架里的评估走 SB3 的 ``evaluate_policy``，它内部调用
``model.predict(obs, state=..., episode_start=..., deterministic=...)``。
只要签名对齐，自研算法就能直接复用 SB3 现成的评估、统计和可视化工具链，
而不需要为它单独写一套。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseAlgorithm(ABC):
    """强化学习算法的统一接口。

    子类需要实现四个方法：``learn`` / ``predict`` / ``save`` / ``load``。
    """

    def __init__(self, env: Any, config: dict[str, Any]) -> None:
        """构造算法。

        Args:
            env: 已包装好的训练环境（Gymnasium 接口）。
            config: 已解析的完整实验配置（对应一个 YAML 文件的内容）。
                算法自身需要的超参从 ``config["algorithm"]`` 里取。
        """
        self.env = env
        self.config = config

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        progress_bar: bool = False,
        reset_num_timesteps: bool = True,
    ) -> "BaseAlgorithm":
        """训练模型，返回自身。

        参数名与 SB3 保持一致，这样训练编排层可以用同一份代码驱动所有算法。
        ``callback`` 接受 SB3 风格的 callback（本项目用它做周期评估和
        checkpoint 保存），自研算法需要在每次 rollout 结束时调用
        ``callback.on_step()``。
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        """对单个观测给出动作。

        Returns:
            ``(action, state)``。``state`` 是循环策略的隐状态，无记忆的算法
            返回 ``None``。这个返回值形状是 SB3 的约定，``evaluate_policy``
            依赖它。
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """保存模型到磁盘。``path`` 不带扩展名。"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(
        cls, path: str | Path, env: Any = None, device: str = "cpu"
    ) -> "BaseAlgorithm":
        """从磁盘加载模型。

        Args:
            path: 模型路径（可带或不带 ``.zip`` 后缀，实现方自行处理）。
            env: 可选的环境，某些算法加载后需要它来恢复空间信息。
            device: 计算设备。
        """
        raise NotImplementedError

    def set_logger(self, logger: Any) -> None:
        """注入日志器。

        默认空实现——自研算法用自己的日志系统（见 ``rl_common/training_log.py``），
        不需要外部注入。SB3 的适配器会覆盖这个方法转交给内部模型。
        """
        return None
