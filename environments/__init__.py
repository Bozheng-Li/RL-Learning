"""统一的环境构造。

以前每个环境一个目录（``cartpole/``、``taxi/``、``minigrid/`` ...），各自带一份
几乎相同的 ``train.py``/``play.py``/``visualize.py``。现在环境构造集中在这里，
由配置里的 ``environment`` 段决定用哪个环境、怎么包装——新增一个环境只需加一份
YAML，不再需要新建目录。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from .wrappers import (
    MiniGridFlatObservation,
    MiniGridNoveltyBonus,
    MiniGridProgressShaping,
)

__all__ = ["make_environment", "apply_environment_attributes"]


def apply_environment_attributes(env: gym.Env, attributes: dict[str, Any]) -> None:
    """按配置覆盖环境内部属性。

    环境构造完之后有些东西只能通过直接改属性来调整。改完之后观测空间往往不再
    匹配，所以下面针对 CartPole 之类的经典环境做了同步修正。
    """
    target = env.unwrapped
    for name, value in attributes.items():
        if not hasattr(target, name):
            raise AttributeError(
                f"Environment {type(target).__name__} has no configurable attribute {name!r}"
            )
        setattr(target, name, value)

    # CartPole 的质量/长度是派生量，改了基本参数必须重算，否则物理参数不一致。
    if hasattr(target, "masscart") and hasattr(target, "masspole"):
        target.total_mass = target.masscart + target.masspole
    if hasattr(target, "masspole") and hasattr(target, "length"):
        target.polemass_length = target.masspole * target.length
    # 改了角度/位置阈值后，观测空间的上下界要跟着变，否则归一化会出错。
    if hasattr(target, "x_threshold") and hasattr(target, "theta_threshold_radians"):
        high = np.asarray(
            [
                target.x_threshold * 2,
                np.inf,
                target.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )
        target.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)


def make_environment(
    config: dict[str, Any],
    *,
    monitor_path: Path | None = None,
    render_mode: str | None = None,
    exploration: bool = True,
) -> gym.Env:
    """按配置构造环境。

    Args:
        config: 完整实验配置。
        monitor_path: 若提供，则套一层 ``Monitor`` 把每个 episode 的回报与长度
            写进 CSV。训练、周期评估、最终评估各写一份，便于事后对照。
        render_mode: 覆盖配置里的渲染模式（录制轨迹时用 ``"rgb_array"``）。
        exploration: 是否启用奖励塑形。**训练时 True，评估与轨迹录制时 False**，
            这样报告的回报始终是纯任务奖励。对没有奖励类包装器的环境没有影响。

    Raises:
        ImportError: 环境所属的包（minigrid / ale_py）未安装。
        ValueError: 配置里出现了不支持的包装器名。
    """
    env_config = config["environment"]
    kwargs = deepcopy(env_config.get("kwargs", {}))
    if kwargs.get("render_mode") is None:
        kwargs.pop("render_mode", None)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    max_steps = env_config.get("max_episode_steps")
    if max_steps is not None:
        kwargs["max_episode_steps"] = int(max_steps)

    # 下面两个包的作用是向 Gymnasium 注册环境 id，导入即生效。
    if str(env_config["id"]).startswith("MiniGrid"):
        try:
            import minigrid  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "MiniGrid environments require the `minigrid` package. "
                "Install it with `python -m pip install minigrid`."
            ) from error
    if str(env_config["id"]).startswith("ALE/"):
        try:
            import ale_py  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "Atari environments require `gymnasium[atari]`."
            ) from error

    env = gym.make(env_config["id"], **kwargs)
    # 属性必须在包装之前设置：包装器构造时会读取 observation_space，
    # 而改属性可能改变观测空间（见上面 apply_environment_attributes）。
    apply_environment_attributes(env, env_config.get("attributes", {}))

    for wrapper_name in env_config.get("wrappers", []):
        wrapper_name = str(wrapper_name).lower()
        if wrapper_name in {"minigrid_flat", "minigrid_flat_observation"}:
            env = MiniGridFlatObservation(env)
        elif wrapper_name in {"minigrid_novelty", "minigrid_exploration"}:
            env = MiniGridNoveltyBonus(
                env,
                scale=env_config.get("novelty_scale", 0.1),
                enabled=exploration,
            )
        elif wrapper_name in {"minigrid_progress", "minigrid_shaping"}:
            env = MiniGridProgressShaping(
                env,
                pickup_bonus=env_config.get("shaping_pickup_bonus", 0.5),
                door_bonus=env_config.get("shaping_door_bonus", 0.5),
                enabled=exploration,
            )
        elif wrapper_name in {"flatten_observation", "flatten"}:
            env = gym.wrappers.FlattenObservation(env)
        else:
            raise ValueError(
                f"Unsupported environment wrapper {wrapper_name!r}; "
                "choose minigrid_flat, minigrid_novelty, minigrid_progress "
                "or flatten_observation"
            )

    if monitor_path is not None:
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        monitor_config = env_config.get("monitor", {})
        env = Monitor(
            env,
            filename=str(monitor_path),
            allow_early_resets=monitor_config.get("allow_early_resets", True),
            info_keywords=tuple(monitor_config.get("info_keywords", [])),
            reset_keywords=tuple(monitor_config.get("reset_keywords", [])),
        )
    return env
