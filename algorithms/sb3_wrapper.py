"""把 Stable-Baselines3 的算法适配到本项目的 ``BaseAlgorithm`` 接口。

保留 SB3 有两个作用：

1. **对照基线**。自己实现的 PPO 到底对不对？在同一环境、同一超参、同一种子下
   和 SB3 的 PPO 比学习曲线，是最直接的验证方式。只跑通不算数，量级要对得上。
2. **覆盖没自研的算法**。本项目只自研策略梯度主线（REINFORCE / A2C / PPO），
   DQN、SAC、TD3、RecurrentPPO 继续用成熟实现。

配置里用前缀区分两种来源，一眼能看出跑的是哪一个：

    algorithm:
      name: PPO          # 自研实现（algorithms/ppo.py）
      # name: SB3-PPO    # Stable-Baselines3 的对照实现
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from .base import BaseAlgorithm

# sb3-contrib 是可选依赖，只有循环策略（RecurrentPPO）需要它。
try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - 取决于是否安装 sb3-contrib
    RecurrentPPO = None


#: SB3 算法名 -> 类。键名带 ``SB3-`` 前缀，与自研算法区分开。
SB3_ALGORITHMS: dict[str, type] = {
    "SB3-A2C": A2C,
    "SB3-DQN": DQN,
    "SB3-PPO": PPO,
    "SB3-SAC": SAC,
    "SB3-TD3": TD3,
}
if RecurrentPPO is not None:
    SB3_ALGORITHMS["SB3-RECURRENTPPO"] = RecurrentPPO

#: 激活函数名 -> torch 模块。YAML 里写字符串，这里转换成可调用的类。
ACTIVATIONS: dict[str, Any] = {}


def _activation_table() -> dict[str, Any]:
    """延迟导入 torch，避免在没装 torch 时导入本模块就失败。"""
    global ACTIVATIONS
    if not ACTIVATIONS:
        from torch import nn

        ACTIVATIONS = {
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
    return ACTIVATIONS


def _convert_policy_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    """把 YAML 里的字符串配置转成 SB3 需要的对象。

    目前只处理 ``activation_fn``：YAML 里写 ``relu``，SB3 要的是 ``nn.ReLU``。
    """
    policy_kwargs = deepcopy(values)
    activation_name = policy_kwargs.get("activation_fn")
    if isinstance(activation_name, str):
        table = _activation_table()
        try:
            policy_kwargs["activation_fn"] = table[activation_name.lower()]
        except KeyError as error:
            supported = ", ".join(sorted(table))
            raise ValueError(
                f"Unsupported activation_fn {activation_name!r}; choose one of: {supported}"
            ) from error
    return policy_kwargs


def _expand_noise_value(value: float | list[float], size: int) -> np.ndarray:
    """把标量或列表形式的噪声参数展开成定长向量。"""
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 0:
        return np.full(size, float(values), dtype=np.float32)
    if values.size != size:
        raise ValueError(f"Action-noise vector has size {values.size}, expected {size}")
    return values.reshape(size)


def profile_key(name: str) -> str:
    """把注册表里的算法名映射到 YAML profile 的键名。

    注册表用 ``SB3-PPO`` 区分实现来源，但配置里的 profile 键是不带前缀的
    ``PPO``。这带来一个有用的后果：**自研实现和 SB3 实现共用同一份超参**。

    这正是对照实验需要的——同环境、同超参、只换实现，学习曲线的差异才能
    归因到实现本身。如果两套实现各用一份超参，比较就失去意义了。
    """
    upper = name.upper()
    return upper[4:] if upper.startswith("SB3-") else upper


def resolve_profile(config: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    """从配置里取出某个算法的 profile。

    YAML 结构形如::

        algorithm:
          name: SB3-PPO
          profiles:
            PPO:              # 键名不带 SB3- 前缀，自研与对照共用
              policy: MlpPolicy
              policy_kwargs: {...}
              kwargs: {...}

    Returns:
        ``(policy, kwargs)``。``kwargs`` 已做类型转换，可直接传给算法构造函数。

    Raises:
        ValueError: profile 缺失，或 profile 里出现了无法识别的键。
    """
    algorithm_config = config["algorithm"]
    profiles = algorithm_config.get("profiles", {})
    key = profile_key(name)
    # 大小写不敏感地匹配 profile 名，与配置里 algorithm.name 的处理保持一致。
    profile_key_name = next(
        (candidate for candidate in profiles if str(candidate).upper() == key), None
    )
    if profile_key_name is None:
        raise ValueError(f"No algorithm.profiles.{key} section exists in the YAML file")

    profile = deepcopy(profiles[profile_key_name])
    policy = profile.pop("policy", algorithm_config.get("policy", "MlpPolicy"))
    kwargs = deepcopy(profile.pop("kwargs", {}))
    policy_kwargs = profile.pop("policy_kwargs", {})
    if profile:
        unknown = ", ".join(profile)
        raise ValueError(f"Unknown keys in algorithm profile {name}: {unknown}")
    if policy_kwargs:
        kwargs["policy_kwargs"] = _convert_policy_kwargs(policy_kwargs)
    return policy, kwargs


def _check_action_space(name: str, env: Any) -> None:
    """提前拦住「算法与动作空间不匹配」这类配置错误。

    不做这个检查的话，错误会在训练中途以难以理解的形式爆出来。
    """
    action_space = env.action_space
    if name.endswith("DQN") and not isinstance(action_space, spaces.Discrete):
        raise ValueError("DQN requires a discrete action space")
    if name.endswith(("SAC", "TD3")) and not isinstance(action_space, spaces.Box):
        raise ValueError(
            f"{name} requires a continuous Box action space. For LunarLander, set "
            "environment.kwargs.continuous: true."
        )


class SB3Algorithm(BaseAlgorithm):
    """SB3 算法的适配器。

    内部持有一个 SB3 模型，把 ``BaseAlgorithm`` 的调用转发过去。构造时完成
    profile 解析和参数校验，之后所有接口都与自研算法一致。
    """

    def __init__(self, env: Any, config: dict[str, Any], sb3_class: type) -> None:
        super().__init__(env, config)
        algorithm_config = config["algorithm"]
        name = str(algorithm_config["name"]).upper()

        policy, kwargs = resolve_profile(config, name)
        _check_action_space(name, env)

        # 动作噪声：仅连续动作空间可用，用于 SAC/TD3 这类确定性策略的探索。
        noise_config = kwargs.pop("action_noise", None)
        if noise_config is not None:
            if not isinstance(env.action_space, spaces.Box):
                raise ValueError(
                    "action_noise can only be used with a continuous action space"
                )
            size = int(np.prod(env.action_space.shape))
            mean = _expand_noise_value(noise_config.get("mean", 0.0), size)
            sigma = _expand_noise_value(noise_config.get("sigma", 0.1), size)
            noise_type = str(noise_config.get("type", "normal")).lower()
            if noise_type == "normal":
                kwargs["action_noise"] = NormalActionNoise(mean=mean, sigma=sigma)
            elif noise_type in {"ornstein_uhlenbeck", "ou"}:
                kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=mean, sigma=sigma
                )
            else:
                raise ValueError(
                    "action_noise.type must be 'normal' or 'ornstein_uhlenbeck'"
                )

        kwargs.update(
            seed=int(config["experiment"].get("seed", 42)),
            device=algorithm_config.get("device", "cpu"),
            verbose=int(algorithm_config.get("verbose", 1)),
        )
        self.policy_name = policy
        self.model = sb3_class(policy, env, **kwargs)

    # ---- 接口转发 ----

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        progress_bar: bool = False,
        reset_num_timesteps: bool = True,
    ) -> "SB3Algorithm":
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
        )
        return self

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        # SB3 的 predict 已经接受 state / episode_start（非循环算法会忽略它们），
        # 所以这里可以原样转发。
        return self.model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    @classmethod
    def load(
        cls, path: str | Path, env: Any = None, device: str = "cpu"
    ) -> "SB3Algorithm":
        raise NotImplementedError(
            "SB3 模型的加载请走 algorithms.load_algorithm()，"
            "它会根据配置里的算法名选择正确的类。"
        )

    def set_logger(self, logger: Any) -> None:
        self.model.set_logger(logger)
