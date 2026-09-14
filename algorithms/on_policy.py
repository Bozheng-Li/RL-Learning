"""策略梯度类算法的公共实现。

``REINFORCE`` / ``A2C`` / ``PPO`` 三个算法共享这里的骨架：环境交互、rollout 收集、
计数器维护、模型的保存与加载。子类只需要实现两件事：

- ``collect_rollout()`` —— 怎么攒一批经验（长度和是否要完整回合）
- ``update()`` —— 拿到经验后怎么算损失、更新参数

**这里刻意不复用 SB3 的训练循环**。用别人写好的 ``learn()`` 学不到东西；
亲手写一遍「采样 → 算优势 → 算损失 → 反向传播」这条链路，才是这个项目里自研
算法的意义。

同时，这个基类对外提供了一套与 SB3 ``BaseAlgorithm`` 兼容的接口
（``num_timesteps`` / ``logger`` / ``get_env()`` / ``save()``），这样它就能被框架里
现成的 checkpoint、日志、可视化工具驱动，而不用为两种实现各写一套外围代码。
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.logger import Logger
from torch.distributions import Categorical

from .base import BaseAlgorithm
from .networks import build_policy


def action_log_prob(distribution: Any, action: torch.Tensor) -> torch.Tensor:
    """计算动作的对数概率，多维动作会按维度求和。

    这个细节很容易踩坑：高斯分布（连续动作）对**每个动作维度**各返回一个对数概率，
    形状是 ``(batch, action_dim)``；而分类分布（离散动作）返回 ``(batch,)``。

    多自由度动作的联合概率是各维概率之积，取对数后就是各维对数概率之和。
    所以这里按最后一维求和，得到形状 ``(batch,)`` 的联合对数概率。

    不处理这一步的话，在 BipedalWalker（4 维）、CarRacing（3 维）这类环境上
    会直接报 ``a Tensor with 4 elements cannot be converted to Scalar``，
    而单自由度的 CartPole 却看不出问题。
    """
    log_prob = distribution.log_prob(action)
    return log_prob.sum(dim=-1) if log_prob.dim() > 1 else log_prob


def action_entropy(distribution: Any) -> torch.Tensor:
    """计算策略熵，多维动作同样按维度求和（与 log_prob 保持一致）。"""
    entropy = distribution.entropy()
    return entropy.sum(dim=-1) if entropy.dim() > 1 else entropy


class OnPolicyAlgorithm(BaseAlgorithm):
    """on-policy 策略梯度算法的公共基类。

    "on-policy" 的含义是：**训练用的数据必须来自当前策略**。每次更新完参数，
    之前收集的经验就作废了（它由旧策略产生，分布已经对不上）。这也是为什么
    这类算法的样本效率通常低于 DQN 那种 off-policy 方法——数据用一次就扔。
    """

    def __init__(self, env: Any, config: dict[str, Any]) -> None:
        super().__init__(env, config)
        algorithm_config = config["algorithm"]
        profile = self._resolve_profile(algorithm_config)

        self.device = torch.device(algorithm_config.get("device", "cpu"))
        self.gamma = float(profile.get("gamma", 0.99))
        self.learning_rate = float(profile.get("learning_rate", 3e-4))
        self.ent_coef = float(profile.get("ent_coef", 0.0))
        self.seed = int(config["experiment"].get("seed", 42))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.policy = build_policy(
            env.observation_space,
            env.action_space,
            hidden_sizes=profile.get("hidden_sizes", [64, 64]),
            activation=profile.get("activation_fn"),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

        # ---- 与 SB3 兼容的接口，供框架的日志/回调/可视化使用 ----
        self.num_timesteps = 0
        self.n_calls = 0
        self.logger: Logger | None = None
        # CheckpointCallback 会检查这两个属性；自研算法没有回放缓冲区。
        self.replay_buffer = None
        self.save_replay_buffer = False

        self._last_observation: np.ndarray | None = None

    # ---- 子类需要实现的部分 ----

    @abstractmethod
    def collect_rollout(self, callback: Any = None, total_timesteps: int = 0) -> None:
        """采集一批经验存入 ``self.buffer``。"""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """用缓冲区里的数据更新参数。

        Returns:
            要写进 ``progress.csv`` 的训练信号，例如
            ``{"train/policy_gradient_loss": ..., "train/entropy_loss": ...}``。
        """

    # ---- 公共工具 ----

    #: profile 缺失时是否允许退回默认超参。超参少的算法（REINFORCE）可以放宽，
    #: 这样不必为它也写一份 profile。
    require_profile: bool = True
    #: 允许缺失时使用的兜底超参。
    default_profile: dict[str, Any] = {}

    def _resolve_profile(self, algorithm_config: dict[str, Any]) -> dict[str, Any]:
        """读取本算法的 profile。

        **自研实现与 SB3 实现读的是同一份 profile**，这是刻意设计的：对照实验
        要求同环境、同超参、只换实现，学习曲线的差异才能归因到实现本身。

        因此这里除了 ``kwargs`` 里的优化器超参，还要兼容 SB3 风格的
        ``policy_kwargs.net_arch`` / ``policy_kwargs.activation_fn``，
        把它们翻译成自研网络需要的 ``hidden_sizes`` / ``activation_fn``。
        """
        name = str(algorithm_config["name"]).upper()
        profiles = algorithm_config.get("profiles", {})
        key = next((k for k in profiles if str(k).upper() == name), None)
        if key is None:
            if type(self).require_profile:
                raise ValueError(
                    f"配置里没有 algorithm.profiles.{name}；自研算法同样需要一份 profile，"
                    f"或者把 name 改成 SB3-{name} 用 SB3 的实现。"
                )
            return dict(type(self).default_profile)

        raw = profiles[key]
        profile: dict[str, Any] = dict(raw.get("kwargs", {}))

        if "hidden_sizes" in raw:
            profile["hidden_sizes"] = list(raw["hidden_sizes"])
        else:
            net_arch = raw.get("policy_kwargs", {}).get("net_arch")
            if net_arch is not None:
                # SB3 的 net_arch 可以写成 int 或 list，统一成 list。
                profile["hidden_sizes"] = (
                    list(net_arch) if isinstance(net_arch, (list, tuple)) else [int(net_arch)]
                )

        activation = raw.get("activation_fn") or raw.get("policy_kwargs", {}).get(
            "activation_fn"
        )
        if activation is not None:
            profile["activation_fn"] = activation
        return profile

    def _record_logs(self, logs: dict[str, float]) -> None:
        """把训练信号写进日志（SB3 的 logger 会落成 progress.csv）。"""
        if self.logger is None:
            return
        for key, value in logs.items():
            self.logger.record(key, float(value))
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)

    def _begin_rollout(self) -> np.ndarray:
        """取当前观测（必要时重置环境），返回它。"""
        if self._last_observation is None:
            observation, _ = self.env.reset(seed=self.seed)
            self._last_observation = np.asarray(observation, dtype=np.float32)
        return self._last_observation

    def _step_env(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """执行一步并处理回合结束后的自动重置。

        Gymnasium 在回合结束后要求显式 ``reset()``，这里统一处理，让上层逻辑
        只需要关心「拿到下一个观测、奖励、是否结束」。
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done:
            observation, _ = self.env.reset()
        self._last_observation = np.asarray(observation, dtype=np.float32)
        return self._last_observation, float(reward), done, info

    def _log_env_step(self, callback: Any, locals_: dict[str, Any]) -> bool:
        """通知回调「又走了一步」。

        Returns:
            False 表示回调要求提前终止训练。
        """
        self.num_timesteps += 1
        if callback is None:
            return True
        callback.update_locals(locals_)
        return bool(callback.on_step())

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        progress_bar: bool = False,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        """训练主循环。

        结构是「采样一段 → 更新一次」反复进行。两种算法只是节奏不同：

        - REINFORCE 每次采样**完整回合**（必须如此，它要用蒙特卡洛回报）
        - A2C / PPO 每次采样固定 ``n_steps`` 步（可以用自举，不必等回合结束）

        ``progress_bar`` / ``log_interval`` 参数保留是为了接口与 SB3 一致，
        自研实现里暂未使用（日志按 rollout 粒度记录）。
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.n_calls = 0
        self._last_observation = None
        self._stop_training = False

        if callback is not None:
            callback.init_callback(self)
            # 注意这里必须传字典而非 None：SB3 的 on_training_start 会直接把
            # self.locals 赋成这个参数，传 None 会让后续 update_locals 崩溃。
            callback.on_training_start(
                locals_={"total_timesteps": int(total_timesteps)}, globals_={}
            )

        try:
            while self.num_timesteps < total_timesteps and not self._stop_training:
                if callback is not None:
                    callback.on_rollout_start()
                self.collect_rollout(callback=callback, total_timesteps=total_timesteps)
                if callback is not None:
                    callback.on_rollout_end()
                if self._stop_training:
                    break
                logs = self.update()
                self._record_logs(logs)
        finally:
            if callback is not None:
                callback.on_training_end()
        return self

    # ---- BaseAlgorithm 接口 ----

    def set_logger(self, logger: Any) -> None:
        self.logger = logger

    def get_env(self) -> Any:
        """SB3 回调会通过它拿环境（我们不做向量化，直接返回）。"""
        return self.env

    def get_vec_normalize_env(self) -> None:
        """自研算法不做观测归一化，恒返回 None。"""
        return None

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        """单步推理。

        返回值的形状**刻意模仿 SB3 的约定**，否则框架里的评估工具会用不了：

        - 输入是单个观测时，离散动作返回形状 ``(1,)`` 的数组，连续动作返回
          ``(action_dim,)``；
        - 输入本身是批量的（首维存在）时，返回保持批量维度。

        这一点很关键：SB3 的 ``evaluate_policy`` 会把环境包成 ``DummyVecEnv``，
        然后做 ``self.actions[env_idx]`` 取动作。如果这里返回一个裸 int，
        就会报 ``'int' object is not subscriptable``。

        ``state`` / ``episode_start`` 是为循环策略（LSTM）预留的参数，自研的
        三个算法都是无记忆的，用不到；保留它们是为了接口完全一致。
        """
        obs_array = np.asarray(observation, dtype=np.float32)
        expected_ndim = len(self.env.observation_space.shape)
        batched = obs_array.ndim > expected_ndim
        if not batched:
            obs_array = obs_array[np.newaxis, ...]

        with torch.no_grad():
            tensor = torch.as_tensor(obs_array, device=self.device)
            distribution = self.policy.distribution(tensor)
            # 离散动作取概率最大的类别；连续动作取分布的均值。
            if isinstance(distribution, Categorical):
                action = (
                    distribution.probs.argmax(dim=-1)
                    if deterministic
                    else distribution.sample()
                )
            else:
                action = distribution.mean if deterministic else distribution.sample()
            array = action.cpu().numpy()

        is_discrete = isinstance(self.env.action_space, spaces.Discrete)
        action_array = (
            array.reshape(-1).astype(np.int64) if is_discrete else array.astype(np.float32)
        )
        if batched:
            return action_array, None
        # 单个观测：离散保持 (1,)（与 SB3 一致），连续去掉 batch 维。
        return (action_array.reshape(-1), None) if is_discrete else (action_array[0], None)

    def _extra_state(self) -> dict[str, Any]:
        """子类要额外保存的状态（比如 A2C/PPO 的 critic 网络）。

        默认没有。这样设计是为了让所有算法都存进**同一个文件**，
        而不是「policy 一个文件、critic 另一个文件」——后者在加载时很容易漏掉。
        """
        return {}

    def _load_extra(self, extra: dict[str, Any]) -> None:
        """加载 ``_extra_state`` 存下的内容。"""
        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        # 统一成 .pt。SB3 的 CheckpointCallback 会按 `xxx_steps.zip` 命名传进来
        # （它假设所有模型都是 SB3 格式），直接拼接会得到 `xxx_steps.zip.pt`。
        path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "extra": self._extra_state(),
                "num_timesteps": self.num_timesteps,
                "config": self.config,
                "algorithm": type(self).__name__,
            },
            str(path),
        )

    @classmethod
    def load(
        cls, path: str | Path, env: Any = None, device: str = "cpu"
    ) -> "OnPolicyAlgorithm":
        path = Path(path)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        if env is None:
            # 重新可视化时可能不传环境，按配置重建一个来恢复网络结构。
            from environments import make_environment

            env = make_environment(config)
        instance = cls(env, config)
        instance.policy.load_state_dict(payload["policy"])
        if "optimizer" in payload:
            instance.optimizer.load_state_dict(payload["optimizer"])
        instance._load_extra(payload.get("extra", {}))
        instance.num_timesteps = int(payload.get("num_timesteps", 0))
        return instance
