"""A2C —— Advantage Actor-Critic。

A2C 就是 REINFORCE 加了两样东西，理解这两样东西就理解了它。

## 改进一：用基线降方差

REINFORCE 直接拿回报 G_t 加权梯度，方差极大。A2C 改成用**优势**：

    A_t = G_t - V(s_t)

含义是「这个动作比该状态下的平均水平好多少」。减去基线 V(s) **不改变梯度的
期望**（因为 E[V(s)·∇log π] = 0），但能大幅降低方差——回报里那些与动作无关的
"运气"成分被减掉了。

V(s) 由一个单独的网络（critic）学习，训练目标是回归实际回报：
``value_loss = MSE(V(s_t), G_t)``。

## 改进二：自举，不必等回合结束

REINFORCE 必须跑完整回合才能算 G_t。A2C 用 n 步回报 + 自举：

    A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

这样只要攒够 ``n_steps`` 步就能更新一次，回合超长的任务也能训练。
λ 控制偏差-方差权衡（见 ``buffers.py`` 里的详细说明）。

## 损失的三部分

    loss = -A_t·log π(a_t|s_t)   +   c_v·(V(s_t) - G_t)²   -   c_e·H(π(·|s_t))
            └── actor：让好动作更可能 ──┘   └─ critic：估值更准 ─┘   └ 熵：保持探索 ┘

第三项（熵正则）常被忽略但很重要：没有它，策略会过早变得确定，陷入局部最优。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import register
from .buffers import RolloutBuffer
from .networks import ValueNetwork
from .on_policy import OnPolicyAlgorithm, action_entropy, action_log_prob


@register
class A2C(OnPolicyAlgorithm):
    """同步优势 Actor-Critic。"""

    def __init__(self, env: Any, config: dict[str, Any]) -> None:
        super().__init__(env, config)
        profile = self._resolve_profile(config["algorithm"])

        self.n_steps = int(profile.get("n_steps", 5))
        self.gae_lambda = float(profile.get("gae_lambda", 1.0))
        self.vf_coef = float(profile.get("vf_coef", 0.5))
        self.max_grad_norm = float(profile.get("max_grad_norm", 0.5))

        observation_shape = tuple(np.asarray(env.observation_space.shape))
        action_shape: tuple[int, ...] = (
            () if hasattr(env.action_space, "n") else tuple(env.action_space.shape)
        )

        self.critic = ValueNetwork(
            env.observation_space,
            hidden_sizes=profile.get("hidden_sizes", [64, 64]),
            activation=profile.get("activation_fn"),
        ).to(self.device)
        # critic 单独一个优化器：它的学习率和 actor 未必该相同。
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )
        self.buffer = RolloutBuffer(
            self.n_steps,
            observation_shape,
            action_shape,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
        )

    def collect_rollout(self, callback: Any = None, total_timesteps: int = 0) -> None:
        """收集固定 ``n_steps`` 步经验。

        与 REINFORCE 的关键区别：这里不要求回合结束。轨迹中间被打断也没关系，
        优势估计会用自举把截断处的价值接上。
        """
        self.buffer.reset()
        observation = self._begin_rollout()
        episode_start = 1.0  # 第一次采样的观测算作一个新回合的开始

        for _ in range(self.n_steps):
            if self._stop_training:
                break

            tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            with torch.no_grad():
                distribution = self.policy.distribution(tensor)
                action = distribution.sample()
                log_prob = action_log_prob(distribution, action)
                value = self.critic(tensor).squeeze(-1)

            if hasattr(self.env.action_space, "n"):
                # 离散动作：缓冲区统一用 float32 存，取出时再转回 long。
                action_value: Any = int(action.item())
                stored_action = np.array(float(action_value), dtype=np.float32)
            else:
                action_value = action.squeeze(0).cpu().numpy().astype(np.float32)
                stored_action = np.asarray(action_value, dtype=np.float32)

            next_observation, reward, done, info = self._step_env(action_value)

            self.buffer.add(
                observation=observation,
                action=stored_action,
                reward=reward,
                episode_start=episode_start,
                value=float(value.item()),
                log_prob=float(log_prob.reshape(-1)[0].item()),
            )

            self._stop_training = not self._log_env_step(
                callback,
                {
                    "actions": action_value,
                    "values": value.cpu().numpy(),
                    "log_probs": log_prob.cpu().numpy(),
                    "rewards": reward,
                    "dones": done,
                    "infos": [info],
                    "observations": next_observation,
                },
            )

            observation = next_observation
            # 这一步结束了回合 → 下一步就是新回合的开始。
            episode_start = 1.0 if done else 0.0

        # 轨迹末尾的自举值：用最后一个观测的 V 估计收尾。
        with torch.no_grad():
            last_tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            last_value = float(self.critic(last_tensor).item())
        self.buffer.compute_returns_and_advantages(last_value, episode_start)

    def update(self) -> dict[str, float]:
        """一次梯度更新。"""
        if self.buffer.pos == 0:
            return {}

        observations = self.buffer.all_observations()
        actions = self._actions_to_tensor(self.buffer.get("actions"))
        old_log_probs = self.buffer.all_log_probs()
        advantages = self.buffer.all_advantages()
        returns = self.buffer.all_returns()

        distribution = self.policy.distribution(observations)
        log_probs = action_log_prob(distribution, actions)
        entropy = action_entropy(distribution).mean()

        # actor：优势越大，该动作的概率越该提高。
        # .detach() 很关键——优势只作为「权重」，不该把梯度回传到 critic 上。
        actor_loss = -(log_probs * advantages.detach()).mean()

        values = self.critic(observations).squeeze(-1)
        critic_loss = F.mse_loss(values, returns)

        # 熵是「策略的不确定程度」，我们要最大化它，所以最小化它的负数。
        entropy_bonus = -entropy

        loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_bonus

        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.critic_optimizer.step()

        # 解释方差：优势估计里有多少能被 V 解释。接近 1 说明 critic 学得准，
        # 接近 0 或为负说明 critic 基本没学到东西（稀疏奖励下常见）。
        with torch.no_grad():
            var_returns = returns.var()
            explained_variance = (
                float(1 - advantages.var() / var_returns) if var_returns > 0 else 0.0
            )

        return {
            "train/policy_gradient_loss": float(actor_loss.item()),
            "train/value_loss": float(critic_loss.item()),
            "train/entropy_loss": float(-entropy.item()),
            "train/explained_variance": explained_variance,
            "train/learning_rate": self.learning_rate,
            # 回合回报取自基类的 ep_info_buffer。这里曾经写的是
            # `returns.mean()`——那是**折扣回报**的均值（critic 的 target），
            # 不是回合总回报，两者在 CartPole 上能差一个数量级。
            **self._episode_logs(),
        }

    def _actions_to_tensor(self, actions: np.ndarray) -> torch.Tensor:
        if hasattr(self.env.action_space, "n"):
            return torch.as_tensor(
                np.asarray(actions).reshape(-1).astype(np.int64), device=self.device
            )
        return torch.as_tensor(
            np.asarray(actions, dtype=np.float32), device=self.device
        )

    def _extra_state(self) -> dict[str, Any]:
        """critic 和它的优化器一起存，保证一个文件装下整个算法状态。"""
        return {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def _load_extra(self, extra: dict[str, Any]) -> None:
        if "critic" in extra:
            self.critic.load_state_dict(extra["critic"])
        if "critic_optimizer" in extra:
            self.critic_optimizer.load_state_dict(extra["critic_optimizer"])
