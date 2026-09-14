"""PPO —— 近端策略优化（Proximal Policy Optimization）。

PPO 是目前最常用的策略梯度算法。它相对 A2C 只多了一个机制——**裁剪（clip）**，
但这个机制解决了 on-policy 方法最要命的效率问题。理解 clip 就理解了 PPO。

## 问题：on-policy 数据只能用一次，太浪费

A2C 收集一批数据、更新一次、然后全部丢掉。但收集数据（跑环境）往往比更新
（算梯度）贵得多。能不能把同一批数据**反复训练几轮**？

不能直接用。因为更新后策略变了，而数据是**旧策略**采出来的。用新策略去算旧数据的
梯度，需要重要性采样修正：

    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)          ← 新旧策略的概率比

于是目标变成 ``E[ r_t · A_t ]``。问题是：**r_t 会失控**。如果某个动作在新策略下
概率变得极小或极大，r_t 会跑到几十上百，梯度爆炸，一次更新就把策略带偏，
之后再怎么训练都救不回来。

## 解决：把 r_t 夹在 [1-ε, 1+ε] 之间

    L_clip = E[ min( r_t·A_t ,  clip(r_t, 1-ε, 1+ε)·A_t ) ]

分两种情况看这个 min：

- **A_t > 0**（这个动作比平均好，想提高它的概率）：r_t 增大会让目标增大，
  但被 clip 卡在 1+ε。也就是说，**一旦把该动作的概率提高了约 ε 倍，
  再提高就不再有梯度**——不贪。
- **A_t < 0**（这个动作比平均差，想降低它的概率）：r_t 减小会让目标增大，
  同样被卡在 1-ε。**一旦降低了约 ε 倍就不再多降**——也不贪。

取 min 的用意是让最终目标始终是**悲观的下界**：只有当新策略确实更好时
（在裁剪范围内）才给梯度，跑出范围就切断。这样反复训练同一批数据几轮
（``n_epochs``）也不会偏离太远，样本效率因此显著提升。

ε 通常取 0.1~0.2。它控制的是"每轮更新允许策略走多远"。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import register
from .a2c import A2C
from .on_policy import action_entropy, action_log_prob


@register
class PPO(A2C):
    """带裁剪的近端策略优化。

    继承 A2C 是因为两者结构几乎相同（都是 actor + critic + GAE），
    差别只在 ``update``：PPO 多一个概率比、一次裁剪，并且把同一批数据
    反复训练 ``n_epochs`` 轮。
    """

    def __init__(self, env: Any, config: dict[str, Any]) -> None:
        super().__init__(env, config)
        profile = self._resolve_profile(config["algorithm"])
        self.clip_range = float(profile.get("clip_range", 0.2))
        # 同一批数据重复训练几轮。这是 PPO 效率提升的来源——
        # 但轮数太多会让策略偏离旧策略太远，即使有 clip 也会过拟合这批数据。
        self.n_epochs = int(profile.get("n_epochs", 10))
        self.batch_size = int(profile.get("batch_size", 64))
        # 是否对优势做标准化。这是 PPO 的常用技巧，能让训练稳定不少。
        self.normalize_advantage = bool(profile.get("normalize_advantage", True))

    def update(self) -> dict[str, float]:
        """多轮次、分批次的裁剪更新。"""
        if self.buffer.pos == 0:
            return {}

        observations = self.buffer.all_observations()
        actions = self._actions_to_tensor(self.buffer.get("actions"))
        old_log_probs = self.buffer.all_log_probs()
        advantages = self.buffer.all_advantages()
        returns = self.buffer.all_returns()

        if self.normalize_advantage and advantages.numel() > 1:
            # 标准化让优势的均值为 0、方差为 1。
            # 这样 A_t 的绝对尺度不影响梯度步长，超参更容易跨环境迁移。
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = {
            "policy_gradient_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        updates = 0

        for _ in range(self.n_epochs):
            # 每轮重新打乱切分，避免同一批样本被反复以相同顺序训练。
            for batch in self.buffer.iterate_minibatches(self.batch_size):
                index = batch["index"]
                batch_observations = batch["observations"]
                batch_actions = actions[index]
                batch_old_log_probs = old_log_probs[index]
                batch_advantages = advantages[index]
                batch_returns = returns[index]

                distribution = self.policy.distribution(batch_observations)
                new_log_probs = action_log_prob(distribution, batch_actions)
                entropy = action_entropy(distribution).mean()

                # 概率比 r_t = π_new / π_old，用对数差再取指数更数值稳定。
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # 未裁剪的替代目标
                surrogate_1 = ratio * batch_advantages
                # 裁剪后的替代目标
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                surrogate_2 = clipped_ratio * batch_advantages
                # 取 min：目标是悲观下界，只有确实变好才给梯度。
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                values = self.critic(batch_observations).squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                self.critic_optimizer.step()

                # 两个诊断指标：
                # approx_kl 衡量新旧策略差多远，涨太快说明步长过大；
                # clip_fraction 是被裁剪的样本比例，长期接近 1 说明 ε 设小了。
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_range).float().mean()
                    )
                totals["policy_gradient_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy_loss"] += float(-entropy.item())
                totals["approx_kl"] += float(approx_kl.item())
                totals["clip_fraction"] += float(clip_fraction.item())
                updates += 1

        if updates == 0:
            return {}
        logs = {f"train/{key}": value / updates for key, value in totals.items()}
        logs["train/learning_rate"] = self.learning_rate
        logs["rollout/ep_rew_mean"] = float(
            self.buffer.get("rewards").mean() if self.buffer.pos else 0.0
        )
        return logs
