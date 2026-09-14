"""REINFORCE —— 最基础的策略梯度算法。

这是学习策略梯度的起点，因为它只有一行核心公式，没有任何额外的组件。

## 核心思想

策略 π_θ(a|s) 是一个带参数 θ 的分布。我们希望调整 θ，让**高回报轨迹里出现过的
动作更可能被选中**。策略梯度定理给出了梯度的精确形式：

    ∇J(θ) = E[ Σ_t  ∇log π_θ(a_t|s_t) · G_t ]

其中 G_t 是从第 t 步开始到回合结束的折扣回报。

直观理解这一项 ``∇log π(a|s) · G``：它沿着「提高该动作概率」的方向 ∂log π/∂θ 走，
走得远近由回报 G 决定。回报高就走一大步，回报低（或为负）就反向走。
所以整个过程就是「试错后强化做得好的动作」。

## 三个实现细节

**为什么要取 log**：∇log π 而不是 ∇π。因为 log 的梯度可以用采样估计，
而 ∇π 需要遍历所有可能的动作（对连续动作空间根本做不到）。

**为什么加负号**：框架做的是「最小化损失」，而我们要「最大化回报」，
所以把 ``-log_prob * G`` 当作损失，最小化它就等于最大化目标。

**为什么必须是完整回合**：G_t 是从 t 到回合结束的实际累计回报，所以必须等回合
结束才知道。这带来两个后果：(1) 只能用于有终止状态的任务；(2) 方差很大——
同一条轨迹的运气成分完全体现在 G 里。这正是 A2C 要解决的问题。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import register
from .buffers import compute_returns_to_go
from .on_policy import OnPolicyAlgorithm, action_entropy, action_log_prob


@register
class REINFORCE(OnPolicyAlgorithm):
    """蒙特卡洛策略梯度。

    与后面两个算法的关系：A2C = REINFORCE + 基线 + 自举，PPO = A2C + 裁剪。
    先把它跑通，再看每一步改进解决了什么问题，比直接读 PPO 清楚得多。
    """

    # REINFORCE 的超参就三个，配置里没写 profile 时用这组默认值即可。
    require_profile = False
    default_profile = {
        "gamma": 0.99,
        "learning_rate": 0.001,
        "episodes_per_update": 4,
    }

    def __init__(self, env: Any, config: dict[str, Any]) -> None:
        super().__init__(env, config)
        profile = self._resolve_profile(config["algorithm"])
        # 攒够多少个回合再更新一次。梯度是回合级的，攒几个能降低方差。
        self.episodes_per_update = int(profile.get("episodes_per_update", 4))
        # 当前正在收集的回合：(观测, 动作, 该动作的 log 概率, 奖励)
        self._trajectory: list[tuple[np.ndarray, Any, float, float]] = []
        # 本次更新要用的所有完整回合
        self.trajectories: list[list[tuple[np.ndarray, Any, float, float]]] = []

    def collect_rollout(self, callback: Any = None, total_timesteps: int = 0) -> None:
        """收集 ``episodes_per_update`` 个完整回合。

        注意这里**必须**走完整回合，不能像 A2C 那样固定步数截断——REINFORCE 用的
        是蒙特卡洛回报，回合没结束就不算知道 G_t。
        """
        self.trajectories: list[list[tuple[np.ndarray, Any, float, float]]] = []
        collected_episodes = 0

        while collected_episodes < self.episodes_per_update and not self._stop_training:
            observation = self._begin_rollout()
            self._trajectory = []
            done = False

            while not done and not self._stop_training:
                # 采样动作，并记下它的 log 概率——梯度里要用。
                tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    distribution = self.policy.distribution(tensor)
                    action = distribution.sample()
                    log_prob = action_log_prob(distribution, action)

                if hasattr(self.env.action_space, "n"):
                    action_value: Any = int(action.item())
                else:
                    action_value = action.squeeze(0).cpu().numpy().astype(np.float32)

                next_observation, reward, done, info = self._step_env(action_value)
                self._trajectory.append(
                    (observation, action_value, float(log_prob.reshape(-1)[0].item()), reward)
                )
                # 通知回调；callback 里挂着周期评估和 checkpoint 保存。
                self._stop_training = not self._log_env_step(
                    callback,
                    {
                        "actions": action_value,
                        "rewards": reward,
                        "dones": done,
                        "infos": [info],
                        "observations": next_observation,
                    },
                )
                observation = next_observation

            if self._trajectory:
                self.trajectories.append(self._trajectory)
                collected_episodes += 1

    def update(self) -> dict[str, float]:
        """用蒙特卡洛回报做一次策略梯度更新。

        对每个回合算出 ``G_t``，然后最小化 ``-Σ log π(a_t|s_t) · G_t``。
        """
        observations: list[np.ndarray] = []
        actions: list[Any] = []
        log_probs: list[float] = []
        returns: list[float] = []

        for trajectory in self.trajectories:
            rewards = [step[3] for step in trajectory]
            # 折扣回报，从后往前累加（见 buffers.compute_returns_to_go）。
            discounted = compute_returns_to_go(rewards, self.gamma)
            for (observation, action, log_prob, _), g_t in zip(trajectory, discounted):
                observations.append(observation)
                actions.append(action)
                log_probs.append(log_prob)
                returns.append(float(g_t))

        if not returns:
            return {}

        obs_tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32), device=self.device
        )
        action_tensor = self._actions_to_tensor(actions)
        log_prob_tensor = torch.as_tensor(
            log_probs, dtype=torch.float32, device=self.device
        )
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # 用「采样时的动作」重新算一遍 log 概率。
        # 看起来多余（采样时就有一个），但那时它带着计算图，而且参数已经更新过。
        # 重新前向一次才是当前参数下的正确梯度。
        distribution = self.policy.distribution(obs_tensor)
        current_log_probs = action_log_prob(distribution, action_tensor)

        # 策略梯度损失：负号是因为要最大化回报。
        policy_loss = -(current_log_probs * return_tensor).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        entropy = action_entropy(distribution).mean()
        return {
            "train/policy_gradient_loss": float(policy_loss.item()),
            "train/entropy_loss": float(entropy.item()),
            "rollout/ep_rew_mean": float(np.mean(returns)),
        }

    def _actions_to_tensor(self, actions: list[Any]) -> torch.Tensor:
        if hasattr(self.env.action_space, "n"):
            return torch.as_tensor(actions, dtype=torch.long, device=self.device)
        return torch.as_tensor(
            np.asarray(actions, dtype=np.float32), device=self.device
        )
