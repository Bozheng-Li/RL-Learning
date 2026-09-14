"""经验缓冲区与优势估计。

``RolloutBuffer`` 存放一段 rollout 的经验，并负责算 **GAE（广义优势估计）**。
这是 A2C 和 PPO 共用的核心数据结构，也是理解这两个算法最关键的一环。

## 为什么需要优势，而不是直接用回报

策略梯度的原始形式是「用回报 G_t 加权 log 概率」：

    ∇J ≈ E[ G_t · ∇log π(a_t|s_t) ]

问题在于 G_t 的**方差极大**——同一个状态，运气不同回报可能天差地别，导致梯度噪声
很大。解决办法是减去一个基线 b(s)：

    ∇J ≈ E[ (G_t - b(s_t)) · ∇log π(a_t|s_t) ]

减去基线**不改变梯度的期望**（因为 E[b(s)·∇log π] = 0），但能显著降低方差。
最常用的基线就是价值函数 V(s)，于是 ``G_t - V(s_t)`` 就是**优势** A(s_t, a_t)：
"这个动作比平均水平好多少"。

## GAE 在做什么

直接算 A_t = G_t - V(s_t) 仍然要用到整条轨迹的回报，方差大。GAE 用一个可调的
参数 λ ∈ [0,1] 在**偏差与方差之间插值**：

    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)          ← 单步 TD 误差
    A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...   ← 指数加权的 TD 误差和

- **λ = 0**：A_t = δ_t，只用一步自举。偏差大（依赖 V 的准确性）但方差小。
- **λ = 1**：A_t = G_t - V(s_t)，退化成蒙特卡洛。方差大但无偏。
- **0 < λ < 1**：折中，实践中通常取 0.95 左右。

实现上是从轨迹**末尾倒着递推**，见 ``compute_returns_and_advantages``。
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


class RolloutBuffer:
    """定长的 rollout 缓冲区，带 GAE 优势估计。

    生命周期是「填满 → 算优势 → 取批次更新 → 清空」，如此循环。
    """

    def __init__(
        self,
        size: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        *,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> None:
        """
        Args:
            size: 能容纳多少步经验（对应算法的 ``n_steps``）。
            observation_shape: 单个观测的形状。
            action_shape: 单个动作的形状（离散动作是 ``()``）。
            gamma: 折扣因子 γ。
            gae_lambda: GAE 的 λ。
            device: 张量所在设备。
        """
        self.size = int(size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.device = device
        self.observation_shape = tuple(observation_shape)
        self.action_shape = tuple(action_shape)

        self.observations = np.zeros((self.size, *self.observation_shape), dtype=np.float32)
        self.actions = np.zeros((self.size, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.size,), dtype=np.float32)
        # episode_starts[t] 标记第 t 步是否是一个新回合的第一步。
        # GAE 靠它切断跨回合的自举——上一回合的终局奖励不该被算进下一回合。
        self.episode_starts = np.zeros((self.size,), dtype=np.float32)
        self.values = np.zeros((self.size,), dtype=np.float32)
        self.log_probs = np.zeros((self.size,), dtype=np.float32)

        self.advantages = np.zeros((self.size,), dtype=np.float32)
        self.returns = np.zeros((self.size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def reset(self) -> None:
        """清空缓冲区，准备收集下一段 rollout。"""
        self.pos = 0
        self.full = False

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        episode_start: float,
        value: float,
        log_prob: float,
    ) -> None:
        """存入一步经验。"""
        if self.pos >= self.size:
            raise IndexError("RolloutBuffer 已满，应当先 compute_returns_and_advantages")
        self.observations[self.pos] = np.asarray(observation, dtype=np.float32)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32)
        self.rewards[self.pos] = float(reward)
        self.episode_starts[self.pos] = float(episode_start)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)
        self.pos += 1
        if self.pos == self.size:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value: float, last_episode_start: float
    ) -> None:
        """用 GAE 算出每一步的优势与回报，就地写入缓冲区。

        从末尾倒着递推。每一步：

            δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
            A_t = δ_t + γλ·(1-done)·A_{t+1}

        其中 ``(1-done)`` 是终止掩码：回合结束后就没有"下一个状态"了，
        自举项必须归零，否则会把下一个回合的价值错误地算进来。

        Args:
            last_value: 最后一步之后那个状态的 V 估计，用于给末端的自举收尾。
            last_episode_start: 最后一步之后是否紧跟着新回合的开始。
        """
        last_gae = 0.0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                # 边界情况：轨迹末尾要用外部传入的 last_value 自举。
                next_non_terminal = 1.0 - last_episode_start
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_value = self.values[step + 1]

            # 单步 TD 误差
            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )
            # 指数加权的历史 TD 误差和
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        # 回报 = 优势 + 价值。PPO 用回报训练价值网络，
        # 这样价值网络的回归目标就带上了实际观测到的奖励信息。
        self.returns = self.advantages + self.values

    def get(self, key: str) -> np.ndarray:
        """取出某个字段的数组。"""
        return getattr(self, key)[: self.pos]

    def iterate_minibatches(
        self, batch_size: int, *, shuffle: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        """把整段 rollout 随机切成若干小批次，供 PPO 的多轮更新使用。

        PPO 的典型流程是「同一批数据反复训练若干轮（n_epochs）」，每轮都重新打乱
        切分，这样一次采样能产生多次梯度更新，样本效率比只更新一次高得多。

        Yields:
            每批数据的张量字典，键名与缓冲区字段一致，额外附上 ``advantages`` /
            ``returns``，以及 ``index`` —— 该批次在整段 rollout 中的原始下标，
            调用方用它去取其它按位置对齐的数组（如动作、旧 log 概率）。
        """
        indices = np.arange(self.pos)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.pos, batch_size):
            chunk = indices[start : start + batch_size]
            if chunk.size == 0:
                continue
            yield {
                "index": torch.as_tensor(chunk, dtype=torch.long, device=self.device),
                "observations": self._to_tensor(self.observations[chunk]),
                "actions": self._to_tensor(self.actions[chunk]),
                "log_probs": self._to_tensor(self.log_probs[chunk]),
                "values": self._to_tensor(self.values[chunk]),
                "advantages": self._to_tensor(self.advantages[chunk]),
                "returns": self._to_tensor(self.returns[chunk]),
            }

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        # 离散动作存成 (batch,)，网络要的是形状正确的张量；这里统一保证至少一维。
        return tensor

    # ---- 便于算法读取整段数据的便捷属性 ----

    def all_observations(self) -> torch.Tensor:
        return self._to_tensor(self.observations[: self.pos])

    def all_actions(self) -> torch.Tensor:
        # 离散动作需要切成 long 才能喂给 Categorical 算 log_prob。
        return torch.as_tensor(
            self.actions[: self.pos], dtype=torch.float32, device=self.device
        )

    def all_log_probs(self) -> torch.Tensor:
        return self._to_tensor(self.log_probs[: self.pos])

    def all_advantages(self) -> torch.Tensor:
        return self._to_tensor(self.advantages[: self.pos])

    def all_returns(self) -> torch.Tensor:
        return self._to_tensor(self.returns[: self.pos])


def compute_returns_to_go(rewards: list[float], gamma: float) -> np.ndarray:
    """蒙特卡洛折扣回报，用于 REINFORCE。

    定义是 ``G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T``，
    即"从第 t 步开始到回合结束，实际拿到的折扣总回报"。

    REINFORCE 用的就是它——不做自举、不依赖价值函数，因此是**无偏但高方差**的。
    对比 PPO 用的 GAE（有偏但低方差），能直观看到偏差-方差权衡在代码上的样子。

    实现上从末尾倒着累加，这样是 O(T) 而不是 O(T²)。

    Args:
        rewards: 一条完整回合的每步奖励。
        gamma: 折扣因子。

    Returns:
        与 rewards 等长的数组，第 t 项是 G_t。
    """
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for step in reversed(range(len(rewards))):
        running = rewards[step] + gamma * running
        returns[step] = running
    return returns
