"""策略网络与价值网络。

三个算法（REINFORCE / A2C / PPO）共用这里的网络定义，区别只在于怎么用它们算损失。

**策略网络的输出**取决于动作空间的类型，这是连续控制与离散控制最本质的区别：

- 离散动作（CartPole、MiniGrid）：网络输出每个动作的 **logits**，动作从
  ``Categorical(logits)`` 采样。所谓"策略"就是一个分类分布。
- 连续动作（Pendulum）：网络输出一个**高斯分布**的均值 ``mu`` 和标准差 ``sigma``，
  动作从 ``Normal(mu, sigma)`` 采样。标准差在训练中会逐渐变小，策略随之变确定。

**价值网络**输出 ``V(s)``：从状态 s 出发、按当前策略行动的期望折扣回报。它不直接
决定动作，只用来做基线（A2C）或算优势（PPO），降低策略梯度的方差。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical, Normal

#: YAML 里可以写的激活函数名。
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def resolve_activation(name: str | None) -> type[nn.Module]:
    if name is None:
        return nn.Tanh
    try:
        return ACTIVATIONS[str(name).lower()]
    except KeyError as error:
        raise ValueError(
            f"不支持的激活函数 {name!r}；可选：{', '.join(sorted(ACTIVATIONS))}"
        ) from error


def mlp(
    sizes: list[int], activation: type[nn.Module], *, output_activation: bool = False
) -> nn.Sequential:
    """搭一个全连接网络。

    Args:
        sizes: 每层的维度，例如 ``[obs_dim, 64, 64, out_dim]``。
        activation: 隐藏层激活函数。
        output_activation: 是否给最后一层也加激活。策略网络的最后一层不能加——
            激活会把 logits 或 mu 限制在值域内，破坏表达能力。
    """
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        is_last = index == len(sizes) - 2
        if not is_last or output_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


def _flatten_dim(space: spaces.Space) -> int:
    """把观测空间的形状压成一维长度。"""
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    raise ValueError(
        f"自研算法目前只支持 Box 观测空间（向量或图像），收到 {type(space).__name__}"
    )


class ImageExtractor(nn.Module):
    """图像观测的特征提取器（简易 CNN）。

    Atari / CarRacing 这类环境给的是 ``(H, W, C)`` 的 uint8 图像。这里用一个小卷积塔
    把它压成向量再送进 MLP。不做任何花哨的架构——目的是让自研算法在这些环境上也
    跑得起来，性能对标 SB3 的 ``CnnPolicy`` 不是本阶段的重点。
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128) -> None:
        super().__init__()
        # Gymnasium 的图像是 (H, W, C)，PyTorch 的卷积要 (C, H, W)，所以先转置。
        channels = int(observation_space.shape[-1])
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 卷积输出维度要跑一次才知道，用一个假输入探出来。
        with torch.no_grad():
            sample = torch.zeros(1, channels, *observation_space.shape[:2])
            conv_out = int(self.conv(sample).shape[1])
        self.head = nn.Sequential(nn.Linear(conv_out, features_dim), nn.ReLU())
        self.features_dim = features_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 观测是 uint8 的 (H, W, C)，转成 float 并换到 (C, H, W)。
        x = obs.float().permute(0, 3, 1, 2) / 255.0
        return self.head(self.conv(x))


def build_feature_extractor(
    observation_space: spaces.Box, features_dim: int
) -> tuple[nn.Module, int]:
    """按观测类型返回 (特征提取器, 输出维度)。

    三维观测当作图像处理，一维当作状态向量处理。
    """
    if len(observation_space.shape) == 3:
        extractor = ImageExtractor(observation_space, features_dim)
        return extractor, features_dim
    dim = _flatten_dim(observation_space)
    return nn.Flatten(), dim


class DiscretePolicy(nn.Module):
    """离散动作策略：输出每个动作的 logits，动作从分类分布采样。

    训练时用 ``sample()`` 采样（保持探索），评估时用 ``argmax``（取概率最大的动作）。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_actions: int,
        *,
        hidden_sizes: list[int] = (64, 64),
        activation: str | None = None,
        features_dim: int = 128,
    ) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.extractor, feat_dim = build_feature_extractor(observation_space, features_dim)
        # 最后一层不加激活：logits 可以是任意实数。
        self.net = mlp([feat_dim, *hidden_sizes, n_actions], act)
        self.n_actions = n_actions
        self._init_weights()

    def _init_weights(self) -> None:
        # 正交初始化 + 小输出增益是策略梯度的常规做法：初始策略接近均匀分布，
        # 避免一开始就过于自信而抑制探索。
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.constant_(last.bias, 0.0)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(self.extractor(obs))
        return Categorical(logits=logits)

    def forward(self, obs: torch.Tensor) -> Categorical:
        return self.distribution(obs)


class ContinuousPolicy(nn.Module):
    """连续动作策略：输出高斯分布的均值与标准差。

    实现要点：

    - 网络只输出 ``mu``；``sigma`` 是一个**独立的可学习参数**（不依赖输入），
      这是连续控制里的常见做法，比让网络同时输出 mu 和 sigma 更稳定。
    - ``sigma`` 经过 ``softplus`` 保证为正，且被夹在 ``[min_std, max_std]`` 之间，
      防止训练中出现极小方差导致 ``log_prob`` 爆炸。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_dim: int,
        *,
        hidden_sizes: list[int] = (64, 64),
        activation: str | None = None,
        features_dim: int = 128,
        log_std_init: float = -0.5,
        min_std: float = 1e-3,
        max_std: float = 10.0,
    ) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.extractor, feat_dim = build_feature_extractor(observation_space, features_dim)
        self.net = mlp([feat_dim, *hidden_sizes, action_dim], act)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.action_dim = action_dim
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mu = self.net(self.extractor(obs))
        std = torch.exp(self.log_std.clamp(np.log(self.min_std), np.log(self.max_std)))
        # 把 std 广播到和 mu 同形状，Normal 要求两者可广播。
        return Normal(mu, std.expand_as(mu))

    def forward(self, obs: torch.Tensor) -> Normal:
        return self.distribution(obs)


class ValueNetwork(nn.Module):
    """状态价值网络 V(s)。

    输出是标量，**不加激活**：价值可以是任意实数（MountainCar 的回报就是负的）。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        hidden_sizes: list[int] = (64, 64),
        activation: str | None = None,
        features_dim: int = 128,
    ) -> None:
        super().__init__()
        act = resolve_activation(activation)
        self.extractor, feat_dim = build_feature_extractor(observation_space, features_dim)
        self.net = mlp([feat_dim, *hidden_sizes, 1], act)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 返回 (batch, 1)，调用方一般会 squeeze 成 (batch,)。
        return self.net(self.extractor(obs))


def build_policy(
    observation_space: spaces.Box,
    action_space: spaces.Space,
    *,
    hidden_sizes: list[int] = (64, 64),
    activation: str | None = None,
) -> nn.Module:
    """按动作空间类型构造合适的策略网络。"""
    if isinstance(action_space, spaces.Discrete):
        return DiscretePolicy(
            observation_space,
            int(action_space.n),
            hidden_sizes=list(hidden_sizes),
            activation=activation,
        )
    if isinstance(action_space, spaces.Box):
        return ContinuousPolicy(
            observation_space,
            int(np.prod(action_space.shape)),
            hidden_sizes=list(hidden_sizes),
            activation=activation,
        )
    raise ValueError(
        f"自研算法不支持动作空间 {type(action_space).__name__}；"
        "MultiDiscrete / Dict 之类请改用 SB3 实现。"
    )
