"""环境包装器。

这些包装器各自解决一个具体问题，通过配置里的 ``environment.wrappers`` 列表
按顺序套用，例如::

    environment:
      wrappers: [minigrid_progress, minigrid_flat]

两个奖励类包装器（``MiniGridNoveltyBonus`` / ``MiniGridProgressShaping``）
**只在训练时生效**：评估和轨迹录制会用 ``exploration=False`` 构造环境，
把奖励塑形关掉，这样 ``evaluation.json`` 里的回报和录制的视频反映的都是
未经修饰的任务奖励，而不是「任务奖励 + 探索奖励」。
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MiniGridFlatObservation(gym.ObservationWrapper):
    """保留 MiniGrid 的图像与朝向，去掉文本 mission。

    注意这个包装器**不会**把局部视野变成全局。MiniGrid 的 ``image`` 观测本身
    就是 agent 前方 7x7 的局部视图（``agent_view_size`` 默认 7），这是它的
    部分可观测性来源。这里做的只是把 dict 观测拍平成向量、把 tile id 缩放到
    0~1，好让 MLP 能用；被丢掉的只有 mission 文本。
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        image_space = env.observation_space["image"]
        direction_space = env.observation_space["direction"]
        image_size = int(np.prod(image_space.shape))
        direction_size = int(direction_space.n)
        low = np.zeros(image_size + direction_size, dtype=np.float32)
        high = np.ones(image_size + direction_size, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        image = np.asarray(observation["image"], dtype=np.float32).reshape(-1)
        # MiniGrid tile id 是小整数，缩放到 0~1 有助于 MLP 优化。
        image = image / 10.0
        direction = np.zeros(self.observation_space.shape[0] - image.size, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0
        return np.concatenate([image, direction]).astype(np.float32)


class MiniGridNoveltyBonus(gym.Wrapper):
    """基于计数的探索奖励，状态离散为 ``(x, y, direction)``。

    MiniGrid 可达的状态空间小到可以精确制表，所以用计数比用学出来的 RND
    预测器更合适：不需要梯度更新、没有额外超参、奖励含义可解释。对 DoorKey
    而言关键的一点是计数**包含朝向**——开门需要同时满足特定的位置和朝向，
    对联合状态给奖励才能真正推动智能体去填补没试过的朝向组合。

    计数跨回合累积，因此一个状态被访问得越多奖励越低，智能体持续被推向
    尚未探索的配置。

    **重要：这个方案在 DoorKey-6x6/8x8 上实测是失败的**（300k 步成功率仍为 0）。
    失败原因是结构性的：状态 novelty 只奖励「访问新状态」，不奖励「在该状态下
    按对动作」。智能体可以无数次站在门旁面朝门，却一直按「前进」而不是
    ``toggle``。这里保留它是作为反例——真正解决问题的是下面的
    ``MiniGridProgressShaping``。
    """

    def __init__(
        self, env: gym.Env, *, scale: float = 0.1, enabled: bool = True
    ) -> None:
        super().__init__(env)
        self.scale = float(scale)
        self.enabled = bool(enabled)
        target = env.unwrapped
        width = int(getattr(target, "width", 1)) or 1
        height = int(getattr(target, "height", 1)) or 1
        self._counts = np.zeros((width, height, 4), dtype=np.float64)

    def step(
        self, action: Any
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["task_reward"] = float(reward)
        if not self.enabled:
            return observation, reward, terminated, truncated, info
        target = self.env.unwrapped
        x, y = (int(value) for value in target.agent_pos)
        direction = int(target.agent_dir)
        self._counts[x, y, direction] += 1.0
        bonus = self.scale / math.sqrt(self._counts[x, y, direction])
        info["exploration_bonus"] = float(bonus)
        return observation, float(reward) + bonus, terminated, truncated, info


class MiniGridProgressShaping(gym.Wrapper):
    """为 DoorKey 的不可逆进度事件提供中间奖励。

    状态 novelty 对 DoorKey 不够用：智能体把探索预算花光了，门却始终没开，
    因为 ``pickup`` 和 ``toggle`` 的价值要在几十步之后才兑现。直接给这两个
    里程碑发奖励，信用分配才有东西可以依附。

    这两件事出现在每一条成功轨迹上，所以这个塑形不会改变最优策略是什么。
    和 novelty 一样它只在训练时生效，评估时报告的仍是未塑形的任务奖励。
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        pickup_bonus: float = 0.5,
        door_bonus: float = 0.5,
        enabled: bool = True,
    ) -> None:
        super().__init__(env)
        self.pickup_bonus = float(pickup_bonus)
        self.door_bonus = float(door_bonus)
        self.enabled = bool(enabled)
        self._acquired_key = False
        self._opened_door = False

    def reset(self, **kwargs: Any) -> Any:
        self._acquired_key = False
        self._opened_door = False
        return self.env.reset(**kwargs)

    @staticmethod
    def _any_door_open(target: Any) -> bool:
        for x in range(int(target.width)):
            for y in range(int(target.height)):
                cell = target.grid.get(x, y)
                if cell is not None and cell.type == "door" and cell.is_open:
                    return True
        return False

    def step(
        self, action: Any
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["task_reward"] = float(reward)
        if not self.enabled:
            return observation, reward, terminated, truncated, info

        target = self.env.unwrapped
        bonus = 0.0
        if target.carrying is not None and not self._acquired_key:
            self._acquired_key = True
            bonus += self.pickup_bonus
        if self._acquired_key and not self._opened_door and self._any_door_open(target):
            self._opened_door = True
            bonus += self.door_bonus
        if bonus:
            info["shaping_bonus"] = float(bonus)
        return observation, float(reward) + bonus, terminated, truncated, info
