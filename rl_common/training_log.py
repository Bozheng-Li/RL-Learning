"""训练日志系统。

除了 SB3 自动产出的 `progress.csv` 与 Monitor 的 `*.monitor.csv`，这里补上两类
现有工具覆盖不到、但排查问题时最需要的信息：

**``logs/episodes.csv``** —— 每个 episode 一行的明细。Monitor 的 CSV 只有
``r,l,t`` 三列，缺 episode 编号与累计步数的对齐，做跨运行对比时不好用。

**``logs/diagnostics.csv``** —— 周期性记录**动作分布**。

第二项是带着教训加的。排查 DoorKey-8x8 为什么学不会时，决定性证据是训练后策略的
动作分布：``pickup`` 占 37.9%、``drop`` 占 31.9%、``toggle`` 占 **0.0%**。
这直接说明问题是策略坍缩而不是探索不足。但当时没有现成工具，只能临时写脚本。
现在它在训练过程中自动记录，零额外开销——统计的是 rollout 里策略真实采样出的动作，
不需要为了诊断再跑一遍推理。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

#: progress.csv 的列。自研算法按这个顺序写，与 SB3 的列名保持一致，
#: 这样可视化和跨运行对比可以用同一套读取代码。
PROGRESS_HEADER = [
    "time/total_timesteps",
    "time/fps",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/explained_variance",
    "train/approx_kl",
    "train/clip_fraction",
    "train/learning_rate",
]


class ProgressWriter:
    """写 ``logs/progress.csv``。

    自研算法用它记录每个 rollout 的训练信号。SB3 算法不需要——SB3 的 logger
    会自己写这个文件，列名刻意保持一致。
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=PROGRESS_HEADER)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(
            {key: row.get(key, "") for key in PROGRESS_HEADER}
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class EpisodeLogCallback(BaseCallback):
    """把每个 episode 的明细写进 ``logs/episodes.csv``。

    数据来源是 Monitor 包装器塞进 ``info["episode"]`` 的统计，所以只需要读
    ``self.locals["infos"]``，不产生额外开销。
    """

    def __init__(self, config: dict[str, Any], run_dir: Path) -> None:
        super().__init__()
        self.path = run_dir / "logs" / "episodes.csv"
        self._handle: Any = None
        self._writer: Any = None
        self._episode_index = 0

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=["episode", "timestep", "return", "length", "elapsed"],
        )
        self._writer.writeheader()

    def _on_step(self) -> bool:
        # Monitor 在 episode 结束时往 info 里塞 "episode" 字段。
        for info in self.locals.get("infos", []) or []:
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode is None:
                continue
            self._writer.writerow(
                {
                    "episode": self._episode_index,
                    "timestep": int(self.num_timesteps),
                    "return": float(episode["r"]),
                    "length": int(episode["l"]),
                    "elapsed": float(episode["t"]),
                }
            )
            self._episode_index += 1
        return True

    def _on_training_end(self) -> None:
        if self._handle is not None:
            self._handle.close()


class DiagnosticsCallback(BaseCallback):
    """周期性记录动作分布与策略统计到 ``logs/diagnostics.csv``。

    动作分布直接从 rollout 里累积，不额外跑推理。对离散动作空间记录每个动作的
    采样占比；对连续动作空间记录均值与标准差。
    """

    def __init__(
        self, config: dict[str, Any], run_dir: Path, *, interval: int = 10_000
    ) -> None:
        super().__init__()
        self.path = run_dir / "logs" / "diagnostics.csv"
        self.interval = max(1, int(interval))
        self._handle: Any = None
        self._writer: Any = None
        self._counts: dict[int, int] = {}
        self._total = 0
        self._action_sum: np.ndarray | None = None
        self._action_sq_sum: np.ndarray | None = None
        self._action_count = 0
        self._last_write = 0

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        header = ["timestep", "samples", "action_space"]
        header += [f"action_{index}_frac" for index in range(self._action_dim())]
        self._writer = csv.DictWriter(self._handle, fieldnames=header + [
            "action_mean", "action_std",
        ])
        self._writer.writeheader()

    def _action_dim(self) -> int:
        """离散动作空间的动作个数；连续空间返回 1 作为占位。"""
        space = self.training_env.action_space
        if hasattr(space, "n"):
            return int(space.n)
        return 1

    def _is_discrete(self) -> bool:
        return hasattr(self.training_env.action_space, "n")

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            array = np.asarray(actions)
            if self._is_discrete():
                for value in array.reshape(-1):
                    index = int(value)
                    self._counts[index] = self._counts.get(index, 0) + 1
                    self._total += 1
            else:
                flat = array.reshape(len(array), -1).astype(np.float64)
                if self._action_sum is None:
                    self._action_sum = np.zeros(flat.shape[1])
                    self._action_sq_sum = np.zeros(flat.shape[1])
                self._action_sum += flat.sum(axis=0)
                self._action_sq_sum += (flat**2).sum(axis=0)
                self._action_count += flat.shape[0]

        if self.num_timesteps - self._last_write >= self.interval:
            self._record()
        return True

    def _record(self) -> None:
        row: dict[str, Any] = {
            "timestep": int(self.num_timesteps),
            "action_space": "discrete" if self._is_discrete() else "continuous",
        }
        if self._is_discrete():
            row["samples"] = self._total
            for index in range(self._action_dim()):
                count = self._counts.get(index, 0)
                row[f"action_{index}_frac"] = (
                    count / self._total if self._total else 0.0
                )
        else:
            row["samples"] = self._action_count
            if self._action_count and self._action_sum is not None:
                mean = self._action_sum / self._action_count
                var = self._action_sq_sum / self._action_count - mean**2
                row["action_mean"] = float(np.mean(mean))
                row["action_std"] = float(np.mean(np.sqrt(np.maximum(var, 0.0))))
        self._writer.writerow(row)
        self._handle.flush()
        # 统计窗口清零，下一条记录反映的是最近这个区间的分布。
        self._counts.clear()
        self._total = 0
        self._action_sum = None
        self._action_sq_sum = None
        self._action_count = 0
        self._last_write = self.num_timesteps

    def _on_training_end(self) -> None:
        if self._handle is not None:
            if self._total or self._action_count:
                self._record()
            self._handle.close()
