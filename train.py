"""统一的训练入口。

以前每个环境目录下都有一份几乎相同的 ``train.py``（11 份，只有配置路径不同）。
现在由 ``--config`` 决定跑什么，新增实验只需要加一份 YAML。

用法::

    # 最简：跑配置里写好的环境与算法
    python train.py --config cartpole

    # 做对照实验：同一环境换算法、换种子，不用改 YAML
    python train.py --config cartpole --algorithm SB3-PPO --seed 0
    python train.py --config cartpole --algorithm SB3-DQN --seed 0

    # 自研实现 vs SB3 实现（同名 profile，同超参，只换实现）
    python train.py --config cartpole --algorithm PPO --set output.directory=outputs/ppo_native
    python train.py --config cartpole --algorithm SB3-PPO --set output.directory=outputs/ppo_sb3

    # 任意深度的覆盖（值按 YAML 语法解析，所以能写列表和布尔）
    python train.py --config minigrid_doorkey_hard --set training.total_timesteps=2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from rl_common import train_from_config  # noqa: E402
from rl_common.cli import add_common_arguments, collect_overrides  # noqa: E402
from rl_common.config import resolve_config_path  # noqa: E402


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
    )
    args = parser.parse_args()
    train_from_config(
        resolve_config_path(args.config), overrides=collect_overrides(args)
    )


if __name__ == "__main__":
    main()
