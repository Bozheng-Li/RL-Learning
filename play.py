"""统一的策略演示入口：加载训练好的模型，录制轨迹。

用法::

    python play.py --config cartpole
    python play.py --config minigrid_doorkey_hard --model outputs/minigrid_doorkey_hard/best_model/best_model.zip

默认加载运行目录里的最终模型；用 ``--model`` 可以指定任意模型文件，
比如回放某个 checkpoint 看训练中途的策略长什么样::

    python play.py --config cartpole --model outputs/cartpole/checkpoints/cartpole_10000_steps.pt

注意：如果训练时用 ``--set`` 覆盖过输出目录，这里要传同样的覆盖，
否则会去原始配置的目录里找结果。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from rl_common import visualize_from_config  # noqa: E402
from rl_common.cli import add_common_arguments, collect_overrides  # noqa: E402
from rl_common.config import resolve_config_path  # noqa: E402


def main() -> None:
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
    )
    parser.add_argument("--model", type=Path, help="指定模型文件，默认用运行目录里的最终模型")
    args = parser.parse_args()

    visualize_from_config(
        resolve_config_path(args.config),
        include_training=False,
        include_trajectories=True,
        model_path=args.model,
        overrides=collect_overrides(args),
    )


if __name__ == "__main__":
    main()
