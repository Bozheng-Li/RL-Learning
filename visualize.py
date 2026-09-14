"""统一的可视化入口：不重新训练，重画训练曲线、重录轨迹。

用法::

    python visualize.py --config cartpole               # 图和轨迹都重做
    python visualize.py --config cartpole --training-only
    python visualize.py --config cartpole --trajectory-only

改过绘图逻辑（颜色、平滑窗口、子图布局）之后用这个重新出图，不必重训。

注意：如果训练时用 ``--set`` 覆盖过输出目录，这里要传同样的覆盖。
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
    parser.add_argument("--training-only", action="store_true", help="只重画训练曲线")
    parser.add_argument("--trajectory-only", action="store_true", help="只重录轨迹")
    args = parser.parse_args()
    if args.training_only and args.trajectory_only:
        parser.error("--training-only 与 --trajectory-only 互斥")

    visualize_from_config(
        resolve_config_path(args.config),
        include_training=not args.trajectory_only,
        include_trajectories=not args.training_only,
        model_path=args.model,
        overrides=collect_overrides(args),
    )


if __name__ == "__main__":
    main()
