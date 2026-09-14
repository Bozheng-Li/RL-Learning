"""命令行参数的公共部分。

三个入口（``train.py`` / ``play.py`` / ``visualize.py``）共享同一套参数，这样
「训练时怎么指定的，演示和可视化时就能照抄同一行」——否则用 ``--set`` 改了输出目录
之后，``play.py`` 会去找原始配置里的目录而报「训练结果不存在」，很让人困惑。
"""

from __future__ import annotations

import argparse
from typing import Any

import yaml


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """加上三个入口共用的参数。"""
    parser.add_argument(
        "--config",
        required=True,
        help="配置名（如 cartpole）或 YAML 路径（如 config/cartpole.yaml）",
    )
    parser.add_argument("--algorithm", help="覆盖 algorithm.name，用于算法对照实验")
    parser.add_argument("--seed", type=int, help="覆盖 experiment.seed，用于多种子验证")
    parser.add_argument("--timesteps", type=int, help="覆盖 training.total_timesteps")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="任意点号路径覆盖，可重复。例如 --set output.directory=outputs/my_run",
    )
    return parser


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """把命令行参数整理成点号路径覆盖表。

    值的类型用 YAML 语法推断，所以 ``2000`` 是 int、``true`` 是 bool、
    ``[1, 2]`` 是列表。
    """
    overrides: dict[str, Any] = {}
    if getattr(args, "algorithm", None):
        overrides["algorithm.name"] = args.algorithm
    if getattr(args, "seed", None) is not None:
        overrides["experiment.seed"] = args.seed
    if getattr(args, "timesteps", None) is not None:
        overrides["training.total_timesteps"] = args.timesteps
    for item in getattr(args, "set", []) or []:
        key, separator, raw = item.partition("=")
        if not separator:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到 {item!r}")
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides
