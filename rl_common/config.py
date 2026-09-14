"""配置加载、路径解析与运行目录管理。

配置是整个项目的唯一入口：选哪个环境、用哪个算法、超参是什么、结果写到哪里，
全部由 YAML 决定。这一层只负责把这些信息读进来并解析成可用的路径。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

#: 项目根目录（本文件位于 <root>/rl_common/config.py）。
REPO_ROOT = Path(__file__).resolve().parents[1]

#: 配置里必须存在的顶层段，缺任何一个都在加载时报错而不是等到训练中途。
REQUIRED_SECTIONS = ("experiment", "environment", "algorithm", "training", "output")

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取并校验 YAML 配置。

    Args:
        config_path: 配置文件的路径。

    Returns:
        解析后的配置字典，额外带一个 ``_config_path`` 键记录来源路径
        （重新可视化时靠它定位运行目录）。

    Raises:
        ValueError: 文件内容不是映射，或缺少必需的段。
    """
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    config["_config_path"] = str(path)
    return config


def resolve_repo_path(value: str | Path) -> Path:
    """把配置里的相对路径解析成相对项目根目录的绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_config_path(value: str | Path) -> Path:
    """把命令行给的配置名解析成实际路径。

    三种写法都支持::

        config/cartpole.yaml   相对项目根的路径
        /abs/path/to/x.yaml    绝对路径
        cartpole               简写，自动补成 config/cartpole.yaml

    Raises:
        FileNotFoundError: 都找不到时，列出 config/ 下可用的配置。
    """
    path = Path(value)
    if path.is_file():
        return path
    candidate = REPO_ROOT / "config" / f"{value}.yaml"
    if candidate.is_file():
        return candidate
    # 兜底再试一次相对项目根的路径（用户可能在别的工作目录下执行）。
    candidate = resolve_repo_path(value)
    if candidate.is_file():
        return candidate

    available = sorted(p.stem for p in (REPO_ROOT / "config").glob("*.yaml"))
    raise FileNotFoundError(
        f"找不到配置 {value!r}。可用配置：{', '.join(available)}"
    )


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    """按点号路径就地覆盖配置项。

    做对照实验时改的参数往往只有一个，反复编辑 YAML 很别扭。命令行传覆盖即可::

        {"algorithm.name": "SB3-PPO", "experiment.seed": 0}

    **路径必须已经存在**，不允许新建键。这是刻意的：``--set training.total_time_steps=100``
    这种拼写错误如果被静默接受，训练会照常用默认值跑，而用户以为改过了。
    报错信息会列出该层可用的键，便于对照。

    Args:
        config: 已解析的配置，会被就地修改。
        overrides: 点号路径 -> 新值。

    Raises:
        KeyError: 路径中有任何一段不存在。
    """
    for dotted_path, value in overrides.items():
        keys = dotted_path.split(".")
        target: Any = config
        for depth, key in enumerate(keys):
            if not isinstance(target, dict) or key not in target:
                prefix = ".".join(keys[:depth]) or "<root>"
                available = ", ".join(sorted(target)) if isinstance(target, dict) else "-"
                raise KeyError(
                    f"无法覆盖 {dotted_path!r}：{prefix} 下没有 {key!r}"
                    f"（可用：{available}）"
                )
            if depth == len(keys) - 1:
                target[key] = value
                logger.debug("配置覆盖 %s = %r", dotted_path, value)
            else:
                target = target[key]


def apply_torch_threads(config: dict[str, Any]) -> None:
    """按 ``experiment.torch_threads`` 限制 torch 的线程数。

    torch 默认按 CPU 核心数开线程。对这里用的小型 MLP/LSTM 策略来说这是**负优化**：
    跨线程同步的开销超过并行带来的收益。在 256 核机器上实测，RecurrentPPO 用单线程
    比默认快约一倍（默认 128 线程 224 FPS，单线程 462 FPS）。

    这个选项是 opt-in 而不是全局默认，因为对像素环境用的 CNN 策略结论正好相反——
    那里多线程是有收益的。
    """
    threads = config.get("experiment", {}).get("torch_threads")
    if threads is None:
        return
    torch.set_num_threads(int(threads))
    logger.debug("torch 线程数限制为 %s", threads)


def run_directory(config: dict[str, Any], *, create_timestamp: bool = False) -> Path:
    """确定这次运行的输出目录。

    配置里 ``output.timestamped`` 为真时，每次训练新建一个时间戳子目录，
    不覆盖历史结果；为假时所有运行共用同一个目录（默认行为，便于反复迭代）。

    Args:
        create_timestamp: 训练路径传 True（生成新目录）；重新可视化时传 False，
            此时会复用最近一次运行或从 ``resolved_config.yaml`` 反推。
    """
    output = config["output"]
    base = resolve_repo_path(output["directory"])
    if not output.get("timestamped", False):
        return base

    if create_timestamp:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return base / stamp

    # 重新可视化：优先跟着已加载的 resolved_config.yaml 走，否则退回最近一次运行。
    source_path = Path(config.get("_config_path", ""))
    if source_path.name == "resolved_config.yaml" and source_path.parent.exists():
        return source_path.parent
    runs = sorted(path for path in base.glob("*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No timestamped runs found below: {base}")
    return runs[-1]


def write_resolved_config(config: dict[str, Any], run_dir: Path) -> None:
    """把生效的配置存一份到运行目录。

    这样做的好处是：事后回看某次实验时，不必去猜当时用的是哪份配置——
    即使原配置后来被改过，运行目录里也留着当时的那一份。
    """
    resolved = deepcopy(config)
    resolved.pop("_config_path", None)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)
