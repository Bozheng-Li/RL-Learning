"""运行目录的发现、状态判定与数据读取。

这一层是纯读的：扫描 ``outputs/``、读 ``evaluation.json`` / ``resolved_config.yaml`` /
各日志文件，全部函数都不依赖 Streamlit，可以脱离界面验证。

刻意**没有**复用 ``compare.discover_runs``：它只扫 ``outputs/`` 的一层，看不到嵌套的
时间戳目录，而且会把「一个运行」的定义与 CLI 的参数解析纠缠在一起。这里只沿用它的
判据——「目录里有 ``evaluation.json`` 才算一个完整运行」——并自己实现扫描。
（``compare.py`` 曾经在找不到匹配时 ``raise SystemExit``，现已改成 ``FileNotFoundError``，
由 CLI 入口自己捕获；那对常驻进程是个雷，见该文件的 ``discover_runs``。）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from . import live, paths

RunStatus = Literal["not_started", "running", "completed", "failed"]

#: 算法族的展示顺序。对比与分组都按这个顺序排，而不是按字母序——
#: 它大致对应「从策略梯度到 actor-critic 再到 off-policy」的学习顺序。
FAMILY_ORDER: tuple[str, ...] = ("REINFORCE", "A2C", "PPO", "DQN", "SAC", "TD3")


def algorithm_family(name: str | None) -> str:
    """把算法名归到族。

    ``SB3-PPO`` 与自研 ``PPO`` 读的是同一份 profile，只是实现不同，所以归到同一个
    族里做对照；实现上的差异用算法全名区分，不在族这一层拆开。
    """
    token = (name or "").upper().replace("SB3-", "").replace("SB3", "")
    for family in FAMILY_ORDER:
        if token == family or token.endswith(family):
            return family
    return "OTHER"


#: 状态的中文显示名。
STATUS_LABELS: dict[str, str] = {
    "not_started": "未开始",
    "running": "运行中",
    "completed": "已完成",
    "failed": "失败",
}


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #


def _is_run_dir(path: Path) -> bool:
    """与 ``compare.discover_runs`` 相同的判据：有 ``evaluation.json`` 才算跑完。"""
    return path.is_dir() and (path / paths.RUN_MARKER).exists()


def _has_run_artifacts(path: Path) -> bool:
    """更宽松的判据：还没跑完（没有 evaluation.json）但已经开工的运行目录。"""
    return path.is_dir() and (path / paths.START_MARKER).exists()


def scan_runs(root: Path = paths.OUTPUTS_ROOT, *, depth: int = 2) -> list[Path]:
    """找出根目录下所有「运行」目录，按修改时间倒序。

    扫两层是必须的：非时间戳的运行在 ``outputs/<配置名>/``，开了
    ``output.timestamped`` 的则在 ``outputs/<配置名>/<时间戳>/``。
    """
    if not root.is_dir():
        return []
    found: list[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if _is_run_dir(entry) or _has_run_artifacts(entry):
            found.append(entry)
        elif depth >= 2 and not entry.name.startswith("."):
            found.extend(
                child
                for child in entry.iterdir()
                if child.is_dir() and (_is_run_dir(child) or _has_run_artifacts(child))
            )
    return sorted(found, key=lambda path: _safe_mtime(path), reverse=True)


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- #
# 状态判定
# --------------------------------------------------------------------------- #


def _pid_alive(pid: int) -> bool:
    """用 ``kill(pid, 0)`` 探活，不需要 psutil。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在，只是不属于当前用户。
        return True
    except OSError:
        return False
    return True


def run_status(run_dir: Path, *, exited_code: int | None = None, pid: int | None = None) -> RunStatus:
    """判定一个运行目录的状态。

    项目没有显式状态文件，只能靠文件存在性加修改时间推。最容易错的一点：
    ``output.timestamped: false`` 时同一个目录会被反复复用，**上一轮遗留的
    ``evaluation.json`` 会一直躺在那里**。所以不能只看它存不存在，要比 mtime：
    只有它比 ``resolved_config.yaml`` 新，才说明是这一轮刚写出来的。

    Args:
        exited_code: 由 WebUI 启动且已经退出时的退出码（None 表示不知道）。
        pid: 由 WebUI 启动的进程号，用于探活。
    """
    resolved = run_dir / paths.START_MARKER
    summary = run_dir / paths.RUN_MARKER

    # 1. 有进程号且还活着 —— 最可靠的「运行中」。
    if pid is not None and _pid_alive(pid):
        return "running"
    # 2. 有进程号但已退出 —— 退出码说话。
    if exited_code is not None:
        return "completed" if exited_code == 0 else "failed"

    # 3. 没有进程信息（服务重启后接管，或不是 WebUI 启动的）：看文件时间。
    if summary.exists():
        if not resolved.exists() or summary.stat().st_mtime >= resolved.stat().st_mtime:
            return "completed"
    if resolved.exists():
        newest = live.newest_log_mtime(run_dir)
        if newest is None:
            # 刚起步、日志还没落盘，给它一点时间。
            newest = resolved.stat().st_mtime
        return "running" if (time.time() - newest) < live.STALE_SECONDS else "failed"
    return "not_started"


# --------------------------------------------------------------------------- #
# 读单个运行
# --------------------------------------------------------------------------- #


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


#: 优先用 libyaml 的 C 实现：同一个 ``resolved_config.yaml`` 实测 10.1ms → 1.2ms。
#: 列表页每次刷新要读几百份配置（639 个运行时约 6.4 秒），这个差别是肉眼可见的。
#: PyYAML 没编进 C 扩展时回退到纯 Python 版，只是慢一点，结果完全一致。
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=_YAML_LOADER)
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class RunInfo:
    """列表页需要的一行。"""

    name: str
    path: Path
    status: RunStatus
    mtime: float
    environment: str | None = None
    algorithm: str | None = None
    seed: int | None = None
    total_timesteps: int | None = None
    latest_timestep: int | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    config_name: str | None = None
    #: 算法族（REINFORCE/A2C/PPO/...），用于跨运行分组。由算法名派生，不单独存储。
    family: str = "OTHER"
    device: str | None = "cpu"
    #: 这次运行实际用的动作空间（``discrete`` / ``continuous`` / ``unknown``）。
    #: 连续与离散的奖励尺度不可比，所以排行榜不能把两者混在一根轴上排。
    action_space: Literal["discrete", "continuous", "unknown"] = "unknown"
    #: ``resolved_config`` 里 ``experiment.name`` 的原值。基准运行就是配置名本身，
    #: 变体运行是 ``<配置名>__<变体名>``。
    experiment: str | None = None
    #: 变体名；``None`` 表示基准（没套任何覆盖）。
    #:
    #: 排行榜与筛选器都靠它把「同一个算法的不同配置」分开成行。`None`
    #: 与空串不区分——`experiment_name()` 对基准返回配置名，看不出变体名。
    variant: str | None = None


def variant_of_experiment(name: Any) -> str | None:
    """``experiment.name`` → 变体名；基准（没有变体）返回 ``None``。

    约定与 ``webui/variants.py::experiment_name`` 对称：``<配置名>__<变体名>``。
    只认一个 ``__``；``experiment.name`` 里没有它的一律当基准，包括历史运行里
    ``lunarlander_baseline`` 这种单下划线的写法——那是配置名的一部分，不是变体。
    """
    from . import variants  # noqa: PLC0415  —— 同包内延迟导入，避免循环

    _, variant = variants.parse_experiment(str(name) if name else None)
    return variant


def load_summary(run_dir: Path) -> dict[str, Any]:
    """读取一次运行的评估摘要，不导入绘图库。"""
    try:
        summary = read_json(run_dir / paths.RUN_MARKER)
        if summary is None:
            return {}
        summary["_run"] = run_dir.name
        return summary
    except (ImportError, OSError, json.JSONDecodeError):
        return {}


def load_curve(run_dir: Path) -> tuple[Any, Any] | None:
    """读取训练曲线（周期评估优先，退回训练 monitor）。"""
    try:
        eval_path = run_dir / "evaluation" / "evaluations.npz"
        if eval_path.exists():
            loaded = np.load(eval_path)
            return loaded["timesteps"], loaded["results"].mean(axis=1)

        monitor = run_dir / "logs" / "train.monitor.csv"
        if not monitor.exists():
            return None
        import csv

        with monitor.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
        if not rows:
            return None
        lengths = np.asarray([int(row["l"]) for row in rows])
        rewards = np.asarray([float(row["r"]) for row in rows])
        return np.cumsum(lengths), rewards
    except (ImportError, OSError, ValueError, KeyError):
        return None


def describe_run(
    run_dir: Path, *, exited_code: int | None = None, pid: int | None = None
) -> RunInfo:
    """读出一个运行目录的全部摘要信息。

    环境/算法/种子/总步数一律取自 ``resolved_config.yaml``（这次运行**实际生效**的
    配置），而不是 ``config/*.yaml``——界面上用 ``--set`` 改过的值不会回写原配置，
    拿原配置会显示成旧值。
    """
    resolved = read_yaml(run_dir / paths.START_MARKER)
    summary = load_summary(run_dir)

    algorithm_cfg = resolved.get("algorithm") or {}
    environment_cfg = resolved.get("environment") or {}
    experiment_cfg = resolved.get("experiment") or {}
    training_cfg = resolved.get("training") or {}

    # 只从 progress.csv 尾部取最后一个步数：列表页要一次性描述几百个运行，
    # 全量解析每个文件（最大 5.6 MB）是页面加载时间的主要来源。
    latest = live.latest_step(run_dir / "logs" / "progress.csv")

    algorithm = algorithm_cfg.get("name") or summary.get("algorithm")
    device = algorithm_cfg.get("device") or "cpu"
    # 变体身份只认 experiment.name：它由 variants.experiment_name 写成
    # ``<配置名>__<变体名>``，同时也落进 evaluation.json。目录名里只有 slug，
    # 而且历史目录名各式各样，反解它会把基准误判成变体。
    experiment = experiment_cfg.get("name") or summary.get("experiment")
    return RunInfo(
        name=run_dir.name,
        path=run_dir,
        status=run_status(run_dir, exited_code=exited_code, pid=pid),
        mtime=_safe_mtime(run_dir),
        environment=environment_cfg.get("id") or summary.get("environment"),
        algorithm=algorithm,
        seed=experiment_cfg.get("seed", summary.get("seed")),
        total_timesteps=training_cfg.get("total_timesteps", summary.get("timesteps")),
        latest_timestep=latest,
        summary=summary,
        config_name=config_for_run(run_dir, resolved),
        family=algorithm_family(algorithm),
        device=device,
        action_space=run_action_space(resolved),
        experiment=str(experiment) if experiment else None,
        variant=variant_of_experiment(experiment),
    )


def list_runs(
    root: Path = paths.OUTPUTS_ROOT,
    *,
    pattern: str | None = None,
) -> list[RunInfo]:
    """扫描并描述所有运行。

    ``describe_run`` 里读步数走的是 ``live.latest_step``（倒着读 ``progress.csv`` 尾部），
    639 个运行一共 0.07 秒量级，所以这里不再提供「跳过进度」的开关——曾经有过一个
    ``with_progress=False``，但它只是**读完之后**把值抹掉，省不下 IO，反而误导调用方。
    """
    if not root.is_dir():
        return []
    infos = [describe_run(run_dir) for run_dir in scan_runs(root)]
    if pattern:
        from fnmatch import fnmatch  # noqa: PLC0415

        infos = [info for info in infos if fnmatch(info.name, pattern)]
    return infos


def runs_by_environment(runs: list[RunInfo]) -> dict[str, list[RunInfo]]:
    """按环境 id 分组，环境按名称排序，组内保持原有（时间倒序）顺序。

    没有环境信息的运行归到「未标注环境」，避免在界面上消失。
    """
    grouped: dict[str, list[RunInfo]] = {}
    for info in runs:
        grouped.setdefault(info.environment or "未标注环境", []).append(info)
    return dict(sorted(grouped.items()))


def leaderboard_key(info: RunInfo) -> tuple[str, str]:
    """排行榜的分组键：``(实现, 变体)``。

    折叠的粒度是**实现**而不是算法族：``PPO`` 与 ``SB3-PPO`` 是两个不同的实现，
    折叠成一个族名就把本项目最核心的对照维度丢掉了。而**变体必须一起进键**——
    同一个算法的 ``lr-1e-3`` 与 ``lr-1e-4`` 是两组不同的配置，折叠后只会留下回报
    最高的那一个，另一组连同它的种子离散度一起消失。

    基准的变体位是空串，所以「只有基准运行」时这个键退化成 ``(algorithm, "")``，
    与改动前逐字段等价。
    """
    return (info.algorithm or info.family, info.variant or "")


def baseline_algorithms(runs: list[RunInfo]) -> set[str]:
    """跑过**基准**（没套任何变体）的实现集合。

    「补齐对照」和总览页的灰色占位条都用它判断缺口。刻意不看变体：一个算法只跑了
    lr 扫描而没有基准时，那些曲线没有对照物——扫描里的每个点都要跟基准比才有意义，
    所以这时应该仍然提示去补基准。反过来，只跑了基准而没有变体时不算缺口——变体是
    可选的深挖，不是必备的对照。
    """
    return {info.algorithm for info in runs if info.algorithm and not info.variant}


def environment_leaderboard(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """同一个环境内，按 ``(算法, 变体)`` 取最好的最终评估，按回报从高到低排。

    每个 ``(算法, 变体)`` 组合保留最好的一次代表。变体参与分组是必须的：变体批次里
    同一个算法会有十几份配置，不分开的话只剩回报最高的那一行，整个扫描的意义就没了。

    每个条目还带 ``action_space`` 与离散度（``spread`` / ``low`` / ``high``）。
    连续动作空间（SAC/TD3）与离散的奖励尺度不可比，且单看一个均值看不出稳不稳。
    ``spread`` 是**同一个 (算法, 变体) 内的多种子离散度**——变体之间的差异由行与行
    的 ``mean`` 对比体现，两者不能混成一个数。

    返回的每一行是
    ``{"family", "algorithm", "variant", "label", "mean", "std", "spread", "low",
    "high", "count", "seed", "action_space", "run"}``。
    """
    best: dict[tuple[str, str], RunInfo] = {}
    for info in runs:
        reward = info.summary.get("mean_reward")
        if reward is None:
            continue
        key = leaderboard_key(info)
        current = best.get(key)
        if current is None or float(reward) > float(current.summary.get("mean_reward", float("-inf"))):
            best[key] = info

    # 同一个 (算法, 变体) 可能跑了多个种子（目录不同、名字相同）。有多个时用样本标准差
    # 做误差带，只有一个时不给误差带——分母为 0 的伪标准差比没有更误导。
    values_by_key: dict[tuple[str, str], list[float]] = {}
    for info in runs:
        reward = info.summary.get("mean_reward")
        if reward is None:
            continue
        values_by_key.setdefault(leaderboard_key(info), []).append(float(reward))

    rows = []
    for key, info in best.items():
        values = values_by_key.get(key, [])
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else None
        mean = float(info.summary.get("mean_reward", 0.0))
        algorithm = info.algorithm or key[0]
        rows.append({
            "family": info.family,
            "algorithm": algorithm,
            "variant": info.variant,
            # 显示名：变体行写成「PPO · lr-1e-3」。``theme.algo_bars`` 认这个键，
            # 缺省才退回 ``algorithm``，所以显示层不需要为变体改一行。
            "label": f"{algorithm} · {info.variant}" if info.variant else algorithm,
            "mean": mean,
            "std": float(info.summary.get("std_reward", 0.0)),
            "spread": spread,
            "low": mean - spread if spread is not None else None,
            "high": mean + spread if spread is not None else None,
            "count": len(values),
            "seed": info.seed,
            "action_space": info.action_space,
            "run": info.name,
        })
    rows.sort(key=lambda row: row["mean"], reverse=True)
    return rows


def family_groups(entries: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """把排行榜条目按算法族分组，族按 ``FAMILY_ORDER``（学习顺序）排。

    跨运行对比页必须分组，不能把两百多条一次性摊开：变体批次跑完一个环境有
    11 个实现 × 十几个变体，一条平铺的条形墙读不出任何东西。按族切开之后每组
    只有几行，色也一致（``theme.family_color``），才看得出「同族的两个实现差多少」
    与「这个算法对哪个参数敏感」。

    族内保持传入顺序——``environment_leaderboard`` 已经按回报从高到低排过了，
    组内再按回报排一次没有意义，还会把「最好的那个」从第一个挪走。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry.get("family") or "OTHER"), []).append(entry)
    ordered = [family for family in FAMILY_ORDER if family in grouped]
    tail = sorted(family for family in grouped if family not in set(ordered))
    return [(family, grouped[family]) for family in ordered + tail]


def best_run(runs: list[RunInfo]) -> RunInfo | None:
    """一组运行里最终评估最好的那一次；都没有评估时返回 ``None``。"""
    scored = [info for info in runs if info.summary.get("mean_reward") is not None]
    if not scored:
        return None
    return max(scored, key=lambda info: float(info.summary.get("mean_reward", float("-inf"))))


def find_run(name: str, root: Path = paths.OUTPUTS_ROOT) -> Path | None:
    """按显示名定位运行目录，支持 ``parent/child`` 形式。"""
    candidate = root / name
    if _is_run_dir(candidate) or _has_run_artifacts(candidate):
        return candidate
    for run_dir in scan_runs(root):
        if run_dir.name == name:
            return run_dir
    return None


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


def list_config_names() -> list[str]:
    """``config/`` 下所有配置的简写名。"""
    if not paths.CONFIG_ROOT.is_dir():
        return []
    return sorted(path.stem for path in paths.CONFIG_ROOT.glob("*.yaml"))


def load_config_dict(name: str) -> dict[str, Any]:
    """读一份实验配置（用配置简写名）。"""
    from rl_common.config import load_config  # noqa: PLC0415  延迟：会拉进 torch

    return load_config(paths.CONFIG_ROOT / f"{name}.yaml")


def available_profiles(config: dict[str, Any]) -> list[str]:
    """当前配置里预置了超参的算法名。

    训练表单的算法下拉**必须**用这个，而不是 ``available_algorithms()``：
    选到一个没有对应 profile 的算法，训练会在启动后抛
    ``ValueError: No algorithm.profiles.X section``。
    """
    profiles = (config.get("algorithm") or {}).get("profiles") or {}
    return sorted(profiles)


def probe_action_space(config: dict[str, Any]) -> Literal["discrete", "continuous", "unknown"]:
    """从配置猜环境的动作空间类型，用于提前拦住不匹配的算法。

    这不是权威判断（真正的检查在环境构造之后），但能把「DQN 跑连续环境」这类
    错误在点按钮之前就暴露出来，而不是等训练启动几十秒后才发现。
    """
    kwargs = (config.get("environment") or {}).get("kwargs") or {}
    value = kwargs.get("continuous")
    if isinstance(value, bool):
        return "continuous" if value else "discrete"

    env_id = str((config.get("environment") or {}).get("id", "")).lower()
    if "pendulum" in env_id or "bipedal" in env_id or "carracing" in env_id:
        return "continuous"
    if env_id.startswith("ale/") or "minigrid" in env_id:
        return "discrete"
    if any(token in env_id for token in ("cartpole", "mountaincar", "acrobot",
                                         "lunarlander", "taxi")):
        # LunarLander 默认是离散的（连续要显式配 continuous: true，上面已处理）。
        return "discrete"
    return "unknown"


def needs_continuous(algorithm: str) -> bool:
    """SAC/TD3 只支持连续动作空间，选到它们时训练参数要带上 ``continuous: true``。"""
    return algorithm.upper().endswith(("SAC", "TD3"))


#: 动作空间的中文短名，界面上标注在算法旁边。
ACTION_SPACE_LABELS: dict[str, str] = {
    "discrete": "离散动作",
    "continuous": "连续动作",
    "unknown": "动作空间未知",
}


def run_action_space(resolved: dict[str, Any]) -> Literal["discrete", "continuous", "unknown"]:
    """从**一次运行的 resolved 配置**推出它实际用的动作空间。

    与 ``probe_action_space`` 的区别：那个看的是 ``config/*.yaml``（打算怎么跑），
    这个看的是运行时真正生效的配置（实际怎么跑的）——发起训练时会为 SAC/TD3 自动把
    ``continuous`` 翻成 true，只看原配置会得出相反的结论。
    """
    return probe_action_space(resolved)


def compatible_algorithms(config: dict[str, Any]) -> list[str]:
    """这个环境上能跑的全部算法：自研的加上 SB3 的，按学习顺序排。

    只保留配置里预置了 profile 的算法族（profile 键不带 SB3- 前缀，自研与 SB3
    读同一份超参）。SAC/TD3 要求配置里有 ``environment.kwargs.continuous`` 这个键，
    因为训练时会把它翻成 ``true``。
    """
    profiles = set(available_profiles(config))
    has_continuous_switch = isinstance(
        ((config.get("environment") or {}).get("kwargs") or {}).get("continuous"), bool
    )
    ordered = ["REINFORCE", "A2C", "SB3-A2C", "PPO", "SB3-PPO", "DQN", "SB3-DQN",
               "SAC", "SB3-SAC", "TD3", "SB3-TD3"]
    result = []
    is_discrete = probe_action_space(config) == "discrete"
    for name in ordered:
        family = name.replace("SB3-", "")
        # REINFORCE 无需显式 profile 配置即可使用内置超参（适用于离散动作空间）
        if family == "REINFORCE":
            if is_discrete:
                result.append(name)
            continue
        if family not in profiles:
            continue
        if needs_continuous(name) and not has_continuous_switch:
            continue
        result.append(name)
    return result


_CONFIG_MAP_CACHE: tuple[float, dict[str, str]] | None = None
_CONFIG_MAP_TTL = 300.0

#: 配置目录映射的缓存有效期（秒）。配置文件很少变，但改完不该等太久才生效。
def _config_directory_map() -> dict[str, str]:
    """``{run_directory 绝对路径: 配置名}``，用于反查一个运行来自哪份配置。

    ``resolved_config.yaml`` 里**没有**记录原始配置文件名（写盘前被 pop 掉了），
    所以只能拿每份配置算出它的输出目录再比对。13 份配置要加载 13 次 YAML，
    而列表页的每个运行都要查一次，所以这里带一个进程内缓存——它是配置文件的
    纯函数，缓存没有副作用。
    """
    global _CONFIG_MAP_CACHE

    now = time.time()
    if _CONFIG_MAP_CACHE is not None and now - _CONFIG_MAP_CACHE[0] < _CONFIG_MAP_TTL:
        return _CONFIG_MAP_CACHE[1]

    try:
        from rl_common.config import load_config, run_directory  # noqa: PLC0415
    except (ImportError, ModuleNotFoundError):
        # The WebUI can still browse existing output folders without the optional
        # environment stack installed. Config reverse lookup is best-effort only.
        _CONFIG_MAP_CACHE = (now, {})
        return {}

    mapping: dict[str, str] = {}
    for stem in list_config_names():
        try:
            config = load_config(paths.CONFIG_ROOT / f"{stem}.yaml")
            mapping[str(run_directory(config))] = stem
        except (OSError, ValueError, KeyError):
            continue
    _CONFIG_MAP_CACHE = (now, mapping)
    return mapping


def clear_caches() -> None:
    """清掉本模块的进程内缓存（界面上「清空缓存」按钮会调）。"""
    global _CONFIG_MAP_CACHE
    _CONFIG_MAP_CACHE = None
    from . import gpus  # noqa: PLC0415  —— 避免顶层导入牵出 torch

    gpus.clear_cache()


def match_config_name(
    directory: str, names: list[str] | None = None
) -> str | None:
    """从平铺的目录名反推配置名：**最长前缀优先**，且必须落在 ``_`` 边界上。

    这是 ``config_for_run`` 的主判据。原来的做法是「拿每份配置的默认输出目录去比对」，
    而 WebUI 发起训练时一定会 ``--set output.directory=outputs/<名字>``，默认目录
    永远对不上——实测 33 个运行全部返回 ``None``，于是总览页的环境占位条、对比页的
    「补齐对照」、详情页的「重新出图」三处一起哑掉。

    判据的两个细节都是必需的：

    - **最长优先**：``minigrid`` / ``minigrid_doorkey`` / ``minigrid_doorkey_hard``
      共存时，短匹配会把 ``minigrid_doorkey_hard_ppo_s42_x`` 判成 ``minigrid``。
    - **``_`` 边界**：单纯的前缀匹配会让 ``lunarlanderfoo`` 命中 ``lunarlander``。
    """
    candidates = list_config_names() if names is None else names
    matched = [
        name for name in candidates if directory == name or directory.startswith(name + "_")
    ]
    return max(matched, key=len) if matched else None


def config_for_run(
    run_dir: Path, resolved: dict[str, Any] | None = None
) -> str | None:
    """反查一个运行目录来自哪份配置；查不到返回 ``None``（界面上显示「自定义」）。

    主判据是**目录名前缀**——WebUI 与 ``variants.py`` 发起的运行都是平铺的
    ``<配置名>_<算法>_s<种子>[_<变体>]_<时间戳>``，名字本身就是答案，实测 639/639
    命中，而且纯字符串比较，不付任何 IO。

    前缀认不出来时才退回 ``resolved_config.yaml`` 里记的 ``output.directory`` 精确
    匹配：那条对「目录名与配置名无关」的形态（命令行 ``--set output.directory=outputs/ppo_native``）
    才可靠。顺序反过来是有代价的——建那张映射表要 ``import torch``（1.9 秒），
    而它几乎永远不会命中，因为 WebUI 发起训练时一定要 ``--set output.directory``。

    Args:
        resolved: 调用方已经读过的 ``resolved_config.yaml``。``describe_run`` 一定会
            传它——那份 YAML 已经被读进来解析环境/算法/种子了，再读第二遍纯属浪费
            （实测 639 个运行多付 0.8 秒）。
    """
    matched = match_config_name(run_dir.name)
    if matched:
        return matched
    if resolved is None:
        resolved = read_yaml(run_dir / paths.START_MARKER)
    directory = (resolved.get("output") or {}).get("directory")
    if directory:
        return _config_directory_map().get(str(paths.resolve_repo_path(directory)))
    return None


# --------------------------------------------------------------------------- #
# 日志与产物
# --------------------------------------------------------------------------- #


def load_diagnostics(run_dir: Path) -> pd.DataFrame | None:
    """读 ``diagnostics.csv``（动作分布诊断）。

    列数随动作空间变化，且中间夹着一个字符串列 ``action_space``：
    - 离散环境：``action_0_frac`` ... ``action_{n-1}_frac``，``action_mean/std`` 为空
    - 连续环境：只有 ``action_mean`` / ``action_std``

    必须按最后一行的 ``action_space`` 值决定用哪一组，并把那个字符串列剔掉——
    留着它会让 ``st.line_chart`` 因为 object dtype 报错。
    """
    path = run_dir / "logs" / "diagnostics.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    if frame.empty or "timestep" not in frame.columns:
        return None

    kind = ""
    if "action_space" in frame.columns:
        values = frame["action_space"].dropna()
        if not values.empty:
            kind = str(values.iloc[-1])

    value_columns = [
        name for name in frame.columns if name not in {"timestep", "samples", "action_space"}
    ]
    if kind == "discrete":
        value_columns = [name for name in value_columns if name.endswith("_frac")]
    elif kind == "continuous":
        value_columns = [name for name in value_columns if name in {"action_mean", "action_std"}]
    if not value_columns:
        return None

    table = frame.loc[:, ["timestep", *value_columns]]
    return table.apply(pd.to_numeric, errors="coerce")


def load_episodes(run_dir: Path) -> pd.DataFrame | None:
    """读 ``episodes.csv``。

    注意这个文件**不 flush**（块缓冲，训练结束才落盘），所以只能用于已经跑完的
    运行，不能拿来做实时进度。
    """
    path = run_dir / "logs" / "episodes.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    return frame if not frame.empty else None


def load_evaluations(run_dir: Path) -> tuple[Any, Any, Any] | None:
    """读 ``evaluation/evaluations.npz``，返回 ``(timesteps, results, ep_lengths)``。"""
    path = run_dir / "evaluation" / "evaluations.npz"
    if not path.exists():
        return None
    try:
        import numpy as np  # noqa: PLC0415

        data = np.load(path)
        return data["timesteps"], data["results"], data["ep_lengths"]
    except (OSError, KeyError, ValueError):
        return None


def trajectory_summary(run_dir: Path) -> list[dict[str, Any]]:
    """读 ``trajectories/summary.json``（每个回合的回报、步数、视频路径）。"""
    data = read_json(run_dir / "trajectories" / "summary.json")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        episodes = data.get("episodes")
        if isinstance(episodes, list):
            return [item for item in episodes if isinstance(item, dict)]
    return []


def list_artifacts(run_dir: Path) -> dict[str, list[Path]]:
    """按类别列出运行目录里的产物文件。"""
    groups: dict[str, list[Path]] = {
        "模型": [],
        "训练图表": [],
        "轨迹": [],
        "日志": [],
        "其他": [],
    }
    if not run_dir.is_dir():
        return groups

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        suffix = path.suffix.lower()
        parts = relative.parts

        if any(part == "checkpoints" for part in parts) or any(
            part == "best_model" for part in parts
        ):
            groups["模型"].append(path)
        elif suffix in {".pt", ".zip"}:
            groups["模型"].append(path)
        elif "visualizations" in parts and suffix == ".png":
            groups["训练图表"].append(path)
        elif "trajectories" in parts:
            groups["轨迹"].append(path)
        elif "logs" in parts or "evaluation" in parts:
            groups["日志"].append(path)
        elif relative.name in {paths.START_MARKER, paths.RUN_MARKER}:
            groups["日志"].append(path)
        else:
            groups["其他"].append(path)

    return {name: files for name, files in groups.items() if files}


def human_size(num_bytes: int) -> str:
    """把字节数格式化成人类可读的形式。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"
