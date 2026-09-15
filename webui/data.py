"""运行目录的发现、状态判定与数据读取。

这一层是纯读的：扫描 ``outputs/``、读 ``evaluation.json`` / ``resolved_config.yaml`` /
各日志文件，全部函数都不依赖 Streamlit，可以脱离界面验证。

刻意**没有**复用 ``compare.discover_runs``：它在找不到匹配时 ``raise SystemExit``
（继承 ``BaseException``，``except Exception`` 抓不住，会把服务进程带崩），而且只扫
``outputs/`` 的一层，看不到嵌套的时间戳目录。这里只沿用它的判据——「目录里有
``evaluation.json`` 才算一个完整运行」——并自己实现扫描。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from . import live, paths

RunStatus = Literal["not_started", "running", "completed", "failed"]

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


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
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


def load_summary(run_dir: Path) -> dict[str, Any]:
    """复用 ``compare.load_summary``，失败时返回空 dict。"""
    try:
        import compare  # noqa: PLC0415  —— 顶层模块，靠 paths.py 注入的 sys.path

        return compare.load_summary(run_dir)
    except (ImportError, OSError, json.JSONDecodeError):
        return {}


def load_curve(run_dir: Path) -> tuple[Any, Any] | None:
    """复用 ``compare.load_curve``（周期评估优先，退回训练 monitor）。"""
    try:
        import compare  # noqa: PLC0415

        return compare.load_curve(run_dir)
    except (ImportError, OSError):
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

    latest: int | None = None
    progress = live.read_progress_csv(run_dir / "logs" / "progress.csv")
    if progress is not None:
        steps = live.column(progress, "steps")
        if steps is not None:
            clean = steps.dropna()
            if not clean.empty:
                latest = int(clean.iloc[-1])

    return RunInfo(
        name=run_dir.name,
        path=run_dir,
        status=run_status(run_dir, exited_code=exited_code, pid=pid),
        mtime=_safe_mtime(run_dir),
        environment=environment_cfg.get("id") or summary.get("environment"),
        algorithm=algorithm_cfg.get("name") or summary.get("algorithm"),
        seed=experiment_cfg.get("seed", summary.get("seed")),
        total_timesteps=training_cfg.get("total_timesteps", summary.get("timesteps")),
        latest_timestep=latest,
        summary=summary,
        config_name=config_for_run(run_dir),
    )


def list_runs(
    root: Path = paths.OUTPUTS_ROOT,
    *,
    pattern: str | None = None,
    with_progress: bool = True,
) -> list[RunInfo]:
    """扫描并描述所有运行。

    Args:
        with_progress: 是否读每个运行的 ``progress.csv`` 拿最新步数。列表页需要，
            但纯筛选场景可以关掉省 IO。
    """
    if not root.is_dir():
        return []
    infos: list[RunInfo] = []
    for run_dir in scan_runs(root):
        info = describe_run(run_dir)
        if not with_progress:
            info.latest_timestep = None
        infos.append(info)
    if pattern:
        from fnmatch import fnmatch  # noqa: PLC0415

        infos = [info for info in infos if fnmatch(info.name, pattern)]
    return infos


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

    from rl_common.config import load_config, run_directory  # noqa: PLC0415

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


def config_for_run(run_dir: Path) -> str | None:
    """反查一个运行目录来自哪份配置；查不到返回 ``None``（界面上显示「自定义」）。

    用 ``--set output.directory=...`` 跑出来的运行查不到，这是正常的。
    """
    resolved = read_yaml(run_dir / paths.START_MARKER)
    directory = (resolved.get("output") or {}).get("directory")
    if not directory:
        return None
    from rl_common.config import resolve_repo_path  # noqa: PLC0415

    return _config_directory_map().get(str(resolve_repo_path(directory)))


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
