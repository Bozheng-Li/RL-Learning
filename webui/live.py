"""训练过程中的实时数据：进度、ETA、训练信号、日志。

这里只做「读正在被写入的文件」这一件事，但它比看上去微妙：

- ``progress.csv`` 的**列集合与顺序每次运行都不同**（SB3 按首次 record 的顺序落列），
  而且新列出现时 SB3 会 ``seek(0)`` 整体重写文件。轮询时可能读到半截内容。
- ``episodes.csv`` 不能用于实时（块缓冲，训练结束才落盘），所以进度只走 ``progress.csv``。
- 自研算法不写 ``time/fps``，ETA 得有回退方案。

所有函数都不依赖 Streamlit，可以脱离界面直接跑。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import paths

#: 语义名 -> ``progress.csv`` 里的真实列名。
#:
#: 取值一律经由这张表，**绝不按列位置索引**：不同算法的列集合与顺序都不一样，
#: 按位置取会静默拿到错的列。
PROGRESS_COLUMNS: dict[str, str] = {
    "steps": "time/total_timesteps",
    "reward": "rollout/ep_rew_mean",
    "ep_len": "rollout/ep_len_mean",
    "eval_reward": "eval/mean_reward",
    "eval_len": "eval/mean_ep_length",
    # 只有 SB3 系（含 SB3-PPO/DQN/SAC/TD3）会写这两列，自研 REINFORCE/A2C/PPO 都没有。
    "fps": "time/fps",
    "elapsed": "time/time_elapsed",
    # PPO 系（自研与 SB3 都有，因为读的是同一份 logger 契约）
    "kl": "train/approx_kl",
    "clip": "train/clip_fraction",
    "entropy": "train/entropy_loss",
    "pg_loss": "train/policy_gradient_loss",
    "value_loss": "train/value_loss",
    "lr": "train/learning_rate",
    # DQN / SAC / TD3
    "loss": "train/loss",
    "n_updates": "train/n_updates",
    "exploration": "rollout/exploration_rate",
}

#: 语义名 -> 界面上显示的中文标签。
SIGNAL_LABELS: dict[str, str] = {
    "reward": "回合回报（ep_rew_mean）",
    "ep_len": "回合长度",
    "eval_reward": "周期评估回报",
    "kl": "近似 KL",
    "clip": "PPO 裁剪比例",
    "entropy": "策略熵",
    "pg_loss": "策略梯度损失",
    "value_loss": "价值损失",
    "lr": "学习率",
    "loss": "TD 损失",
    "n_updates": "更新次数",
    "exploration": "探索率 ε",
    "fps": "FPS",
}

#: 判断训练是否「卡死」的静默阈值（秒）。
STALE_SECONDS = 300.0

# 上一次成功解析的快照：路径 -> (文件大小, DataFrame)。
# SB3 重写 progress.csv 时会先 seek(0) 截断，这期间读到的可能是半截文件，
# 用上一次的结果兜底。多会话共享没有副作用——这只是按路径索引的只读快照。
_LAST_GOOD: dict[str, tuple[int, pd.DataFrame]] = {}
_LAST_GOOD_LOCK = threading.Lock()


def read_progress_csv(path: Path) -> pd.DataFrame | None:
    """按列名读取 ``progress.csv``，容忍正在被重写的半截文件。

    返回 ``None`` 表示文件还不存在或完全读不出内容。
    """
    if not path.exists():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None

    key = str(path)
    with _LAST_GOOD_LOCK:
        cached = _LAST_GOOD.get(key)
    # 文件比上次读到的还小 —— 只可能是 SB3 正在重写它（加新列时会截断重来），
    # 此刻的内容不可信，直接用上一次的快照。
    if cached is not None and size < cached[0]:
        return cached[1]

    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError):
        return cached[1] if cached else None

    if frame.empty:
        return None

    # 表头确定之后才出现的新列，pandas 会把多出来的字段塞进 "Unnamed: N"。
    # 留着只会污染画图，直接丢。
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    frame = frame.dropna(how="all")
    if frame.empty:
        return cached[1] if cached else None

    with _LAST_GOOD_LOCK:
        _LAST_GOOD[key] = (size, frame)
    return frame


def column(frame: pd.DataFrame | None, key: str) -> pd.Series | None:
    """按语义名取一列；列不存在时返回 ``None`` 而不是抛 ``KeyError``。

    这是刻意的：调用方拿到的 progress.csv 来自哪个算法是运行期才知道的，
    让「这列没有」成为正常返回值，比到处 try/except 干净。
    """
    if frame is None:
        return None
    name = PROGRESS_COLUMNS.get(key, key)
    if name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce")


def available_signals(frame: pd.DataFrame | None) -> list[str]:
    """列出这个运行**真的有数据**的信号（按语义名）。

    用它来决定界面画哪些图：DQN 没有近似 KL，自研 PPO 没有 FPS，固定列名会
    画出空白图或者直接报错。
    """
    if frame is None:
        return []
    found = []
    for key in PROGRESS_COLUMNS:
        series = column(frame, key)
        if series is not None and series.notna().any():
            found.append(key)
    return found


@dataclass(frozen=True)
class Progress:
    """一次训练此刻的进度快照。"""

    done: int
    total: int
    fraction: float
    eta_seconds: float | None
    latest: dict[str, float] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.total > 0 and self.done >= self.total


def _estimate_eta(frame: pd.DataFrame, done: int, total: int, run_dir: Path) -> float | None:
    """估算剩余秒数，三级回退。

    没有任何一级能给出估计时返回 ``None``（界面上显示「未知」比显示一个编造的
    数字好）。
    """
    if total <= 0 or done <= 0 or done >= total:
        return None
    remaining = total - done

    # 第一级：SB3 系自带的 FPS，最准。自研算法没有这一列。
    fps = column(frame, "fps")
    if fps is not None:
        clean = fps.dropna()
        if not clean.empty and clean.iloc[-1] > 0:
            return remaining / float(clean.iloc[-1])

    # 第二级：已用时间换算平均速度。同样只有 SB3 系会写。
    elapsed = column(frame, "elapsed")
    if elapsed is not None:
        clean = elapsed.dropna()
        if not clean.empty and clean.iloc[-1] > 0:
            return remaining * (float(clean.iloc[-1]) / done)

    # 第三级：墙钟兜底。resolved_config.yaml 是训练一开始就写的，拿它当起点。
    start = run_dir / paths.START_MARKER
    if start.exists():
        wall = time.time() - start.stat().st_mtime
        if wall > 0:
            return remaining * (wall / done)
    return None


def compute_progress(run_dir: Path, total_timesteps: int | None) -> Progress | None:
    """从 ``progress.csv`` 算出当前进度。

    ``total_timesteps`` 应当来自该次运行的 ``resolved_config.yaml``，而不是原始
    配置——界面上改过步数之后，原始 YAML 里的值已经不是这次训练的目标了。
    """
    frame = read_progress_csv(run_dir / "logs" / "progress.csv")
    if frame is None:
        return None
    steps = column(frame, "steps")
    if steps is None:
        return None
    clean = steps.dropna()
    if clean.empty:
        return None

    done = int(clean.iloc[-1])
    total = int(total_timesteps or 0)
    latest: dict[str, float] = {}
    for key in available_signals(frame):
        series = column(frame, key)
        values = series.dropna() if series is not None else None
        if values is not None and not values.empty:
            latest[key] = float(values.iloc[-1])

    return Progress(
        done=done,
        total=total,
        fraction=(done / total) if total > 0 else 0.0,
        eta_seconds=_estimate_eta(frame, done, total, run_dir),
        latest=latest,
    )


def format_eta(seconds: float | None) -> str:
    """把剩余秒数格式化成「1 小时 23 分」这样的中文短语。"""
    return _format_span(seconds) if seconds is not None else "未知"


def format_duration(seconds: float | None) -> str:
    """把已流逝秒数格式化。与 :func:`format_eta` 同格式，语义上是「已经过去了多久」。"""
    return _format_span(seconds) if seconds is not None else "-"


def _format_span(seconds: float) -> str:
    if seconds < 0:
        return "未知"
    total = int(seconds)
    if total < 60:
        return f"{total} 秒"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} 分 {secs} 秒"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} 小时 {minutes} 分"
    days, hours = divmod(hours, 24)
    return f"{days} 天 {hours} 小时"


def signal_frame(
    run_dir: Path, *, keys: list[str] | None = None, max_points: int = 3000
) -> pd.DataFrame | None:
    """把 ``progress.csv`` 转成 ``x=总步数``、``y=各信号`` 的表格，供画图用。

    超过 ``max_points`` 行会等间隔抽稀：300k 步的 Atari 运行有上万行，
    全量塞给浏览器会卡。
    """
    frame = read_progress_csv(run_dir / "logs" / "progress.csv")
    if frame is None:
        return None
    steps = column(frame, "steps")
    if steps is None:
        return None

    if keys is None:
        keys = [key for key in available_signals(frame) if key != "steps"]
    if not keys:
        return None

    columns: dict[str, pd.Series] = {"steps": steps}
    for key in keys:
        series = column(frame, key)
        if series is not None:
            columns[key] = series

    table = pd.DataFrame(columns).dropna(subset=["steps"])
    if table.empty:
        return None
    if len(table) > max_points:
        table = table.iloc[:: max(1, len(table) // max_points)]
    return table.set_index("steps")


def tail_text(path: Path, *, lines: int = 200, max_bytes: int = 256_000) -> str:
    """读日志文件的尾部若干行。

    只从尾部读固定字节数：Atari 那种长训练加上 verbose 输出，日志能到几十 MB，
    整个读进内存会把页面拖死。从中间截断可能落在 UTF-8 多字节字符中间，
    用 ``errors="replace"`` 兜住（代价只是首行可能有一个乱码字符）。
    """
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(chunk.splitlines()[-lines:])


def newest_log_mtime(run_dir: Path) -> float | None:
    """运行目录里日志文件的最新修改时间，用于判断训练是否还在推进。"""
    logs = run_dir / "logs"
    if not logs.is_dir():
        return None
    stamps = [path.stat().st_mtime for path in logs.glob("*.csv")]
    return max(stamps) if stamps else None
