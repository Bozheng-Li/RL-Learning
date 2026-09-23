"""跨运行对比：把多次训练的结果汇总成表格与叠加曲线。

做对照实验时会跑出一堆运行目录，逐个打开 ``evaluation.json`` 手工抄数字既慢又容易
出错。这个脚本扫描 ``outputs/`` 下的结果，按需筛选，输出一张对比表；曲线也可以
叠在一张图里，直观看出谁收敛得更快。

用法::

    # 指定若干次运行（可以用多次 --runs）
    python compare.py --runs cartpole_ppo --runs cartpole_a2c

    # 按通配符筛选
    python compare.py --pattern 'cartpole*'

    # 全部
    python compare.py --all

    # 只出表，不画图
    python compare.py --pattern 'cartpole*' --no-plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def discover_runs(root: Path, *, names: list[str], pattern: str | None, all_runs: bool) -> list[Path]:
    """找出要对比的运行目录。

    一个「运行」是指含有 ``evaluation.json`` 的目录——那是训练完整跑完的标志。

    找不到东西时抛 ``FileNotFoundError`` 而不是 ``SystemExit``：后者继承自
    ``BaseException``，``except Exception`` 抓不住，一旦这段逻辑被 WebUI 之类
    的常驻进程复用就会把服务带崩。``main()`` 自己捕获后 ``sys.exit(1)``，
    命令行的退出行为不变。
    """
    if names:
        candidates = [root / name for name in names]
        missing = [path for path in candidates if not (path / "evaluation.json").exists()]
        if missing:
            available = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
            raise FileNotFoundError(
                f"这些运行没有结果：{', '.join(str(p) for p in missing)}\n"
                f"outputs/ 下现有：{', '.join(available) or '(空)'}"
            )
        return candidates
    if not root.exists():
        raise FileNotFoundError(f"输出目录不存在：{root}")

    found = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "evaluation.json").exists()
    )
    if pattern:
        found = [path for path in found if path.match(pattern)]
    if not found and not all_runs:
        raise FileNotFoundError(
            f"没有匹配的运行。outputs/ 下现有：{', '.join(p.name for p in root.iterdir())}"
        )
    return found


def load_summary(run_dir: Path) -> dict:
    with (run_dir / "evaluation.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["_run"] = run_dir.name
    return summary


def load_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """读取训练曲线（累计步数, 回报）。

    优先用周期评估的结果（噪声小），没有就退回 Monitor 的训练记录。
    """
    eval_path = run_dir / "evaluation" / "evaluations.npz"
    if eval_path.exists():
        data = np.load(eval_path)
        steps = data["timesteps"]
        means = data["results"].mean(axis=1)
        return steps, means

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


def variant_of(summary: dict) -> str:
    """从 ``evaluation.json`` 的 ``experiment`` 字段里取出变体名；基准返回空串。

    约定与 ``webui/variants.py::experiment_name`` 对称（``<配置名>__<变体名>``）。
    命令行这边只做一次 ``rsplit``，不导入 ``webui`` 包——``compare.py`` 要在
    最小依赖下能跑（它连绘图库都是延迟导入的）。
    """
    name = str(summary.get("experiment") or "")
    head, separator, tail = name.rpartition("__")
    return tail if separator and head and tail else ""


def print_table(summaries: list[dict]) -> None:
    header = f"{'运行':<26}{'算法':<13}{'环境':<24}{'变体':<18}{'均值':>9}{'标准差':>9}{'回合':>6}{'步数':>10}"
    print(header)
    print("-" * len(header))
    for summary in sorted(summaries, key=lambda s: -s.get("mean_reward", float("-inf"))):
        print(
            f"{summary['_run']:<26}"
            f"{str(summary.get('algorithm', '-')):<13}"
            f"{str(summary.get('environment', '-')):<24}"
            f"{(variant_of(summary) or '基准'):<18}"
            f"{summary.get('mean_reward', float('nan')):>9.2f}"
            f"{summary.get('std_reward', float('nan')):>9.2f}"
            f"{summary.get('episodes', 0):>6}"
            f"{summary.get('timesteps', 0):>10}"
        )


def plot_curves(runs: list[Path], summaries: list[dict], output: Path) -> Path | None:
    """把所有运行的曲线叠在一张图里。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plotted = 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    curve_axis, final_axis = axes

    for run_dir, summary in zip(runs, summaries):
        curve = load_curve(run_dir)
        if curve is not None:
            steps, values = curve
            curve_axis.plot(steps, values, linewidth=1.8, label=run_dir.name)
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("没有任何运行带训练曲线，跳过绘图")
        return None

    # 图内标签用英文：matplotlib 默认字体不含中文，否则会渲染成方块。
    # 终端表格可以放心用中文。
    curve_axis.set(
        title="Training curve (periodic evaluation mean)",
        xlabel="Timesteps",
        ylabel="Mean reward",
    )
    curve_axis.legend(fontsize=9)
    curve_axis.grid(alpha=0.3)

    # 右侧画最终评估的均值 ± 标准差，直接看离散程度。
    labels = [summary["_run"] for summary in summaries]
    means = [summary.get("mean_reward", 0.0) for summary in summaries]
    stds = [summary.get("std_reward", 0.0) for summary in summaries]
    final_axis.barh(labels, means, xerr=stds, color="#3377aa", alpha=0.8)
    final_axis.set(title="Final evaluation (error bar = std)", xlabel="Mean reward")
    final_axis.grid(alpha=0.3, axis="x")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"对比图已保存：{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", action="append", default=[], help="运行目录名，可重复")
    parser.add_argument("--pattern", help="按通配符筛选，如 'cartpole*'")
    parser.add_argument("--all", action="store_true", help="对比 outputs/ 下全部运行")
    parser.add_argument(
        "--outputs", type=Path, default=REPO_ROOT / "outputs", help="输出根目录"
    )
    parser.add_argument("--no-plot", action="store_true", help="只打印表格")
    parser.add_argument("--figure", type=Path, help="对比图保存路径")
    args = parser.parse_args()

    try:
        runs = discover_runs(
            args.outputs, names=args.runs, pattern=args.pattern, all_runs=args.all
        )
    except FileNotFoundError as error:
        # CLI 入口负责把「找不到」变成退出码 1 与一句人话；库函数只管抛。
        raise SystemExit(str(error)) from None
    if not runs:
        raise SystemExit("没有找到任何运行")

    summaries = [load_summary(run_dir) for run_dir in runs]
    print_table(summaries)

    if not args.no_plot:
        figure = args.figure or (args.outputs / "comparison.png")
        plot_curves(runs, summaries, figure)


if __name__ == "__main__":
    main()
