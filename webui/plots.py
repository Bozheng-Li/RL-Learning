"""需要 matplotlib 的图。

大部分图表用 Streamlit 原生的 ``st.line_chart`` / ``st.area_chart`` 就够了，
而且更好——它们走浏览器渲染，中文不会变方块，开销也小得多（实时刷新路径上
每 2 秒重建一个 matplotlib Figure 会稳定泄漏内存）。

这里只放原生组件画不了的：带误差棒的条形图、多运行叠加需要精确控制样式的情况。

**图内文字一律用英文**：matplotlib 的默认字体不含 CJK，中文会渲染成方块。
这是项目既有约定（见 ``compare.py`` 里的同名注释）。中文只出现在 Streamlit 的
标题、标签、说明文字里。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from . import data  # noqa: E402


def fig_to_png(figure: Any, *, dpi: int = 150) -> bytes:
    """把 Figure 渲染成 PNG 字节并**总是**关闭它。

    ``finally`` 里的 ``plt.close`` 不是可选的：调用方在定时刷新的 fragment 里跑，
    漏关一次就是每轮泄漏一个 Figure。返回 bytes 而不是让调用方用 ``st.pyplot``，
    也正是为了彻底摆脱 pyplot 的全局状态。
    """
    try:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(figure)


#: 默认色板。**只在调用方没给 ``color_of`` 时用**——按列表下标取色会把「同一个算法的
#: 多个变体」画成七色彩虹里互不相干的两条。正式路径走 ``series_for``，按算法族取色。
_PALETTE = ("#e4572e", "#2563eb", "#0f9d8a", "#7c3aed", "#d97706", "#db2777", "#69757b")

#: 同族多条曲线时轮换的线型。色盲友好，灰度打印下也分得开。
_LINESTYLES = ("-", "--", ":", "-.")


@dataclass(frozen=True)
class PlotSeries:
    """一条曲线需要的最小绘图信息。

    ``label`` 是图例文字（**英文**，见模块注释），``variant`` 只用来决定线型与透明度，
    不参与显示——变体名可能是中文，进 matplotlib 会变方块。
    """

    run_dir: Path
    label: str
    color: str
    linestyle: str = "-"
    alpha: float = 1.0
    variant: str | None = None


def series_for(
    runs: Sequence[data.RunInfo], *, color_of: Callable[[str], str] | None = None
) -> list[PlotSeries]:
    """把运行列表编成绘图序列：**同族同色，同族内的不同 (实现, 变体) 换线型**。

    只靠颜色不够用：一个算法跑十个变体时七色板会绕回来撞色，而撞色的两条恰好是
    同一个算法，看起来就像「同一条线画了两遍」。线型是第二维度。

    线型的分配是**按 (实现, 变体) 的字典序排完再编号**，不是按传入顺序——传入顺序
    取决于运行的修改时间，加一个新的运行会让所有线型整体平移，同一张图前后两次刷新
    就长得不一样了。排完序再编号，只要页面上选中的那一组运行不变，得到的就是同一张图。

    线型只有四种，同族超过四组时开始绕回；绕回的那几条靠 ``alpha`` 区分（每条递减 0.1，
    到 0.35 为止）。这是刻意的取舍：变体扫描里同一个族最多也就十来条，
    再多本来就该收窄筛选而不是硬塞进一张图。

    图例文字在变体名是中文时省略变体——图内文字必须英文（matplotlib 默认字体不含 CJK），
    中文变体名会渲染成方块。这种情况下列表里会出现同名标签，但颜色与线型仍然分得开，
    而中文变体名在列表与详情页里是正常显示的。
    """
    ordered = sorted(runs, key=lambda info: (info.algorithm or "", info.variant or ""))
    # 每个族内，把不同的 (实现, 变体) 编号——同一个键每次都会拿到同一个线型。
    keys_by_family: dict[str, set[tuple[str, str]]] = {}
    for info in ordered:
        keys_by_family.setdefault(info.family or "OTHER", set()).add(
            (info.algorithm or "", info.variant or "")
        )
    slots = {
        family: {key: index for index, key in enumerate(sorted(keys))}
        for family, keys in keys_by_family.items()
    }

    out: list[PlotSeries] = []
    for info in ordered:
        family = info.family or "OTHER"
        key = (info.algorithm or "", info.variant or "")
        index = slots[family][key]
        label = info.algorithm or family
        if info.variant and info.variant.isascii():
            label = f"{label} · {info.variant}"
        out.append(PlotSeries(
            run_dir=info.path,
            label=label,
            color=(color_of or _family_color)(family),
            linestyle=_LINESTYLES[index % len(_LINESTYLES)],
            alpha=max(0.35, 1.0 - 0.1 * index),
            variant=info.variant,
        ))
    return out


def _family_color(family: str) -> str:
    """延迟导入 ``theme``：它顶层 import streamlit，而绘图本该能在纯脚本里跑。"""
    from . import theme  # noqa: PLC0415

    return theme.family_color(family)


def comparison_figure(
    run_dirs: list[Path] | None = None,
    *,
    series: Sequence[PlotSeries] | None = None,
    width: float = 13.0,
) -> bytes | None:
    """多运行对比图：左侧曲线叠加，右侧最终评估均值 ± 标准差。

    两条入口：给 ``series``（推荐，来自 ``series_for``，带族色与线型），或给
    ``run_dirs``（退回按顺序取色板）。传 ``series`` 时 ``run_dirs`` 被忽略。

    数据完全来自 ``data.load_curve`` / ``load_summary``（也就是 ``compare.py`` 用的
    同一套读取函数），但没有复用 ``compare.plot_curves``——那个函数会 ``mkdir``
    再 ``savefig`` 写盘，而这里是每次页面重跑都要调用的，不能有写副作用。
    """
    if series is None:
        series = [
            PlotSeries(run_dir=path, label=path.name, color=_PALETTE[index % len(_PALETTE)])
            for index, path in enumerate(run_dirs or [])
        ]
    if not series:
        return None

    curves: list[tuple[str, Any, Any, PlotSeries]] = []
    summaries: list[tuple[str, float, float, PlotSeries]] = []
    for item in series:
        curve = data.load_curve(item.run_dir)
        if curve is not None:
            curves.append((item.label, curve[0], curve[1], item))
        summary = data.load_summary(item.run_dir)
        if summary:
            summaries.append(
                (
                    item.label,
                    float(summary.get("mean_reward", 0.0)),
                    float(summary.get("std_reward", 0.0)),
                    item,
                )
            )

    if not curves and not summaries:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(width, 5.2), constrained_layout=True)
    figure.patch.set_facecolor("#ffffff")
    curve_axis, final_axis = axes

    for name, steps, values, item in curves:
        curve_axis.plot(steps, values, linewidth=1.9, label=name, color=item.color,
                        linestyle=item.linestyle, alpha=item.alpha)
    curve_axis.set(
        title="Training curve (periodic evaluation mean)",
        xlabel="Timesteps",
        ylabel="Mean reward",
    )
    curve_axis.grid(alpha=0.25, color="#dfe5e2")
    curve_axis.spines[["top", "right"]].set_visible(False)
    if curves:
        curve_axis.legend(fontsize=8, frameon=False)

    if summaries:
        labels = [item[0] for item in summaries]
        means = [item[1] for item in summaries]
        stds = [item[2] for item in summaries]
        colors = [item[3].color for item in summaries]
        final_axis.barh(labels, means, xerr=stds, color=colors, alpha=0.9, height=0.62,
                        error_kw={"ecolor": "#69757b", "capsize": 3, "lw": 1})
        final_axis.axvline(0, color="#182126", linewidth=0.8)
        final_axis.set(title="Final evaluation (error bar = std)", xlabel="Mean reward")
        final_axis.grid(alpha=0.25, axis="x", color="#dfe5e2")
        final_axis.spines[["top", "right"]].set_visible(False)

    return fig_to_png(figure)
