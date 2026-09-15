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
from pathlib import Path
from typing import Any

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


def comparison_figure(run_dirs: list[Path], *, width: float = 13.0) -> bytes | None:
    """多运行对比图：左侧曲线叠加，右侧最终评估均值 ± 标准差。

    数据完全来自 ``data.load_curve`` / ``load_summary``（也就是 ``compare.py`` 用的
    同一套读取函数），但没有复用 ``compare.plot_curves``——那个函数会 ``mkdir``
    再 ``savefig`` 写盘，而这里是每次页面重跑都要调用的，不能有写副作用。
    """
    if not run_dirs:
        return None

    curves: list[tuple[str, Any, Any]] = []
    summaries: list[tuple[str, float, float]] = []
    for run_dir in run_dirs:
        curve = data.load_curve(run_dir)
        if curve is not None:
            curves.append((run_dir.name, curve[0], curve[1]))
        summary = data.load_summary(run_dir)
        if summary:
            summaries.append(
                (
                    run_dir.name,
                    float(summary.get("mean_reward", 0.0)),
                    float(summary.get("std_reward", 0.0)),
                )
            )

    if not curves and not summaries:
        return None

    figure, axes = plt.subplots(1, 2, figsize=(width, 5.2), constrained_layout=True)
    curve_axis, final_axis = axes

    for name, steps, values in curves:
        curve_axis.plot(steps, values, linewidth=1.7, label=name)
    curve_axis.set(
        title="Training curve (periodic evaluation mean)",
        xlabel="Timesteps",
        ylabel="Mean reward",
    )
    curve_axis.grid(alpha=0.3)
    if curves:
        curve_axis.legend(fontsize=9)

    if summaries:
        labels = [item[0] for item in summaries]
        means = [item[1] for item in summaries]
        stds = [item[2] for item in summaries]
        # 用列表推导逐个上色，避免不同 matplotlib 版本对 color= 列表的支持差异。
        colors = ["#3182bd" if mean >= 0 else "#de2d26" for mean in means]
        final_axis.barh(labels, means, xerr=stds, color=colors, alpha=0.85,
                        error_kw={"ecolor": "#444", "capsize": 3, "lw": 1})
        final_axis.axvline(0, color="#333", linewidth=1)
        final_axis.set(
            title="Final evaluation (error bar = std)",
            xlabel="Mean reward",
        )
        final_axis.grid(alpha=0.3, axis="x")

    return fig_to_png(figure)
