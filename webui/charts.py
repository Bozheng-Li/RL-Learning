"""altair 图构造层：对比页的分面曲线、详情页的全程面板。

这一层**不 import streamlit**，只返回 altair 的 Chart 对象，所以能脱离界面单测
（``webui/tests/test_charts.py`` 直接断言 ``.to_dict()``）。配色走延迟导入
``theme``——它顶层 import streamlit。

与 ``plots.py``（matplotlib）的分工：

- **屏幕上看到的一切图都由这里构造**。altair 走浏览器渲染，**中文不会变方块**
  （与 matplotlib 默认字体不含 CJK 的处境完全不同），所以这里的标题、轴名、
  图例都可以直接用中文。
- ``plots.py`` 只保留「导出一张静态 PNG 插进报告」这一条后备路径。

两条路径共用 ``series_style`` 的配色与线型规则，保证同一个变体在导出的 PNG 与
屏幕上的图里读到的是同一种线型。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import altair as alt
import pandas as pd

#: 线型序列。同一个族里不同的 ``(实现, 变体)`` 按**字典序**编号后循环取用——
#: 排序编号而不是按传入顺序编号，是为了让前后两次刷新画出来的是同一张图
#: （传入顺序取决于目录 mtime，新加一个运行会让所有线型整体平移）。
_DASHES: tuple[tuple[int, ...], ...] = (
    (1, 0),        # solid
    (6, 3),        # dashed
    (2, 2),        # dotted
    (6, 3, 2, 3),  # dashdot
)

#: 基准线的线宽。基准是唯一能跨实现对读的一组，画粗一点让它在面板里站得住。
_BASELINE_WIDTH = 3.0

#: 变体线宽。变体是「同一个算法的另一档配置」，比基准细，视觉上从属。
_VARIANT_WIDTH = 1.8

#: 刷选之外的曲线压暗到这个不透明度。0.2 仍看得见轮廓，但不会跟选区里抢注意力。
_DIM_OPACITY = 0.16
_FOCUS_OPACITY = 0.95

#: 误差带的填充透明度。带是面不是线，同一个不透明度在面上看起来重得多，
#: 所以它单独一套数值，不跟着曲线的 0.95/0.16 走。
_BAND_OPACITY = 0.18

#: 刷选之外误差带保留的比例。带压到 0 会让那段区间整个消失（看上去像数据缺了），
#: 留一点轮廓才是「变暗」而不是「没有」。
_DIM_RATIO = 0.35


@dataclass(frozen=True)
class CurveSeries:
    """一条曲线的样式：标签、族、颜色、线型、线宽。

    ``variant`` 为 ``None`` 表示基准。``label`` 是图例文字，**可以是中文**——
    altair 在浏览器里渲染，没有 matplotlib 那个字体问题。
    """

    label: str
    family: str
    variant: str | None
    color: str
    dash: tuple[int, ...]
    width: float

    @property
    def is_baseline(self) -> bool:
        return self.variant is None


def series_style(runs: Sequence[Any], *, label_of: Callable[[Any], str] | None = None) -> list[CurveSeries]:
    """把运行列表编成曲线样式：**同族同色，同族内的不同 (实现, 变体) 换线型**。

    规则与 ``plots.series_for`` 一致（那条路径用于导出 PNG）：

    - 色按**族**取（``theme.family_color``）——一眼认出「这几条属于同一个算法族」；
    - 线型按族内的 ``(实现, 变体)`` 字典序编号，四种循环；
    - 基准恒为实线且更粗，变体细一些，视觉上从属；
    - 变体线型**跳过实线**：同一个面板里基准与一个变体撞成同一条实线就分不开了。

    返回的列表按 ``runs`` 的传入顺序，与调用方给的数据行一一对应。
    """
    ordered = sorted(runs, key=lambda info: (info.algorithm or "", info.variant or ""))
    slots: dict[str, dict[tuple[str, str], int]] = {}
    for info in ordered:
        family = getattr(info, "family", None) or "OTHER"
        slots.setdefault(family, {})
    for family, table in slots.items():
        keys = sorted(
            {
                (getattr(info, "algorithm", None) or "", getattr(info, "variant", None) or "")
                for info in ordered
                if (getattr(info, "family", None) or "OTHER") == family
            }
        )
        for index, key in enumerate(keys):
            table[key] = index

    out: list[CurveSeries] = []
    for info in runs:
        family = getattr(info, "family", None) or "OTHER"
        key = (getattr(info, "algorithm", None) or "", getattr(info, "variant", None) or "")
        index = slots.get(family, {}).get(key, 0)
        label = label_of(info) if label_of else _default_label(info)
        out.append(
            CurveSeries(
                label=label,
                family=family,
                variant=getattr(info, "variant", None),
                color=_family_color(family),
                dash=_DASHES[index % len(_DASHES)],
                width=_BASELINE_WIDTH if getattr(info, "variant", None) is None else _VARIANT_WIDTH,
            )
        )
    return out


def _default_label(info: Any) -> str:
    algorithm = getattr(info, "algorithm", None) or getattr(info, "family", None) or "?"
    variant = getattr(info, "variant", None)
    return f"{algorithm} · {variant}" if variant else str(algorithm)


def _family_color(family: str) -> str:
    """延迟导入 ``theme``：它顶层 import streamlit，而这一层要能脱离界面单测。"""
    from . import theme  # noqa: PLC0415

    return theme.family_color(family)


def _dedupe(labels: Sequence[str]) -> list[str]:
    """按首次出现去重，保持顺序。altair 的 scale domain 不接受重复项。"""
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def comparison_curves(
    frame: pd.DataFrame,
    styles: Sequence[CurveSeries],
    *,
    facet_field: str = "算法",
    columns: int = 2,
    height: int = 200,
    width: int = 340,
    x_title: str = "训练步数",
    y_title: str | None = None,
) -> alt.Chart:
    """按算法分面的学习曲线：每个面板里叠着这个算法的各档配置。

    ``frame`` 需要这些列：``算法``（分面）、``步数``（x）、``回报``（y）、
    ``曲线``（图例，与 ``styles`` 的 ``label`` 对应）。

    三件事同时成立才谈得上「读得出来」：

    - **分面**：一个面板一个算法。面板之间互不干扰，「哪个算法学得快」一眼可见。
    - **面板内独立 y 轴**（``resolve_scale(y="independent")``）：REINFORCE 的回报能到
      -600，跟 DQN 摆在一根轴上会让后者压成一条直线。
    - **跨面板共享一把刷子**（``selection_interval`` 加在分面之前，参数提升到分面层）：
      在任意一个面板里横向拖选一个步数区间，其它面板同步变暗，于是「同一个训练阶段里
      各算法分别在哪」可以直接对读。

    颜色与线型的 domain 显式给定：变体名可能不是 ASCII，交给 altair 自己排序会让
    图例顺序在两次刷新之间跳动。
    """
    labels = _dedupe([item.label for item in styles])
    color_of = {item.label: item.color for item in styles}
    dash_of = {item.label: list(item.dash) for item in styles}
    width_of = {item.label: item.width for item in styles}

    brush = alt.selection_interval(encodings=["x"], name="compare_brush")
    base = (
        alt.Chart(frame)        .mark_line(strokeCap="round")
        .encode(
            x=alt.X("步数:Q", title=x_title),
            y=alt.Y("回报:Q", title=y_title),
            color=alt.Color(
                "曲线:N",
                title="运行",
                scale=alt.Scale(domain=labels, range=[color_of[label] for label in labels]),
            ),
            strokeDash=alt.StrokeDash(
                "曲线:N",
                scale=alt.Scale(
                    domain=labels, range=[dash_of[label] for label in labels]
                ),
                legend=None,
            ),
            strokeWidth=alt.StrokeWidth(
                "曲线:N",
                scale=alt.Scale(
                    domain=labels, range=[width_of[label] for label in labels]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{facet_field}:N", title="算法"),
                alt.Tooltip("曲线:N", title="运行"),
                alt.Tooltip("步数:Q", title="步数", format=","),
                alt.Tooltip("回报:Q", title="回报", format=".1f"),
            ],
            opacity=alt.condition(brush, alt.value(_FOCUS_OPACITY), alt.value(_DIM_OPACITY)),
        )
        .add_params(brush)
        .properties(height=height, width=width)
    )
    return base.facet(
        facet=alt.Facet(f"{facet_field}:N", title=None), columns=columns
    ).resolve_scale(y="independent")


def stack(
    panels: Sequence[alt.Chart],
    *,
    brush: alt.Parameter | None = None,
    spacing: int = 6,
) -> alt.Chart:
    """把若干面板竖着摞起来，**独立 y 轴**，共用一个刷子参数。

    ``brush`` 不给就现造一个。传进来的每个面板必须已经把同一个 ``brush`` 加进了
    自己的 ``add_params``（``line_panel`` / ``line_with_band`` 的 ``brush=`` 参数
    就是干这个的），altair 会自动去重成一个同名参数，于是拖动任意一个面板，
    其余面板同步变暗。
    """
    if not panels:
        raise ValueError("至少需要一个面板")
    shared = brush or alt.selection_interval(encodings=["x"], name="journey_brush")
    if brush is None:
        # 现造的刷子还没被任何面板引用，这里补挂到最上面一个，否则它不生效。
        panels = [panels[0].add_params(shared), *panels[1:]]
    with warnings.catch_warnings():
        # altair 见到配置相同的 selection 会自动合并成一个，并为此发一条 UserWarning。
        # **这里要的就是合并**（同一个参数对象被多个面板引用，拖动才联动），
        # 所以把这条噪音按下去——不清掉的话，这一页每渲染一次就刷一行警告。
        warnings.filterwarnings(
            "ignore", message="Automatically deduplicated selection parameter"
        )
        return alt.vconcat(*panels, spacing=spacing).resolve_scale(y="independent")


def line_panel(
    frame: pd.DataFrame,
    *,
    color: str,
    dash: tuple[int, ...] = (1, 0),
    width: float = 1.6,
    height: int = 132,
    panel_width: int = 880,
    x_title: str = "训练步数",
    y_title: str | None = None,
    brush: alt.Parameter | None = None,
    series_field: str | None = None,
    title: str | None = None,
) -> alt.Chart:
    """一张曲线面板。``brush`` 给了就跟着它一起变暗（用于跨面板联动）。"""
    encodings: dict[str, Any] = {
        "x": alt.X("步数:Q", title=x_title),
        "y": alt.Y("值:Q", title=y_title),
    }
    if series_field:
        encodings["color"] = alt.Color(f"{series_field}:N", title=None)
        encodings["strokeDash"] = alt.StrokeDash(
            f"{series_field}:N",
            scale=alt.Scale(domain=["实线", "虚线"], range=[list(dash), [6, 3]]),
            legend=None,
        )
    else:
        encodings["color"] = alt.value(color)
    if brush is not None:
        encodings["opacity"] = alt.condition(
            brush, alt.value(_FOCUS_OPACITY), alt.value(_DIM_OPACITY)
        )
    chart = alt.Chart(frame).mark_line(strokeCap="round").encode(**encodings)
    if brush is not None:
        # 参数提升：多个面板共用同一个参数对象，拖动一个面板会让其余面板同步。
        chart = chart.add_params(brush)
    chart = chart.properties(height=height, width=panel_width)
    if title is not None:
        chart = chart.properties(
            title=alt.TitleParams(text=title, anchor="start", fontSize=12)
        )
    return chart


def journey_lines(
    panels: Sequence[tuple[str, pd.DataFrame]],
    *,
    height: int = 132,
    panel_width: int = 880,
    x_title: str = "训练步数",
    brush: alt.Parameter | None = None,
) -> alt.Chart:
    """把若干 ``(标题, 数据)`` 竖着摞成共用一把刷子的面板组。

    ``数据`` 需要 ``步数`` 与 ``值`` 两列。每个面板**独立 y 轴**——回报是几百的量级，
    KL 是小数点后三位，共用一根轴等于只画得出回报那一条。

    只有最下面一个面板画 x 轴标题：上面几个的刻度标签仍然保留（对读时间要的），
    但重复的五遍轴名只会占地方。
    """
    shared = brush or alt.selection_interval(encodings=["x"], name="journey_brush")
    charts: list[alt.Chart] = []
    last = len(panels) - 1
    for index, (title, frame) in enumerate(panels):
        charts.append(
            line_panel(
                frame,
                color="#2563eb",
                height=height,
                panel_width=panel_width,
                x_title=x_title if index == last else "",
                y_title=None,
                brush=shared,
                title=title,
            )
        )
    return stack(charts, brush=shared)


def action_areas(
    frame: pd.DataFrame,
    *,
    value_field: str = "占比",
    category_field: str = "动作",
    value_format: str = "%",
    height: int = 170,
    panel_width: int = 880,
    x_title: str = "训练步数",
    brush: alt.Parameter | None = None,
    title: str | None = None,
) -> alt.Chart:
    """动作分布的堆叠面积图（**长表**：``步数`` / ``动作`` / ``占比``）。

    离散动作空间里各动作占比天然和为 1，堆叠起来正好是一条 100% 的带子；
    某个颜色被挤没了就是「这个动作再也不被选中」，是策略坍缩最直接的证据。

    长表而不是宽表：altair 的 ``color`` 要的是「类别进一列」，宽表的每一个动作列
    都得单独配一次色，动作数一变就得改代码。
    """
    encodings: dict[str, Any] = {
        "x": alt.X("步数:Q", title=x_title),
        "y": alt.Y(
            f"{value_field}:Q",
            stack="zero",
            title=None,
            axis=alt.Axis(format=value_format),
        ),
        "color": alt.Color(f"{category_field}:N", title=None),
    }
    if brush is not None:
        encodings["opacity"] = alt.condition(
            brush, alt.value(_FOCUS_OPACITY), alt.value(_DIM_OPACITY)
        )
    chart = alt.Chart(frame).mark_area(opacity=0.85).encode(**encodings)
    chart = chart.properties(height=height, width=panel_width)
    if brush is not None:
        chart = chart.add_params(brush)
    if title is not None:
        chart = chart.properties(
            title=alt.TitleParams(text=title, anchor="start", fontSize=12)
        )
    return chart


def bars(
    frame: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    color_field: str | None = None,
    color: str = "#2563eb",
    baseline: float = 0.0,
    height: int = 200,
    panel_width: int = 880,
    x_title: str = "回合",
    y_title: str = "回报",
) -> alt.Chart:
    """一根一根的柱状图。用于「最终评估的 10 个回合」。

    ``baseline`` 画一条横向参考线（传均值就是「这 10 回合的均值」），没有它，
    十根孤立的柱子读不出「这个模型稳不稳」。
    """
    encodings: dict[str, Any] = {
        "x": alt.X(f"{x_field}:O", title=x_title),
        "y": alt.Y(f"{y_field}:Q", title=y_title),
    }
    if color_field:
        encodings["color"] = alt.Color(f"{color_field}:N", title=None)
    else:
        encodings["color"] = alt.value(color)
    chart = alt.Chart(frame).mark_bar().encode(**encodings)
    rule = (
        alt.Chart(pd.DataFrame({y_field: [baseline]}))
        .mark_rule(color="#bf3c48", strokeDash=[4, 3], size=1.5)
        .encode(y=alt.Y(f"{y_field}:Q"))
    )
    return (chart + rule).properties(height=height, width=panel_width)


def line_with_band(
    frame: pd.DataFrame,
    *,
    lower: str,
    upper: str,
    middle: str,
    height: int = 200,
    panel_width: int = 880,
    x_title: str = "训练步数",
    y_title: str = "评估回报",
    brush: alt.Parameter | None = None,
    title: str | None = None,
) -> alt.Chart:
    """均值线 + 半透明误差带。周期评估每档只有几个回合，没有带就看不出可信度。

    给了 ``brush`` 就跟着它一起变暗，可以并进 ``stack`` 的面板组里——
    这时误差带与均值线**当成一个整体**压暗：带不跟着暗的话，刷选之外的那段会变成
    「线没了但区间还在」，读起来像是那一段只有区间是可信的，恰恰相反。
    带本身就是 0.16 的填充，所以它走的是自己的一套明暗（不是曲线的 0.95 / 0.16）。
    """
    if brush is not None:
        curve_dim = {
            "opacity": alt.condition(
                brush, alt.value(_FOCUS_OPACITY), alt.value(_DIM_OPACITY)
            )
        }
        band_opacity = alt.condition(
            brush, alt.value(_BAND_OPACITY), alt.value(_BAND_OPACITY * _DIM_RATIO)
        )
    else:
        curve_dim = {}
        band_opacity = alt.value(_BAND_OPACITY)
    band = (
        alt.Chart(frame)
        .mark_area(color="#2563eb")
        .encode(
            x=alt.X("步数:Q", title=x_title),
            y=alt.Y(f"{middle}:Q", title=y_title),
            y2=alt.Y2(f"{upper}:Q"),
            opacity=band_opacity,
        )
    )
    lower_line = (
        alt.Chart(frame)
        .mark_line(color="#2563eb", strokeWidth=1.0, strokeDash=[3, 3])
        .encode(
            x=alt.X("步数:Q"),
            y=alt.Y(f"{lower}:Q"),
            **curve_dim,
        )
    )
    middle_line = (
        alt.Chart(frame)
        .mark_line(color="#2563eb", strokeWidth=2.0)
        .encode(x=alt.X("步数:Q"), y=alt.Y(f"{middle}:Q"), **curve_dim)
    )
    chart = (band + lower_line + middle_line).properties(height=height, width=panel_width)
    if brush is not None:
        chart = chart.add_params(brush)
    if title is not None:
        chart = chart.properties(
            title=alt.TitleParams(text=title, anchor="start", fontSize=12)
        )
    return chart
