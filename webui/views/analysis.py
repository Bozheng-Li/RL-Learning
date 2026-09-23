"""变体分析：同一个算法对哪个参数敏感，以及这些结论有多可信。

这一页读的是**变体批次**——一次 ``variants.py`` 调用产生的几百个运行。它回答的问题
与「跨运行对比」页不同：那边问「哪个算法好」，这里问「同一个算法把 lr 换一档会怎样」。

页面刻意把**不确定性放在最前面**，而不是塞进附录。原因很实在：这批数据里，配置逐
字段相同的两次运行，3 种子平均分也能差出中位 24 分（见 ``analysis.twin_null``），
所以不先划出「多大才算有东西」的线，后面每一行数字都会被读成效应。
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from webui import analysis, data, nav, theme
from webui.views import _shared

#: 「哪些变体值得细看」那一节的判定门槛。
_STRONG_CONSENSUS = 0.75
_STRONG_P = 0.05

#: 维度在界面上出现的顺序。名字由 ``analysis.DIMENSIONS`` 定义，这里只定展示顺序。
_DIMENSION_ORDER = ("学习超参", "环境扰动", "训练预算", "评估种子", "其它")


def _overview(runs: list[data.RunInfo]) -> None:
    """这一批跑了什么。"""
    batch, stamp = analysis.select_batch(runs)
    if not batch:
        theme.empty_state("没有找到变体批次", "先跑一轮 variants.py，再回来看这一页。")
        return

    variants = {info.variant for info in batch if info.variant}
    algorithms = {info.algorithm for info in batch if info.algorithm}
    theme.stat_row([
        ("批次", stamp or "-", "目录名尾部的时刻"),
        ("运行", len(batch), f"{len(algorithms)} 个算法"),
        ("变体", len(variants), "含基准"),
        ("终端", len({analysis.dimension_of(v) for v in variants}), "改动类别"),
    ])


def _reliability(result: dict[str, Any]) -> None:
    """先把「多小的 Δ 不能当回事」划出来，再让用户看效应表。

    这不是方法学附录。这批数据里配置完全相同的两次运行（只落在不同型号的卡上），
    3 种子平均分也能差出中位 24 分——不先说这件事，下面每一行 Δ 都会被当成效应读。
    """
    theme.section_head(
        "先看噪声底线",
        "配置一字未改时能差多少 —— 低于这条线的 Δ 不当结论看",
    )

    repro = result["repro"]
    twin = result["twin_null"]
    same = repro["placebo_same_model"]
    cross = repro["placebo_cross_model"]

    left, middle, right = st.columns(3)
    with left:
        st.metric(
            "同型号的安慰剂 Δ（中位）",
            f"{same['median']:.1f}" if same["n"] else "—",
            help="变体与基准落在同一型号的卡上、又没改任何训练参数时的 |Δ| 中位数。"
                 "实测这一档逐位相同——它是纯粹「评估那 10 个回合的运气」。",
        )
    with middle:
        st.metric(
            "跨型号的安慰剂 Δ（中位）",
            f"{cross['median']:.1f}" if cross["n"] else "—",
            help="同样没改训练，但两侧落在不同型号的卡上。这里还多一层硬件偏置。",
        )
    with right:
        st.metric(
            "配对里跨型号的比例",
            f"{repro['cross_model_share']:.0%}" if repro["pairs"] else "—",
            help=f"{repro['cross_model_pairs']}/{repro['pairs']} 个 (变体, 种子) 配对"
                 "两侧不在同一型号的卡上。这个比例决定「只报同型号子集」能保留多少数据。",
        )

    if twin.get("n"):
        theme.caption(
            f"<strong>零分布（最硬的一条）</strong>：把配置逐字段相同的两次运行（"
            f"<code>SAC</code> 与 <code>SB3-SAC</code> 这对孪生，本项目里它们解析到同一个"
            f"SB3 实现，超参一字不差）拿来对比，{twin['n']} 组「3 种子平均 Δ」的 |Δ| 中位 "
            f"{twin['median']:.1f}、P90 {twin['p90']:.1f}、最大 {twin['max']:.1f}。"
            f"其中 {twin['share_over_100']:.0%} 超过 100 分。"
            f"<strong>所以 3 种子平均 Δ 要到几十上百分才值得当回事</strong>——"
            f"配置没改也能到这个量级。"
        )
        theme.caption(
            "这条噪声的来源已经钉死：<strong>同一型号的卡上跑同一个配置逐位相同</strong>"
            "（孪生对照里同型号的 "
            f"{sum(row['同型号逐位相同'] for row in result['twin'])}/"
            f"{sum(row['同型号配对'] for row in result['twin'])} 组全部逐位相同），"
            "<strong>换型号才分叉</strong>。它是确定性的硬件偏置，不是碰运气——"
            "事后没法从单个运行里剔掉，只能靠绑卡时让对照与处理落在同型号的卡上来回避。"
        )


def _twin_table(result: dict[str, Any]) -> None:
    rows = result["twin"]
    frame = pd.DataFrame([
        {
            "算法": row["算法"],
            "对照实现": row["对照实现"],
            "是不是同一个实现": "是（孪生）" if row["同名同实现"] else "否（两个实现）",
            "同型号配对": row["同型号配对"],
            "同型号逐位相同": row["同型号逐位相同"],
            "跨型号配对": row["跨型号配对"],
            "跨型号逐位相同": row["跨型号逐位相同"],
            "型号未知": row["型号未知"],
        }
        for row in rows
    ])
    with st.expander("可复现性明细：孪生对照表", expanded=False):
        theme.caption(
            "`SAC` / `TD3` / `DQN` 在本项目里**不是自研的**——`resolve_name` 把这三个名字"
            "回退到 `SB3-*`，所以它们与 `SB3-SAC` / `SB3-TD3` / `SB3-DQN` 是**同一个实现"
            "的两次运行**，超参逐字段相同。它们之间的差纯粹反映跑在了哪台卡上："
            "**同型号的配对全部逐位相同，跨型号的大多分叉。**"
            "`A2C` / `PPO` 两行是**两个不同实现**（自研 vs SB3），差异本来就该有，"
            "不是反例——所以它们的两列不能读成可复现性。"
        )
        st.dataframe(frame, hide_index=True, width="stretch", height=theme.fit_height(len(frame)))


def _variant_cards(result: dict[str, Any]) -> None:
    """按维度分组，每个变体一行总账：方向一致性、显著个数、平均 Δ。"""
    summaries = result["summaries"]
    scales = result["scales"]
    usable = [item for item in summaries if not analysis.is_side_quest(item.variant)]

    theme.section_head(
        "变体总账",
        f"{len(usable)} 个变体 × 各自算法 —— 方向一致性比单个 z 更值得信",
    )
    if scales:
        theme.caption(
            "判据：每个 (变体, 算法) 的 3 种子平均 Δ 除以该算法自己的噪声尺度 σ 再乘 √3，"
            "得 <code>z</code>；<code>|z| ≥ 2</code> 记作显著。<strong>σ 按算法分别估</strong>——"
            "REINFORCE 的回报在 -1000 量级，噪声本来就跟 DQN 差一个数量级。"
            "但零分布说明单个 z 只是线索，<strong>真正有力的是跨算法方向一致</strong>。"
        )

    by_dimension: dict[str, list[analysis.VariantSummary]] = {}
    for item in usable:
        by_dimension.setdefault(item.dimension, []).append(item)

    order = [name for name in _DIMENSION_ORDER if name in by_dimension]
    order += [name for name in by_dimension if name not in order]

    for dimension in order:
        items = by_dimension[dimension]
        theme.section_head(f"{dimension}", f"{len(items)} 个变体")
        for item in items:
            _variant_card(item, scales)


def _variant_card(item: analysis.VariantSummary, scales: dict[str, float]) -> None:
    """一个变体的卡片：一句结论 + 逐算法的 Δ 条。"""
    strong = item.consensus >= _STRONG_CONSENSUS and item.sign_p < _STRONG_P
    if strong and item.negative > item.positive:
        tone, verdict = "danger", (
            f"一致有害 — {item.negative}/{item.algorithms} 个算法变差，符号检验 p={item.sign_p:.3f}"
        )
    elif strong:
        tone, verdict = "success", (
            f"一致有利 — {item.positive}/{item.algorithms} 个算法变好，符号检验 p={item.sign_p:.3f}"
        )
    elif item.significant == 0:
        tone, verdict = "neutral", "分不出来 — 与噪声同量级"
    else:
        tone, verdict = "neutral", f"{item.direction}，但方向不一致（p={item.sign_p:.3f}）"

    tones = {
        "success": (theme.SUCCESS, theme.SUCCESS_SOFT),
        "danger": (theme.DANGER, theme.DANGER_SOFT),
        "neutral": (theme.MUTED, theme.SUBTLE),
    }
    color, background = tones[tone]
    rows = "".join(_delta_row(effect, scales) for effect in item.effects)
    st.markdown(
        f'<div class="rl-card" style="margin-bottom:.9rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem">'
        f'<span style="font-weight:700;font-size:1rem">{html.escape(item.variant)}</span>'
        f'<span style="color:{color};background:{background};padding:.15rem .55rem;'
        f'border-radius:999px;font-size:.76rem;white-space:nowrap">{html.escape(verdict)}</span>'
        f'</div>'
        f'<div style="color:{theme.MUTED};font-size:.78rem;margin:.3rem 0 .5rem">'
        f'{item.algorithms} 个算法 · 平均 Δ {item.mean_delta:+.1f} 分 · '
        f'{item.harmful} 个显著变差、{item.favorable} 个显著变好</div>'
        + rows
        + "</div>",
        unsafe_allow_html=True,
    )


def _delta_row(effect: analysis.Effect, scales: dict[str, float]) -> str:
    """一个算法的完整一行（含 ``rl-algo-row`` 外壳）。"""
    scale = scales.get(effect.algorithm) or 0.0
    # 条长按「Δ 是噪声尺度的几倍」算，而不是按绝对分——各算法的分数尺度差得远
    # （REINFORCE 在 -1000 量级，DQN 在 0 附近），同一根轴量不出可比性。
    norm = min(1.0, abs(effect.mean_delta) / (3.0 * scale)) if scale else 0.0
    color = theme.MUTED
    if effect.significant:
        color = theme.DANGER if effect.mean_delta < 0 else theme.SUCCESS
    width = max(4.0, norm * 100) if norm > 0 else 2.0
    flag = ""
    if effect.mixed_cards:
        flag = (
            f'<span style="color:{theme.WARNING};font-size:.68rem;margin-left:.4rem" '
            'title="全部配对都跨了型号，Δ 的绝对值不可信，只有方向还能看">⚠跨型号</span>'
        )
    elif effect.cross_model:
        flag = (
            f'<span style="color:{theme.MUTED};font-size:.68rem;margin-left:.4rem">'
            f'{effect.cross_model}/{effect.n} 跨型号</span>'
        )
    value = f"{effect.mean_delta:+.1f}" + (f" · z={effect.z:+.1f}" if effect.z is not None else "")
    return (
        '<div class="rl-algo-row">'
        f'<span class="rl-algo-name">{html.escape(effect.algorithm)}</span>'
        '<span class="rl-algo-track">'
        f'<span class="rl-algo-bar" style="width:{width:.1f}%;background:{color}"></span>'
        '</span>'
        f'<span class="rl-algo-value" style="width:auto;min-width:6.4rem;'
        f'font-weight:600">{value}</span>{flag}'
        '</div>'
    )


def _side_quests(result: dict[str, Any]) -> None:
    """预算与评估种子：它们不能当敏感性 Δ 读，各自回答一个不同的问题。"""
    budget = result["budget"]
    if budget:
        _budget_section(budget)

    seeds = result["seeds"]
    if seeds:
        with st.expander("基准的跨种子离散度（不换任何参数，只换种子）", expanded=False):
            theme.caption(
                "这是「同一个配置换个随机种子」的尺度。种子在这里指的是**训练**的随机性"
                "（初始权重、环境重置、采样），与 `evalseed-*` 变体换的**评估**初始条件不同。"
            )
            frame = pd.DataFrame(seeds)
            st.dataframe(
                frame, hide_index=True, width="stretch",
                height=theme.fit_height(len(frame)),
                column_config={
                    "均值": st.column_config.NumberColumn(format="%.1f"),
                    "标准差": st.column_config.NumberColumn(format="%.1f"),
                    "最低": st.column_config.NumberColumn(format="%.1f"),
                    "最高": st.column_config.NumberColumn(format="%.1f"),
                },
            )


def _budget_section(budget: dict[str, dict[int, float]]) -> None:
    """训练预算：多少步够用。"""
    theme.section_head("训练预算", "多少步够用 —— 与敏感性 Δ 是两个问题，不能混读")
    theme.caption(
        "预算变体改的是训练步数，所以<strong>不能</strong>拿它与 20 万步基准算敏感性 Δ"
        "——那会把「训练不足」读成「这个参数有害」。这里只并排看各步数档的最终回报，"
        "最大的那一档就是基准本身的 20 万步。"
    )

    pivot = pd.DataFrame(budget).T
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot.index.name = "算法"
    st.dataframe(
        pivot, width="stretch", height=theme.fit_height(len(pivot)),
        column_config={
            step: st.column_config.NumberColumn(label=f"{step // 1000}k", format="%.1f")
            for step in pivot.columns
        },
    )
    st.download_button(
        "导出预算表",
        data=pivot.to_csv().encode("utf-8-sig"),
        file_name=f"budget_{result_stamp()}.csv",
        mime="text/csv",
    )


def result_stamp() -> str:
    """当前批次的时间戳，供下载文件名用。"""
    return str(st.session_state.get("_analysis_stamp") or "variants")


def _detail_table(result: dict[str, Any]) -> None:
    rows = result["table"]
    if not rows:
        return
    with st.expander(f"完整敏感性表（{len(rows)} 行，可导出）", expanded=False):
        theme.caption(
            "每一行是一个 (变体, 算法)。<code>Δ均值</code> 用全部种子，混着跨型号的硬件"
            "偏置；<code>Δ均值（同型号）</code> 只用可复现的同型号配对，数干净但种子少；"
            "<code>跨型号配对数</code> 说明这一行有多少配对是跨型号的。"
        )
        frame = pd.DataFrame(rows)
        frame = frame.sort_values(["维度", "变体", "Δ均值"], ascending=[True, True, True])
        st.dataframe(
            frame, hide_index=True, width="stretch", height=theme.fit_height(len(frame)),
            column_config={
                "Δ均值": st.column_config.NumberColumn(format="%+.1f"),
                "Δ均值（同型号）": st.column_config.NumberColumn(format="%+.1f"),
                "Δ最小": st.column_config.NumberColumn(format="%+.1f"),
                "Δ最大": st.column_config.NumberColumn(format="%+.1f"),
                "Δ波动": st.column_config.NumberColumn(format="%.1f"),
                "z": st.column_config.NumberColumn(format="%+.1f"),
            },
        )
        st.download_button(
            "导出这张表",
            data=frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"sensitivity_{result_stamp()}.csv",
            mime="text/csv",
        )


def _render_gap(runs: list[data.RunInfo]) -> None:
    """往上补一个变体批次的入口。"""
    batch, _ = analysis.select_batch(runs)
    covered = {info.variant for info in batch if info.variant}
    available = analysis.available_variants(runs)
    missing = [name for name in available if name not in covered]
    if not available:
        return
    if not missing:
        theme.caption(f"变体集里的 {len(available)} 个变体都已经有结果了。")
        return
    theme.section_head(
        "还没跑的变体",
        f"{len(missing)} 个：{'、'.join(missing[:12])}" + ("…" if len(missing) > 12 else ""),
    )
    if st.button("去「发起训练」页跑变体", key="analysis-fill-variants", width="stretch"):
        nav.goto("发起训练")
        st.rerun()


def render() -> None:
    theme.page_header(
        "变体分析",
        "同一个算法对哪个参数敏感 —— 以及这些结论有多可信。",
        eyebrow="Reinforce / sensitivity",
    )

    runs = [info for info in _shared.load_runs() if info.summary]
    if not runs:
        theme.empty_state(
            "还没有跑完的运行",
            "变体分析读的是一次 variants.py 产生的整批运行。先到「发起训练」页跑一批。",
        )
        return

    has_variants = any(info.variant for info in runs)
    if not has_variants:
        theme.empty_state(
            "这个目录里还没有变体运行",
            "变体分析需要「同一个算法的多份配置」。用 variants.py 跑一轮，"
            "或到「发起训练」页的变体模式发起。",
        )
        return

    _overview(runs)
    result = analysis.analyse(runs)
    st.session_state["_analysis_stamp"] = result["stamp"] or "variants"

    _reliability(result)
    _twin_table(result)
    _variant_cards(result)
    _side_quests(result)
    _detail_table(result)
    _render_gap(runs)
