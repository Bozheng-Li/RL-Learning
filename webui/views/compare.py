"""跨运行对比：先选一个环境、再选一类改动，然后看图与榜。

这一页只回答两个问题，顺序也是这个顺序：

**「哪个算法好」** —— 基准档。每个算法本来的水平，也是唯一能跨实现对读的一组。

**「同一个算法换一档配置会怎样」** —— 变体档。这里不再把六百个运行摊在一屏上，而是
按**改动类别**（学习超参 / 环境扰动 / 训练预算 / 评估种子）分组，一次只看一类。
一类里同一个算法只有二到五个版本，曲线图上叠得下、排行榜上也读得出，
「同算法多版本」从「一屏几十条同色线」变成「一条基准加几条线型不同的对照」。

曲线图用 altair 分面（一个面板一个算法、面板内独立 y 轴、跨面板共享刷子），
不再把上百条线叠进一张 matplotlib 图——那种图连图例都盖住半张画面。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from webui import analysis, charts, data, nav, theme
from webui.views import _shared

#: 「落地录像」一行放几个。算法多的时候会折成多行，超过这个数每格就小得看不清了。
_VIDEO_COLUMNS = 4

#: 最多放几个录像。变体批次跑完一个环境有几百个运行，每个都塞一个 ``st.video``
#: 会把页面拖死，也没人看得过来。按最终回报挑最好的几个。
_VIDEO_LIMIT = 12

#: 每个族最多画几条条。按维度筛选之后每族只剩「一条基准 + 该维度的几个版本」，
#: 正常不会触发；「全部」档下仍会，保留一个上限免得条形墙又回来。
_BAR_LIMIT = 20

#: 曲线图最多画几个算法的面板。分面是 2 列，8 个算法 = 4 行，再多人就要滚半天，
#: 而面板一多，每个面板的高度也就压到读不出曲线形状了。
_ALGO_LIMIT = 8

#: 「基准」与「全部」两个不由 ``analysis.DIMENSIONS`` 定义的档位。
_DIM_BASELINE = "基准"
_DIM_ALL = "全部"


def _dimensions_available(runs: list[data.RunInfo]) -> list[str]:
    """这个环境上真的有的维度，按 ``analysis.DIMENSIONS`` 的顺序。

    只列真实存在的：一个还没跑过预算扫描的环境，界面上不该出现「训练预算」这个
    档位——点进去是空的，比没有这个选项更让人困惑。
    """
    present = {analysis.dimension_of(info.variant) for info in runs if info.variant}
    order = [name for name, _prefixes in analysis.DIMENSIONS if name in present]
    order += sorted(present - set(order) - {"基准"})
    return [_DIM_BASELINE, *order, _DIM_ALL]


def _dimension_runs(runs: list[data.RunInfo], dimension: str) -> list[data.RunInfo]:
    """这个维度下的运行集合。

    - ``基准`` → 只有基准（没套任何变体覆盖的运行）；
    - 某个变体维度 → **该维度的变体，外加那些变体所属算法的基准**。基准必须跟着来：
      脱离基准，`lr-1e-3` 的回报既不知道算高还是算低，也不知道是它自己还是别人的贡献；
    - ``全部`` → 原样返回。六百个运行会一起铺开，界面上要提示这是刻意动作。
    """
    if dimension == _DIM_BASELINE:
        return [info for info in runs if not info.variant]
    if dimension == _DIM_ALL:
        return list(runs)

    picked = [
        info for info in runs if info.variant and analysis.dimension_of(info.variant) == dimension
    ]
    algorithms = {info.algorithm for info in picked if info.algorithm}
    baselines = [info for info in runs if not info.variant and info.algorithm in algorithms]
    return [*baselines, *picked]


def _dimension_algorithms(runs: list[data.RunInfo], limit: int = _ALGO_LIMIT) -> list[str]:
    """默认选中哪些算法：**按族轮流取**，每族先取运行数最多的那个实现。

    按族轮流而不是按目录顺序取前几个：变体批次里跑得最多的那几个算法全挤在一个族
    （PPO 有两套实现 + 七个变体），按顺序取会落成「这个环境只有 PPO」。
    按族轮流保证每个族都有代表。

    粒度是**算法名**而不是运行名，所以「同配置的多个种子」天然整组进出——截半个
    种子组会让曲线图上那一条凭空少一半种子，读不出离散度。
    """
    buckets: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for info in runs:
        if not info.algorithm:
            continue
        counts[info.algorithm] = counts.get(info.algorithm, 0) + 1
    for info in runs:
        if info.algorithm:
            buckets.setdefault(info.family or "OTHER", [])
            if info.algorithm not in buckets[info.family or "OTHER"]:
                buckets[info.family or "OTHER"].append(info.algorithm)

    order = [family for family in data.FAMILY_ORDER if family in buckets]
    order += sorted(family for family in buckets if family not in set(order))
    for family in order:
        buckets[family].sort(key=lambda name: (-counts.get(name, 0), name))

    picked: list[str] = []
    depth = 0
    while len(picked) < limit:
        added = False
        for family in order:
            if depth >= len(buckets[family]):
                continue
            name = buckets[family][depth]
            if name in picked or len(picked) >= limit:
                continue
            picked.append(name)
            added = True
        if not added:
            break
        depth += 1
    return picked


def _curve_frame(
    runs: list[data.RunInfo],
) -> tuple[pd.DataFrame, list[charts.CurveSeries], list[str]]:
    """把运行编成 altair 要的长表，返回 ``(表, 样式, 没有曲线的运行名)``。

    一列一行（长表）而不是一列一条曲线（宽表）：altair 的分面、图例、刷选都建立在
    「类别进一列」的形状上。

    没有任何曲线数据的运行**单独列出来**交给界面说明，而不是静默丢掉——用户明明
    选了这个运行却在图上找不到它，比少一条线更难排查。
    """
    styles = charts.series_style(runs)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for info, style in zip(runs, styles):
        curve = data.load_curve(info.path)
        if curve is None:
            missing.append(info.name)
            continue
        steps, values = curve
        if len(steps) == 0:
            missing.append(info.name)
            continue
        for step, value in zip(steps, values):
            rows.append(
                {
                    "算法": info.algorithm or info.family,
                    "曲线": style.label,
                    "运行": info.name,
                    "步数": float(step),
                    "回报": float(value),
                }
            )
    return pd.DataFrame(rows), styles, missing


def _render_curves(chosen: list[data.RunInfo]) -> None:
    """学习曲线：分面 + 独立 y 轴 + 跨面板刷子。这一页的主视图，不再折叠。"""
    frame, styles, missing = _curve_frame(chosen)
    theme.section_head(
        "学习曲线",
        f"{len(styles)} 条曲线 · 按算法分面 · 每个面板一根独立的 y 轴",
    )
    if frame.empty:
        st.info("这些运行都还没有可画的曲线（`evaluation/evaluations.npz` 与 `logs/train.monitor.csv` 都为空的运行不参与绘图）。")
        return
    panel_columns = 1 if len(styles) <= 1 else 2
    chart = charts.comparison_curves(frame, styles, columns=panel_columns)
    st.altair_chart(chart, width="stretch")
    theme.caption(
        "每个面板是一个算法，面板里叠着它在当前这档下跑过的各份配置；"
        "**基准是粗实线，变体是细的虚/点线**，同色代表同一个算法族。"
        "在任意一个面板里横向拖选一段步数，其余面板会同步变暗，"
        "于是「同一个训练阶段里各算法分别在哪」可以对读。"
        "面板的 y 轴各自独立——REINFORCE 的回报能到 −600，跟 DQN 共用一根轴会把它压成直线。"
    )
    if missing:
        theme.caption(
            f"⚠️ 有 {len(missing)} 个运行没有曲线数据，没有出现在图上："
            + "、".join(missing[:6])
            + ("…" if len(missing) > 6 else "")
        )


def _family_buckets(
    runs: list[data.RunInfo],
) -> list[tuple[str, list[list[data.RunInfo]]]]:
    """把运行组织成 ``[(族, [ (算法,变体) 组, ... ]), ...]``。

    族按 ``FAMILY_ORDER``（学习顺序）排，族内各组按它们在 ``runs`` 里首次出现的位置排。
    """
    groups: dict[tuple[str, str | None], list[data.RunInfo]] = {}
    for info in runs:
        groups.setdefault((info.algorithm or "", info.variant), []).append(info)
    by_family: dict[str, list[list[data.RunInfo]]] = {}
    for items in groups.values():
        by_family.setdefault(items[0].family or "OTHER", []).append(items)
    order = [family for family in data.FAMILY_ORDER if family in by_family]
    order += sorted(family for family in by_family if family not in set(order))
    return [(family, by_family[family]) for family in order]


def _render_fill_gap(environment: str, runs: list[data.RunInfo]) -> None:
    """把「这个环境上还没跑过的算法」一键预填到发起训练页。

    入口收敛：总览页每张卡只有一个「打开对比」，补齐动作放在这里——用户已经在看这个
    环境的成绩，缺口在哪里最清楚。
    """
    config_name = next((info.config_name for info in runs if info.config_name), None)
    if not config_name:
        return
    try:
        supported = data.compatible_algorithms(data.load_config_dict(config_name))
    except (OSError, ValueError):
        return
    # 只把**基准**运行当作「这个算法跑过了」。变体不算：一个算法只跑了 lr 扫描、
    # 却没有基准时，扫描出的曲线没有对照物，这里应该仍然提示去补基准。
    done = data.baseline_algorithms(runs)
    missing = [name for name in supported if name not in done]
    if not missing:
        theme.caption(f"这个环境上能跑的 {len(supported)} 个算法都已经有结果了。")
        return

    theme.section_head("补齐对照", f"还缺 {len(missing)} 个算法：{'、'.join(missing)}")
    if st.button("用这份配置去跑这些算法", key=f"fill::{environment}", width="stretch"):
        nav.preset(**{
            "train_config": config_name,
            f"train_algorithms::{config_name}": missing,
        })
        nav.goto("发起训练")
        st.rerun()


def _result_table(runs: list[data.RunInfo]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "运行": info.name,
            "算法": info.algorithm or "-",
            "变体": info.variant or "基准",
            "族": info.family,
            "动作空间": data.ACTION_SPACE_LABELS.get(info.action_space, info.action_space),
            "设备": info.device or "cpu",
            "种子": info.seed if info.seed is not None else "-",
            "步数": info.summary.get("timesteps", info.total_timesteps),
            "均值回报": info.summary.get("mean_reward"),
            "标准差": info.summary.get("std_reward"),
            "最小": info.summary.get("min_reward"),
            "最大": info.summary.get("max_reward"),
            "回合数": info.summary.get("episodes"),
        }
        for info in runs
    ])


def _render_leaderboard(runs: list[data.RunInfo], dimension: str) -> None:
    """按**算法族**分组的排行榜：同族内每个 ``(算法, 变体)`` 取最好成绩。

    分组不是排版偏好。两百多行平铺成一面条形墙之后——条的宽度各不相同却共用一根
    归一化轴，色也是六族轮着用——除了「最高的那根最长」什么都读不出来。按族切开之后：

    - 组内每根条只跟同族的比，归一化轴**只在组内**，长度是可信的（族与族之间不比长度，
      比数字，族色只是帮你把同一族的几根条认成一组）。
    - 同族里出现两个实现（``PPO`` 与 ``SB3-PPO``）时，直接读出实现差异；这是本项目
      最核心的对照维度，混在两百行里就彻底消失了。
    - **基准与变体分开着色**：基准用饱和的族色，变体用同色半透明。同一个算法在这个
      维度下的几条并排时，一眼能指出哪一条是参照物。

    连续与离散动作空间的奖励尺度不可比，所以族之间本来就该分开读。
    """
    board = data.environment_leaderboard(runs)
    if not board:
        return
    groups = data.family_groups(board)
    variants = {entry["variant"] for entry in board if entry["variant"]}
    theme.section_head(
        "算法排名",
        f"{len(groups)} 个算法族 · {len(board)} 条"
        + ("（每族内：基准 + 这一类改动下的各份配置）" if variants else "（只有基准）"),
    )

    spaces = {entry["action_space"] for entry in board}
    if len(spaces) > 1:
        theme.caption(
            "⚠️ 这个环境里混了两种动作空间，奖励尺度不可直接比较，所以按族分开排。"
        )
    for family, entries in groups:
        shown = entries[:_BAR_LIMIT]
        hidden = len(entries) - len(shown)
        names = [entry["variant"] or "基准" for entry in shown]
        implementations = sorted({entry["algorithm"] for entry in shown})
        theme.caption(
            f'<span style="color:{theme.family_color(family)};font-weight:700">{family}</span>'
            f" · {'、'.join(implementations)}"
            f" · 本组 {len(names)} 条：{'、'.join(names)}"
            + (f" · 另有 {hidden} 条在下面的结果表里" if hidden > 0 else "")
        )
        _bars(shown)
    if variants:
        theme.caption(
            "同一个算法的多根条是**同一个算法的不同配置**（" + dimension + "这一类里的各档值），"
            "不是不同算法——它们的差距读作「这个算法对那个参数有多敏感」，"
            "与跨算法的差距不是一回事。变体之间的完整对照（含噪声底线与显著性）在「变体分析」页。"
        )
    _leaderboard_note(board)


def _bars(entries: list[dict[str, Any]]) -> None:
    # 归一化轴**只在这一组内**：跨族的回报尺度不可比（离散环境里 SAC/TD3 高出
    # 一个量级），拿全局最大值归一化会让几个族都缩成一条线。
    scale = max((abs(entry["mean"]) for entry in entries), default=0.0)
    # 图例用 label（变体名在其中），色按族取；变体压成半透明，基准保持饱和——
    # 同色系深浅两档，一眼能指出哪一条是参照物。
    styled = []
    for entry in entries:
        color = theme.family_color(entry["family"])
        if entry["variant"]:
            color = theme.fade(color, 0.42)
        styled.append({**entry, "color": color})
    theme.legend(
        (entry.get("label") or entry["algorithm"], entry["color"]) for entry in styled
    )
    theme.card(theme.algo_bars(styled, scale=scale))


def _leaderboard_note(board: list[dict[str, Any]]) -> None:
    """一句「这一族里最好的是谁」，**按族说**而不是按全表说。

    全表第一名的措辞在这一页是错的：不同族的条长不可比（REINFORCE 在这套数据里最差
    到 −615，拿它去当尺度基准会把别的族全压平），所以「全环境领先」这个说法本身
    就没有可比性。改成逐族点名，并且明确说出「族与族只比数字，不比条长」。
    """
    leaders = []
    for family, entries in data.family_groups(board):
        if not entries:
            continue
        top = entries[0]
        leaders.append(f"{family} 族 <strong>{top.get('label') or top['algorithm']}</strong>"
                       f"（{top['mean']:.2f}）")
    text = "各族当前最好的一次： " + "；".join(leaders) + "。"
    multi = next((entry for entry in board if entry["count"] > 1), None)
    if multi:
        label = multi.get("label") or multi["algorithm"]
        text += (
            f" 注意 <strong>{label}</strong> 跑了 {multi['count']} 个种子，"
            f"均值 {multi['mean']:.2f} ± {multi['spread']:.2f}（样本标准差），"
            "这才是可以拿来判断「稳不稳」的数。"
        )
    text += " 单看均值分不出「真的更好」和「种子好」；不同族的条长不可比，只比数字。"
    theme.caption(text)


def _render_videos(runs: list[data.RunInfo]) -> None:
    """播放训练后录制的轨迹视频，按算法排成网格。

    视频是训练结束时 ``visualization.trajectory`` 自动录的（``record_video: true``），
    同一套评估种子，所以并排看的是「同样的开局，不同算法怎么落地」。

    按 ``_VIDEO_COLUMNS`` 折行而不是挤成一行：一行放 11 个的时候每格只有 100 多像素宽，
    LunarLander 的着陆器小到看不清，对比就失去意义了。

    最多放 ``_VIDEO_LIMIT`` 个（按最终回报挑最好的）。每个 ``st.video`` 都会让浏览器
    去拉一个 mp4；选了两三百个运行时全放会让页面长时间转圈。
    """
    columns_data = []
    ranked = sorted(
        runs,
        key=lambda info: float(info.summary.get("mean_reward") or float("-inf")),
        reverse=True,
    )
    for info in ranked:
        episodes = data.trajectory_summary(info.path)
        directory = info.path / "trajectories"
        video = None
        for item in episodes:
            candidate = directory / f"episode_{item.get('episode', 1)}.mp4"
            if candidate.exists():
                video = (candidate, item)
                break
        if video is None:
            found = sorted(directory.glob("episode_*.mp4")) if directory.is_dir() else []
            if found:
                video = (found[0], {})
        if video is not None:
            columns_data.append((info, *video))
        if len(columns_data) >= _VIDEO_LIMIT:
            break
    if not columns_data:
        return

    note = (
        f"同一评估种子下各算法的一次飞行；按最终回报取前 {len(columns_data)} 个"
        + (f"（共 {len(runs)} 个运行有录像）" if len(runs) > len(columns_data) else "")
    )
    theme.section_head("落地录像", note)
    for start in range(0, len(columns_data), _VIDEO_COLUMNS):
        batch = columns_data[start : start + _VIDEO_COLUMNS]
        columns = st.columns(_VIDEO_COLUMNS, gap="small")
        for column, (info, path, item) in zip(columns, batch):
            reward = info.summary.get("mean_reward")
            dev_tag = f" [{info.device}]" if info.device else ""
            variant_tag = f" · {info.variant}" if info.variant else ""
            with column:
                st.markdown(
                    f'<div style="font-weight:700;color:{theme.family_color(info.family)}">'
                    f'{info.algorithm}<span style="font-size:0.72rem;color:#69757b;font-weight:normal">'
                    f'{variant_tag}{dev_tag}</span></div>',
                    unsafe_allow_html=True,
                )
                caption = f"评估均值 {reward:.1f}" if isinstance(reward, (int, float)) else ""
                if item.get("return") is not None:
                    caption += f" · 这一回 {item['return']:.0f}"
                st.caption(caption or info.name)
                st.video(str(path))


def _render_table(chosen: list[data.RunInfo], environment: str) -> None:
    """完整结果表 + 两个导出：CSV（数据）与静态叠加 PNG（插图）。"""
    with st.expander(f"完整结果表（{len(chosen)} 次运行）", expanded=False):
        table = _result_table(chosen).sort_values("均值回报", ascending=False, na_position="last")
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=theme.fit_height(len(table)),
            column_config={
                "均值回报": st.column_config.NumberColumn(format="%.2f"),
                "标准差": st.column_config.NumberColumn(format="%.2f"),
                "最小": st.column_config.NumberColumn(format="%.2f"),
                "最大": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.download_button(
            "导出这张表",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"comparison_{environment.replace('/', '_')}.csv",
            mime="text/csv",
        )
        st.divider()
        st.caption(
            "要插进报告里的静态图用下面这个。它走 matplotlib，"
            "**图内文字是英文**（绘图库默认字体不含中文），与上面的交互图是两套实现。"
        )
        # 默认不生成：每轮页面重跑都重建一个 matplotlib Figure 是纯浪费，
        # 而且这条路上的字体问题（中文变方块）与 altair 那条路完全不同。
        if st.checkbox("生成静态叠加图（PNG）", key=f"static::{environment}"):
            from webui import plots

            figure = plots.comparison_figure(series=plots.series_for(chosen))
            if figure is None:
                st.info("这些运行都没有可画的训练曲线。")
            else:
                st.image(figure, width="stretch")
                st.download_button(
                    "下载这张对比图", data=figure, file_name="comparison.png", mime="image/png",
                )


def render() -> None:
    theme.page_header(
        "跨运行对比",
        "先选一个环境，再选一类改动：看这个环境上各算法学得怎么样，以及同一个算法换一档配置会怎样。",
        eyebrow="Reinforce / compare",
    )

    runs = _shared.load_runs()
    completed = [info for info in runs if info.summary]
    if not completed:
        theme.empty_state("还没有跑完的运行", "到「发起训练」在同一个环境上跑几个算法，再回来对比。")
        return

    groups = data.runs_by_environment(completed)
    environments = list(groups)
    preferred = st.session_state.get("compare_environment")
    index = environments.index(preferred) if preferred in environments else 0

    environment = st.selectbox(
        "环境",
        environments,
        index=index,
        key="compare_environment",
        format_func=lambda name: f"{name}　·　{len(groups[name])} 次运行",
    )
    in_environment = groups[environment]

    # 第一道筛选：一类改动一次。变体批次跑完一个环境有六百个运行、二十个变体名，
    # 一屏之内平铺就是在看一面墙；按改动类别切开之后，一类里同一个算法只有二到五档。
    dimension = st.radio(
        "对比维度",
        _dimensions_available(in_environment),
        index=0,
        horizontal=True,
        key=f"compare_dimension::{environment}",
        help="「基准」= 没套任何变体覆盖的运行，是唯一能跨算法对读的一组。"
             "其余各档是同一类改动下的各份配置，会带上它们各自的基准一起显示。",
    )

    scoped = _dimension_runs(in_environment, dimension)
    if not scoped:
        st.info(f"这个环境上还没有「{dimension}」这一类的结果。")
        return
    if dimension == _DIM_ALL and len(scoped) > 60:
        st.warning(
            f"「全部」档下有 {len(scoped)} 个运行，图上会有很多条线。"
            "建议按改动类别看——同一类里同一个算法只有几档，读得出来。"
        )

    # 第二道筛选：要看哪几个算法。粒度是算法名，所以「同配置的多个种子」整组进出。
    algorithm_names = sorted({info.algorithm for info in scoped if info.algorithm})
    default_algorithms = _dimension_algorithms(scoped)
    algorithms = st.multiselect(
        "要看的算法",
        options=algorithm_names,
        default=[name for name in default_algorithms],
        key=f"compare_algorithms::{environment}::{dimension}",
        help=f"默认按算法族轮换取 ≤{_ALGO_LIMIT} 个，保证每个族都有代表。"
             "每个算法的全部种子会整组进出。",
    )
    if not algorithms:
        st.warning("至少选一个算法。")
        return

    chosen = [info for info in scoped if info.algorithm in set(algorithms)]
    families = {info.family for info in chosen}
    if len(families) < 2:
        st.info(
            "这些运行都属于同一个算法族（自研与 SB3 实现读同一份超参）。"
            "同族对照看实现差异，跨族对照看算法差异。"
        )

    best = max((info.summary.get("mean_reward", 0) for info in chosen), default=0)
    baseline_count = sum(1 for info in chosen if not info.variant)
    theme.stat_row([
        ("环境", environment.split("/")[-1][:16], environment),
        ("对比维度", dimension, "一类改动一次"),
        ("运行", len(chosen), f"{len(algorithms)} 个算法"),
        ("其中基准", baseline_count, "跨算法可对读的那组"),
        ("最好", f"{best:.2f}", "最终评估均值"),
    ])

    # 先看图：这一页最该先看的是「学得怎么样」，曲线于是排在排行榜前面。
    _render_curves(chosen)
    _render_leaderboard(chosen, dimension)
    _render_videos(chosen)
    _render_fill_gap(environment, in_environment)
    _render_table(chosen, environment)
