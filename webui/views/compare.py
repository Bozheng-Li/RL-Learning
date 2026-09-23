"""跨运行对比：先选一个环境，再看这个环境上各算法的表现。

跨环境的奖励尺度完全不同，所以默认以环境为入口，而不是让用户从全部运行里盲选。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from webui import data, nav, theme
from webui.views import _shared

#: 「落地录像」一行放几个。算法多的时候会折成多行，超过这个数每格就小得看不清了。
_VIDEO_COLUMNS = 4

#: 最多放几个录像。变体批次跑完一个环境有几百个运行，每个都塞一个 ``st.video``
#: 会把页面拖死，也没人看得过来。按最终回报挑最好的几个。
_VIDEO_LIMIT = 12

#: 学习曲线最多画几条（软上限：足够放下「11 个实现 × 多种子」的基准批次）。
_CURVE_LIMIT = 96

#: 曲线里最多保留几组 ``(算法, 变体)``。**按组取舍而不是按条**：同一个配置的
#: 多个种子是一组，拆开画就只剩一半种子，读不出离散度。基准批次只有 11 组，
#: 全留下；变体批次有 202 组，超过这个数就按族轮流取。
_CURVE_GROUPS = 18

#: 每个族最多画几条条。变体扫描的长尾（一个算法十几个变体）全画出来就是一面墙，
#: 想看全的用下面的结果表或「变体分析」页。
_BAR_LIMIT = 14

#: 「纳入对比」下拉在窄范围下默认全选；范围大于这个数时默认只选基准那几条，
#: 免得一进来就是几百个名字。
_DEFAULT_SELECT_LIMIT = 24

#: 对比范围。**默认只看基准**：变体批次一跑就是几百个运行，全选之后排行榜有
#: 两百多条、曲线图六百多条线，什么都读不出来。要看变体是刻意的动作，不是默认。
_SCOPES: dict[str, str] = {
    "只看基准": "baseline",
    "只看变体": "variant",
    "全部": "all",
}


def _family_buckets(
    runs: list[data.RunInfo],
) -> list[tuple[str, list[list[data.RunInfo]]]]:
    """把运行组织成 ``[(族, [ (算法,变体) 组, ... ]), ...]``。

    族按 ``FAMILY_ORDER``（学习顺序）排，族内各组按它们在 ``runs`` 里首次出现的位置排。
    这两个使用方共享同一套组织方式：

    - ``_curve_sets``：曲线图按族轮流取组，保证每个族都出现在图里。
    - ``_default_selection``：多选框的默认值按族轮流取组，保证默认选中不是一个
      「目录顺序排到哪算哪」的偏斜子集。
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


def _default_selection(runs: list[data.RunInfo], limit: int) -> list[str]:
    """多选框的默认选中项。

    名字少就全选。名字多时**基准优先**：基准是「各算法本来的水平」，也是唯一能跨算法
    对读的那一组。当前范围内一条基准都没有（「只看变体」）时退到按族轮换取代表组——
    变体批次有 573 个变体运行，按目录顺序取前 24 个会落成「两个族、十几个变体」这种
    偏斜子集，看起来像这个环境只有这两个族。按族轮流取，保证每个族、每个算法都有代表。

    只取**完整**的 ``(算法, 变体)`` 组：截半个组的种子会让曲线图里那一条凭空少一半
    种子，读不出离散度。凑不满上限就少选几个，也好过拆组。
    """
    baselines = [info.name for info in runs if not info.variant]
    if baselines:
        return baselines
    picked: list[str] = []
    seen: set[str] = set()
    buckets = _family_buckets(runs)
    max_depth = max((len(groups) for _family, groups in buckets), default=0)
    depth = 0
    while len(picked) < limit and depth < max_depth:
        for _family, groups in buckets:
            if depth >= len(groups):
                continue
            fresh = [info.name for info in groups[depth] if info.name not in seen]
            if not fresh or len(fresh) > limit - len(picked):
                continue
            picked.extend(fresh)
            seen.update(fresh)
            if len(picked) >= limit:
                break
        depth += 1
    return picked


def _curve_sets(chosen: list[data.RunInfo]) -> tuple[list[data.RunInfo], int]:
    """按**算法族**给曲线图挑一组代表，返回 ``(选中的运行, 被挤掉的条数)``。

    曲线图跟条形图是同一个问题：变体批次里选几百个运行就是六百条线，图例能盖住
    半张图。取舍的粒度是 ``(算法, 变体)`` **组**而不是单条运行——同一组的多个种子
    拆开画就只剩一半种子，读不出离线度。组**内部**要么全留要么全走。

    先在组之间按族轮流取（每族先取自己的第一组），保证每个算法族都在图里出现过，
    而不是「回报最高的那个族的十几组」把整张图占满；组选完之后再受 ``_CURVE_LIMIT``
    约束，超了就按优先级砍——一轮一轮取的时候越晚取到的越先砍，同族同轮的按传入
    顺序（也就是用户在 multiselect 里的顺序）。
    """
    groups: dict[tuple[str, str | None], list[data.RunInfo]] = {}
    for info in chosen:
        groups.setdefault((info.algorithm or "", info.variant), []).append(info)
    if len(groups) <= _CURVE_GROUPS and len(chosen) <= _CURVE_LIMIT:
        return chosen, 0

    buckets = _family_buckets(chosen)
    max_depth = max((len(group_list) for _family, group_list in buckets), default=0)

    picked_groups: list[list[data.RunInfo]] = []
    depth = 0
    budget = 0
    while len(picked_groups) < _CURVE_GROUPS and depth < max_depth:
        added = False
        for _family, group_list in buckets:
            if depth >= len(group_list):
                continue
            size = len(group_list[depth])
            if budget + size > _CURVE_LIMIT:
                continue
            picked_groups.append(group_list[depth])
            budget += size
            added = True
            if len(picked_groups) >= _CURVE_GROUPS:
                break
        if not added:
            break
        depth += 1

    picked = [info for group in picked_groups for info in group]
    # 保持用户在 multiselect 里的顺序，图例顺序才稳定（不然每轮刷新都在跳）。
    rank = {info.name: index for index, info in enumerate(chosen)}
    picked.sort(key=lambda info: rank[info.name])
    return picked, len(chosen) - len(picked)


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


def _render_leaderboard(runs: list[data.RunInfo]) -> None:
    """按**算法族**分组的排行榜：同族内每个 ``(算法, 变体)`` 取最好成绩。

    分组不是排版偏好。变体批次跑完一个环境会有两百多行（11 个实现 × 十几个变体），
    平铺成一面条形墙之后——条的宽度各不相同却共用一根归一化轴，色也是六族轮着用——
    除了「最高的那根最长」什么都读不出来。按族切开之后每组只剩几行：

    - 组内每根条只跟同族的比，归一化轴**只在组内**，长度是可信的（族与族之间不比长度，
      比数字，族色只是帮你把同一族的几根条认成一组）。
    - 同族里出现两个实现（``PPO`` 与 ``SB3-PPO``）时，直接读出实现差异；这是本项目
      最核心的对照维度，混在两百行里就彻底消失了。

    连续与离散动作空间的奖励尺度不可比，所以族之间本来就该分开读。

    组内超过 ``_BAR_LIMIT`` 条（变体扫描的长尾）时只画回报最高的那些，并把隐藏的
    条数写在旁边——完整的一百多条在下面的结果表与「变体分析」页里，这里保留可读性。
    """
    board = data.environment_leaderboard(runs)
    if not board:
        return
    groups = data.family_groups(board)
    variants = {entry["variant"] for entry in board if entry["variant"]}
    subtitle = (
        f"{len(groups)} 个算法族 · {len(board)} 条；每族内按最好的一次排"
        + ("；同一算法的多条是同算法不同配置" if variants else "")
    )
    theme.section_head("算法排名", subtitle)

    spaces = {entry["action_space"] for entry in board}
    if len(spaces) > 1:
        theme.caption(
            "⚠️ 这个环境里混了两种动作空间，奖励尺度不可直接比较，所以按族分开排。"
        )
    for family, entries in groups:
        shown = entries[:_BAR_LIMIT]
        hidden = len(entries) - len(shown)
        implementations = sorted({entry["algorithm"] for entry in shown})
        theme.caption(
            f'<span style="color:{theme.family_color(family)};font-weight:700">{family}</span>'
            f" · 显示 {len(shown)} / {len(entries)} 条 · {'、'.join(implementations)}"
            f" · 轴只在这组内归一化"
            + (f" · 另有 {hidden} 条在下面的结果表里" if hidden > 0 else "")
        )
        _bars(shown)
    if variants:
        theme.caption(
            "同一个算法的多根条是**同一个算法的不同配置**（变体），不是不同算法——"
            "它们的差距读作「这个算法对那个超参有多敏感」，与跨算法的差距不是一回事。"
            "变体之间的完整对照（含噪声底线与显著性）在「变体分析」页。"
        )
    _leaderboard_note(board)




def _bars(entries: list[dict[str, Any]]) -> None:
    # 归一化轴**只在这一组内**：跨族的回报尺度不可比（离散环境里 SAC/TD3 高出
    # 一个量级），拿全局最大值归一化会让几个族都缩成一条线。
    scale = max((abs(entry["mean"]) for entry in entries), default=0.0)
    # 图例用 label（变体名在其中），色仍按族取——同算法的多个变体是同一色系，
    # 一眼能看出「这几条属于同一个算法」。
    theme.legend(
        (entry.get("label") or entry["algorithm"], theme.family_color(entry["family"]))
        for entry in entries
    )
    theme.card(theme.algo_bars(entries, scale=scale))


def _leaderboard_note(board: list[dict[str, Any]]) -> None:
    winner = board[0]
    multi = next((entry for entry in board if entry["count"] > 1), None)
    seed = f"，种子 {winner['seed']}" if winner.get("seed") is not None else ""
    text = (
        f"全环境领先的是 <strong>{winner.get('label') or winner['algorithm']}</strong>"
        f"（{winner['family']} 族{seed}）。"
        "注意这只是一次运行的最好成绩——单看均值分不出「真的更好」和「种子好」。"
    )
    if multi:
        label = multi.get("label") or multi["algorithm"]
        text += (
            f" {label} 跑了 {multi['count']} 个种子，"
            f"均值 {multi['mean']:.2f} ± {multi['spread']:.2f}（样本标准差），"
            "这才是可以拿来判断「稳不稳」的数。"
        )
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
            with column:
                st.markdown(
                    f'<div style="font-weight:700;color:{theme.family_color(info.family)}">'
                    f'{info.algorithm}<span style="font-size:0.72rem;color:#69757b;font-weight:normal">{dev_tag}</span></div>',
                    unsafe_allow_html=True,
                )
                caption = f"评估均值 {reward:.1f}" if isinstance(reward, (int, float)) else ""
                if item.get("return") is not None:
                    caption += f" · 这一回 {item['return']:.0f}"
                st.caption(caption or info.name)
                st.video(str(path))


def render() -> None:
    theme.page_header(
        "跨运行对比",
        "先选一个环境，再看这个环境上各算法学得怎么样。不同环境的奖励尺度不能放在一起比。",
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

    # 对比范围：变体批次一跑就是几百个运行，一进来全选的话排行榜有两百多条、
    # 曲线图六百条线，什么都读不出来。「只看基准」是默认，要看变体是刻意的动作。
    scope = st.radio(
        "对比范围",
        list(_SCOPES),
        index=list(_SCOPES).index("只看基准"),
        horizontal=True,
        key=f"compare_scope::{environment}",
        help="「基准」= 没套任何变体覆盖的运行。变体之间的系统性对照在「变体分析」页。",
    )
    scope_key = _SCOPES[scope]
    scoped = [
        info for info in in_environment
        if scope_key == "all"
        or (scope_key == "baseline" and not info.variant)
        or (scope_key == "variant" and info.variant)
    ]
    if not scoped:
        st.info(
            "这个范围内没有运行"
            + ("。" if scope_key != "variant" else "——变体批次还没跑，或还没跑完。")
        )
        return

    names = [info.name for info in scoped]
    # 名字少就全选；名字多（变体批次）按基准 / 按族轮换取代表。否则每次进来都要先
    # 手动取消几百个勾选才能开始看，而 multiselect 的默认值只在建键那一刻生效。
    default = names if len(names) <= _DEFAULT_SELECT_LIMIT else _default_selection(
        scoped, _DEFAULT_SELECT_LIMIT
    )
    picked = st.multiselect(
        "纳入对比的运行",
        options=names,
        default=default,
        key=f"compare_runs::{environment}::{scope_key}",
        help="默认不选全部：变体批次有几百个运行，先看基准，再按需加。",
    )
    if not picked:
        st.warning("至少选一个运行。")
        return

    chosen = [info for info in scoped if info.name in picked]
    families = {info.family for info in chosen}
    algorithms = {info.algorithm for info in chosen}
    if len(families) < 2 and len(algorithms) < 2:
        st.info("这个环境目前只有一个算法的结果。再跑一个不同的算法，对比才有意义。")
    elif len(families) < 2:
        st.info(
            "这些运行都属于同一个算法族（自研与 SB3 实现读同一份超参）。"
            "同族对照看实现差异，跨族对照看算法差异。"
        )

    best = max((info.summary.get("mean_reward", 0) for info in chosen), default=0)
    theme.stat_row([
        ("环境", environment.split("/")[-1][:16], environment),
        ("运行", len(chosen), f"{len(algorithms)} 个算法"),
        ("最好", f"{best:.2f}", "最终评估均值"),
        ("种子", len({info.seed for info in chosen}), "不同随机种子"),
    ])

    _render_leaderboard(chosen)
    _render_videos(chosen)
    _render_fill_gap(environment, in_environment)

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

    with st.expander("学习曲线", expanded=len(chosen) > 1):
        curves, dropped = _curve_sets(chosen)
        if dropped:
            theme.caption(
                f"曲线太多会糊成一团，这里按算法族轮流取了 {len(curves)} 条"
                f"（挤掉 {dropped} 条）。想看某几条的完整曲线，到「运行详情」页看。"
            )
        # 这里刻意不用 `return`：裸 return 会把同一个 expander 之后的内容（下载按钮）
        # 一起跳过，于是「绘图库缺失」和「没有曲线」两种情况下连下载按钮都不出现。
        figure = None
        try:
            from webui import plots

            figure = plots.comparison_figure(series=plots.series_for(curves))
        except (ImportError, ModuleNotFoundError):
            st.info("当前环境的绘图库不可用。安装兼容 NumPy 的 matplotlib 后即可生成对比图。")
        if figure is None:
            if not any(info.summary for info in curves):
                st.info("这些运行都没有可画的训练曲线。")
        else:
            st.image(figure, width="stretch")
            theme.caption(
                "左图是周期评估均值；右图是最终评估的均值与标准差。"
                "图内文字为英文，绘图库默认字体不含中文。"
            )
            st.download_button(
                "下载这张对比图", data=figure, file_name="comparison.png", mime="image/png",
            )
