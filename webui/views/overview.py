"""运行总览：以环境为单位看实验。

同一个环境上的不同算法才有可比性——各环境的奖励尺度完全不同，所以首页按环境分组，
每张卡里只比较这个环境内的算法，而不是把所有运行摊成一张大表。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from webui import data, live, nav, theme
from webui.views import _shared

#: 环境卡上最多列出几个「还没跑过」的算法。再多就只剩名字没有条了，卡会被撑高。
_PLACEHOLDER_LIMIT = 4


def _missing_algorithms(config_name: str | None, runs: list[data.RunInfo]) -> list[str]:
    """这个环境上还没跑过**基准**的算法，按学习顺序排。

    变体不算数：一个算法只跑了 lr 扫描而没有基准时，那些曲线没有对照物。
    """
    if not config_name:
        return []
    try:
        supported = data.compatible_algorithms(data.load_config_dict(config_name))
    except (OSError, ValueError):
        return []
    done = data.baseline_algorithms(runs)
    return [name for name in supported if name not in done]


def _summary_table(runs: list[data.RunInfo]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "运行": info.name, "状态": data.STATUS_LABELS[info.status],
            "环境": info.environment or "-", "算法": info.algorithm or "-",
            "变体": info.variant or "基准",
            "种子": info.seed if info.seed is not None else "-",
            "步数": info.latest_timestep or info.total_timesteps or "-",
            "最终回报": info.summary.get("mean_reward"), "波动": info.summary.get("std_reward"),
            "评估回合": info.summary.get("episodes"),
        }
        for info in runs
    ])


def _apply_filters(runs: list[data.RunInfo]) -> list[data.RunInfo]:
    with st.expander("筛选与定位", expanded=False):
        left, middle, right = st.columns([1.15, 1, 1])
        pattern = left.text_input("运行名称", value="*", placeholder="例如 cartpole*")
        environments = sorted({item.environment for item in runs if item.environment})
        picked_envs = middle.multiselect("环境", environments)
        algorithms = sorted({item.algorithm for item in runs if item.algorithm})
        picked_algos = right.multiselect("算法", algorithms)
        # 变体单独一格：变体批次跑起来后同一个算法会有十几行，只按算法筛等于没筛。
        # 没有变体运行时不渲染这个控件——空的多选框只会占地方。
        variants = sorted({item.variant for item in runs if item.variant})
        picked_variants: list[str] = []
        if variants:
            picked_variants = st.multiselect(
                "变体", ["基准", *variants], key="overview_variants",
                help="「基准」= 没套任何变体覆盖的运行。留空表示不限。",
            )
    wants_baseline = not picked_variants or "基准" in picked_variants
    wanted_variants = {name for name in picked_variants if name != "基准"}
    filtered = [
        item for item in runs
        if (not picked_envs or item.environment in picked_envs)
        and (not picked_algos or item.algorithm in picked_algos)
        and (
            not picked_variants
            or (item.variant in wanted_variants)
            or (not item.variant and wants_baseline)
        )
    ]
    if pattern and pattern != "*":
        from fnmatch import fnmatch
        filtered = [item for item in filtered if fnmatch(item.name, pattern)]
    return filtered


def _render_environment_cards(runs: list[data.RunInfo]) -> None:
    """每个环境一张卡：组内各算法的最好回报横向比较。"""
    groups = data.runs_by_environment(runs)
    theme.section_head("按环境看", f"{len(groups)} 个环境 · 条形只在同一张卡内可比")

    environments = list(groups)
    for start in range(0, len(environments), 3):
        row = environments[start:start + 3]
        columns = st.columns(len(row), gap="medium")
        for column, environment in zip(columns, row):
            items = groups[environment]
            board = data.environment_leaderboard(items)
            best = board[0] if board else None
            # 卡上只放**基准**的条：变体批次里这张卡会有两百条，卡被撑到一屏都滚不完，
            # 而且族与族的回报尺度不同，混在一根轴上长度本来就不可比。变体的系统性
            # 对照在「变体分析」页，卡上只留一句数目提示。
            shown = [entry for entry in board if not entry["variant"]]
            variants = len(board) - len(shown)
            if not shown:
                # 只有变体、没有基准时（少见），退化成每族留最好的一条。
                seen: set[str] = set()
                shown = [
                    entry for entry in board
                    if not (entry["family"] in seen or seen.add(entry["family"]))
                ]
                variants = len(board) - len(shown)
            config_name = next((item.config_name for item in items if item.config_name), None)
            missing = _missing_algorithms(config_name, items) if config_name else []
            notes = []
            if variants:
                notes.append(f"另有 {variants} 条变体，见「变体分析」页")
            if len(missing) > _PLACEHOLDER_LIMIT:
                notes.append(f"另有 {len(missing) - _PLACEHOLDER_LIMIT} 个算法没列出")
            with column:
                st.markdown(
                    theme.env_card(
                        environment,
                        runs=len(items),
                        algorithms=len({item.algorithm for item in items if item.algorithm}),
                        best_name=best["label"] if best else None,
                        best_value=best["mean"] if best else None,
                        entries=shown,
                        placeholders=missing[:_PLACEHOLDER_LIMIT],
                        note=" · ".join(notes),
                    ),
                    unsafe_allow_html=True,
                )
                # 单一主操作：进这个环境的对比页。补齐缺失算法在对比页里做，
                # 免得每张卡上并列两个按钮、各自的入口都通向不同的地方。
                if st.button("打开对比", key=f"cmp::{environment}", width="stretch"):
                    st.session_state["compare_environment"] = environment
                    nav.goto("跨运行对比")
                    st.rerun()


def render() -> None:
    theme.page_header(
        "运行总览",
        "同一个环境上的不同算法才有可比性，所以这里按环境分组，而不是把所有运行摊成一张表。",
        eyebrow="Reinforce / overview",
    )
    runs = _shared.load_runs()
    action_left, action_right = st.columns([5, 1])
    action_left.caption(f"产物目录 · {_shared.outputs_root()}")
    if action_right.button("发起对照训练", type="primary", width="stretch"):
        nav.goto("发起训练")
        st.rerun()
    if not runs:
        theme.empty_state(
            "还没有发现实验运行",
            "到「发起训练」选一份配置，一次跑齐这个环境上的多个算法；或检查侧栏的产物目录。",
        )
        return

    counts = {status: 0 for status in data.STATUS_LABELS}
    for info in runs:
        counts[info.status] += 1
    groups = data.runs_by_environment(runs)
    multi = sum(1 for items in groups.values() if len({item.family for item in items}) > 1)
    completed = [item for item in runs if item.summary.get("mean_reward") is not None]
    best = max(completed, key=lambda item: float(item.summary.get("mean_reward", float("-inf"))),
               default=None)
    theme.stat_row([
        ("环境", len(groups), f"共 {len(runs)} 次运行"),
        ("可对比", multi, "同一环境上跑过不止一个算法"),
        ("进行中", counts["running"], f"已完成 {counts['completed']} 次"),
        ("单次最高回报", f"{best.summary.get('mean_reward'):.1f}" if best else "-",
         best.environment if best else "暂无评估"),
    ])

    _render_environment_cards(runs)

    filtered = _apply_filters(runs)
    if not filtered:
        st.warning("没有运行匹配当前筛选条件。")
        return

    theme.section_head("全部运行", f"{len(filtered)} 个 · 点选一行后进入详情")
    table = _summary_table(filtered)
    event = st.dataframe(
        table, width="stretch", height=theme.fit_height(len(table), max_height=520),
        hide_index=True, on_select="rerun", selection_mode="single-row",
        column_config={
            "最终回报": st.column_config.NumberColumn(format="%.2f"),
            "波动": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        picked = filtered[selected_rows[0]]
        st.session_state["selected_run"] = picked.name
        left, right = st.columns([4, 1])
        left.markdown(f"已选中 **{picked.name}** · 切换到运行详情查看曲线、轨迹和配置。")
        if right.button("查看详情", width="stretch"):
            nav.goto("运行详情")
            st.rerun()

    export_left, export_right = st.columns([1, 5])
    export_left.download_button(
        "导出 CSV", data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="runs.csv", mime="text/csv", width="stretch",
    )
    export_right.caption("跨环境的回报尺度不可直接比较；对比请在同一张环境卡里看。")

    running = [item for item in filtered if item.status == "running"]
    if running:
        theme.section_head("进行中的训练", "实时读取 progress.csv")
        for info in running:
            progress = live.compute_progress(info.path, info.total_timesteps)
            left, right = st.columns([5, 1], vertical_alignment="center")
            if progress is None:
                left.progress(0.0, text=f"{info.name} · 正在初始化环境")
                right.caption("剩余时间未知")
                continue
            left.progress(min(progress.fraction, 1.0),
                          text=f"{info.name} · {progress.done:,} / {progress.total:,} 步")
            right.caption(f"预计剩余 {live.format_eta(progress.eta_seconds)}")
    theme.caption(f"当前显示 {len(filtered)} 个运行；最后扫描目录：<code>{_shared.outputs_root()}</code>")
