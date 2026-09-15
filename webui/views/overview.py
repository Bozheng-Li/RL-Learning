"""运行总览：一屏看完 outputs 下所有实验的状态与结果。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from webui import data, live
from webui.views import _shared


def _summary_table(runs: list[data.RunInfo]) -> pd.DataFrame:
    rows = []
    for info in runs:
        summary = info.summary
        rows.append(
            {
                "运行": info.name,
                "状态": data.STATUS_LABELS[info.status],
                "环境": info.environment or "-",
                "算法": info.algorithm or "-",
                "种子": info.seed if info.seed is not None else "-",
                "配置": info.config_name or "自定义",
                "总步数": info.total_timesteps or "-",
                "已训练": info.latest_timestep if info.latest_timestep is not None else "-",
                "均值回报": summary.get("mean_reward"),
                "标准差": summary.get("std_reward"),
                "最小": summary.get("min_reward"),
                "最大": summary.get("max_reward"),
                "评估回合": summary.get("episodes"),
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    st.header("运行总览")

    runs = _shared.load_runs()
    if not runs:
        st.info(
            "没有发现任何运行。\n\n"
            "「运行」的判据是目录里存在 `evaluation.json`（训练跑完的标志），"
            "或者已经写了 `resolved_config.yaml`（正在训练）。"
        )
        return

    # ---- 统计卡片 ----
    counts = {status: 0 for status in data.STATUS_LABELS}
    for info in runs:
        counts[info.status] += 1
    columns = st.columns(4)
    for column, status in zip(columns, ("completed", "running", "failed", "not_started")):
        column.metric(data.STATUS_LABELS[status], counts[status])

    st.divider()

    # ---- 筛选 ----
    with st.expander("筛选", expanded=False):
        left, middle, right = st.columns(3)
        pattern = left.text_input("名称通配符", value="*", help="例如 cartpole* 或 minigrid_*")
        environments = sorted({info.environment for info in runs if info.environment})
        picked_envs = middle.multiselect("环境", environments)
        algorithms = sorted({info.algorithm for info in runs if info.algorithm})
        picked_algos = right.multiselect("算法", algorithms)

    filtered = [
        info
        for info in runs
        if (not picked_envs or info.environment in picked_envs)
        and (not picked_algos or info.algorithm in picked_algos)
    ]
    if pattern and pattern != "*":
        from fnmatch import fnmatch  # noqa: PLC0415

        filtered = [info for info in filtered if fnmatch(info.name, pattern)]

    if not filtered:
        st.warning("没有运行匹配当前筛选条件。")
        return

    # ---- 表格 ----
    table = _summary_table(filtered)
    event = st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "均值回报": st.column_config.NumberColumn(format="%.2f"),
            "标准差": st.column_config.NumberColumn(format="%.2f"),
            "最小": st.column_config.NumberColumn(format="%.2f"),
            "最大": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        picked = filtered[selected_rows[0]]
        if st.session_state.get("selected_run") != picked.name:
            st.session_state["selected_run"] = picked.name
        st.caption(f"已选中 **{picked.name}** — 切到「运行详情」页查看。")

    st.download_button(
        "导出当前表格为 CSV",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="runs.csv",
        mime="text/csv",
    )

    # ---- 运行中的运行：进度条 ----
    running = [info for info in filtered if info.status == "running"]
    if running:
        st.divider()
        st.subheader("进行中")
        for info in running:
            progress = live.compute_progress(info.path, info.total_timesteps)
            left, right = st.columns([3, 1])
            if progress is None:
                left.progress(0.0, text=f"{info.name}：还没写出任何训练数据")
                right.metric("预计剩余", "未知")
                continue
            # 步数偶尔会略微超过总数（rollout 粒度导致），进度条要夹住上限。
            left.progress(
                min(progress.fraction, 1.0),
                text=f"{info.name}：{progress.done:,} / {progress.total:,} 步",
            )
            right.metric("预计剩余", live.format_eta(progress.eta_seconds))

    # ---- 环境分布小结 ----
    st.divider()
    st.caption(
        f"共 {len(filtered)} 个运行；输出根目录：`{_shared.outputs_root()}`。"
        "跨环境比较回报没有意义——各环境的奖励尺度完全不同。"
    )
