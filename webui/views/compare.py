"""跨运行对比：把多次实验的曲线叠在一起，结果并排看。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from webui import data, plots, theme
from webui.views import _shared


def _result_table(runs: list[data.RunInfo]) -> pd.DataFrame:
    rows = []
    for info in runs:
        summary = info.summary
        rows.append(
            {
                "运行": info.name,
                "环境": summary.get("environment") or info.environment or "-",
                "算法": summary.get("algorithm") or info.algorithm or "-",
                "种子": summary.get("seed", info.seed),
                "步数": summary.get("timesteps", info.total_timesteps),
                "均值回报": summary.get("mean_reward"),
                "标准差": summary.get("std_reward"),
                "最小": summary.get("min_reward"),
                "最大": summary.get("max_reward"),
                "回合数": summary.get("episodes"),
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    theme.page_header("跨运行对比", "把多次实验的曲线叠在一起，结果并排看")

    runs = _shared.load_runs()
    completed = [info for info in runs if info.summary]
    if len(completed) < 1:
        st.info("还没有跑完的运行可以对比。")
        return

    names = [info.name for info in completed]
    # 默认选最近完成的几个，省得每次手动挑。
    default = names[: min(5, len(names))]
    picked = st.multiselect(
        "选择要对比的运行",
        options=names,
        default=default,
        key="compare_runs",
        help="选同一环境的不同算法/种子才有可比性——各环境的奖励尺度完全不同。",
    )
    if not picked:
        st.warning("至少选一个运行。")
        return

    chosen = [info for info in completed if info.name in picked]
    environments = {info.summary.get("environment") for info in chosen}
    if len(environments) > 1:
        st.warning(
            "选中的运行来自**不同的环境**（"
            + "、".join(str(item) for item in sorted(environments) if item)
            + "），奖励尺度不可直接比较，这张图更适合看「学习速度」而不是「谁更好」。"
        )

    st.subheader("结果表")
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
        "导出对比表为 CSV",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="comparison.csv",
        mime="text/csv",
    )

    st.subheader("曲线与最终评估")
    figure = plots.comparison_figure([info.path for info in chosen])
    if figure is None:
        st.info("这些运行都没有可画的训练曲线。")
        return
    st.image(figure, width="stretch")
    st.caption(
        "左图取自 `evaluation/evaluations.npz`（周期评估均值，噪声比训练回合小）；"
        "右图是最终评估的均值与标准差。"
    )

    # 只在用户点导出时才落盘——平时的重绘不能往 outputs/ 里写东西。
    st.download_button(
        "下载这张对比图",
        data=figure,
        file_name="comparison.png",
        mime="image/png",
    )
