"""单个运行的详情：曲线、信号、诊断、评估、轨迹、产物、配置。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from webui import data, jobs, live, paths, theme
from webui.views import _shared


def _relaunch_hint() -> None:
    st.caption(
        "这些按钮以子进程方式调用 `visualize.py`，不改动模型与日志。"
        "跑完到「训练监控」页看输出。"
    )


def _render_regenerate(run_dir: Path, config_name: str | None) -> None:
    """重新出图 / 重录轨迹。"""
    if not config_name:
        st.info(
            "这个运行的输出目录是用 `--set output.directory=...` 指定的，"
            "无法反查出对应的配置名，因此不能从这里重新出图。"
            "请回命令行执行 `python visualize.py --config <配置> --set output.directory=...`。"
        )
        return

    _relaunch_hint()
    left, middle, right = st.columns(3)
    if left.button("重新画训练曲线", width="stretch"):
        argv = jobs.build_visualize_argv(config_name, run_dir, trajectories=False)
        jobs.launch(argv, label=f"visualize:{run_dir.name}", kind="visualize", run_dir=run_dir)
        st.success("已启动，去「训练监控」页看进度。")
    if middle.button("重新录轨迹", width="stretch"):
        argv = jobs.build_visualize_argv(config_name, run_dir, trajectories=True)
        jobs.launch(argv, label=f"play:{run_dir.name}", kind="visualize", run_dir=run_dir)
        st.success("已启动，去「训练监控」页看进度。")
    with right.popover("指定 checkpoint 重录", width="stretch"):
        checkpoints = sorted((run_dir / "checkpoints").glob("*.*")) if (run_dir / "checkpoints").is_dir() else []
        options = [str(path) for path in checkpoints]
        if not options:
            st.caption("这个运行没有保存 checkpoint。")
            return
        chosen = st.selectbox("选择模型文件", options)
        if st.button("用这个模型重录轨迹"):
            argv = jobs.build_visualize_argv(
                config_name, run_dir, trajectories=True, model_path=Path(chosen)
            )
            jobs.launch(argv, label=f"play:{run_dir.name}", kind="visualize", run_dir=run_dir)
            st.success("已启动，去「训练监控」页看进度。")


def _tab_curve(info: data.RunInfo) -> None:
    image = info.path / "visualizations" / "training_results.png"
    if image.exists():
        st.image(str(image), caption="训练四联图", width="stretch")
    else:
        st.caption("还没有训练曲线图。用上面的按钮生成，或等训练跑完。")

    curve = data.load_curve(info.path)
    if curve is None:
        return
    steps, values = curve
    st.caption("下图取自 `evaluation/evaluations.npz`（没有则退回 `logs/train.monitor.csv`）。")
    st.line_chart(pd.DataFrame({"回报": values}, index=steps), height=280)


def _tab_signals(info: data.RunInfo) -> None:
    frame = live.read_progress_csv(info.path / "logs" / "progress.csv")
    if frame is None:
        st.info("还没有 `logs/progress.csv`。")
        return

    signals = [key for key in live.available_signals(frame) if key != "steps"]
    if not signals:
        st.info("这个运行的 `progress.csv` 里没有可画的信号列。")
        return

    st.caption(
        "`progress.csv` 的**列集合每个算法都不一样**（DQN 有探索率没有 KL，"
        "自研算法没有 FPS），所以这里只列出这个运行真的有的列。"
    )
    default = [key for key in ("reward", "eval_reward") if key in signals] or signals[:2]
    picked = st.multiselect(
        "要画的信号",
        options=signals,
        default=default,
        format_func=lambda key: live.SIGNAL_LABELS.get(key, key),
    )
    if not picked:
        return
    _shared.signal_charts(info.path, picked, height=230)

    latest = live.compute_progress(info.path, info.total_timesteps)
    if latest and latest.latest:
        st.caption("最后一行各信号的取值：")
        st.dataframe(
            pd.DataFrame(
                [{"信号": live.SIGNAL_LABELS.get(k, k), "数值": v}
                 for k, v in latest.latest.items() if k in {"reward", "fps", "kl", "entropy",
                                                            "clip", "exploration", "loss"}]
            ),
            hide_index=True,
            width="stretch",
        )


def _tab_diagnostics(info: data.RunInfo) -> None:
    table = data.load_diagnostics(info.path)
    if table is None:
        st.info("还没有 `logs/diagnostics.csv`，或这个运行没开诊断。")
        return

    st.caption(
        "动作分布是排查策略坍缩最直接的证据：曲线塌平或某个动作掉到 0，"
        "说明策略退化，而不是「训练不充分」。"
    )
    frame = table.set_index("timestep")
    if any(name.endswith("_frac") for name in frame.columns):
        st.area_chart(frame, height=340)
        st.caption("最后一个窗口的动作频率：")
        st.dataframe(
            frame.tail(1).T.rename(columns={frame.index[-1]: "占比"}),
            width="stretch",
        )
    else:
        st.line_chart(frame, height=340)


def _tab_evaluations(info: data.RunInfo) -> None:
    loaded = data.load_evaluations(info.path)
    if loaded is None:
        st.info("还没有 `evaluation/evaluations.npz`。")
        return
    steps, results, lengths = loaded
    means = results.mean(axis=1)
    stds = results.std(axis=1)
    st.line_chart(
        pd.DataFrame(
            {
                "评估均值": means,
                "上界 (+1σ)": means + stds,
                "下界 (-1σ)": means - stds,
            },
            index=steps,
        ),
        height=320,
    )
    st.caption(f"共 {len(steps)} 次周期评估，每次 {results.shape[1]} 个回合。")
    detail_table = pd.DataFrame(
        {
            "步数": steps,
            "均值": means,
            "标准差": stds,
            "最小": results.min(axis=1),
            "最大": results.max(axis=1),
            "平均长度": lengths.mean(axis=1),
        }
    )
    st.dataframe(
        detail_table,
        hide_index=True,
        width="stretch",
        height=theme.fit_height(len(detail_table), max_height=420),
    )


def _tab_trajectories(info: data.RunInfo) -> None:
    episodes = data.trajectory_summary(info.path)
    directory = info.path / "trajectories"
    if not episodes and not directory.is_dir():
        st.info("还没有轨迹记录。用上面的「重新录轨迹」生成。")
        return

    if episodes:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "回合": item.get("episode"),
                        "回报": item.get("return"),
                        "步数": item.get("steps"),
                        "种子": item.get("seed"),
                        "终止": item.get("terminated"),
                        "截断": item.get("truncated"),
                    }
                    for item in episodes
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    for index, item in enumerate(episodes, start=1):
        number = item.get("episode", index)
        with st.expander(
            f"第 {number} 回合　·　回报 {item.get('return')}　·　{item.get('steps')} 步",
            expanded=index == 1,
        ):
            video = directory / f"episode_{number}.mp4"
            image = directory / f"episode_{number}.png"
            left, right = st.columns(2)
            if video.exists():
                left.video(str(video))
            else:
                left.caption("没有录制视频（配置里 `record_video` 为 false）。")
            if image.exists():
                right.image(str(image), width="stretch")

            table_path = directory / f"episode_{number}.csv"
            if table_path.exists():
                try:
                    st.dataframe(
                        pd.read_csv(table_path).head(500), height=240, width="stretch"
                    )
                    st.caption("只显示前 500 行；完整数据在 `trajectories/episode_N.csv`。")
                except (pd.errors.ParserError, OSError):
                    pass


def _tab_artifacts(info: data.RunInfo) -> None:
    groups = data.list_artifacts(info.path)
    if not groups:
        st.info("这个目录里还没有产物。")
        return
    for name, files in groups.items():
        with st.expander(f"{name}（{len(files)}）", expanded=name == "模型"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "文件": str(path.relative_to(info.path)),
                            "大小": data.human_size(path.stat().st_size),
                        }
                        for path in files
                        if path.exists()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )


def _tab_config(info: data.RunInfo) -> None:
    resolved = data.read_yaml(info.path / paths.START_MARKER)
    if resolved:
        st.caption("`resolved_config.yaml` —— 这次运行**实际生效**的完整配置。")
        st.json(resolved)
    else:
        st.info("没有 `resolved_config.yaml`。")

    summary = info.summary
    if summary:
        with st.expander("evaluation.json"):
            st.json(summary)


def render() -> None:
    theme.page_header("运行详情", "曲线、诊断、轨迹回放与产物清单")

    runs = _shared.load_runs()
    info = _shared.run_selector(runs, key="detail_run")
    if info is None:
        return
    st.session_state["selected_run"] = info.name

    # 一行徽章代替 st.metric：metric 默认字号极大，几个并排会把标题压下去，
    # 视觉重心反了，还白占纵向空间。
    theme.meta_row(
        [
            ("状态", data.STATUS_LABELS[info.status]),
            ("算法", info.algorithm or "-"),
            ("环境", info.environment or "-"),
            ("配置", info.config_name or "自定义目录"),
            ("种子", str(info.seed) if info.seed is not None else "-"),
        ]
    )
    summary = info.summary
    if summary:
        theme.meta_row(
            [
                ("最终评估", f"{summary.get('mean_reward', 0):.2f} ± {summary.get('std_reward', 0):.2f}"),
                ("评估回合", str(summary.get("episodes", "-"))),
                ("训练步数", f"{summary.get('timesteps', info.total_timesteps or 0):,}"),
            ]
        )
    st.write("")

    if info.status == "running":
        progress = live.compute_progress(info.path, info.total_timesteps)
        if progress is not None:
            st.progress(min(progress.fraction, 1.0),
                        text=f"{progress.done:,} / {progress.total:,} 步")

    tabs = st.tabs(["训练曲线", "训练信号", "动作诊断", "周期评估",
                    "轨迹回放", "产物", "配置与日志"])
    with tabs[0]:
        _tab_curve(info)
    with tabs[1]:
        _tab_signals(info)
    with tabs[2]:
        _tab_diagnostics(info)
    with tabs[3]:
        _tab_evaluations(info)
    with tabs[4]:
        _tab_trajectories(info)
    with tabs[5]:
        _tab_artifacts(info)
    with tabs[6]:
        _render_regenerate(info.path, info.config_name)
        st.divider()
        _tab_config(info)
