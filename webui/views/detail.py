"""单个运行的详情：全程训练复盘、信号、诊断、周期评估、轨迹、产物、配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from webui import charts, data, jobs, live, paths, theme, variants
from webui.views import _shared

#: 「训练全程」面板组里，训练过程信号的**固定**三格。这三条回答「学得怎么样」，
#: 其余信号（KL、熵、探索率…）按运行真的有的列动态追加。
_JOURNEY_CORE: tuple[str, ...] = ("reward", "eval_reward", "ep_len")

#: 动态追加的候选信号，按「先看这里最该看的」排序。
_JOURNEY_EXTRA: tuple[str, ...] = (
    "kl", "clip", "entropy", "pg_loss", "value_loss", "loss", "exploration", "lr",
)

#: 面板组的总上限（含固定的三格）。再多就是一屏滚不完，且每格压到读不出形状。
_JOURNEY_MAX_PANELS = 8


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


def _journey_frame(info: data.RunInfo) -> pd.DataFrame | None:
    """这次训练的信号表，index 是总步数，列名是**语义名**（`reward` / `entropy`…）。

    走 ``live.signal_frame``：它内含 ``progress.csv`` 的整份解析与超过 3000 行的等间隔
    抽稀（一次 200k 步的 PPO 有 4 万行）。信号名的取法一律经由 ``live.PROGRESS_COLUMNS``
    的映射表，**绝不按列位置索引**——每个算法写出来的列集合都不一样。
    """
    frame = live.signal_frame(info.path, keys=None)
    if frame is None or frame.empty:
        return None
    if frame.index.name != "steps":
        return None
    return frame


def _panel(series: pd.Series) -> pd.DataFrame:
    """一列信号 → altair 要的长表（``步数`` / ``值``）。"""
    table = pd.DataFrame(
        {
            "步数": series.index.astype(float),
            "值": pd.to_numeric(series, errors="coerce"),
        }
    )
    return table.dropna()


def _render_journey(info: data.RunInfo) -> None:
    """**一次训练从头到尾**的整合视图。这是「运行详情」的第一个 tab。

    为什么要有这一格：原来的第一个 tab 只是一张离线四联图 PNG（`training_results.png`）
    加一条 ``st.line_chart``，而它下面的六个 tab 各看一段，横向上互不相连——要回答
    「这次训练到底发生了什么」，得在七个 tab 之间来回切，还得自己把时间轴对起来。

    这里的顺序就是训练实际发生的时间顺序：

    1. 主干曲线：回合回报 + 周期评估，两格共用一根时间轴；
    2. 面板组：回合长度、训练过程信号（按这份运行真的有的列动态出）、动作分布，
       共用一个 x 轴与一把跨面板刷子——拖一段步数，所有面板同步聚焦到那一段；
    3. 周期评估的均值 ± 1σ（这一格回答「主干那条线有多可信」）；
    4. 最终评估的 10 个回合（`evaluation.json` 的 ``episode_rewards``，逐回合展开）；
    5. 落地录像；
    6. 离线四联图收进 expander。

    数据缺哪一块就跳过哪一块并写一句「为什么没有」，不留白格，也不静默消失。
    """
    frame = _journey_frame(info)
    curve = data.load_curve(info.path)

    if frame is None and curve is None:
        st.info(
            "这次运行还没有可画的训练数据。"
            "`logs/progress.csv`（实时信号）与 `evaluation/evaluations.npz`（周期评估）都不存在——"
            "训练刚开始时就是这样，开跑一会儿再回来。"
        )
        _render_static_figure(info)
        return

    if frame is not None:
        _render_main_curve(frame)
        brush = alt.selection_interval(encodings=["x"], name="journey_brush")
        panels = _journey_panels(frame)
        if panels:
            theme.section_head(
                "过程中发生了什么",
                f"{len(panels)} 格 · 拖动任意一格刷选步数区间，其余格同步聚焦",
            )
            # 面板超过三格就折成两列（每格窄一点），否则一屏滚不完，
            # 也失去了「竖着对齐同一个训练阶段」的意义。
            if len(panels) <= 3:
                st.altair_chart(
                    charts.journey_lines(panels, height=118, brush=brush), width="stretch"
                )
            else:
                st.altair_chart(
                    charts.journey_lines(panels, height=110, panel_width=430, brush=brush),
                    width="stretch",
                )
            theme.caption(
                "每一格一根独立的 y 轴，但共用一把刷子：在任意一格里横向拖选一段步数，"
                "其余格会在同一区间里高亮。于是「熵掉下去的那一刻，回合长度是不是同时涨了」"
                "可以直接看。信号清单是**按这份运行真的写出来的列**动态列的，"
                "不同算法不一样（DQN 有探索率没有 KL，自研算法没有 FPS）。"
            )
        _render_action_areas(info, brush)
    else:
        st.info("没有 `logs/progress.csv`（还在初始化，或这次运行没写过程日志），只画得出周期评估。")

    _render_eval_band(info)
    _render_final_episodes(info)
    _render_trajectory_block(info)
    _render_static_figure(info)


def _render_main_curve(frame: pd.DataFrame) -> None:
    """主干曲线：回合回报 + 周期评估，两格竖着对齐，共用一根 x 轴。

    ``rollout/ep_rew_mean`` 理论上每份运行都有，但**不能假设**：一次只写了几行、
    或者写日志的路径被裁掉过，``reward`` 列就可能整个缺席。缺席时退到这份运行真的
    有的第一条固定信号，而不是 ``KeyError``——页面炸掉比少一条线严重得多。
    """
    main_key = next(
        (key for key in _JOURNEY_CORE if key in frame.columns and frame[key].notna().any()),
        None,
    )
    if main_key is None:
        return
    theme.section_head("主干曲线", "一条时间轴贯穿整场训练 · 两格各自一根 y 轴")
    panels: list[Any] = [
        charts.line_panel(
            _panel(frame[main_key]),
            color="#2563eb",
            height=180,
            x_title="",
            title=f"{live.SIGNAL_LABELS.get(main_key, main_key)}"
                  "（训练过程中逐步累积的滑动平均，点密、抖）",
        )
    ]
    if "eval_reward" in frame.columns and frame["eval_reward"].notna().any():
        panels.append(
            charts.line_panel(
                _panel(frame["eval_reward"]),
                color="#bf3c48",
                height=140,
                x_title="训练步数",
                title="周期评估均值 · eval/mean_reward（每档固定几个回合，点稀、噪声小）",
            )
        )
    st.altair_chart(charts.stack(panels), width="stretch")
    theme.caption(
        "上面一格是训练过程中的回合回报（抖动大但点密，看得出「什么时候开始学会」）；"
        "下面一格是周期评估的均值（点稀但噪声小，是判断「学到没有」的主证据）。"
        "两格竖向对齐表示同一时刻。"
    )


def _journey_panels(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """面板组的成员：三条固定信号 + 这份运行真的有的过程信号（受总数上限约束）。

    哪些过程信号存在**每个算法都不一样**（实测同一份配置下：PPO 有 kl/clip/entropy/
    pg_loss/value_loss，DQN 有 exploration/loss，SAC/TD3 有 critic_loss/ent_coef，
    自研 REINFORCE 只有 entropy/pg_loss），所以这里按 ``frame.columns`` 判断，
    绝不硬编码清单，也绝不按列位置取。
    """
    panels: list[tuple[str, pd.DataFrame]] = []
    for key in _JOURNEY_CORE:
        if key in frame.columns:
            table = _panel(frame[key])
            if not table.empty:
                panels.append((live.SIGNAL_LABELS.get(key, key), table))
    for key in _JOURNEY_EXTRA:
        if len(panels) >= _JOURNEY_MAX_PANELS:
            break
        if key in frame.columns:
            table = _panel(frame[key])
            if not table.empty:
                panels.append((live.SIGNAL_LABELS.get(key, key), table))
    return panels


def _render_action_areas(info: data.RunInfo, brush: Any) -> None:
    """动作分布单独一格：离散动作是占比堆叠，连续动作是均值线。

    它**不进面板组**，因为量纲与那些信号完全不同（占比是 0..1，回报是几百），
    塞进同一组会让人以为它也在同一个刻度上。
    """
    table = data.load_diagnostics(info.path)
    if table is None or table.empty:
        return
    folded = _diagnostics_frame(table)
    if folded.empty:
        return
    theme.section_head("动作分布", "策略坍缩最直接的证据 · 某一色消失就是这个动作不再被选")
    # 离散动作空间是 ``action_*_frac``，和为 1，纵轴按百分比格式化；
    # 连续动作空间是 ``action_mean`` / ``action_std``，纵轴是原始量纲，不加百分号。
    discrete = any(name.endswith("_frac") for name in table.columns)
    st.altair_chart(
        charts.action_areas(
            folded,
            value_format="%" if discrete else "",
            height=180,
            brush=brush,
        ),
        width="stretch",
    )
    if discrete:
        theme.caption(
            "纵轴是各动作被选中的比例（合计 100%）。某个颜色被挤没了，"
            "说明策略退化成「永远选那一个动作」，而不是「训练不充分」——"
            "曲线塌平的时候先来这里看一眼，就能分开这两种情况。"
        )
    else:
        theme.caption(
            "连续动作空间没有「占比」可言，这里画的是动作各维度的均值随训练的变化。"
        )


def _diagnostics_frame(table: pd.DataFrame) -> pd.DataFrame:
    """动作诊断表 → 一张可以叠起来的占比面积图（多列合成一列「动作」）。

    离散动作的列名是 ``action_0_frac``，做法是把 ``action_`` 前缀与 ``_frac`` 后缀
    都去掉，剩下的就是动作编号（LunarLander 是 0..3）。保留编号而不是套一层中文名：
    编号与文档里的动作表对得上，中文名一旦写错反而误导。

    连续的 ``action_mean`` / ``action_std`` 是**动作各维度**的统计量，不是某几个动作的
    占比，所以原样保留整列名——把前缀后缀照剥会得到 ``mean`` / ``std`` 两个看不出是
    什么的名字。
    """
    value_columns = [name for name in table.columns if name != "timestep"]
    pieces = []
    for name in value_columns:
        if name.endswith("_frac"):
            label = name.removeprefix("action_").removesuffix("_frac")
        else:
            label = name
        pieces.append(
            pd.DataFrame({
                "步数": table["timestep"].astype(float),
                "动作": label,
                "占比": pd.to_numeric(table[name], errors="coerce"),
            })
        )
    if not pieces:
        return pd.DataFrame({"步数": [], "动作": [], "占比": []})
    return pd.concat(pieces).dropna()


def _render_eval_band(info: data.RunInfo) -> None:
    """周期评估的均值 ± 1σ：主干那条线到底有多可信。"""
    loaded = data.load_evaluations(info.path)
    if loaded is None:
        return
    steps, results, lengths = loaded
    means = results.mean(axis=1)
    stds = results.std(axis=1)
    theme.section_head(
        "周期评估的可信区间",
        f"{len(steps)} 次评估 · 每次 {results.shape[1]} 个回合",
    )
    band_frame = pd.DataFrame(
        {
            "步数": pd.to_numeric(pd.Series(steps), errors="coerce"),
            "均值": means,
            "上界 (+1σ)": means + stds,
            "下界 (-1σ)": means - stds,
        }
    ).dropna()
    st.altair_chart(
        charts.line_with_band(
            band_frame, lower="下界 (-1σ)", upper="上界 (+1σ)", middle="均值"
        ),
        width="stretch",
    )
    theme.caption(
        "带的宽度是**同一档评估里几个回合之间的离散度**（±1σ）。带很宽就说明这个分数"
        "不能只看均值——同一次评估里最好与最差的回合能差出上百点。"
    )
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
    with st.expander(f"逐档评估明细（{len(steps)} 档）", expanded=False):
        st.dataframe(
            detail_table,
            hide_index=True,
            width="stretch",
            height=theme.fit_height(len(detail_table), max_height=420),
        )


def _render_final_episodes(info: data.RunInfo) -> None:
    """最终评估的**逐回合**展开。

    ``evaluation.json`` 里的 ``episode_rewards`` / ``episode_lengths`` 是这次运行的
    最后 10 个回合的具体得分——两个数组一直都在，但此前界面一行都没读过，只显示了
    它们的均值与标准差。有了逐回合展开，「均值 132 但其中一回合是 −17」这种事才看得见，
    而那正是「这个模型能不能用」的关键。
    """
    summary = info.summary
    rewards = summary.get("episode_rewards")
    lengths = summary.get("episode_lengths")
    if not isinstance(rewards, list) or not rewards:
        return
    lengths = lengths if isinstance(lengths, list) else []
    mean = summary.get("mean_reward")
    std = summary.get("std_reward")
    theme.section_head(
        "最终评估的每一个回合",
        f"{len(rewards)} 个回合 · 均值 {mean:.2f}"
        + (f" ± {std:.2f}" if isinstance(std, (int, float)) else ""),
    )

    episodes = pd.DataFrame(
        {
            "回合": [f"第 {index} 回" for index in range(1, len(rewards) + 1)],
            "回报": pd.to_numeric(pd.Series(rewards), errors="coerce"),
        }
    ).dropna()
    if episodes.empty:
        return
    st.altair_chart(
        charts.bars(
            episodes,
            x_field="回合",
            y_field="回报",
            baseline=float(mean) if isinstance(mean, (int, float)) else 0.0,
            height=210,
        ),
        width="stretch",
    )
    theme.caption(
        "一根柱子是一个回合的得分，红色虚线是这 10 个回合的均值。"
        "**同一个模型在不同开局下的差距，就是这一格的柱高差**——柱高差得很开，"
        "说明均值这个数不能单独看。"
    )
    detail = pd.DataFrame(
        {
            "回合": list(range(1, len(rewards) + 1)),
            "回报": rewards,
            "步数": lengths + ["-"] * (len(rewards) - len(lengths)),
        }
    )
    st.dataframe(detail, hide_index=True, width="stretch")


def _render_trajectory_block(info: data.RunInfo) -> None:
    """落地录像：训练结束后录的几次飞行，放在全程视图里而不是单独的 tab。"""
    episodes = data.trajectory_summary(info.path)
    directory = info.path / "trajectories"
    if not episodes and not directory.is_dir():
        return
    theme.section_head("落地录像", "同一套评估种子的几次飞行 · 过程在图里，结果在这里")
    _render_episodes(episodes, directory)


def _render_static_figure(info: data.RunInfo) -> None:
    """离线四联图（``visualizations/training_results.png``）——附件，不是第一屏。"""
    image = info.path / "visualizations" / "training_results.png"
    if not image.exists():
        return
    with st.expander("离线四联图（`visualizations/training_results.png`）", expanded=False):
        st.image(str(image), width="stretch")
        st.caption(
            "`visualize.py` 生成的四联图：训练回报（含滑动平均）、回合长度、"
            "周期评估、最终评估分布。图内文字是英文——matplotlib 默认字体不含中文。"
            "它不含交互，放这里当附件；上面各格是它的可交互版本。"
        )


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


def _render_episodes(episodes: list[dict[str, Any]], directory: Path) -> None:
    """轨迹记录的表 + 逐个回合的展开面板（视频 / 首帧 / 前 500 行 CSV）。

    抽成独立函数是因为有两个入口读同一份数据：「训练全程」这一格里要直接看到落地过程
    （全程视图的最后一环就是「它最后飞成什么样」），「轨迹回放」tab 则要单独细看。
    两处渲染体一模一样，抄两遍的结果一定是改一处忘一处。
    """
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


def _tab_trajectories(info: data.RunInfo) -> None:
    episodes = data.trajectory_summary(info.path)
    directory = info.path / "trajectories"
    if not episodes and not directory.is_dir():
        st.info("还没有轨迹记录。用上面的「重新录轨迹」生成。")
        return
    _render_episodes(episodes, directory)


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


def _render_variant_diff(info: data.RunInfo) -> None:
    """「相对原配置改了什么」——变体运行最该先看的一段。

    变体批次跑起来后，光看回报排序看不出 ``lr-1e-3`` 与 ``lr-1e-4`` 差在哪。这里拿
    ``resolved_config.yaml``（训练启动那一刻写下的真相）跟当前的原配置逐叶子比，
    把变化的点号路径列出来。
    """
    if not info.variant or not info.config_name:
        return
    resolved = data.read_yaml(info.path / paths.START_MARKER)
    if not resolved:
        return
    try:
        base = data.load_config_dict(info.config_name)
    except (OSError, ValueError):
        return

    from webui import variants  # noqa: PLC0415  —— 只有这一小段需要

    changed = variants.diff_against_base(resolved, base)
    if not changed:
        theme.caption("这个变体的配置与原配置逐字段相同（可能只改了由批次参数承载的步数）。")
        return
    theme.section_head("相对原配置改了什么", f"对比 config/{info.config_name}.yaml")
    st.dataframe(
        pd.DataFrame([
            {"键": key, "原值": _show(old), "生效值": _show(new)}
            for key, old, new in changed
        ]),
        hide_index=True,
        width="stretch",
        height=theme.fit_height(len(changed)),
    )


def _show(value: Any) -> str:
    """diff 表里的值：缺席用一句话表示，其余用 YAML 标量写法（列表也读得懂）。"""
    if value is variants.MISSING:
        return "（配置里没有这个键）"
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    return str(value)


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
    theme.page_header(
        "运行详情",
        "一次训练从头到尾的复盘：主干曲线、联动信号、动作分布、误差带、逐回合得分、录像，"
        "以及产物与配置。",
        eyebrow="Reinforce / run detail",
    )

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
            ("设备", info.device or "cpu"),
            ("配置", info.config_name or "自定义目录"),
            ("变体", info.variant or "基准"),
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

    tabs = st.tabs(["训练全程", "训练信号", "动作诊断", "周期评估",
                    "轨迹回放", "产物", "配置与日志"])
    with tabs[0]:
        _render_journey(info)
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
        _render_variant_diff(info)
        _tab_config(info)
