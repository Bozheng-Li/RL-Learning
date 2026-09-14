"""训练结果可视化与策略轨迹录制。

输出三类东西：

- **训练曲线**（``visualizations/training_results.png``）：回报、回合长度、
  周期评估、最终评估分布，四张子图。
- **策略轨迹**（``trajectories/episode_N.*``）：录成 MP4、画出观测/动作/奖励
  曲线、同时保存 npz 与 csv 原始数据。
- **轨迹摘要**（``trajectories/summary.json``）：每条的回报与步数。

绘制一律用 Agg 后端，所以在无桌面的服务器上也能跑。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from algorithms import load_algorithm, resolve_model_file  # noqa: E402
from environments import make_environment  # noqa: E402

from .config import load_config, resolve_repo_path, run_directory  # noqa: E402


def _load_monitor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 Monitor 写出的 CSV，返回 (累计步数, 每个 episode 回报, 长度)。"""
    if not path.exists():
        return np.array([]), np.array([]), np.array([])
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    rewards = np.asarray([float(row["r"]) for row in rows], dtype=float)
    lengths = np.asarray([int(row["l"]) for row in rows], dtype=int)
    timesteps = np.cumsum(lengths)
    return timesteps, rewards, lengths


def _rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """滑动平均。返回 (对应的索引, 均值)，索引用于和步数轴对齐。"""
    if values.size == 0:
        return np.array([]), np.array([])
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=float) / window
    means = np.convolve(values, kernel, mode="valid")
    indices = np.arange(window - 1, values.size)
    return indices, means


def plot_training_results(config: dict[str, Any], run_dir: Path) -> Path:
    """画四联图：训练回报、回合长度、周期评估、最终评估分布。"""
    visual_config = config.get("visualization", {}).get("training", {})
    output_dir = run_dir / visual_config.get("output_subdirectory", "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / visual_config.get("filename", "training_results.png")
    timesteps, rewards, lengths = _load_monitor(run_dir / "logs" / "train.monitor.csv")
    smoothing_window = int(visual_config.get("smoothing_window", 20))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    reward_axis, length_axis, periodic_axis, final_axis = axes.flat
    if rewards.size:
        # 原始曲线画淡一些，滑动平均画粗——单看原始曲线在 RL 里几乎读不出趋势。
        reward_axis.plot(timesteps, rewards, alpha=0.3, color="#3377aa", label="episode")
        indices, smoothed = _rolling_mean(rewards, smoothing_window)
        reward_axis.plot(
            timesteps[indices], smoothed, color="#cc3311", linewidth=2, label="rolling mean"
        )
        reward_axis.legend()
        length_axis.plot(timesteps, lengths, color="#228833", alpha=0.7)
    else:
        reward_axis.text(0.5, 0.5, "No training monitor data", ha="center")
        length_axis.text(0.5, 0.5, "No episode-length data", ha="center")
    reward_axis.set(title="Training episode reward", xlabel="Timesteps", ylabel="Reward")
    length_axis.set(title="Training episode length", xlabel="Timesteps", ylabel="Steps")

    # 周期评估比训练曲线更可信：它是固定回合数的确定性评估，噪声小得多。
    evaluation_enabled = config.get("evaluation", {}).get("enabled", True)
    evaluation_path = run_dir / "evaluation" / "evaluations.npz"
    if evaluation_enabled and evaluation_path.exists():
        evaluation = np.load(evaluation_path)
        eval_steps = evaluation["timesteps"]
        eval_rewards = evaluation["results"]
        means = eval_rewards.mean(axis=1)
        stds = eval_rewards.std(axis=1)
        periodic_axis.plot(eval_steps, means, color="#aa4499")
        periodic_axis.fill_between(eval_steps, means - stds, means + stds, alpha=0.2)
    else:
        periodic_axis.text(0.5, 0.5, "No periodic evaluation data", ha="center")
    periodic_axis.set(title="Periodic evaluation", xlabel="Timesteps", ylabel="Reward")

    summary_path = run_dir / "evaluation.json"
    if evaluation_enabled and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        final_rewards = summary.get("episode_rewards", [])
        bins = min(10, max(1, len(final_rewards)))
        final_axis.hist(final_rewards, bins=bins, color="#ee7733", edgecolor="black")
        final_axis.axvline(
            summary["mean_reward"], color="#000000", linestyle="--", label="mean"
        )
        final_axis.legend()
    else:
        final_axis.text(0.5, 0.5, "No final evaluation data", ha="center")
    final_axis.set(title="Final evaluation rewards", xlabel="Reward", ylabel="Episodes")

    title = config["experiment"].get("name", config["environment"]["id"])
    fig.suptitle(f"{title} - {str(config['algorithm']['name']).upper()}", fontsize=16)
    fig.savefig(output_path, dpi=int(visual_config.get("dpi", 150)))
    plt.close(fig)
    print(f"Saved training visualization: {output_path}")
    return output_path


def _model_path(config: dict[str, Any], run_dir: Path) -> Path:
    """返回要加载的模型文件（不含扩展名，扩展名由算法决定）。"""
    trajectory = config.get("visualization", {}).get("trajectory", {})
    if trajectory.get("model", "final") == "best":
        return run_dir / "best_model" / "best_model"
    return run_dir / f"{config['output'].get('model_name', 'final_model')}"


def _observation_features(
    observation: Any, trajectory_config: dict[str, Any]
) -> np.ndarray:
    """把观测压成适合画图的标量序列，避免把图像展开成成百上千个子图。"""
    if isinstance(observation, dict):
        values = []
        for key in sorted(observation):
            value = observation[key]
            array = np.asarray(value)
            if array.dtype.kind in "OUS":
                continue
            values.extend(array.astype(float).reshape(-1))
        return np.asarray(values, dtype=float)

    array = np.asarray(observation)
    if array.dtype.kind in "OUS":
        return np.array([], dtype=float)
    array = array.astype(float)
    if trajectory_config.get("observation_mode", "auto") == "summary":
        features = [
            float(array.mean()),
            float(array.std()),
            float(array.min()),
            float(array.max()),
        ]
        if array.ndim >= 3:
            features.extend(float(value) for value in array.mean(axis=tuple(range(array.ndim - 1))))
        return np.asarray(features, dtype=float)
    return array.reshape(-1)


def _trajectory_labels(
    config: dict[str, Any], size: int, observation: Any
) -> list[str]:
    trajectory = config.get("visualization", {}).get("trajectory", {})
    labels = trajectory.get("observation_labels", [])
    if isinstance(observation, np.ndarray) and observation.ndim >= 2 and len(labels) != size:
        labels = ["observation_mean", "observation_std", "observation_min", "observation_max"]
        if observation.ndim >= 3:
            labels.extend(f"channel_{index}_mean" for index in range(observation.shape[-1]))
    if len(labels) != size:
        return [f"observation_{index}" for index in range(size)]
    return [str(label) for label in labels]


def _write_trajectory_csv(
    path: Path,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    labels: list[str],
) -> None:
    action_width = actions.shape[1]
    fieldnames = ["step", *labels]
    fieldnames += [f"action_{index}" for index in range(action_width)]
    fieldnames += ["reward", "cumulative_reward"]
    cumulative = np.cumsum(rewards)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(len(rewards)):
            row: dict[str, float | int] = {"step": step}
            row.update(
                {label: float(observations[step, index]) for index, label in enumerate(labels)}
            )
            row.update(
                {
                    f"action_{index}": float(actions[step, index])
                    for index in range(action_width)
                }
            )
            row["reward"] = float(rewards[step])
            row["cumulative_reward"] = float(cumulative[step])
            writer.writerow(row)


def _plot_trajectory(
    path: Path,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    labels: list[str],
    phase_plots: list[dict[str, Any]],
    dpi: int,
) -> None:
    plot_count = observations.shape[1] + 2 + len(phase_plots)
    columns = 2
    rows = math.ceil(plot_count / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(14, max(5, rows * 2.6)), constrained_layout=True
    )
    flat_axes = np.atleast_1d(axes).ravel()
    state_steps = np.arange(len(observations))
    transition_steps = np.arange(len(rewards))
    for index, label in enumerate(labels):
        flat_axes[index].plot(state_steps, observations[:, index], color="#3377aa")
        flat_axes[index].set(title=label, xlabel="Step", ylabel="Value")

    action_axis = flat_axes[observations.shape[1]]
    for index in range(actions.shape[1]):
        action_axis.step(
            transition_steps,
            actions[:, index],
            where="post",
            label=f"action_{index}",
        )
    if actions.shape[1] > 1:
        action_axis.legend()
    action_axis.set(title="Actions", xlabel="Step", ylabel="Action")

    reward_axis = flat_axes[observations.shape[1] + 1]
    cumulative_axis = reward_axis.twinx()
    reward_line = reward_axis.plot(
        transition_steps, rewards, color="#3377aa", alpha=0.75, label="step reward"
    )
    cumulative_line = cumulative_axis.plot(
        transition_steps,
        np.cumsum(rewards),
        color="#ee7733",
        label="cumulative reward",
    )
    reward_axis.legend(reward_line + cumulative_line, ["step reward", "cumulative reward"])
    reward_axis.set(title="Rewards", xlabel="Step", ylabel="Step reward")
    cumulative_axis.set_ylabel("Cumulative reward")

    for offset, phase_plot in enumerate(phase_plots):
        x_index = int(phase_plot["x"])
        y_index = int(phase_plot["y"])
        if not (0 <= x_index < observations.shape[1]):
            raise ValueError(f"phase_plots x index out of range: {x_index}")
        if not (0 <= y_index < observations.shape[1]):
            raise ValueError(f"phase_plots y index out of range: {y_index}")
        phase_axis = flat_axes[observations.shape[1] + 2 + offset]
        phase_axis.plot(
            observations[:, x_index], observations[:, y_index], color="#3377aa", alpha=0.5
        )
        phase_axis.scatter(
            observations[:, x_index],
            observations[:, y_index],
            c=state_steps,
            cmap="viridis",
            s=10,
        )
        phase_axis.set(
            title=phase_plot.get("title", f"{labels[x_index]} vs {labels[y_index]}"),
            xlabel=phase_plot.get("xlabel", labels[x_index]),
            ylabel=phase_plot.get("ylabel", labels[y_index]),
        )
    for axis in flat_axes[plot_count:]:
        axis.set_visible(False)
    fig.suptitle(f"Trajectory: return={rewards.sum():.2f}, steps={len(rewards)}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def generate_trajectories(
    config: dict[str, Any], model: Any, run_dir: Path
) -> list[dict[str, Any]]:
    """用训练好的策略跑若干回合，录视频、存数据、画曲线。"""
    trajectory = config.get("visualization", {}).get("trajectory", {})
    episodes = int(trajectory.get("episodes", 1))
    if episodes <= 0:
        raise ValueError("visualization.trajectory.episodes must be positive")
    record_video = trajectory.get("record_video", True)
    show_window = trajectory.get("show_window", False)
    if record_video and show_window:
        raise ValueError(
            "record_video and show_window cannot both be true; use video for headless runs"
        )
    render_mode = "rgb_array" if record_video else ("human" if show_window else None)
    # exploration=False：轨迹展示的是纯任务策略，不含训练用的探索奖励。
    env = make_environment(config, render_mode=render_mode, exploration=False)
    output_dir = run_dir / trajectory.get("output_subdirectory", "trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"].get("seed", 42))
    seed += int(trajectory.get("seed_offset", 20_000))
    max_steps = trajectory.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    if max_steps is not None and max_steps <= 0:
        raise ValueError("visualization.trajectory.max_steps must be positive or null")
    fps = int(trajectory.get("fps", env.metadata.get("render_fps", 30)))
    video_format = str(trajectory.get("video_format", "mp4")).lower()
    if video_format not in {"mp4", "gif"}:
        raise ValueError("visualization.trajectory.video_format must be mp4 or gif")

    summaries: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            observations = [_observation_features(observation, trajectory)]
            raw_observations = [observation]
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            terminated = truncated = False
            writer = None
            video_path = output_dir / f"episode_{episode + 1}.{video_format}"
            if record_video:
                writer_kwargs = (
                    {
                        "fps": fps,
                        "macro_block_size": int(trajectory.get("macro_block_size", 8)),
                    }
                    if video_format == "mp4"
                    else {"duration": 1.0 / fps, "loop": 0}
                )
                writer = imageio.get_writer(video_path, mode="I", **writer_kwargs)
                writer.append_data(env.render())

            # 循环策略（如 RecurrentPPO）的 LSTM 隐状态必须在每个回合边界重置，
            # 否则录下来的第二条轨迹会带着第一条的记忆。无记忆算法会忽略这两个
            # 参数而非报错——这也是 SB3 自家 evaluate_policy 驱动所有算法的方式。
            episode_start = np.ones((1,), dtype=bool)
            recurrent_states: tuple[np.ndarray, ...] | None = None
            step = 0
            try:
                while not (terminated or truncated):
                    if max_steps is not None and step >= max_steps:
                        break
                    action, recurrent_states = model.predict(
                        observation,
                        state=recurrent_states,
                        episode_start=episode_start,
                        deterministic=trajectory.get("deterministic", True),
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        # 非向量化环境下 SB3 可能返回单元素 ndarray，
                        # 而 Gymnasium 的离散环境要的是 int。
                        action = int(np.asarray(action).reshape(-1)[0])
                    observation, reward, terminated, truncated, _ = env.step(action)
                    actions.append(np.asarray(action, dtype=float).reshape(-1))
                    rewards.append(float(reward))
                    observations.append(_observation_features(observation, trajectory))
                    raw_observations.append(observation)
                    if writer is not None:
                        writer.append_data(env.render())
                    step += 1
            finally:
                if writer is not None:
                    writer.close()

            observations_array = np.asarray(observations, dtype=float)
            actions_array = np.asarray(actions, dtype=float).reshape(len(actions), -1)
            rewards_array = np.asarray(rewards, dtype=float)
            labels = _trajectory_labels(config, observations_array.shape[1], observation)
            stem = output_dir / f"episode_{episode + 1}"
            if trajectory.get("save_data", True):
                data = {
                    "observations": observations_array,
                    "actions": actions_array,
                    "rewards": rewards_array,
                }
                if trajectory.get("save_raw_observations", False):
                    try:
                        data["raw_observations"] = np.asarray(raw_observations)
                    except (TypeError, ValueError):
                        # 字典或文本观测已经由标量特征表示过了。
                        pass
                np.savez_compressed(stem.with_suffix(".npz"), **data)
                _write_trajectory_csv(
                    stem.with_suffix(".csv"),
                    observations_array,
                    actions_array,
                    rewards_array,
                    labels,
                )
            if trajectory.get("save_plot", True):
                _plot_trajectory(
                    stem.with_suffix(".png"),
                    observations_array,
                    actions_array,
                    rewards_array,
                    labels,
                    trajectory.get("phase_plots", []),
                    int(trajectory.get("dpi", 150)),
                )
            summary = {
                "episode": episode + 1,
                "seed": seed + episode,
                "steps": len(rewards),
                "return": float(rewards_array.sum()),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "video": str(video_path) if record_video else None,
            }
            summaries.append(summary)
            print(
                f"Trajectory {episode + 1}: return={summary['return']:.2f}, "
                f"steps={summary['steps']}"
            )
    finally:
        env.close()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    return summaries


def visualize_from_config(
    config_path: str | Path,
    *,
    include_training: bool = True,
    include_trajectories: bool = True,
    model_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """不重新训练，直接用已有模型重画图、重录轨迹。

    Args:
        overrides: 与训练时相同的点号路径覆盖。必须一致，否则会去找另一个
            输出目录（比如训练时用 ``--set output.directory=outputs/a``，
            这里也要传同样的覆盖才能找到结果）。
    """
    config = load_config(config_path)
    if overrides:
        from .config import apply_overrides

        apply_overrides(config, overrides)
    run_dir = run_directory(config)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}. Train a model first.")
    training_visualization = config.get("visualization", {}).get("training", {})
    trajectory_visualization = config.get("visualization", {}).get("trajectory", {})
    if include_training and training_visualization.get("enabled", True):
        plot_training_results(config, run_dir)
    if include_trajectories and trajectory_visualization.get("enabled", True):
        if model_path is not None:
            path = resolve_repo_path(model_path)
        else:
            path = resolve_model_file(
                _model_path(config, run_dir), str(config["algorithm"]["name"])
            )
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Model not found below {_model_path(config, run_dir)}. Train a model first."
            )
        model = load_algorithm(
            path, config, device=config["algorithm"].get("device", "cpu")
        )
        generate_trajectories(config, model, run_dir)
    return run_dir
