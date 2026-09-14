"""训练编排。

这一层负责把配置、环境、算法、回调、日志、可视化串起来，是整个项目的调度中心。
算法本身的实现放在 ``algorithms/``，环境构造放在 ``environments/``，
日志放在 ``rl_common/training_log.py``。

一次完整训练的产物（都在 ``output.<directory>`` 下）：

    logs/train.monitor.csv        每个训练 episode 的回报与长度
    logs/progress.csv             每个 rollout 的训练信号（损失、熵、KL、解释方差...）
    logs/episodes.csv             每个 episode 的明细（含累计步数与种子）
    logs/diagnostics.csv          周期性记录的动作分布与策略统计
    evaluation/evaluations.npz    周期性评估的原始结果
    evaluation.json               最终评估的汇总
    checkpoints/*.zip             周期性模型快照
    best_model/best_model.zip     周期评估中最优的模型
    visualizations/*.png          训练曲线
    trajectories/*.{mp4,png,npz,csv}  策略轨迹
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure

from algorithms import build_algorithm, resolve_model_file
from environments import make_environment

from .config import (
    apply_overrides,
    apply_torch_threads,
    load_config,
    run_directory,
    write_resolved_config,
)
from .training_log import DiagnosticsCallback, EpisodeLogCallback
from .visualization import generate_trajectories, plot_training_results


def _build_callbacks(
    config: dict[str, Any], run_dir: Path, eval_env: gym.Env | None
) -> CallbackList | None:
    """组装训练回调。

    这里复用 SB3 的 ``CheckpointCallback`` / ``EvalCallback``——自研算法实现了
    ``num_timesteps`` / ``get_env()`` / ``save()`` / ``predict()``，因此同样能被
    它们驱动，不必为两种实现各写一套。
    """
    callbacks: list[Any] = []
    checkpoint = config["training"].get("checkpoint", {})
    if checkpoint.get("enabled", False):
        callbacks.append(
            CheckpointCallback(
                save_freq=int(checkpoint["save_freq"]),
                save_path=str(run_dir / checkpoint.get("directory", "checkpoints")),
                name_prefix=checkpoint.get("name_prefix", "checkpoint"),
                save_replay_buffer=checkpoint.get("save_replay_buffer", False),
                save_vecnormalize=checkpoint.get("save_vecnormalize", False),
            )
        )

    evaluation = config.get("evaluation", {})
    if eval_env is not None and evaluation.get("enabled", True):
        eval_frequency = int(evaluation.get("eval_freq", 0))
        if eval_frequency > 0:
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(run_dir / "best_model"),
                    log_path=str(run_dir / "evaluation"),
                    eval_freq=eval_frequency,
                    n_eval_episodes=int(evaluation.get("n_eval_episodes", 5)),
                    deterministic=evaluation.get("deterministic", True),
                    render=False,
                    warn=False,
                )
            )

    # 下面两个是自研的日志回调，对所有算法都生效。
    callbacks.append(EpisodeLogCallback(config, run_dir))
    diagnostics = config.get("training", {}).get("diagnostics", {})
    if diagnostics.get("enabled", True):
        callbacks.append(
            DiagnosticsCallback(
                config,
                run_dir,
                interval=int(diagnostics.get("interval", 10_000)),
            )
        )

    return CallbackList(callbacks) if callbacks else None


def train_from_config(
    config_path: str | Path, *, overrides: dict[str, Any] | None = None
) -> Path:
    """读配置、套用命令行覆盖、然后训练。

    Args:
        config_path: YAML 配置路径。
        overrides: 点号路径覆盖，如 ``{"algorithm.name": "SB3-PPO"}``。
    """
    config = load_config(config_path)
    if overrides:
        apply_overrides(config, overrides)
    return train(config)


def train(config: dict[str, Any]) -> Path:
    """按已解析的配置训练一个模型，返回运行目录。"""
    apply_torch_threads(config)
    run_dir = run_directory(config, create_timestamp=True)
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if config["output"].get("save_resolved_config", True):
        write_resolved_config(config, run_dir)

    seed = int(config["experiment"].get("seed", 42))
    train_env = make_environment(config, monitor_path=logs_dir / "train.monitor.csv")
    train_env.reset(seed=seed)
    train_env.action_space.seed(seed)

    evaluation = config.get("evaluation", {})
    eval_env = None
    if evaluation.get("enabled", True) and int(evaluation.get("eval_freq", 0)) > 0:
        eval_env = make_environment(
            config,
            monitor_path=logs_dir / "periodic_eval.monitor.csv",
            # 评估时关掉奖励塑形，否则评估回报会混入探索奖励。
            exploration=False,
        )
        eval_env.reset(seed=seed + int(evaluation.get("seed_offset", 10_000)))

    model = build_algorithm(train_env, config)
    # SB3 算法靠这一步产出 logs/progress.csv（损失、熵、KL、解释方差等训练信号）。
    # 自研算法的 set_logger 是空实现——它们在自己的训练循环里用 ProgressWriter
    # 写同样格式的 progress.csv，列名刻意保持一致。
    logger_formats = config["output"].get("logger_formats", ["stdout", "csv"])
    model.set_logger(configure(str(logs_dir), logger_formats))
    callbacks = _build_callbacks(config, run_dir, eval_env)
    training = config["training"]

    try:
        model.learn(
            total_timesteps=int(training["total_timesteps"]),
            callback=callbacks,
            log_interval=int(training.get("log_interval", 1)),
            progress_bar=training.get("progress_bar", False),
            reset_num_timesteps=training.get("reset_num_timesteps", True),
        )
        model_stem = run_dir / config["output"].get("model_name", "final_model")
        model.save(model_stem)
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    saved_model = resolve_model_file(model_stem, str(config["algorithm"]["name"]))
    if evaluation.get("enabled", True):
        summary = _run_final_evaluation(config, model, run_dir, saved_model)
        print(
            f"Final evaluation: {summary['mean_reward']:.2f} +/- "
            f"{summary['std_reward']:.2f} over {summary['episodes']} episodes"
        )
    print(f"Saved model: {saved_model}")

    visualization = config.get("visualization", {})
    if visualization.get("training", {}).get("enabled", True):
        plot_training_results(config, run_dir)
    trajectory = visualization.get("trajectory", {})
    if trajectory.get("enabled", True) and trajectory.get(
        "generate_after_training", True
    ):
        generate_trajectories(config, model, run_dir)
    return run_dir


def _run_final_evaluation(
    config: dict[str, Any], model: Any, run_dir: Path, model_path: Path
) -> dict[str, Any]:
    """训练结束后跑一次完整评估，结果写进 ``evaluation.json``。

    这个文件是跨运行对比（``compare.py``）的主要数据来源，所以字段要稳定。
    """
    evaluation = config.get("evaluation", {})
    episodes = int(evaluation.get("final_episodes", 10))
    if episodes <= 0:
        raise ValueError("evaluation.final_episodes must be positive")
    seed = int(config["experiment"].get("seed", 42))
    seed_offset = int(evaluation.get("seed_offset", 10_000))
    eval_env = make_environment(
        config,
        monitor_path=run_dir / "logs" / "final_eval.monitor.csv",
        exploration=False,
    )
    eval_env.reset(seed=seed + seed_offset + 1)
    rewards, lengths = evaluate_deterministic(
        model,
        eval_env,
        episodes=episodes,
        deterministic=evaluation.get("deterministic", True),
    )
    eval_env.close()
    summary = {
        "experiment": config["experiment"].get("name"),
        "environment": config["environment"]["id"],
        "algorithm": str(config["algorithm"]["name"]).upper(),
        "timesteps": int(config["training"]["total_timesteps"]),
        "seed": seed,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "episode_rewards": [float(value) for value in rewards],
        "episode_lengths": [int(value) for value in lengths],
        "model": str(model_path),
    }
    with (run_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def evaluate_deterministic(
    model: Any,
    env: gym.Env,
    *,
    episodes: int,
    deterministic: bool = True,
) -> tuple[list[float], list[int]]:
    """跑固定回合数的评估，返回每回合的回报与长度。

    自己实现而不用 SB3 的 ``evaluate_policy``，是因为要精确控制回合边界上的
    循环策略状态：``RecurrentPPO`` 的 LSTM 隐状态必须在每个回合开始时重置，
    否则第二条轨迹会继承第一条的记忆。
    """
    rewards: list[float] = []
    lengths: list[int] = []
    for _ in range(episodes):
        observation, _ = env.reset()
        states = None
        episode_start = np.ones((1,), dtype=bool)
        done = False
        total = 0.0
        steps = 0
        while not done:
            action, states = model.predict(
                observation,
                state=states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            episode_start = np.zeros((1,), dtype=bool)
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = int(np.asarray(action).reshape(-1)[0])
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(total)
        lengths.append(steps)
    return rewards, lengths
