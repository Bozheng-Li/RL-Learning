"""发起训练：从配置动态生成表单，以子进程方式启动 `train.py`。"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import streamlit as st

from webui import data, jobs, paths

#: 摊平配置时跳过的子树。
_SKIP_KEYS = {"profiles"}

#: 高级覆盖里不重复暴露的键（上面已有专门的控件）。
_MANAGED_KEYS = {
    "algorithm.name",
    "experiment.seed",
    "training.total_timesteps",
    "output.directory",
}


def _flatten_scalars(
    config: dict[str, Any], prefix: str = "", *, skip: set[str] | None = None
) -> dict[str, Any]:
    """把配置摊平成 ``{"training.total_timesteps": 50000, ...}``，只保留标量叶子。

    ``apply_overrides`` 只接受已经存在的键，所以表单必须从配置里**探测**出这些键，
    不能凭空构造。列表与字典整段跳过——它们在输入框里没法编辑。
    """
    skip = skip or set()
    found: dict[str, Any] = {}
    for key, value in config.items():
        if key.startswith("_") or key in _SKIP_KEYS or key in skip:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            found.update(_flatten_scalars(value, f"{path}.", skip=skip))
        elif isinstance(value, bool) or (isinstance(value, (int, float, str)) and value is not None):
            found[path] = value
    return found


def _conflicting_job(run_dir: Path) -> bool:
    """目标目录是否正被一个还活着的任务写入。"""
    for job in jobs.load_registry():
        if job.run_dir and Path(job.run_dir) == run_dir and job.alive:
            return True
    return False


def render() -> None:
    st.header("发起训练")

    names = data.list_config_names()
    if not names:
        st.error(f"在 {paths.CONFIG_ROOT} 下没有找到任何配置。")
        return

    left, right = st.columns([3, 2], gap="large")

    with left:
        config_name = st.selectbox("配置", names, key="train_config")
        try:
            config = data.load_config_dict(config_name)
        except (ValueError, OSError) as error:
            st.error(f"读取配置失败：{error}")
            return

        algorithm_cfg = config.get("algorithm") or {}
        # 算法下拉的数据源必须是这个配置**预置了超参**的算法。
        # 用 available_algorithms() 会列出没有 profile 的算法，选中后训练直接
        # 抛 ValueError: No algorithm.profiles.X section exists。
        profiles = data.available_profiles(config)
        if not profiles:
            st.error("这个配置没有定义任何 algorithm.profiles。")
            return
        default = algorithm_cfg.get("name", profiles[0])
        algorithm = st.selectbox(
            "算法 profile",
            profiles,
            index=profiles.index(default) if default in profiles else 0,
            key="train_algorithm",
        )

        use_sb3 = st.checkbox(
            "使用 Stable-Baselines3 实现（对照基线）",
            value=False,
            key="train_use_sb3",
            help="profile 的键名不带 SB3- 前缀，所以 PPO 与 SB3-PPO 读的是同一份超参，"
                 "只有实现不同——这正是做实现对照实验的前提。",
        )
        selected = f"SB3-{algorithm}" if use_sb3 else algorithm

        experiment_cfg = config.get("experiment") or {}
        training_cfg = config.get("training") or {}
        column_a, column_b = st.columns(2)
        seed = column_a.number_input(
            "随机种子", value=int(experiment_cfg.get("seed", 42)), step=1, key="train_seed"
        )
        timesteps = column_b.number_input(
            "总步数",
            value=int(training_cfg.get("total_timesteps", 10000)),
            min_value=1,
            step=1000,
            key="train_timesteps",
        )

        # 动作空间前置校验：选错的报错要等训练启动几十秒后才出现在日志里，
        # 不如在这里就拦住。
        kind = data.probe_action_space(config)
        upper = selected.upper()
        if kind != "unknown":
            if upper.endswith("DQN") and kind != "discrete":
                st.error(f"DQN 只支持离散动作空间，而 {config_name} 看起来是连续的。")
            if upper.endswith(("SAC", "TD3")) and kind != "continuous":
                st.error(
                    f"SAC/TD3 只支持连续动作空间，而 {config_name} 看起来是离散的。"
                    "LunarLander 之类的环境需要在 `environment.kwargs` 里设 `continuous: true`。"
                )

        with st.expander("输出目录", expanded=True):
            unique = st.checkbox(
                "用独立目录（不覆盖已有结果）",
                value=True,
                key="train_unique",
                help="平铺成 outputs/<配置>_<算法>_s<种子>_<时间戳>。"
                     "不要用嵌套时间戳目录——compare.py 只扫 outputs/ 的一层，会看不到。",
            )
            directory = st.text_input(
                "output.directory",
                value=jobs.suggest_output_directory(
                    config_name, selected, int(seed), unique=unique
                ),
                key="train_directory",
            )

        with st.expander("高级：任意配置覆盖", expanded=False):
            st.caption(
                "只列出这份配置里**已经存在**的标量键——`apply_overrides` 刻意不允许新建键，"
                "拼错的键会直接抛错而不是被静默忽略。"
            )
            scalars = _flatten_scalars(config)
            editable = {k: v for k, v in scalars.items() if k not in _MANAGED_KEYS}
            picked = st.multiselect("要覆盖的键", sorted(editable), key="train_override_keys")
            overrides: dict[str, Any] = {}
            for key in picked:
                current = editable[key]
                widget_key = f"override::{key}"
                if isinstance(current, bool):
                    overrides[key] = st.checkbox(key, value=current, key=widget_key)
                elif isinstance(current, int):
                    overrides[key] = st.number_input(key, value=current, step=1, key=widget_key)
                elif isinstance(current, float):
                    overrides[key] = st.number_input(key, value=current, key=widget_key)
                else:
                    overrides[key] = st.text_input(key, value=str(current), key=widget_key)

    run_dir = paths.resolve_repo_path(directory)

    argv = jobs.build_train_argv(
        config_name,
        algorithm=selected,
        seed=int(seed),
        timesteps=int(timesteps),
        output_directory=directory,
        config=config,
        overrides=overrides,
    )

    with right:
        st.subheader("将要执行的命令")
        st.caption("可以直接复制到终端执行——UI 做的事情和它完全一样。")
        st.code(shlex.join(argv), language="bash", wrap_lines=True)

        st.caption("输出目录")
        # 用 code 而不是 metric：metric 会把长路径渲染成巨大的字号并溢出。
        st.code(directory, language=None)

        conflict = _conflicting_job(run_dir)

    if conflict:
        st.warning(f"`{run_dir}` 正被另一个还在运行的任务写入，先换个目录。")

    if st.button("开始训练", type="primary", disabled=conflict, width="stretch"):
        job = jobs.launch(
            argv,
            label=f"train:{config_name}",
            kind="train",
            run_dir=run_dir,
            total_timesteps=int(timesteps),
        )
        st.session_state["page"] = "训练监控"
        st.session_state["jobs"] = {**st.session_state.get("jobs", {}), job.job_id: job}
        st.success(f"已启动（PID {job.pid}）。训练独立于本页面运行，关掉浏览器也不会中断。")
        st.rerun()
