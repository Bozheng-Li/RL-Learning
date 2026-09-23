"""发起训练：在同一个环境上一次跑齐多个算法，或同一个算法的多份配置。

对照实验的前提是「同环境、同超参、只换算法」。这里一次选多个算法（和多个种子），
每个组合启动一个独立子进程，共用同一个时间戳，事后按目录名就能归到同一组。

第二个模式把「只换算法」换成「只换配置」：同一个算法跑多份超参/环境参数，回答的是
「这个算法对什么敏感」。两个模式共用同一套用卡分流与启动链路，区别只在待启动列表
从「算法 × 种子」变成 ``variants.expand`` 的输出。
"""

from __future__ import annotations

import html
import shlex
import time
from collections.abc import Iterable
from typing import Any, Sequence

import pandas as pd
import streamlit as st

from webui import data, gpus, jobs, nav, paths, theme, variants
from webui.views import _shared

#: 摊平配置时跳过的子树。
_SKIP_KEYS = {"profiles"}

#: 两个训练模式。默认是原来的「算法 × 种子」——变体是刻意的深挖，不是默认路径。
_MODES: dict[str, str] = {
    "算法 × 种子": "base",
    "算法 × 变体 × 种子": "variant",
}

#: 变体表的三列。``variants.variant_from_lines`` 认的就是这三个名字。
_VARIANT_COLUMNS = ("变体名", "覆盖键", "值")

#: 高级覆盖里不重复暴露的键（上面已有专门的控件）。
_MANAGED_KEYS = {
    "algorithm.name",
    "experiment.seed",
    "training.total_timesteps",
    "output.directory",
}

#: 步数的常用档位。精确值仍可在下面直接改。
_TIMESTEP_PRESETS: dict[str, int] = {
    "冒烟 1 万步": 10_000,
    "短跑 5 万步": 50_000,
    "标准 20 万步": 200_000,
    "长跑 100 万步": 1_000_000,
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


def _parse_seeds(text: str) -> list[int]:
    """把 ``"42, 43 44"`` 解析成种子列表；非法片段直接丢掉。"""
    seeds: list[int] = []
    for token in text.replace(",", " ").split():
        try:
            seeds.append(int(token))
        except ValueError:
            continue
    return seeds


def _action_space_problem(config_name: str, config: dict[str, Any], selected: str) -> str | None:
    """动作空间不匹配时返回一句说明，匹配则返回 ``None``。

    SAC/TD3 不在这里拦：启动时会自动把 ``continuous`` 翻成 ``true``。
    """
    if data.needs_continuous(selected):
        return None
    kind = data.probe_action_space(config)
    if kind == "unknown":
        return None
    upper = selected.upper()
    if upper.endswith("DQN") and kind != "discrete":
        return f"{selected}：DQN 只支持离散动作，而 {config_name} 看起来是连续的"
    return None


#: 四个变体维度的快捷预设：点一下把对应行填进表里。
#:
#: 这不是另一套机制，只是替用户敲那几行字——填进去的还是同一张长表里的行，用户可以
#: 直接改。``<ALGO>`` 占位符保持原样，展开由 ``variants.resolve`` 做。
_VARIANT_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "学习超参 lr": {"lr-half": {"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.0001}},
    "网络宽度 net": {"net-256": {
        "algorithm.profiles.<ALGO>.policy_kwargs.net_arch": [256, 256],
    }},
    "环境扰动 wind": {"wind-strong": {
        "environment.kwargs.enable_wind": True,
        "environment.kwargs.wind_power": 15.0,
        "environment.kwargs.turbulence_power": 1.5,
    }},
    "训练预算 budget": {"budget-50k": {"training.total_timesteps": 50_000}},
    "评估种子 evalseed": {"evalseed-0": {"evaluation.seed_offset": 0}},
}


def _variant_rows(catalog: Iterable[variants.Variant]) -> list[dict[str, Any]]:
    """把变体集摊成长表的行（``变体名 / 覆盖键 / 值``）。

    值统一写成 YAML 标量，让用户看得见 ``[64, 64]`` 与 ``true`` 的原样——
    ``variant_from_lines`` 就是按 YAML 把字符串解析回原生类型的。
    """
    return [
        {"变体名": variant.name, "覆盖键": key, "值": jobs.yaml_scalar(value)}
        for variant in catalog
        for key, value in variant.overrides.items()
    ]


def _render_variant_editor(
    config_name: str, config: dict[str, Any], algorithms: list[str]
) -> tuple[list[variants.Variant], list[str]]:
    """变体模式的核心表单：变体集文件 + 长表编辑 + 校验说明。

    返回 ``(变体列表, 校验说明)``。用长表而不是宽表：四个维度的覆盖键各不相同，宽表
    会变成一张大部分为空的格子；长表的一行是「一个变体的一条覆盖」，同名归一组——
    这正是 ``variants.variant_from_lines`` 认的形状。

    编辑只活在 session 里，**不写回** ``config/variants/*.yaml``：服务端不主动往仓库里
    写文件。要留档就用旁边的「下载这份变体集」，自己放进那个目录。
    """
    theme.caption(
        "每个变体是一组点号路径覆盖。路径里的 <code>&lt;ALGO&gt;</code> 展开时替换成"
        "该算法的 profile 键（去掉 <code>SB3-</code> 前缀），所以同一条覆盖能同时用在"
        "自研 <code>PPO</code> 与 <code>SB3-PPO</code> 上。值按 YAML 语法解析，"
        "<code>true</code> / <code>0.001</code> / <code>[64, 64]</code> 都能写。"
    )

    catalog = variants.load_variant_set(config_name)
    listed = sorted(catalog)
    if listed:
        picked = st.multiselect(
            "从变体集文件载入",
            listed,
            default=listed,
            key=f"train_variant_pick::{config_name}",
            help=f"来源是 config/variants/{config_name}.yaml。取消勾选即从表里去掉。",
        )
    else:
        picked = []
        theme.caption(
            f"这份配置没有变体集文件（<code>config/variants/{config_name}.yaml</code>），"
            "可以直接在下面手填。"
        )

    quick = st.pills(
        "快捷预设",
        options=list(_VARIANT_PRESETS),
        selection_mode="multi",
        key=f"train_variant_presets::{config_name}",
        help="点一下把对应行填进下面的表；填进去之后照样可以直接改。",
    ) or []

    # 表的内容是**派生**的：勾了哪些变体、点了哪些预设，就得到哪些行。每次重跑都
    # 把派生集合与表对齐一次——去掉不再勾选的那些变体的行、补上缺的行——这样「取消
    # 勾选」立刻反映到表里，而不是把两处状态留给用户去猜。用户手填的变体名不在已知
    # 集合里，一律保留。
    desired = set(picked) | {
        name for preset in quick for name in _VARIANT_PRESETS[preset]
    }
    known = set(catalog) | {
        name for preset in _VARIANT_PRESETS.values() for name in preset
    }
    rows_key = f"train_variant_rows::{config_name}"
    current = [
        row for row in (st.session_state.get(rows_key) or [])
        if str(row.get("变体名") or "") not in known
        or str(row.get("变体名") or "") in desired
    ]
    derived = _variant_rows(catalog[name] for name in picked)
    for preset in quick:
        derived.extend(_variant_rows(
            variants.Variant(name=name, overrides=dict(payload))
            for name, payload in _VARIANT_PRESETS[preset].items()
        ))
    for row in derived:
        if row not in current:
            current.append(row)

    edited = st.data_editor(
        pd.DataFrame(current, columns=list(_VARIANT_COLUMNS)),
        num_rows="dynamic",
        width="stretch",
        height=theme.fit_height(max(len(current), 4)),
        key=f"train_variant_editor::{config_name}",
        column_config={
            "变体名": st.column_config.TextColumn(
                help="同一个名字的几行属于同一个变体。不要含 __（它是实验名的分隔符）。"
            ),
            "覆盖键": st.column_config.TextColumn(
                help="点号路径，例如 algorithm.profiles.<ALGO>.kwargs.learning_rate"
            ),
            "值": st.column_config.TextColumn(help="按 YAML 解析；列表写成 [64, 64]。"),
        },
    )
    records = edited.to_dict("records")
    st.session_state[rows_key] = records

    catalogue = variants.variant_from_lines(records)
    if catalogue:
        st.download_button(
            "下载这份变体集",
            data=variants.dump_variant_set(catalogue).encode("utf-8"),
            file_name=f"{config_name}.yaml",
            mime="text/yaml",
            help="放进 config/variants/ 就会出现在上面的「从变体集文件载入」里。",
        )

    return catalogue, _variant_notes(catalogue, config, algorithms)


def _variant_notes(
    catalogue: list[variants.Variant], config: dict[str, Any], algorithms: list[str]
) -> list[str]:
    """把变体会在这次批次上遇到的问题说清楚，逐条给原因。

    校验只对**这一批真正会跑的算法**做，而不是配置里全部可跑的算法。差别很大：
    变体集里十几条超参覆盖对 ``REINFORCE`` 都不适用（它没有 profile），按全量算法
    校验就会刷出十几行与本次批次无关的「会跳过」。校验的目的是让用户看清「我点的
    这批里哪几条不会启动」，那就只该说这批。

    两类问题分开说：

    - **完全不成立**的变体（选中的算法上一个都用不了）单独一行，说明它不会启动任何运行；
    - **部分成立**的变体逐算法列出，因为 ``expand`` 是逐组合跳过的。
    """
    if not algorithms:
        return []
    notes: list[str] = []
    for variant in catalogue:
        problems_by_algorithm = {
            algorithm: variants.validate(variant, algorithm, config)
            for algorithm in algorithms
        }
        rejected = {a: p for a, p in problems_by_algorithm.items() if p}
        if len(rejected) == len(algorithms):
            notes.append(
                f"「{variant.name}」在这些算法上都用不了，不会启动任何运行："
                f"{next(iter(rejected.values()))[0]}"
            )
            continue
        for algorithm, problems in rejected.items():
            notes.append(f"「{variant.name}」在 {algorithm} 上会跳过：{problems[0]}")
        # 「覆盖值与默认值相同」分两种，处理不同：
        # - 整个变体被摘空（lr-3e-4 对 PPO）——那是跳过而不是提醒，``expand`` 会以
        #   「与基准相同，已去重」精确到每个组合地报出来，这里不重复说；
        # - 只摘掉其中几条（wind-strong 只改 enable_wind，wind_power 本来就是 15.0）
        #   ——变体照跑，但少了一项覆盖，值得单独说一句。
        accepted = [a for a in algorithms if not rejected.get(a)]
        resolved, warnings = variants.resolve_for(variant, accepted[0], config)
        if resolved:
            notes.extend(warnings)
    return list(dict.fromkeys(notes))


def _planned_runs(
    config_name: str,
    config: dict[str, Any],
    algorithms: list[str],
    seeds: list[int],
    timesteps: int,
    *,
    unique: bool,
    stamp: str,
    choices: list[gpus.DeviceChoice],
    overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """展开「算法 × 种子」成一组待启动的运行。

    ``choices`` 是**每任务一个**的用卡决定（长度 = 算法数 × 种子数），由调用方用
    ``gpus.build_choices`` 算好——用卡策略留在 ``gpus.py`` 一处，这里只负责铺开。
    """
    planned = []
    for index, (algorithm, seed) in enumerate(
        (algorithm, seed) for algorithm in algorithms for seed in seeds
    ):
        choice = choices[index] if index < len(choices) else gpus.CPU_CHOICE
        directory = jobs.suggest_output_directory(
            config_name, algorithm, seed, unique=unique, stamp=stamp
        )
        planned.append({
            "algorithm": algorithm,
            "seed": seed,
            "label": algorithm,
            "variant": None,
            "timesteps": int(timesteps),
            "job_label": f"train:{config_name}:{algorithm}:s{seed}",
            "directory": directory,
            "choice": choice,
            "device": gpus.short_label(choice),
            "problem": _action_space_problem(config_name, config, algorithm),
            "argv": jobs.build_train_argv(
                config_name, algorithm=algorithm, seed=seed, timesteps=timesteps,
                output_directory=directory, config=config, device=choice.device_arg,
                overrides=overrides,
            ),
        })
    return planned


def _planned_variants(
    config_name: str,
    config: dict[str, Any],
    specs: Sequence[variants.Spec],
    *,
    choices: list[gpus.DeviceChoice],
) -> list[dict[str, Any]]:
    """把 ``variants.expand`` 的输出铺成与 ``_planned_runs`` 同形的一批。

    两条链路从头到尾共用同一套东西：``gpus.build_choices`` 分流、``jobs.launch``
    启动、右边那张按卡分组的预览。这里只需要把 ``Spec`` 翻译成同样的字典——
    **GPU 分流那一层一行都不用改**，因为分流只看数量、不看这一批是什么。

    变体的覆盖表与 ``experiment.name`` 一起进 ``--set``：前者是「改了什么」，
    后者是「事后从运行目录里读回变体身份」的唯一依据（见 ``webui/variants.py``）。
    """
    planned = []
    for index, spec in enumerate(specs):
        choice = choices[index] if index < len(choices) else gpus.CPU_CHOICE
        overrides = dict(spec.overrides)
        if spec.experiment:
            overrides["experiment.name"] = spec.experiment
        planned.append({
            "algorithm": spec.algorithm,
            "seed": spec.seed,
            "label": spec.label,
            "variant": spec.variant_name,
            # 预算变体会把总步数改掉，所以每条运行各自带自己的步数：监控页与注册表
            # 里的进度条要用它，混用批次默认步数会让 5 万步的变体显示成跑了 1/4。
            "timesteps": spec.timesteps,
            "job_label": (
                f"train:{config_name}:{spec.algorithm}:s{spec.seed}"
                + (f":{spec.variant.tag}" if spec.variant else "")
            ),
            "directory": spec.directory,
            "choice": choice,
            "device": gpus.short_label(choice),
            "problem": _action_space_problem(config_name, config, spec.algorithm),
            "argv": jobs.build_train_argv(
                config_name, algorithm=spec.algorithm, seed=spec.seed,
                timesteps=spec.timesteps, output_directory=spec.directory,
                config=config, device=choice.device_arg, overrides=overrides,
            ),
        })
    return planned


def render() -> None:
    theme.page_header(
        "发起训练",
        "在同一个环境上一次跑齐多个算法；要让同一个算法跑多份配置，切到变体模式。",
        eyebrow="Reinforce / launch",
    )

    names = data.list_config_names()
    if not names:
        st.error(f"在 {paths.CONFIG_ROOT} 下没有找到任何配置。")
        return

    mode_label = st.segmented_control(
        "训练模式",
        options=list(_MODES),
        default=list(_MODES)[0],
        selection_mode="single",
        key="train_mode",
        help="「算法 × 种子」回答「哪个算法好」；「算法 × 变体 × 种子」回答"
             "「同一个算法对哪个超参敏感」。两者共用同一套显卡分流与启动链路。",
    )
    variant_mode = _MODES.get(mode_label or list(_MODES)[0]) == "variant"

    left, right = st.columns([3, 2], gap="large")

    with left:
        # 对比页可以预填「这个环境上还缺哪些算法」。先取走预填值（取走时会清掉同名
        # 控件键），否则控件带着旧值实例化之后，这里的 default 就会被直接忽略。
        preset_config = nav.take_preset("train_config")
        config_name = st.selectbox(
            "环境配置", names, index=names.index(preset_config) if preset_config in names else 0,
            key="train_config",
        )
        # 预填的算法按配置名分键（见下面 suffix 的说明），所以要在配置定下来之后才取。
        preset_algorithms = nav.take_preset(f"train_algorithms::{config_name}")
        # 跟着配置走的那几个控件，键都带配置名后缀——**每个配置一套独立的控件**。
        #
        # 这不是洁癖，是实测出来的：切了配置只把旧值从 session_state 里删掉**挡不住
        # 前端**。实测从 acrobot 切到 lunarlander，服务端当次就按 20 万步算（右栏
        # 预览也是 20 万），但输入框里还显示着 acrobot 的 10 万；下一次任何重跑，
        # 前端都把这个旧值原样发回来，服务端又按 10 万算。框里写的与真正会跑的
        # 不一致，而且不同环境的步数量级差很多。换个键就换了个新控件，前端没有
        # 旧值可发。
        #
        # 带后缀的另一个后果是切回去时看到的是配置默认值，不是上次在那边手填的值
        # （控件没在当前这一轮渲染，Streamlit 会把它的状态丢掉）。这是想要的行为：
        # 回到一个环境时从它自己的默认值起步，比继承一段几轮前的历史更好猜。
        #
        # 设备分配与「独立目录」不带后缀：它们与配置无关，切环境时本来就该留着。
        suffix = f"::{config_name}"
        try:
            config = data.load_config_dict(config_name)
        except (ValueError, OSError) as error:
            st.error(f"读取配置失败：{error}")
            return

        environment = (config.get("environment") or {}).get("id", "未知环境")
        theme.caption(f"环境 · <code>{environment}</code>")

        # 列出这个环境上能跑的全部算法（自研 + SB3）。SAC/TD3 需要连续动作空间，
        # 启动时会自动把 environment.kwargs.continuous 翻成 true。
        supported = data.compatible_algorithms(config)
        if not supported:
            st.error("这个配置没有定义任何可用的 algorithm.profiles。")
            return

        default_name = (config.get("algorithm") or {}).get("name", supported[0])
        preset_ok = [name for name in (preset_algorithms or []) if name in supported]
        algorithms = st.multiselect(
            "算法",
            supported,
            default=preset_ok or ([default_name] if default_name in supported else supported[:1]),
            key=f"train_algorithms{suffix}",
            help=("自研与 SB3- 前缀的读同一份 profile，只有实现不同。"
                  "选 SAC/TD3 时会自动切换到连续动作空间。")
            + ("变体模式下每个算法都会套上表里的每个变体，加上一条基准。" if variant_mode else ""),
        )

        # 变体表放在算法之后：它要按**选中的算法**逐条校验（某个算法没有 ent_coef 时，
        # 那个变体在它上面会被跳过，这件事必须在点按钮之前说出来）。按配置里全部可跑的
        # 算法校验会刷出一堆与本次批次无关的提示。
        catalogue: list[variants.Variant] = []
        variant_notes: list[str] = []
        if variant_mode:
            catalogue, variant_notes = _render_variant_editor(
                config_name, config, list(algorithms)
            )

        experiment_cfg = config.get("experiment") or {}
        training_cfg = config.get("training") or {}
        column_a, column_b = st.columns(2)
        seed_text = column_a.text_input(
            "随机种子",
            value=str(int(experiment_cfg.get("seed", 42))),
            key=f"train_seeds{suffix}",
            help="多个种子用空格或逗号分隔，例如 42 43 44。每个种子都会单独跑一遍。",
        )
        timesteps = column_b.number_input(
            "总步数",
            value=int(training_cfg.get("total_timesteps", 10000)),
            min_value=1,
            step=1000,
            key=f"train_timesteps{suffix}",
            help="变体里的 training.total_timesteps 会按变体各自覆盖这个值。",
        )
        preset = st.segmented_control(
            "步数档位",
            options=list(_TIMESTEP_PRESETS),
            default=None,
            selection_mode="single",
            key=f"train_preset{suffix}",
            label_visibility="collapsed",
        )
        if preset and preset in _TIMESTEP_PRESETS:
            timesteps = _TIMESTEP_PRESETS[preset]

        inventory = gpus.detect_devices()
        device_opts = gpus.device_options(inventory)
        opt_keys = [option.key for option in device_opts]
        opt_labels = {option.key: option.label for option in device_opts}
        chosen_key = st.selectbox(
            "计算设备分配",
            opt_keys,
            index=0,
            format_func=lambda key: opt_labels.get(key, key),
            key="train_device_choice",
            help=f"检测到 {len(inventory.devices)} 张可用显卡。"
                 "「自动分流」按剩余显存从多到少轮询，一张卡跑满再轮到下一张；"
                 "也可以固定在某一张卡或 CPU。"
                 if inventory.devices
                 else "没有检测到可用显卡，任务会跑在 CPU 上。",
        )
        chosen_device = next(
            option for option in device_opts if option.key == chosen_key
        )

        excluded: list[str] = []
        if chosen_key == gpus.AUTO_KEY:
            excluded = st.multiselect(
                "不使用这些显卡",
                options=[device.key for device in inventory.devices],
                format_func=lambda key: next(
                    (device.short_label for device in inventory.devices if device.key == key), key
                ),
                key="train_excluded_gpus",
                help="留一张卡给别的活，或避开有问题的卡。",
            )
        if inventory.notes:
            theme.caption("；".join(inventory.notes))

        with st.expander("显卡状态", expanded=False):
            _shared.gpu_panel(
                interval=st.session_state.get("poll_interval", 2.0), key="train-gpu"
            )

        with st.expander("输出目录", expanded=False):
            unique = st.checkbox(
                "每个算法一个独立目录（推荐）",
                value=True,
                key="train_unique",
                help="平铺成 outputs/<配置>_<算法>_s<种子>[_<变体>]_<时间戳>，同一批共用一个时间戳。"
                     "不要用嵌套时间戳目录——compare.py 只扫 outputs/ 的一层，会看不到。",
            )
            st.caption("关掉独立目录时，所有算法会写进同一个目录，后启动的会覆盖先启动的。")

        overrides: dict[str, Any] = {}
        if not variant_mode:
            with st.expander("高级：任意配置覆盖", expanded=False):
                st.caption(
                    "只列出这份配置里**已经存在**的标量键——`apply_overrides` 刻意不允许新建键，"
                    "拼错的键会直接抛错而不是被静默忽略。这里的覆盖会应用到这一批的每一次运行。"
                )
                scalars = _flatten_scalars(config)
                editable = {key: value for key, value in scalars.items() if key not in _MANAGED_KEYS}
                picked = st.multiselect(
                    "要覆盖的键", sorted(editable), key=f"train_override_keys{suffix}"
                )
                for key in picked:
                    current = editable[key]
                    # 同 suffix 的道理：覆盖值来自配置，换配置后旧值不该被前端发回来。
                    widget_key = f"override{suffix}::{key}"
                    if isinstance(current, bool):
                        overrides[key] = st.checkbox(key, value=current, key=widget_key)
                    elif isinstance(current, int):
                        overrides[key] = st.number_input(key, value=current, step=1, key=widget_key)
                    elif isinstance(current, float):
                        overrides[key] = st.number_input(key, value=current, key=widget_key)
                    else:
                        overrides[key] = st.text_input(key, value=str(current), key=widget_key)
        else:
            st.caption(
                "变体模式不走「高级覆盖」：每个变体自己就是一组覆盖，两者叠加会让"
                "「表里写了什么」与实际启动的命令对不上。"
            )

    seeds = _parse_seeds(seed_text)
    selected = list(algorithms)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    skipped: list[str] = []
    if variant_mode:
        # 基准显式占一格：没有基准的变体扫描读不出「改这一档是好是坏」。
        combos: list[variants.Variant | None] = [None, *catalogue]
        specs, skipped = variants.expand(
            config, config_name, selected, seeds, combos,
            stamp=stamp, timesteps=int(timesteps), unique=unique,
        )
        count = len(specs)
        choices, assign_notes = (
            gpus.build_choices(chosen_device, count, inventory, excluded=excluded)
            if count else ([], [])
        )
        planned = _planned_variants(config_name, config, specs, choices=choices)
        shape = (
            f"{len(selected)} 算法 × {len(catalogue) + 1} 档（含基准）× {len(seeds)} 种子"
        )
    else:
        count = len(selected) * len(seeds)
        choices, assign_notes = (
            gpus.build_choices(chosen_device, count, inventory, excluded=excluded)
            if count else ([], [])
        )
        planned = _planned_runs(
            config_name, config, selected, seeds, int(timesteps),
            unique=unique, stamp=stamp, choices=choices, overrides=overrides,
        ) if selected and seeds else []
        shape = f"{len(selected)} 算法 × {len(seeds)} 种子"

    with right:
        theme.section_head("这一批", f"{environment}")
        if not algorithms:
            theme.empty_state("还没选算法", "至少选一个算法。选多个才会形成对照。")
        elif not seeds:
            theme.empty_state("种子格式不对", "用空格或逗号分隔整数，例如 42 43 44。")
        elif variant_mode and not catalogue:
            theme.empty_state("表里还没有变体", "从变体集载入，或点上面的快捷预设，或直接在表里手填。")
        else:
            per_card: dict[str, list[str]] = {}
            for item, choice in zip(planned, choices):
                per_card.setdefault(gpus.short_label(choice), []).append(item["label"])
            card_count = len(per_card)
            steps = {item["timesteps"] for item in planned if item["timesteps"]}
            theme.stat_row([
                ("运行数", len(planned), shape),
                ("计算设备", f"{card_count} 张显卡分流" if card_count > 1 else next(iter(per_card), "CPU"),
                 "按剩余显存轮询" if card_count > 1 else "单设备"),
                ("每轮步数", f"{int(timesteps):,}" if not (variant_mode and len(steps) == 1)
                 else f"{int(next(iter(steps))):,}",
                 "各变体自带" if variant_mode and len(steps) > 1
                 else (preset if isinstance(preset, str) else "自定义")),
                ("时间戳", stamp.split("-")[-1], "同一批共用"),
            ])
            for note in assign_notes + inventory.notes:
                theme.caption(note)
            for note in variant_notes:
                theme.caption(f"⚠️ {note}")
            for note in skipped:
                theme.caption(f"⚠️ 已跳过：{note}")
            problems = [item["problem"] for item in planned if item["problem"]]
            for problem in dict.fromkeys(problems):
                st.error(problem)

            # 按卡分组：一张卡一行，列出它这一批要跑的算法，「哪张卡跑什么」一眼可见。
            st.markdown(
                theme.card("".join(
                    '<div class="rl-gpu-head" style="margin:.35rem 0">'
                    f'<span class="rl-gpu-name">{html.escape(card)}</span>'
                    f'<span class="rl-gpu-sub">{html.escape("、".join(names))}</span></div>'
                    for card, names in per_card.items()
                ) or '<div class="rl-gpu-proc">这一批没有可分配的任务。</div>'),
                unsafe_allow_html=True,
            )

            with st.expander("逐条命令行", expanded=False):
                preview = "\n\n".join(
                    f"# [{item['label']}] · 种子 {item['seed']} · 硬件 {item['device']}"
                    f"\n{shlex.join(item['argv'])}"
                    for item in planned
                )
                st.code(preview, language="bash")

    blocked = [item for item in planned if item["problem"]]
    ready = len(planned) > 0 and not blocked

    label = "开始这一批训练" if len(planned) > 1 else "开始训练"
    if st.button(label, type="primary", disabled=not ready, width="stretch"):
        launched = []
        for item in planned:
            choice = item["choice"]
            run_dir = paths.resolve_repo_path(item["directory"])
            job = jobs.launch(
                item["argv"],
                label=item["job_label"],
                kind="train",
                run_dir=run_dir,
                total_timesteps=item["timesteps"] or int(timesteps),
                device=item.get("device"),
                cuda_visible=choice.cuda_visible,
            )
            launched.append(job)
        st.session_state["jobs"] = {
            **st.session_state.get("jobs", {}),
            **{job.job_id: job for job in launched},
        }
        nav.goto("训练监控")
        # 用 toast 而不是 st.success：紧接着就 rerun 了，常规输出会被这次重跑丢掉，
        # 而 toast 是直接推给浏览器的，能活过重跑。
        st.toast(f"已启动 {len(launched)} 个训练，交给计算设备去跑了。", icon="🚀")
        st.rerun()
