"""变体实验：同一环境上跑同一算法的多个配置变体。

「哪个算法好」和「同一个算法对什么敏感」是两个不同的问题。前者靠一次
「算法 × 种子」的批量就能回答，后者需要**同一个算法跑多份配置**——学习率改一档、
网络加宽一层、给环境加上风。

这一层把「变体」变成一等公民：定义、展开、命名、校验都在这里，WebUI 与命令行
共用同一份实现。与 ``webui/gpus.py`` / ``webui/jobs.py`` 一样**不 import
Streamlit，也不 import torch**——这一层要能脱离界面单独验证。

## 变体是什么

一份变体 = 一个名字 + 若干条「点号路径覆盖」。路径里的 ``<ALGO>`` 是占位符，
展开时替换成该算法的 **profile 键**（去掉 ``SB3-`` 前缀），因为在配置里
``SB3-PPO`` 与自研 ``PPO`` 读的是同一份 profile::

    algorithm.profiles.<ALGO>.kwargs.learning_rate: 0.0001

这条覆盖对 ``PPO`` 展开成 ``algorithm.profiles.PPO...``，对 ``SB3-PPO`` 也展开成
``algorithm.profiles.PPO...``——这正是「同超参、只换实现」的对照所要求的。

值走 YAML 原生类型，所以 ``true`` / ``0.001`` / ``[64, 64]`` 都能表达（底层由
``jobs.yaml_scalar`` 序列化，``rl_common.cli`` 再 ``yaml.safe_load`` 回来）。

## 变体身份怎么传下去

不新增任何配置字段、不改 ``rl_common``：变体名写进 ``experiment.name``，
形如 ``lunarlander__lr-1e-3``（双下划线分隔）。这个字段本来就会被写进
``evaluation.json`` 与 ``resolved_config.yaml``，所以事后从运行目录里就能读回来——
``parse_experiment`` 是它的逆运算。同时目录名里也带上变体标识，避免同算法同种子的
两个变体写进同一个目录互相覆盖。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths

#: 变体定义的存放目录（``config/variants/<配置名>.yaml``）。
#:
#: 刻意放在 ``config/`` 的**子目录**里：``data.list_config_names()`` 用的是
#: ``config/*.yaml``，不会把变体集误当成一份实验配置。
VARIANT_ROOT = paths.CONFIG_ROOT / "variants"

#: 路径模板里的算法占位符。展开时替换成 profile 键。
ALGO_TOKEN = "<ALGO>"

#: 实验名的分隔符。配置名里不含它，所以反解是安全的。
EXPERIMENT_SEPARATOR = "__"

#: 变体标识进目录名时截断到多少字符。目录名要能在 ``ls`` 里一眼读完。
_TAG_LIMIT = 24

#: 变体不许覆盖的键——这些由批次自己的控件拥有，两边都写会让预览与实际不一致。
RESERVED_KEYS = frozenset({
    "algorithm.name",
    "experiment.seed",
    "output.directory",
    "output.timestamped",
})

#: 目录名里允许出现的字符，其余一律折成 ``-``。
_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: ``experiment.name`` 后缀的合法字符。**刻意比目录名更严**：实验名会经
#: ``--set`` 进 YAML，还会被 ``data.parse_experiment`` 按 ``__`` 反解，
#: 所以 ``__``、空白、引号一律拒掉，而不是靠事后 rpartition 猜。
_EXPERIMENT_SAFE = re.compile(r"[A-Za-z0-9._-]+")


def profile_key(name: str) -> str:
    """算法名 → 配置里的 profile 键名。

    与 ``algorithms.sb3_wrapper.profile_key`` 语义一致（那边是唯一权威），但这里
    单独实现一份：那个模块顶层 import stable_baselines3，为一个纯字符串变换付
    1.6 秒的导入代价不值得。``webui/tests/test_variants.py`` 里有用例钉住两者一致。
    """
    upper = str(name).upper()
    return upper[4:] if upper.startswith("SB3-") else upper


def name_problems(name: str) -> list[str]:
    """变体名本身的问题（与算法无关）。

    名字有两条硬约束，都在这里拦：

    - 不含 ``__``：那是 ``experiment.name`` 的分隔符，名字里再出现就反解不出来了。
    - 进 ``experiment.name`` 的形态必须只有 ``[A-Za-z0-9._-]``。中文变体名会被
      ``slug`` 折成一串 ``-``，于是 ``experiment.name`` 变成 ``lunarlander__---``，
      ``data.parse_experiment`` 读回来是个没意义的名字，排行榜上每一行都长得一样。
      与其事后猜，不如现在就要求用英文名。
    """
    problems: list[str] = []
    if not name or not name.strip():
        problems.append("变体名不能为空")
        return problems
    if EXPERIMENT_SEPARATOR in name:
        problems.append(f"变体名不能含 {EXPERIMENT_SEPARATOR!r}（它是实验名的分隔符）：{name!r}")
    if _SLUG_SAFE.search(name):
        problems.append(
            f"变体名 {name!r} 里有不能进实验名的字符（只允许字母、数字、``. _ -``）。"
            "空格、斜杠、中文都会被折成短横线，事后分不出是哪一行——请换个英文名。"
        )
    return problems


def slug(name: str) -> str:
    """把变体名变成可以进目录名的标识。

    变体名是人写的，可能带空格、斜杠、中文。目录名不能带 ``/``（会被当成路径分隔符），
    也不该带空格（命令行里要引号）。这里只保留 ``[A-Za-z0-9._-]``，其余折成一个
    ``-``，并截断到 ``_TAG_LIMIT``。

    中文变体名会被折成一串 ``-``——这是可接受的：变体名用英文是仓库既有约定
    （``experiment.name`` 与目录名本来就是英文），中文只出现在界面的说明文字里。
    """
    cleaned = _SLUG_SAFE.sub("-", str(name)).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned[:_TAG_LIMIT].strip("-") or "variant"


def experiment_name(config_name: str, variant: str | None) -> str:
    """变体对应的 ``experiment.name``。

    ``variant`` 为 ``None`` 时返回配置名（也就是「没有变体」的形态）。
    """
    if not variant:
        return config_name
    return f"{config_name}{EXPERIMENT_SEPARATOR}{slug(variant)}"


def parse_experiment(name: str | None) -> tuple[str | None, str | None]:
    """``experiment_name`` 的逆运算，返回 ``(配置名, 变体名)``。

    认不出来时返回 ``(None, None)`` 而不是瞎猜：历史运行里的 ``experiment.name``
    各式各样（有 ``lunarlander_baseline`` 这种下划线单写的），拿它当变体名会把
    基准误判成变体。**只认双下划线**，恰好切一刀。
    """
    if not name:
        return None, None
    parts = str(name).split(EXPERIMENT_SEPARATOR)
    if len(parts) != 2:
        return None, None
    config_part, variant_part = parts[0].strip(), parts[1].strip()
    if not config_part or not variant_part:
        return None, None
    return config_part, variant_part


# --------------------------------------------------------------------------- #
# 定义
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Variant:
    """一个具名的覆盖组合。值保持 YAML 原生类型。"""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        """进目录名与实验名的标识。"""
        return slug(self.name)

    def keys(self) -> list[str]:
        return sorted(self.overrides)

    def describe(self) -> str:
        """给界面用的一行说明，例如 ``lr 0.0003 → 0.001``。"""
        return "、".join(f"{key.split('.')[-1]}={value}" for key, value in self.overrides.items())


def load_variant_set(config_name: str, *, root: Path | None = None) -> dict[str, Variant]:
    """读 ``config/variants/<配置名>.yaml``；没有就返回空。

    文件结构::

        variants:
          lr-1e-3:
            algorithm.profiles.<ALGO>.kwargs.learning_rate: 0.001
          wind-strong:
            environment.kwargs.enable_wind: true
            environment.kwargs.wind_power: 15.0

    读不出来（文件不存在、YAML 坏了、结构不对）一律当作「没有变体集」，
    而不是抛异常——变体集是可选的东西，缺了不该让整页崩掉。
    """
    path = (root or VARIANT_ROOT) / f"{config_name}.yaml"
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("variants")
    if not isinstance(entries, dict):
        return {}

    variants: dict[str, Variant] = {}
    for name, overrides in entries.items():
        if not isinstance(overrides, dict) or not overrides:
            continue
        variants[str(name)] = Variant(name=str(name), overrides=dict(overrides))
    return variants


def dump_variant_set(variants: Iterable[Variant]) -> str:
    """把变体集序列化成 YAML 文本（界面上「下载这份变体集」用）。"""
    body = {"variants": {variant.name: dict(variant.overrides) for variant in variants}}
    return yaml.safe_dump(body, allow_unicode=True, sort_keys=False, default_flow_style=False)


def variant_from_lines(rows: Iterable[Mapping[str, Any]]) -> list[Variant]:
    """从界面上那张长表（``变体名 / 覆盖键 / 值`` 三列）还原出变体集。

    值从表格里取回来一定是字符串，这里按 YAML 语法解析成原生类型——与
    ``rl_common.cli`` 处理 ``--set`` 右值的方式一致，用户填 ``[64, 64]`` 会得到
    一个真正的列表而不是字符串。
    """
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("变体名") or row.get("name") or "").strip()
        key = str(row.get("覆盖键") or row.get("key") or "").strip()
        if not name or not key:
            continue
        raw = row.get("值", row.get("value"))
        try:
            value = yaml.safe_load(raw) if isinstance(raw, str) else raw
        except yaml.YAMLError:
            value = raw
        grouped.setdefault(name, {})[key] = value
    return [Variant(name=name, overrides=overrides) for name, overrides in grouped.items()]


# --------------------------------------------------------------------------- #
# 展开与校验
# --------------------------------------------------------------------------- #


def resolve(variant: Variant, algorithm: str) -> dict[str, Any]:
    """把 ``<ALGO>`` 换成该算法的 profile 键，返回可交给 ``--set`` 的覆盖表。

    展开后**必然**落在该算法自己的 profile 上（``<ALGO>`` 就是按这个算法的
    ``profile_key`` 换的），所以不存在「覆盖写到别的算法的 profile 段、被静默忽略」
    这种失败模式——自由写点号路径的写法才有那个坑。
    """
    key = profile_key(algorithm)
    return {
        path.replace(ALGO_TOKEN, key): value
        for path, value in variant.overrides.items()
    }


def _leaf(config: Mapping[str, Any], dotted: str) -> Any:
    """取点号路径指向的值；路径不存在时返回 ``_MISSING``。"""
    target: Any = config
    for key in dotted.split("."):
        if not isinstance(target, Mapping) or key not in target:
            return _MISSING
        target = target[key]
    return target


def coerce_value(raw: Any, target: Any, path: str) -> tuple[Any, str | None]:
    """按目标叶子的类型收敛取值，返回 ``(值, 警告)``。

    存在的理由是 YAML 1.1 的一个坑，实测::

        yaml.safe_load("1e-3")  -> "1e-3"    # 字符串！
        yaml.safe_load("0.001") -> 0.001     # float

    ``1e-3`` 是超参扫描里最顺手的写法，而它经 ``--set`` 一路下去会变成一个字符串，
    直到 SB3 构造优化器才炸——报错点离输入点十万八千里。这里按目标类型兜一层：
    目标是数值而拿到字符串时尝试转换，成功就静默修正并留一条说明，失败则保持原样
    交给 ``validate`` 去报错。
    """
    if isinstance(raw, str) and isinstance(target, (int, float)) and not isinstance(target, bool):
        try:
            converted = float(raw) if isinstance(target, float) else int(raw)
        except ValueError:
            return raw, None
        return converted, f"{path} 的值 {raw!r} 是字符串，已按目标类型转成 {converted!r}"
    return raw, None


def validate(variant: Variant, algorithm: str, config: Mapping[str, Any]) -> list[str]:
    """检查变体能不能用在某个算法上，返回人话问题清单（空列表 = 通过）。

    不做特例硬编码：一律拿展开后的真实路径去配置里查。这样配置改了什么键、
    哪个算法换了 profile，这里自动跟着变。

    唯一的特例是「算法根本没有 profile」——这时报「没有 profile」比报
    「``algorithm.profiles.REINFORCE.kwargs.learning_rate`` 不存在」有用得多：
    前者说清了「为什么」，后者只是一条路径不存在。
    """
    from .jobs import has_path  # noqa: PLC0415  —— 同包内延迟导入，避免循环

    problems: list[str] = []
    profiles = (config.get("algorithm").get("profiles") if isinstance(config.get("algorithm"), Mapping) else None) or {}
    key = profile_key(algorithm)
    has_profile = key in {str(name).upper() for name in profiles}

    # 与算法无关的名字问题放在最前，且不随算法变化——每个算法都会报一遍同样的错，
    # 调用方去重后只留一条。
    problems.extend(name_problems(variant.name))

    for path, value in resolve(variant, algorithm).items():
        if path in RESERVED_KEYS:
            problems.append(
                f"变体「{variant.name}」不能覆盖 {path}——这个键由批次自己管，"
                "写进去会让预览和实际跑的不一致"
            )
            continue
        if path.startswith("algorithm.profiles."):
            if not has_profile:
                problems.append(
                    f"{algorithm} 在这份配置里没有 profile，不能套用只改超参的变体 "
                    f"「{variant.name}」"
                )
                break
            # 覆盖必须落在**这个算法**的 profile 段上。写到别的算法的段上不会报错，
            # 但那次运行根本不会读它——静默 no-op 比报错危险得多。
            segment = path.split(".")[2] if len(path.split(".")) > 2 else ""
            if segment.upper() != key:
                problems.append(
                    f"变体「{variant.name}」的 {path} 属于 {segment} 的 profile，"
                    f"而这次跑的是 {algorithm}（读 {key}）——这个覆盖会被静默忽略"
                )
                continue
        if not has_path(dict(config), path):
            problems.append(f"配置里没有 {path}（变体「{variant.name}」）")
    return problems


def resolve_for(
    variant: Variant, algorithm: str, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """``resolve`` + 按目标叶子类型收敛取值 + **丢掉空覆盖**，返回 ``(覆盖表, 说明)``。

    调用方假定 ``validate`` 已经放行，所以这里不再判路径是否存在——``_leaf`` 拿不到
    时按原样返回，让问题在训练启动时以原来的方式暴露，而不是在这里被悄悄改掉。

    「空覆盖」这一条是扫描实验里很实际的一个坑：``lr-3e-4`` 对 PPO 而言正好是它的
    默认学习率（``0.0003``），覆盖写了等于没写。这种运行既浪费一张卡，又会在排行榜上
    和基准并排出现两行一模一样的数字，看起来像「复现失败」。这里直接把它摘掉，
    让 ``expand`` 的去重把整格并进基准，而不是靠事后人工剔除。
    """
    resolved = resolve(variant, algorithm)
    notes: list[str] = []
    redundant: list[str] = []
    for path, value in list(resolved.items()):
        target = _leaf(config, path)
        coerced, warning = coerce_value(value, target, path)
        if warning:
            notes.append(warning)
        if target is not _MISSING and coerced == target:
            redundant.append(path)
            continue
        resolved[path] = coerced
    for path in redundant:
        del resolved[path]
    if redundant:
        names = "、".join(path.split(".")[-1] for path in redundant)
        notes.append(f"变体「{variant.name}」的 {names} 与配置里的默认值相同，未产生实际覆盖")
    return resolved, notes


@dataclass(frozen=True)
class Spec:
    """一个待启动的运行的完整描述。

    这是 ``variants`` 层交给启动层的唯一交接类型，与 ``gpus.DeviceChoice`` 的分工
    类似：谁启动谁负责把字段翻译成 argv，这里不做子进程相关的事。
    """

    algorithm: str
    seed: int
    #: ``None`` 表示基准（不加任何覆盖）。
    variant: Variant | None
    #: 已在配置里校验过的覆盖表（``<ALGO>`` 已展开）。
    overrides: dict[str, Any] = field(default_factory=dict)
    #: ``training.total_timesteps`` 被特判到这里，走 ``--timesteps`` 而不是 ``--set``。
    timesteps: int | None = None
    #: 写进 ``experiment.name`` 的值；``None`` 表示不覆盖，用配置里的原值。
    experiment: str | None = None
    directory: str = ""

    @property
    def variant_name(self) -> str | None:
        return self.variant.name if self.variant else None

    @property
    def label(self) -> str:
        """界面与日志里的显示名。"""
        if not self.variant:
            return self.algorithm
        return f"{self.algorithm} · {self.variant.name}"


def _flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """摊平成点号路径 → 标量，用于「相对原配置改了什么」的浅层 diff。"""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if str(key).startswith("_"):
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def diff_against_base(
    resolved: Mapping[str, Any], base: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """比较运行目录里的生效配置与原始配置，返回 ``(键, 原值, 生效值)``。

    这是「这次运行到底改了什么」的唯一可靠来源：``resolved_config.yaml`` 是训练
    启动那一刻写下的，即使原配置后来被改过也留着当时的真相。
    """
    before, after = _flatten(base), _flatten(resolved)
    changed: list[tuple[str, Any, Any]] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key, _MISSING), after.get(key, _MISSING)
        if old != new:
            changed.append((key, old, new))
    return changed


class _Missing:
    """``None`` 是一个合法值，所以缺席需要单独的哨兵。"""

    def __repr__(self) -> str:  # pragma: no cover  —— 只在调试输出里出现
        return "(缺席)"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Missing)


_MISSING = _Missing()

#: ``diff_against_base`` 里表示「这个小节在某一侧整个不存在」的哨兵。公开出来是给
#: 调用方（详情页的 diff 表）判分支用：``None`` 不能担这个职责。
MISSING = _MISSING


def expand(
    config: Mapping[str, Any],
    config_name: str,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    variants: Sequence[Variant | None],
    *,
    stamp: str,
    timesteps: int | None = None,
    unique: bool = True,
) -> tuple[list[Spec], list[str]]:
    """展开「算法 × 变体 × 种子」成一组待启动的运行。

    序列里的 ``None`` 就是基准——**不是**「空列表等于只跑基准」。让基准在同一个序列里
    显式占一格，是为了让「基准与某个变体的覆盖恰好相同」这种情况也能被下面那道去重
    拦下来：手写的 lr 扫描里，某一个档位有相当概率正好等于配置里的默认值。

    返回 ``(待启动列表, 被跳过的组合说明)``。跳过的组合进不了返回的列表——
    与其让一个必然报错的子进程去把整批搞脏，不如在这里就拦下来并说清原因。
    """
    from . import jobs  # noqa: PLC0415  —— 同包内延迟导入，避免循环

    specs: list[Spec] = []
    skipped: list[str] = []
    seen: dict[tuple, str] = {}

    for variant in variants:
        for algorithm in algorithms:
            problems = validate(variant, algorithm, config) if variant else []
            if problems:
                skipped.extend(problems)
                continue
            if variant is None:
                overrides: dict[str, Any] = {}
            else:
                overrides, _ = resolve_for(variant, algorithm, config)
            # total_timesteps 从覆盖表里摘出来：它有一等公民参数（--timesteps），
            # 两个来源同时给会互相打架，界面上也显示成步数档位更直观。
            run_timesteps = timesteps
            if "training.total_timesteps" in overrides:
                try:
                    run_timesteps = int(overrides.pop("training.total_timesteps"))
                except (TypeError, ValueError):
                    skipped.append(
                        f"变体「{variant.name}」的 training.total_timesteps 不是整数，已跳过"
                    )
                    continue
            # 批内去重：名字不同但覆盖完全相同的变体（手写的 lr 扫描里某一个档位有相当
            # 概率正好等于配置里的默认值）会白烧一张卡。指纹里带步数，因为预算变体走的是
            # ``--timesteps`` 而不是 ``--set``，不带上会被误判成「没改任何东西」。
            fingerprint = (
                algorithm, run_timesteps,
                tuple(sorted((key, repr(value)) for key, value in overrides.items())),
            )
            if fingerprint in seen:
                skipped.append(
                    f"变体「{variant.name or '（基准）'}」在 {algorithm} 上的覆盖与"
                    f"「{seen[fingerprint]}」相同，已去重"
                )
                continue
            # 基准不在这一批里时（``--no-baseline``），上面那条去重没有对照物可比。
            # 这里补一条规则：覆盖被摘空、步数也没变，那它就是基准本身。不拦的话会跑出
            # 一批与已有基准逐字段相同的运行，排行榜上并排两行一样的数字。
            if (
                variant is not None and not overrides and run_timesteps == timesteps
                and fingerprint not in seen
            ):
                skipped.append(
                    f"变体「{variant.name}」在 {algorithm} 上没有产生任何实际改动，"
                    "等同于基准，已跳过"
                )
                continue
            seen[fingerprint] = variant.name if variant else "（基准）"
            for seed in seeds:
                specs.append(Spec(
                    algorithm=algorithm,
                    seed=int(seed),
                    variant=variant,
                    overrides=dict(overrides),
                    timesteps=run_timesteps,
                    experiment=experiment_name(config_name, variant.name) if variant else None,
                    directory=jobs.suggest_output_directory(
                        config_name, algorithm, int(seed),
                        unique=unique, stamp=stamp,
                        tag=variant.tag if variant else None,
                    ),
                ))
    return specs, list(dict.fromkeys(skipped))
