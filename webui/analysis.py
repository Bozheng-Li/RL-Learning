"""变体批次的敏感性分析：**同一个算法对哪个参数敏感**。

「哪个算法好」由基线批次回答；「同一个算法把 lr 换一档会怎样」由变体批次回答。这个
模块把 606 个变体运行折成可读的结论，同时把「这些结论有多可信」一并算出来——后者不
是附录，是正文：这套实验的噪声尺度与「两次运行能不能复现」决定了多小的 Δ 值得当回事。

## 噪声有两层，第二层是这一批特有的

**第一层：评估种子。** 训练完用 10 个固定回合评估，换一批初始条件，同一个模型的分
就会变。``evalseed-0`` / ``evalseed-5000`` 是**内建的安慰剂对照**：它们不改训练，
只改评估用的种子偏移，所以它们的 Δ 是这套分析流程的**真实零分布**——比「跨种子
SD」更贴近实际用到的统计量，因为它复刻了同样的配对结构（同批、同种子、跨不跨型号）。

**第二层：显卡型号。** 本机混插了 RTX 5880 Ada / 4090 D / 3090 三种卡。这批 606 个
运行留下了三条硬结论，都由 ``implementation_pairs`` 直接从数据里量出来：

1. 同一个配置、同一个种子，**同一张卡上跑两遍逐位相同**（另跑了一遍对照：5 张卡各跑
   2 遍共 10 次，只有 3 个不同值，且同一个值专属于同一型号的卡）。
2. **同一型号的两张不同物理卡也逐位相同**——``DQN`` / ``SAC`` / ``TD3`` 的孪生对里
   （这三个的 plain 名本来就解析到 SB3 实现，超参逐字段相同），同型号的 56 个配对
   **全部**逐位相同；跨型号的 106 个配对只有 17 个相同。
3. **换了型号就分叉**，而且分叉会被 RL 的混沌性放大：安慰剂 Δ 在同型号下 |Δ| 中位
   16.5 分，跨型号下 49.8 分、最大 1060 分。

所以噪声里混着一份**确定性的硬件偏置**：它不是随机的，事后无法从单个运行里剔除，
而变体批里 274/474 的配对天生就是跨型号的。对策不是假装它不存在，而是：

- ``noise_scale`` 估的是**含跨型号偏置在内的总噪声**，当判据的分母才对跨型号的配对不冤；
- 每张表并排给一个 ``Δ均值（同型号）``——那是可复现子集算出来的干净数，代价是种子少；
- ``Pair.same_model`` / ``Effect.cross_model`` 把这件事带进每一行；
- ``twin_null`` 给出**形状一致的零分布**：配置逐字段相同、只换卡时，3 种子均值 |Δ|
  中位 24.4、P90 95.6。这就是「多小的 Δ 不能当回事」的直接答案。

## 判据

对每个 ``(变体, 算法)``：3 个种子的 Δ 取均值，除以 ``σ_a / √3`` 得到 ``z``，
``|z| ≥ 2`` 才算「分得出来」。``σ_a`` 由该算法自己的 6 个安慰剂 Δ 估出，所以每个
算法有各自的尺度——REINFORCE 的回报能到 -1000 量级，噪声本来就比 DQN 大一个数量级，
用一把尺子量所有算法是错的。

但**不要只看 z**。``twin_null`` 说明「配置完全没变」也能凑出中位 24 分的 3 种子均值
Δ，所以单个 ``(变体, 算法)`` 的显著只是个线索。真正有力的是**跨算法的一致性**：
一个变体如果在 11 个算法里有 10 个都往下走，那它「有害」这件事比任何单个 z 都硬。
``VariantSummary`` 因此同时给方向一致性、显著个数与符号检验的 p 值。
"""

from __future__ import annotations

import math
import statistics as st
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import data, jobs

#: 判显著的门槛。正态近似下 2 对应双侧 ~5%，与「3 个种子」的分辨力相称——再严就会
#: 一个都判不出来，再松就会把安慰剂自己判成效应（实测安慰剂里有 15% 的单个 Δ 超过 100）。
Z_THRESHOLD = 2.0

#: 变体名前缀 → 维度。页面按维度分组，读起来才知道这一串名字是同一类改动。
#:
#: 用前缀而不是变体集文件：分析要能跑在**已经落盘的结果**上，而结果里只有
#: ``experiment.name`` 带出来的变体名。变体集文件改了、删了，历史结果仍然要能分析。
DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("学习超参", ("lr-", "ent-", "gamma-", "net-")),
    ("环境扰动", ("wind-", "gravity-")),
    ("训练预算", ("budget-",)),
    ("评估种子", ("evalseed-",)),
)

#: 预算与评估种子这两类**不能**与 200k 基准的 Δ 一起读：
#:
#: - ``budget-*`` 改的是训练步数，它回答「多少步够用」，与「同预算下换个超参」是
#:   两个问题。把它并进敏感性表里，会把「训练不足」误读成「这个参数有害」。
#: - ``evalseed-*`` 本来就不该有训练效应，它是安慰剂。
_SIDE_QUEST_PREFIXES = ("budget-", "evalseed-")

#: 其中有哪一类连「配对」都不该出现在 Δ 表里。
#:
#: 只有预算：它的 Δ 与基准比没有意义（基准是 200k 步，它自己不是）。
#: ``evalseed-*`` **必须**留在配对里——``noise_scale`` 的 σ 正是从它的 Δ 估出来的，
#: 剔掉它整套判据就没有分母了。所以这两类的处理刻意不同。
_NO_DELTA_PREFIXES = ("budget-",)


def dimension_of(variant: str | None) -> str:
    """变体名 → 维度名；认不出归到「其它」。"""
    if not variant:
        return "基准"
    for name, prefixes in DIMENSIONS:
        if variant.startswith(prefixes):
            return name
    return "其它"


def is_side_quest(variant: str | None) -> bool:
    """是否属于「不能当作敏感性 Δ 读」的那两类（预算 / 评估种子）。"""
    return bool(variant) and str(variant).startswith(_SIDE_QUEST_PREFIXES)


def has_no_delta(variant: str | None) -> bool:
    """是否**连配对都不该做**（只有预算这一类）。

    ``evalseed-*`` 不在这里：它虽然也是 side quest，但它的 Δ 正是 σ 的来源，
    必须留在 ``pair_table`` 的输出里。见 ``_NO_DELTA_PREFIXES``。
    """
    return bool(variant) and str(variant).startswith(_NO_DELTA_PREFIXES)


# --------------------------------------------------------------------------- #
# 批次与配对
# --------------------------------------------------------------------------- #


def batch_stamp(info: data.RunInfo) -> str:
    """从目录名尾部取出批次时间戳（``..._20260922-233256``）。

    批次是「同一次 ``variants.py`` 调用产生的运行」——同时也是「同一套显卡、同一段
    时间、同一份代码」的代名词。跨批比较会把批次差异读成实验效应，所以要能识别它。
    """
    return info.name.rsplit("_", 1)[-1]


def latest_batch(runs: Sequence[data.RunInfo]) -> str | None:
    """运行最多的那个批次。变体批次一次 606 个，基线批次 33 个，取多的那个。"""
    counts: dict[str, int] = {}
    for info in runs:
        counts[batch_stamp(info)] = counts.get(batch_stamp(info), 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda key: counts[key])


def select_batch(
    runs: Sequence[data.RunInfo], stamp: str | None = None
) -> tuple[list[data.RunInfo], str | None]:
    """挑出一个批次的运行。``stamp=None`` 时选最大的那一批。"""
    stamp = stamp or latest_batch(runs)
    if stamp is None:
        return [], None
    return [info for info in runs if batch_stamp(info) == stamp], stamp


def device_of(runs: Sequence[data.RunInfo]) -> dict[str, str]:
    """运行名 → ``CUDA_VISIBLE_DEVICES`` 的值（卡 UUID）。

    只有任务注册表记得「这一次跑在哪张卡上」——``evaluation.json`` 与
    ``resolved_config.yaml`` 里都写的是 ``cuda:0``，因为卡片是用 ``CUDA_VISIBLE_DEVICES``
    隔离的，子进程只看得到一张卡。所以复现性分析必须查注册表。

    注册表里没有的运行（手工命令行发起、没经过 ``webui.jobs.launch``）不给条目，
    调用方按「未知」处理，**不要**猜成 cpu 或 None 参与比较。
    """
    wanted = {info.name for info in runs}
    found: dict[str, str] = {}
    for job in jobs.load_registry():
        if not job.run_dir or not job.cuda_visible:
            continue
        name = Path(job.run_dir).name
        if name in wanted:
            found[name] = job.cuda_visible
    return found


def card_model_of(runs: Sequence[data.RunInfo]) -> dict[str, str]:
    """运行名 → 显卡型号（如 ``GeForce RTX 3090``）。

    注册表里 ``device`` 字段的形态是 ``"GPU 3: GeForce RTX 3090"``，型号取冒号之后。
    型号是**比物理卡更粗**的分组：同一型号的两张卡会挑同一批 kernel，
    跨型号就不会——见 ``implementation_pairs``。
    """
    wanted = {info.name for info in runs}
    found: dict[str, str] = {}
    for job in jobs.load_registry():
        if not job.run_dir:
            continue
        name = Path(job.run_dir).name
        if name in wanted and job.device:
            found[name] = job.device.split(": ", 1)[-1].strip()
    return found


@dataclass(frozen=True)
class Pair:
    """一次「变体运行 - 同算法同种子的基准运行」的配对。"""

    algorithm: str
    variant: str
    seed: int
    baseline: float
    value: float
    #: 变体运行是否与基准运行落在**同一张物理卡**上。跨卡时这个 Δ 里混着显卡型号
    #: 带来的分叉，不是纯粹的变体效应。
    same_device: bool | None = None
    #: 两侧的显卡型号。跨型号才会分叉——同一型号的两张卡实测逐位相同。
    value_card: str | None = None
    baseline_card: str | None = None

    @property
    def delta(self) -> float:
        return self.value - self.baseline

    @property
    def same_model(self) -> bool | None:
        """两侧是不是**同一型号**的卡。这是比 ``same_device`` 更该看的那个判据。

        实测（``implementation_pairs``）：同一型号的两张不同物理卡上跑同一个配置，
        结果是**逐位相同**的；换了型号才分叉。所以「跨卡」本身不制造噪声，
        「跨型号」才制造。
        """
        if self.value_card is None or self.baseline_card is None:
            return None
        return self.value_card == self.baseline_card


def pair_table(
    runs: Sequence[data.RunInfo], *, stamp: str | None = None
) -> tuple[list[Pair], str | None]:
    """把变体运行与同算法、同种子、同批次的基准运行配对。

    同一批内配对是必须的：跨批的同种子运行**不是**可复现的（实测 33 对里 19 对不
    相等，且配置逐字节相同——差异来自当时落到了不同型号的卡上）。拿另一批的基准
    当对照，等于把批次差异算进变体效应。

    预算变体（``budget-*``）**不参与配对**：它与 200k 基准的差回答的是「多少步够用」，
    会污染敏感性表（见 ``has_no_delta``）。评估种子（``evalseed-*``）则必须留着——
    σ 就是从它的 Δ 估出来的。
    """
    batch, stamp = select_batch(runs, stamp)
    baseline: dict[tuple[str, int], data.RunInfo] = {}
    variants: list[data.RunInfo] = []
    for info in batch:
        if info.summary.get("mean_reward") is None or info.seed is None:
            continue
        if info.variant:
            if not has_no_delta(info.variant):
                variants.append(info)
        elif info.algorithm:
            baseline[(info.algorithm, info.seed)] = info

    cards = device_of(batch)
    models = card_model_of(batch)
    pairs: list[Pair] = []
    for info in variants:
        control = baseline.get((info.algorithm, info.seed))
        if control is None:
            continue
        same: bool | None = None
        if info.name in cards and control.name in cards:
            same = cards[info.name] == cards[control.name]
        pairs.append(Pair(
            algorithm=info.algorithm or "?",
            variant=info.variant or "",
            seed=info.seed,
            baseline=float(control.summary["mean_reward"]),
            value=float(info.summary["mean_reward"]),
            same_device=same,
            value_card=models.get(info.name),
            baseline_card=models.get(control.name),
        ))
    return pairs, stamp


# --------------------------------------------------------------------------- #
# 噪声尺度
# --------------------------------------------------------------------------- #

#: 安慰剂变体：只换评估种子，不改训练。它们的 Δ 按定义全是噪声。
PLACEBO_VARIANTS = ("evalseed-0", "evalseed-5000")

#: 「同名但不同前缀」的孪生实现：``plain`` 名在本项目里回退到 ``SB3-*``，所以这两列
#: 是**同一个实现的两次运行**（超参逐字段相同）。它们之间的差是一份免费的可复现性探针。
TWIN_FAMILIES: tuple[tuple[str, str], ...] = (
    ("A2C", "SB3-A2C"), ("PPO", "SB3-PPO"),
    ("DQN", "SB3-DQN"), ("SAC", "SB3-SAC"), ("TD3", "SB3-TD3"),
)

#: 上面这些族里，哪几个才是**真的同一个实现**。
#:
#: ``SAC`` / ``TD3`` / ``DQN`` 在本项目里没有自研版本（``NATIVE_ALGORITHMS`` 只有
#: ``A2C`` / ``PPO`` / ``REINFORCE``），所以 plain 名回退到 SB3，与 ``SB3-*`` 逐字段同
#: 配置、同代码。剩下的 ``A2C`` / ``PPO`` 是**两份不同的代码**，它们之间的差是实现
#: 差异，不是噪声——任何「零分布」性质的统计都**必须**把这两族剔掉，否则量出来的
#: 是「两个算法差多少」，会把零分布抬得虚高。
TRUE_TWIN_ALGORITHMS = frozenset({"DQN", "SAC", "TD3"})


def placebo_deltas(pairs: Iterable[Pair]) -> dict[str, list[float]]:
    """每个算法的安慰剂 Δ 列表。"""
    out: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        if pair.variant in PLACEBO_VARIANTS:
            out[pair.algorithm].append(pair.delta)
    return dict(out)


def noise_scale(pairs: Iterable[Pair], *, same_model_only: bool = False) -> dict[str, float]:
    """每个算法的噪声尺度 σ_a（单次 Δ 的标准差），判 z 的分母。

    用安慰剂（``evalseed-*``）估：它们不改训练，所以 Δ 里**只有**噪声，而且复刻了
    变体运行与基准运行的实际配对结构（同批、同种子、跨不跨型号的卡）。

    ``same_model_only=True`` 时只取同型号的配对，量的是「真正的重复实验抖动」——
    实测这一档接近 0（同型号的配对逐位相同）。默认取全部，量的是**含跨型号偏置在内的
    总噪声**，这才是判断变体效应时该用的尺子：变体批里 60% 的配对天生就是跨型号的，
    拿同型号的窄尺度去判它们，会把硬件偏置判成效应。

    这里刻意**不做**按算法细分的修正之外的任何调整：样本只有 6 个（2 个安慰剂变体 ×
    3 个种子），σ 本身带估计误差，同族的两个算法能差两三倍（实测 PPO 59.9 vs
    SB3-PPO 169.3）。所以它是个**量级尺子**，不是置信区间。``twin_null`` 用完全不同的
    一条路径（配置逐字段相同、只换卡）给出同一个量级的交叉验证，页面两个都摆出来。
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        if pair.variant not in PLACEBO_VARIANTS:
            continue
        if same_model_only and pair.same_model is not True:
            continue
        grouped[pair.algorithm].append(pair.delta)

    out: dict[str, float] = {}
    for algorithm, deltas in grouped.items():
        if len(deltas) > 1:
            out[algorithm] = st.stdev(deltas)
    return out


# --------------------------------------------------------------------------- #
# 效应量
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Effect:
    """一个 ``(变体, 算法)`` 的效应量。"""

    variant: str
    algorithm: str
    n: int
    mean_delta: float
    sd_delta: float | None
    z: float | None
    deltas: tuple[float, ...] = ()
    #: 这些配对里有几个是跨**型号**的卡。跨型号的 Δ 里混着硬件偏置，个数越多，
    #: 这个效应越不该被当成纯粹的参数效应读。
    cross_model: int = 0
    #: 只保留同型号配对的 Δ。同型号的配对是干净的——硬件偏置不在里面。
    clean_deltas: tuple[float, ...] = ()

    @property
    def significant(self) -> bool:
        return self.z is not None and abs(self.z) >= Z_THRESHOLD

    @property
    def verdict(self) -> str:
        """``有利`` / ``有害`` / ``—``（分不出来）。"""
        if not self.significant:
            return "—"
        return "有利" if self.mean_delta > 0 else "有害"

    @property
    def mixed_cards(self) -> bool:
        """全部配对都跨了型号——这个 Δ 的绝对值不可信，只有方向还能看。"""
        return self.n > 0 and self.cross_model == self.n

    @property
    def clean_delta(self) -> float | None:
        """只用同型号配对算的 Δ。同型号的可复现性已经在孪生对照里钉死过了。"""
        if not self.clean_deltas:
            return None
        return st.mean(self.clean_deltas)


def effect_of(
    pairs: Iterable[Pair], variant: str, algorithm: str, scales: Mapping[str, float]
) -> Effect | None:
    """算一个 ``(变体, 算法)`` 的效应量。没有配对时返回 ``None``。"""
    chosen = [
        pair for pair in pairs
        if pair.variant == variant and pair.algorithm == algorithm
    ]
    if not chosen:
        return None
    deltas = [pair.delta for pair in chosen]
    clean = [pair.delta for pair in chosen if pair.same_model is True]
    mean = st.mean(deltas)
    sd = st.stdev(deltas) if len(deltas) > 1 else None
    scale = scales.get(algorithm)
    z = mean / (scale / math.sqrt(len(deltas))) if scale and len(deltas) else None
    return Effect(
        variant=variant,
        algorithm=algorithm,
        n=len(deltas),
        mean_delta=mean,
        sd_delta=sd,
        z=z,
        deltas=tuple(deltas),
        cross_model=sum(1 for pair in chosen if pair.same_model is False),
        clean_deltas=tuple(clean),
    )


def effects(pairs: Sequence[Pair], scales: Mapping[str, float]) -> list[Effect]:
    """全部 ``(变体, 算法)`` 的效应量，按变体名、算法名排。"""
    keys = sorted({(pair.variant, pair.algorithm) for pair in pairs})
    out = []
    for variant, algorithm in keys:
        found = effect_of(pairs, variant, algorithm, scales)
        if found is not None:
            out.append(found)
    return out


def _sign_p_value(wins: int, total: int) -> float:
    """双侧符号检验的精确 p 值（H0: 正负各半）。"""
    if total <= 0:
        return 1.0
    extreme = max(wins, total - wins)
    tail = sum(math.comb(total, k) for k in range(extreme, total + 1))
    return min(1.0, 2.0 * tail / (2 ** total))


@dataclass
class VariantSummary:
    """一个变体在全部算法上的总账。"""

    variant: str
    dimension: str
    algorithms: int = 0
    #: Δ 为正 / 为负的算法个数。
    positive: int = 0
    negative: int = 0
    significant: int = 0
    favorable: int = 0
    harmful: int = 0
    mean_delta: float = 0.0
    #: 符号检验 p 值：方向一致性有多难靠掷硬币凑出来。
    sign_p: float = 1.0
    effects: list[Effect] = field(default_factory=list)

    @property
    def direction(self) -> str:
        if self.positive == self.negative:
            return "无一致方向"
        return "偏有利" if self.positive > self.negative else "偏有害"

    @property
    def consensus(self) -> float:
        """方向一致的比例（多数方向 / 总数）。"""
        if not self.algorithms:
            return 0.0
        return max(self.positive, self.negative) / self.algorithms


def summarize_variants(
    pairs: Sequence[Pair], scales: Mapping[str, float]
) -> list[VariantSummary]:
    """按变体汇总：方向一致性 + 显著个数 + 平均 Δ。"""
    grouped: dict[str, list[Effect]] = defaultdict(list)
    for effect in effects(pairs, scales):
        grouped[effect.variant].append(effect)

    out: list[VariantSummary] = []
    for variant, items in grouped.items():
        summary = VariantSummary(
            variant=variant,
            dimension=dimension_of(variant),
            algorithms=len(items),
            positive=sum(1 for item in items if item.mean_delta > 0),
            negative=sum(1 for item in items if item.mean_delta < 0),
            significant=sum(1 for item in items if item.significant),
            favorable=sum(1 for item in items if item.verdict == "有利"),
            harmful=sum(1 for item in items if item.verdict == "有害"),
            mean_delta=st.mean([item.mean_delta for item in items]),
            effects=sorted(items, key=lambda item: item.mean_delta, reverse=True),
        )
        summary.sign_p = _sign_p_value(summary.positive, summary.algorithms)
        out.append(summary)
    # 排得「最像真效应」的在前：先按方向一致性，再按显著个数，最后按平均 Δ 的绝对值。
    out.sort(key=lambda item: (item.consensus, item.significant, abs(item.mean_delta)),
             reverse=True)
    return out


def sensitivity_table(pairs: Sequence[Pair], scales: Mapping[str, float]) -> list[dict[str, Any]]:
    """页面表格用的扁平行：一个 ``(变体, 算法)`` 一行。

    ``Δ均值（同型号）`` 是只用同型号配对算的 Δ——那些配对是可复现的，所以这个数干净，
    但往往只有一两个种子。``Δ均值`` 用了全部种子，代价是混进了跨型号的硬件偏置。
    两个并排放，「数值」与「可信度」的取舍交给读者，而不是替他选一个。
    """
    rows = []
    for effect in effects(pairs, scales):
        rows.append({
            "变体": effect.variant,
            "维度": dimension_of(effect.variant),
            "算法": effect.algorithm,
            "种子数": effect.n,
            "Δ均值": effect.mean_delta,
            "Δ均值（同型号）": effect.clean_delta,
            "同型号配对数": len(effect.clean_deltas),
            "Δ最小": min(effect.deltas),
            "Δ最大": max(effect.deltas),
            "Δ波动": effect.sd_delta,
            "z": effect.z,
            "判定": effect.verdict,
            "跨型号配对数": effect.cross_model,
        })
    return rows


# --------------------------------------------------------------------------- #
# 预算曲线
# --------------------------------------------------------------------------- #


def budget_curves(runs: Sequence[data.RunInfo]) -> list[dict[str, Any]]:
    """训练预算变体的曲线：每个算法在各步数档的最终回报。

    预算变体**不与 200k 基准算 Δ**——那是两个问题。这里把步数与回报并排放，
    回答的是「多少步够用」，以及「哪些算法 25k 步就已经成型、哪些还完全没学会」。
    """
    rows: list[dict[str, Any]] = []
    bucket: dict[tuple[str, int], list[float]] = defaultdict(list)
    for info in runs:
        variant = info.variant or ""
        if not variant.startswith("budget-"):
            continue
        reward = info.summary.get("mean_reward")
        steps = info.total_timesteps
        if reward is None or not steps:
            continue
        bucket[(info.algorithm or "?", int(steps))].append(float(reward))

    for (algorithm, steps), values in bucket.items():
        rows.append({
            "算法": algorithm,
            "步数": steps,
            "均值回报": st.mean(values),
            "种子数": len(values),
        })
    rows.sort(key=lambda row: (row["算法"], row["步数"]))
    return rows


def budget_pivot(runs: Sequence[data.RunInfo]) -> dict[str, dict[int, float]]:
    """``算法 -> {步数: 均值回报}``，给表格与曲线用。基准的 200k 也并进来当最后一档。"""
    pivot: dict[str, dict[int, float]] = defaultdict(dict)
    for row in budget_curves(runs):
        pivot[row["算法"]][row["步数"]] = row["均值回报"]

    batch, _ = select_batch(runs)
    baseline: dict[str, list[float]] = defaultdict(list)
    for info in batch:
        if info.variant is None and info.summary.get("mean_reward") is not None:
            steps = info.total_timesteps
            if steps:
                baseline[info.algorithm or "?"].append(
                    (int(steps), float(info.summary["mean_reward"]))  # type: ignore[arg-type]
                )
    for algorithm, items in baseline.items():
        by_steps: dict[int, list[float]] = defaultdict(list)
        for steps, reward in items:
            by_steps[steps].append(reward)
        for steps, values in by_steps.items():
            pivot[algorithm][steps] = st.mean(values)
    return dict(pivot)


# --------------------------------------------------------------------------- #
# 可复现性
# --------------------------------------------------------------------------- #


def implementation_pairs(runs: Sequence[data.RunInfo]) -> list[dict[str, Any]]:
    """`plain` 名与 `SB3-` 名之间的「孪生对照」——同时是免费的跨卡可复现性探针。

    ``SAC`` / ``TD3`` / ``DQN`` 在本项目里**不是自研的**（``NATIVE_ALGORITHMS`` 只有
    ``A2C`` / ``PPO`` / ``REINFORCE``），``resolve_name`` 会回退到 ``SB3-*``。所以
    ``SAC`` 与 ``SB3-SAC`` 是**同一个实现的两次运行**，超参也逐字段相同。它们
    「逐位相同」还是「不同」不反映实现差异，只反映跑在了哪张（型号的）卡上。

    实测结论（本批 285 个配对）：**同一型号的卡 → 逐位相同；不同型号 → 分叉。**
    三种卡混插时（5880 Ada / 4090 D / 3090），这解释了跨批「同配置同种子结果不同」
    的全部 19 个案例。也顺便说明 ``A2C`` / ``PPO`` 的孪生对照为什么全是 0：
    它们是**两个不同的实现**（自研 vs SB3），本来就该不同——那两行不是反例。

    分桶按**型号**，不按物理卡：同型号的两张不同卡也逐位相同，按物理卡分桶会把
    「可复现」的一大批错记成「跨卡」。查不到型号的运行（没进注册表）单独计在
    ``型号未知``，不能默认当成同一侧——那会把未知混进结论里。
    """
    batch, _ = select_batch(runs)
    models = card_model_of(batch)
    index: dict[tuple[str, str, int], data.RunInfo] = {}
    for info in batch:
        if info.summary.get("mean_reward") is None or info.seed is None:
            continue
        index[(info.algorithm or "?", info.variant or "", info.seed)] = info

    rows: list[dict[str, Any]] = []
    for native, sb3 in TWIN_FAMILIES:
        same_model_total = same_model_identical = 0
        cross_total = cross_identical = 0
        unknown_total = 0
        for info in batch:
            if (info.algorithm or "?") != native or info.seed is None:
                continue
            twin = index.get((sb3, info.variant or "", info.seed))
            if twin is None:
                continue
            identical = info.summary["mean_reward"] == twin.summary["mean_reward"]
            model = models.get(info.name)
            twin_model = models.get(twin.name)
            if model is None or twin_model is None:
                unknown_total += 1
            elif model == twin_model:
                same_model_total += 1
                same_model_identical += int(identical)
            else:
                cross_total += 1
                cross_identical += int(identical)
        # 「孪生」只有在 plain 名确实回退成 SB3 实现时才成立。``A2C`` / ``PPO`` 有自研
        # 实现，所以那几个 plain 名跑的是**另一份代码**，它们那一行量的是实现差异，
        # 不是可复现性——必须标出来，否则会被读成「跨卡就是不可复现」的反例。
        is_twin = native not in {"A2C", "PPO"}
        rows.append({
            "算法": native,
            "对照实现": sb3,
            "同名同实现": is_twin,
            "同型号配对": same_model_total,
            "同型号逐位相同": same_model_identical,
            "跨型号配对": cross_total,
            "跨型号逐位相同": cross_identical,
            "型号未知": unknown_total,
            "结论": (
                f"同型号 {same_model_identical}/{same_model_total} 逐位相同；"
                f"跨型号 {cross_identical}/{cross_total}"
                if cross_total else f"同型号 {same_model_identical}/{same_model_total} 逐位相同"
            ),
        })
    return rows


def twin_null(runs: Sequence[data.RunInfo]) -> dict[str, Any]:
    """**零分布**：配置逐字段相同时，3 种子均值 Δ 能有多大。

    这是整套分析里最直接的一条基准线，而且比安慰剂更好：

    - 安慰剂（``evalseed-*``）换了评估种子，但**没换训练**——它量的是「评估那 10 个
      回合的运气」。而孪生对（``SAC`` vs ``SB3-SAC`` 这类）训练配置、种子、代码全同，
      唯一的变量是落在哪张（型号的）卡上——它量的正是变体批里实际存在的那份扰动。
    - 统计量的**形状完全一致**：变体表里看的是「3 个种子的 Δ 取均值」，这里也是。
      安慰剂只有 11 个算法级样本，这里一个一个 (算法, 变体) 组合都能出，n 大得多。

    实测（本批）：54 个真孪生组（``DQN`` / ``SAC`` / ``TD3``）的 3 种子均值 |Δ|
    中位 24.4、P90 95.6、最大 5713.7；对照的安慰剂 33 个同型号样本给出中位 16.5
    ——两条独立路径给出同一个量级，所以「3 种子均值 |Δ| 要明显超过 ~50 才值得当回事」
    这个判断有两个来源支撑。

    **同型号的子集全是 0**（56 组，逐位相同）。所以这条零分布之所以不为零，
    来源就是跨型号——也就是这份扰动是**可以靠调度的同质性消掉的**，不是必须接受的。

    ``A2C`` / ``PPO`` 必须剔出去：那两个 plain 名**不是** SB3 实现的别名（有自研版），
    它们与 ``SB3-*`` 的差是实现差异。算进来会把零分布从 24.4 抬到 33.0——虚高。
    """
    batch, _ = select_batch(runs)
    index: dict[tuple[str, str, int], data.RunInfo] = {}
    for info in batch:
        if info.summary.get("mean_reward") is None or info.seed is None:
            continue
        index[(info.algorithm or "?", info.variant or "", info.seed)] = info

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (algorithm, variant, seed), info in index.items():
        if algorithm not in TRUE_TWIN_ALGORITHMS:
            continue
        twin = index.get((f"SB3-{algorithm}", variant, seed))
        if twin is None:
            continue
        grouped[(algorithm, variant)].append(
            float(twin.summary["mean_reward"]) - float(info.summary["mean_reward"])
        )

    means = [st.mean(values) for values in grouped.values()]
    if not means:
        return {"n": 0}
    ordered = sorted(abs(value) for value in means)

    def percentile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "n": len(means),
        "median": st.median(ordered),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "share_over_50": sum(1 for value in ordered if value > 50) / len(ordered),
        "share_over_100": sum(1 for value in ordered if value > 100) / len(ordered),
        "share_over_200": sum(1 for value in ordered if value > 200) / len(ordered),
    }


def reproducibility(pairs: Sequence[Pair]) -> dict[str, Any]:
    """噪声结构：同型号 vs 跨型号的 Δ，以及**安慰剂**在两侧的尺度。

    这个函数是整套分析的前提说明。实测已经钉死：**同一个配置在同一个型号的卡上是
    可复现的（逐位相同），换一个型号的卡就会分叉。** 变体运行与基准运行落在不同
    型号的卡上时，Δ 里就混进了这份分叉——它不是随机的，是确定的硬件偏置，事后无法
    从单个运行里剔除。

    所以按下卡型号分成两桶统计。**但两桶都只统计安慰剂配对**（``evalseed-*``）：
    全部配对里混着真实效应，拿它当「噪声底线」会把 lr-1e-3 这类真差异算进抖动里，
    底线被抬高一倍还多。真实效应的分桶在 ``sensitivity_table`` 里逐行给。

    ``cross_model_share`` 是**全部**配对里跨型号的比例——它说明这份偏置影响面有多大，
    是「为什么不能只报同型号子集」的依据，不是一个噪声数字。
    """
    placebo_pairs = [pair for pair in pairs if pair.variant in PLACEBO_VARIANTS]
    same_placebo = [abs(pair.delta) for pair in placebo_pairs if pair.same_model is True]
    cross_placebo = [abs(pair.delta) for pair in placebo_pairs if pair.same_model is False]
    unknown = [pair for pair in placebo_pairs if pair.same_model is None]
    cross_all = sum(1 for pair in pairs if pair.same_model is False)

    def describe(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        def percentile(q: float) -> float:
            return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        return {
            "n": len(values),
            "median": st.median(values),
            "p90": percentile(0.90),
            "max": ordered[-1],
        }

    return {
        "placebo_same_model": describe(same_placebo),
        "placebo_cross_model": describe(cross_placebo),
        "placebo": describe([abs(pair.delta) for pair in placebo_pairs]),
        "unknown_model": len(unknown),
        "pairs": len(pairs),
        "cross_model_pairs": cross_all,
        "cross_model_share": cross_all / len(pairs) if pairs else 0.0,
    }


def seed_spread(runs: Sequence[data.RunInfo]) -> list[dict[str, Any]]:
    """每个算法基准运行的跨种子离散度——「不换任何参数，只换种子」的尺度。"""
    batch, _ = select_batch(runs)
    grouped: dict[str, list[float]] = defaultdict(list)
    for info in batch:
        if info.variant is None and info.seed is not None:
            reward = info.summary.get("mean_reward")
            if reward is not None:
                grouped[info.algorithm or "?"].append(float(reward))
    rows = []
    for algorithm, values in sorted(grouped.items()):
        rows.append({
            "算法": algorithm,
            "种子数": len(values),
            "均值": st.mean(values),
            "标准差": st.stdev(values) if len(values) > 1 else None,
            "最低": min(values),
            "最高": max(values),
        })
    return rows


# --------------------------------------------------------------------------- #
# 一页拿全
# --------------------------------------------------------------------------- #


def available_variants(runs: Sequence[data.RunInfo]) -> list[str]:
    """这批运行涉及的配置里，变体集文件定义了哪些变体。

    用来回答「还有哪些变体没跑」。取配置名而不是让调用方传：一页里可能有多个环境的
    运行，各自有自己的变体集文件。

    读不出来（文件不存在、YAML 坏）就当那个配置没有变体集——与分析路径一致，
    变体分析必须能在**只有结果、没有定义**的情况下跑起来。
    """
    from . import variants as variants_module  # noqa: PLC0415  —— 同包延迟导入，避免循环

    names: list[str] = []
    for config_name in sorted({info.config_name for info in runs if info.config_name}):
        try:
            catalog = variants_module.load_variant_set(config_name)
        except (OSError, ValueError):
            continue
        for name in catalog:
            if name not in names:
                names.append(name)
    return names


def analyse(
    runs: Sequence[data.RunInfo], *, stamp: str | None = None
) -> dict[str, Any]:
    """分析一次算全，供页面直接取用。"""
    pairs, used = pair_table(runs, stamp=stamp)
    scales = noise_scale(pairs)
    batch, _ = select_batch(runs, stamp)
    return {
        "stamp": used,
        "runs": len(batch),
        "pairs": pairs,
        "scales": scales,
        "summaries": summarize_variants(pairs, scales),
        "table": sensitivity_table(pairs, scales),
        "budget": budget_pivot(runs),
        "twin": implementation_pairs(runs),
        "twin_null": twin_null(runs),
        "repro": reproducibility(pairs),
        "seeds": seed_spread(runs),
        "placebo": placebo_deltas(pairs),
    }
