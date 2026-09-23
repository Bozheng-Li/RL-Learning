"""变体实验的批量入口：同一环境上跑同一算法的多份配置。

「哪个算法好」用一次「算法 × 种子」的批量就能回答；「同一个算法对什么敏感」需要
让同一个算法跑多份配置——学习率换一档、网络加宽一层、给环境加上风。这个入口负责
把变体定义展开成一批运行，并按**探测到的显卡**分派下去。

用法::

    # 用配置里的变体集，扫 PPO 的学习率（两个实现一起，同超参只换实现）
    python variants.py --config lunarlander --variants lr-1e-4 lr-1e-3 \
                       --algorithms PPO SB3-PPO --seeds 42 43

    # 看清单与绑卡，不启动
    python variants.py --config lunarlander --variants wind-strong --dry-run

    # 临时定义一个变体（名=点号路径=值），不写文件
    python variants.py --config lunarlander \
        --variant 'lr-1e-4=algorithm.profiles.<ALGO>.kwargs.learning_rate=0.0001'

    # 流水线：始终保持所有显卡同时有活在跑，一个跑完立刻补下一个，直到全部跑完
    python variants.py --config lunarlander --variants lr-1e-3 net-64 \
                       --algorithms PPO SB3-PPO --seeds 42 43 --pipeline

设计要点：

- **用卡靠探测与分配（``webui/gpus.py``），不写死卡数。** 探测到几张就用几张。
  「一张卡上同时放几个任务」由 ``--per-device`` 给（默认 16，取值理由见下面那个常量），
  所以并行度恒等于 ``探测到的卡数 × --per-device``——本机探测到 5 张就是 80 路并行，
  换台机器自动跟着变。绑卡走 UUID，子进程里的设备恒为 ``cuda:0``。
- **子进程 + 日志落盘复用 ``webui.jobs.launch``。** 这个模块不 import Streamlit，
  可以放心给命令行用；顺带白拿「日志落 ``webui/.state/logs/``」与「进
  ``webui/.state/jobs.json`` 注册表」——命令行发起的批次因此会自动出现在
  「训练监控」页里。
- **``--pipeline`` 是「不要停」的那个模式。** 不是一次把 N 个任务全丢下去让它们抢卡，
  而是每个槽位同时只有一个任务，哪个槽空了立刻补下一个。跑完的算力立刻被接上，
  不会出现「前一半跑完、后一半还在排队占着空闲卡」的空转。
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from webui import gpus, jobs, variants  # noqa: E402
from webui import data as _data  # noqa: E402

#: 一张卡上默认叠几个任务。**是实测出来的，不是拍的**（本机 5 卡，PPO/20 万步）：
#:
#: ============  ====================  ===================
#: 每卡进程数     每卡吞吐（步/分）      相对单进程
#: ============  ====================  ===================
#: 1             31,000                1.0x
#: 4             82,000                2.6x
#: 8             103,000               3.3x
#: 16            105,000               3.4x
#: 32            108,000               3.5x
#: ============  ====================  ===================
#:
#: 16 是拐点：再加一倍进程只多 3%，而 CPU 负载翻倍（本机 256 核，5 卡 × 16 = 80 个
#: 单线程进程的 load1 是 75，×32 = 160 是 155）。**每卡一个进程是明显错的**——
#: 那种配置下 GPU 利用率上不去，因为小网络的瓶颈在 kernel launch 与采样，
#: 不在算力。所以默认取 16 而不是 1。
#:
#: 显存不是约束：单个进程实测最坏 448 MiB（SAC/TD3 的重放缓冲更大，PPO 是 444），
#: 16 × 448 = 7 GB，远低于最小的那张卡（24 GB）。
DEFAULT_PER_DEVICE = 16


def _parse_names(text: str) -> list[str]:
    return [token for token in text.replace(",", " ").split() if token]


def _build_variant_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="配置名，如 lunarlander")
    parser.add_argument(
        "--variants", nargs="*", default=[], metavar="名",
        help="要跑的变体名，取自 config/variants/<配置名>.yaml。留空则只跑基准。",
    )
    parser.add_argument(
        "--variant-set-file", type=Path,
        help="改用别的变体集文件（默认 config/variants/<配置名>.yaml）",
    )
    parser.add_argument(
        "--variant", action="append", default=[], metavar="名=路径=值",
        help="临时定义一个变体，可重复。值与 --set 同规则（YAML 语法）。"
             "路径里的 <ALGO> 会替换成算法的 profile 键。",
    )
    parser.add_argument(
        "--algorithms", nargs="*", default=[],
        help="算法名，留空则用配置里所有可跑的算法。",
    )
    parser.add_argument("--seeds", nargs="*", default=["42"], help="随机种子，默认 42")
    parser.add_argument("--timesteps", type=int, help="覆盖总步数")
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="只跑变体，不跑基准（默认变体与基准一起跑，基准是对照所必需的）。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划与绑卡")
    parser.add_argument(
        "--pipeline", action="store_true",
        help="流水线模式：每个卡槽同时只跑一个任务，空出来立刻补下一个，直到全部跑完。"
             "不加这个参数则一次把所有任务丢下去。",
    )
    parser.add_argument(
        "--poll", type=float, default=5.0, help="流水线模式的轮询间隔秒数（默认 5）"
    )
    parser.add_argument(
        "--per-device", type=int, default=DEFAULT_PER_DEVICE, metavar="N",
        help=f"一张卡上同时跑几个任务（默认 {DEFAULT_PER_DEVICE}）。并行度 = 探测到的卡数 × N。"
             "1 表示一张卡只跑一个。",
    )
    parser.add_argument("--stamp", help="时间戳后缀，默认取当前时刻；同批共用")
    return parser


def _ad_hoc_variants(raw_items: list[str], config: dict) -> list[variants.Variant]:
    """解析 ``--variant 名=路径=值``，按 YAML 语法取值。"""
    import yaml

    grouped: dict[str, dict] = {}
    for item in raw_items:
        name, sep, rest = item.partition("=")
        if not sep:
            raise SystemExit(f"--variant 需要 名=路径=值 形式，收到 {item!r}")
        path, sep2, raw = rest.partition("=")
        if not sep2:
            raise SystemExit(f"--variant 需要 名=路径=值 形式，收到 {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise SystemExit(f"--variant 的值不是合法 YAML：{raw!r}（{error}）") from None
        grouped.setdefault(name.strip(), {})[path.strip()] = value
    return [variants.Variant(name=name, overrides=overrides) for name, overrides in grouped.items()]


def _device_slots(per_device: int) -> list[gpus.DeviceChoice]:
    """探测本机的计算设备，每张卡铺 ``per_device`` 个槽位；没有卡时退到单槽 CPU。

    这是「动态分流」在批处理场景里的落点：卡数是探测出来的，写死卡数的地方一处都没有。
    ``build_choices``（``webui/gpus.py``）是那个唯一的分配入口，这里只是把它当成
    「槽位表」用——批处理要的是「哪个槽空了补哪个」，而不是一次性铺开。

    一张卡叠多个任务是因为这个项目的网络只有几十 MB：显存吃得掉，而小网络的
    GPU 利用率本来就低（大量时间在 kernel launch 与采样上），叠起来填满反而更快。
    """
    inventory = gpus.detect_devices()
    option = next(
        (item for item in gpus.device_options(inventory) if item.key == gpus.AUTO_KEY), None
    )
    if option is None:
        return [gpus.CPU_CHOICE]
    slots, _ = gpus.build_choices(
        option, max(len(inventory.devices), 1) * max(per_device, 1), inventory
    )
    return slots or [gpus.CPU_CHOICE]


def _is_finished(job: jobs.Job) -> bool:
    """收一次退出码；返回它是否已经结束。

    ``Job`` 的接口是 ``refresh()`` + ``finished``，不是 ``wait()``：这个类是给
    WebUI 用的，那边是「每次页面重跑顺手 ``poll()`` 一下」，从来不阻塞。批处理要的是
    等，所以在这里自己轮询出来——``refresh()`` 会在拿到退出码后把 ``_process`` 置空，
    重复调用是安全的幂等操作。

    先 ``refresh()`` 再读 ``finished`` 是必须的顺序：``alive`` 也会 ``poll()``，
    但退出码只有 ``refresh()`` 才记到 ``exit_code`` 上，漏掉这一步会永远看到
    ``exit_code is None``。
    """
    job.refresh()
    return job.finished


@dataclass
class _Running:
    spec: variants.Spec
    job: jobs.Job
    slot: gpus.DeviceChoice
    index: int
    label: str = ""

    def __post_init__(self) -> None:
        self.label = self.job.label


@dataclass
class _Stats:
    launched: int = 0
    done: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        elapsed = time.time() - self.started_at
        return (
            f"已启动 {self.launched} · 完成 {self.done} · 失败 {self.failed}"
            f" · 用时 {elapsed / 60:.1f} 分钟"
        )


def _variant_catalog(config_name: str, explicit: Path | None) -> dict[str, variants.Variant]:
    """读变体集。显式给了文件就按那个文件读，否则按 ``config/variants/<配置名>.yaml``。"""
    if explicit is not None:
        return variants.load_variant_set(explicit.stem, root=explicit.parent)
    return variants.load_variant_set(config_name)


def _plan(config_name: str, args: argparse.Namespace) -> tuple[list[variants.Spec], list[str], dict]:
    config = _data.load_config_dict(config_name)

    supported = {name.upper(): name for name in _data.compatible_algorithms(config)}
    if args.algorithms:
        wanted = []
        unknown = []
        for name in args.algorithms:
            canonical = supported.get(name.upper())
            (wanted if canonical else unknown).append(canonical or name)
        if unknown:
            available = "、".join(sorted(supported.values())) or "（没有）"
            raise SystemExit(
                f"配置 {config_name} 里不能跑的算法：{', '.join(unknown)}。可用的：{available}"
            )
    else:
        wanted = list(supported.values())

    catalog = _variant_catalog(config_name, args.variant_set_file)
    if args.variant_set_file and not catalog:
        raise SystemExit(f"变体集文件里没读到任何变体：{args.variant_set_file}")
    picked = list(catalog.values()) if not args.variants else [
        catalog[name] for name in args.variants if name in catalog
    ]
    unknown = [name for name in args.variants if name not in catalog]
    if unknown:
        known = "、".join(sorted(catalog)) or "（这个配置没有变体集文件）"
        raise SystemExit(f"变体集里没有：{', '.join(unknown)}。可用的：{known}")

    picked.extend(_ad_hoc_variants(args.variant, config))

    seeds = [int(token) for token in _parse_names(" ".join(str(s) for s in args.seeds))]
    stamp = args.stamp or time.strftime("%Y%m%d-%H%M%S")
    # 基准排在最前：它是对照的锚点，而且流水线按顺序下发时基准会先落地，
    # 后续的分析脚本不必等整批跑完就能开始工作。
    combos: list[variants.Variant | None] = [] if args.no_baseline else [None]
    combos.extend(picked)
    specs, skipped = variants.expand(
        config, config_name, wanted, seeds, combos,
        stamp=stamp, timesteps=args.timesteps,
    )
    return specs, skipped, config


def _order_for_pipeline(specs: list[variants.Spec], default_timesteps: int) -> list[variants.Spec]:
    """按步数从多到少排，长的先跑、短的后跑。

    流水线的墙钟时间由「最后一个离开的任务」决定：如果把 2.5 万步的预算变体排在前面，
    那些 20 万步的会拖到最后，收尾阶段就只剩一两个任务占着一整台机器。把长的排前面，
    短的自然堆到队尾，最后几个槽位空得很快，尾巴被压短。

    不按算法分组：槽位是按顺序轮询分派的，连续的任务本来就落到不同的卡上，
    想让「一张卡只跑同一个算法」得改分派逻辑，收益（少几次环境重建）抵不上复杂度。
    同一步数内按算法名排序，只是为了让日志读起来有顺序。
    """
    return sorted(
        specs,
        key=lambda spec: (
            -(spec.timesteps or default_timesteps),
            variants.profile_key(spec.algorithm),
            spec.algorithm,
            spec.seed,
            spec.variant_name or "",
        ),
    )


def _print_plan(specs: list[variants.Spec], slots: list[gpus.DeviceChoice]) -> None:
    from collections import Counter

    print(f"计划启动 {len(specs)} 个运行，{len(slots)} 个计算槽位：")
    per_card = Counter(gpus.short_label(slot) for slot in slots)
    for card, count in per_card.items():
        print(f"  · {card} × {count}")
    print()
    per_variant = Counter(spec.label if spec.variant else f"{spec.algorithm}（基准）" for spec in specs)
    for label, count in per_variant.items():
        print(f"  {label:<40} {count} 个")
    print()
    print(f"{'#':>3}  {'变体':<32} {'种子':>5}  目录")
    for index, spec in enumerate(specs, 1):
        name = spec.variant.name if spec.variant else "（基准）"
        print(f"{index:>3}  {name:<32} {spec.seed:>5}  {spec.directory}")


def _launch(spec: variants.Spec, slot: gpus.DeviceChoice, config_name: str) -> jobs.Job:
    argv = jobs.build_train_argv(
        config_name,
        algorithm=spec.algorithm,
        seed=spec.seed,
        timesteps=spec.timesteps,
        output_directory=spec.directory,
        device=slot.device_arg,
        overrides={**spec.overrides, **({"experiment.name": spec.experiment} if spec.experiment else {})},
    )
    tag = f":{spec.variant.tag}" if spec.variant else ""
    return jobs.launch(
        argv,
        label=f"train:{config_name}:{spec.algorithm}:s{spec.seed}{tag}",
        kind="train",
        run_dir=Path(spec.directory),
        total_timesteps=spec.timesteps,
        device=gpus.short_label(slot),
        cuda_visible=slot.cuda_visible,
    )


def _run_batch(specs: list[variants.Spec], slots: list[gpus.DeviceChoice], config_name: str) -> int:
    """一次把全部任务按槽位轮着丢下去（非流水线模式）。"""
    stats = _Stats()
    running: list[_Running] = []
    for index, spec in enumerate(specs):
        slot = slots[index % len(slots)]
        job = _launch(spec, slot, config_name)
        stats.launched += 1
        running.append(_Running(spec=spec, job=job, slot=slot, index=index))
        print(f"  [{stats.launched}/{len(specs)}] {job.label} → {gpus.short_label(slot)}")
    for item in running:
        while not _is_finished(item.job):
            time.sleep(1.0)
    return 0


def _run_pipeline(
    specs: list[variants.Spec], slots: list[gpus.DeviceChoice], config_name: str, poll: float
) -> int:
    """流水线：每个槽同时只跑一个，空出来立刻补下一个，直到全部跑完。

    这样显卡不会出现「有卡空闲但任务还在排队」的空转——只要队列里还有活，
    每个探测到的卡槽就一直是满的。
    """
    queue = list(specs)
    stats = _Stats()
    busy: dict[int, _Running] = {}
    print(f"流水线模式：{len(queue)} 个任务，{len(slots)} 个槽位，轮询 {poll:g}s")

    while queue or busy:
        # 先补空槽：能跑几个就补几个。
        for slot_index, slot in enumerate(slots):
            if slot_index in busy or not queue:
                continue
            spec = queue.pop(0)
            job = _launch(spec, slot, config_name)
            stats.launched += 1
            busy[slot_index] = _Running(spec=spec, job=job, slot=slot, index=stats.launched)
            print(
                f"  ▶ [{stats.launched}/{len(specs)}] {job.label}"
                f" → {gpus.short_label(slot)}  （剩余队列 {len(queue)}）"
            )

        if not busy:
            break

        # 再收完成的。
        for slot_index, item in list(busy.items()):
            if not _is_finished(item.job):
                continue
            code = item.job.exit_code if item.job.exit_code is not None else -1
            stats.done += 1
            if code != 0:
                stats.failed += 1
            mark = "✔" if code == 0 else f"✘({code})"
            print(
                f"  {mark} [{stats.done}/{len(specs)}] {item.job.label}"
                f"  {gpus.short_label(item.slot)}  · {stats.summary()}"
            )
            del busy[slot_index]

        if queue or busy:
            time.sleep(poll)

    print(f"\n全部结束：{stats.summary()}")
    return 1 if stats.failed else 0


def main() -> int:
    args = _build_variant_parser().parse_args()
    specs, skipped, config = _plan(args.config, args)

    slots = _device_slots(args.per_device)
    if not specs:
        print("没有要跑的运行。检查 --variants / --algorithms / --seeds。")
        return 1

    # 槽位表在这里定一次就用到底，**不在流水线里反复重探**：批次一旦跑起来，卡上占着的
    # 正是我们自己刚放上去的任务，``select_devices`` 会把它们判成「忙」而剔出候选池，
    # 槽位越跑越少，最后自己把自己憋停。探测放在开工前，那时看到的是真实的空卡情况。
    default_timesteps = int((config.get("training") or {}).get("total_timesteps") or 0)
    specs = _order_for_pipeline(specs, default_timesteps)

    _print_plan(specs, slots)
    if skipped:
        print("\n以下组合被跳过（不会启动子进程）：")
        for note in skipped:
            print(f"  · {note}")

    if args.dry_run:
        print("\n--dry-run：只打印，没有启动任何进程。")
        return 0

    print(f"\n开始启动（日志在 {Path(jobs.paths.LOG_ROOT)}）")
    if args.pipeline:
        return _run_pipeline(specs, slots, args.config, args.poll)
    return _run_batch(specs, slots, args.config)


if __name__ == "__main__":
    sys.exit(main())
