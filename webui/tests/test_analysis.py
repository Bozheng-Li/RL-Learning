"""``webui.analysis`` 的纯函数测试。

只用标准库 ``unittest``（跑法见 ``test_gpus.py``）。这一层刻意不 import Streamlit、
不 import torch、也不读磁盘上的 ``outputs/``——全部断言都喂手工构造的 ``RunInfo``，
所以结果与「本机现在恰好有哪些运行」无关，换个仓库也照样绿。

这里钉住的每一条都对应分析里一个真实会读错的点：

- ``batch_stamp`` / ``select_batch``：批次是「同一批显卡、同一段时间、同一份代码」的
  代名词。跨批配对会把批次差异算成实验效应，所以配对必须限在同一批内。
- ``dimension_of`` / ``is_side_quest``：预算与评估种子**不能**与 200k 基准算敏感性 Δ
  ——前者改的是训练步数，后者是安慰剂。混进去会把「训练不足」读成「这个参数有害」。
- ``pair_table``：变体只与**同算法、同种子、同批次**的基准配对。
- ``Pair.same_model``：判据是**卡型号**，不是物理卡——实测同型号的两张不同卡逐位相同。
- ``noise_scale``：默认估的是含跨型号偏置的总噪声；``same_model_only`` 才收窄到同型号。
- ``_sign_p_value``：精确双侧符号检验，不走近似的正态。
- ``implementation_pairs`` / ``twin_null``：``SAC`` 与 ``SB3-SAC`` 是**同一个实现**的两次
  运行（``SAC`` 在本项目里回退到 SB3），只有 ``A2C`` / ``PPO`` 才是两个不同实现——
  这个区分搞错，整张可复现性表的结论就会反过来。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from webui import analysis, data, jobs


def run(
    name: str,
    algorithm: str,
    seed: int,
    reward: float,
    *,
    variant: str | None = None,
    timesteps: int = 200_000,
    has_summary: bool = True,
) -> data.RunInfo:
    """造一行 ``RunInfo``。``name`` 直接当目录名用，批次时间戳就取它尾部的 ``_xxx``。"""
    return data.RunInfo(
        name=name,
        path=Path("/nonexistent") / name,
        status="completed",
        mtime=0.0,
        algorithm=algorithm,
        seed=seed,
        total_timesteps=timesteps,
        summary={"mean_reward": reward} if has_summary else {},
        config_name="demo",
        experiment=(f"demo__{variant}" if variant else "demo"),
        variant=variant,
    )


class FakeRegistry:
    """临时替掉 ``jobs.load_registry``：给运行名喂 ``CUDA_VISIBLE_DEVICES`` 与型号。

    分析层要靠注册表才知道「这一次跑在哪张卡上」——``evaluation.json`` 里写的是
    ``cuda:0``（卡片被 ``CUDA_VISIBLE_DEVICES`` 隔离了），单看结果文件分不出来。
    """

    def __init__(self, cards: dict[str, tuple[str, str]]) -> None:
        #: 运行名 -> (卡 UUID, 注册表里的 device 字符串)
        self.cards = cards

    def __enter__(self) -> "FakeRegistry":
        self._original = jobs.load_registry
        jobs.load_registry = self._load  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        jobs.load_registry = self._original  # type: ignore[assignment]

    def _load(self) -> list[jobs.Job]:
        return [
            jobs.Job(
                job_id=name, label=name, kind="train", argv=[], log_path="",
                started_at="", run_dir=f"/outputs/{name}", pid=0,
                device=f"GPU 1: {model}", cuda_visible=uuid,
            )
            for name, (uuid, model) in self.cards.items()
        ]


class TestBatch(unittest.TestCase):
    def test_stamp_is_the_directory_suffix(self) -> None:
        info = run("lunarlander_ppo_s42_20260922-233256", "PPO", 42, 0.0)
        self.assertEqual(analysis.batch_stamp(info), "20260922-233256")

    def test_latest_batch_picks_the_biggest(self) -> None:
        runs = [
            run(f"a_s42_OLD", "PPO", 42, 0.0),
            run(f"b_s42_NEW", "PPO", 42, 0.0),
            run(f"c_s42_NEW", "PPO", 43, 0.0),
        ]
        self.assertEqual(analysis.latest_batch(runs), "NEW")

    def test_select_batch_filters(self) -> None:
        runs = [
            run("a_s42_OLD", "PPO", 42, 0.0),
            run("b_s42_NEW", "PPO", 42, 0.0),
            run("c_s42_NEW", "PPO", 43, 0.0),
        ]
        batch, stamp = analysis.select_batch(runs, "NEW")
        self.assertEqual(stamp, "NEW")
        self.assertEqual({info.name for info in batch}, {"b_s42_NEW", "c_s42_NEW"})

    def test_empty_input(self) -> None:
        self.assertEqual(analysis.select_batch([]), ([], None))
        self.assertIsNone(analysis.latest_batch([]))


class TestDimensions(unittest.TestCase):
    def test_known_prefixes(self) -> None:
        cases = {
            "lr-1e-3": "学习超参", "ent-0": "学习超参", "gamma-0.95": "学习超参",
            "net-64": "学习超参", "wind-strong": "环境扰动", "gravity-moon": "环境扰动",
            "budget-25k": "训练预算", "evalseed-0": "评估种子",
            "something-else": "其它", None: "基准",
        }
        for variant, expected in cases.items():
            with self.subTest(variant=variant):
                self.assertEqual(analysis.dimension_of(variant), expected)

    def test_side_quests_are_budget_and_evalseed_only(self) -> None:
        self.assertTrue(analysis.is_side_quest("budget-25k"))
        self.assertTrue(analysis.is_side_quest("evalseed-5000"))
        # 敏感性变体一个都不能被当成 side quest，否则会从总账里整批消失。
        for variant in ("lr-1e-3", "ent-0", "gamma-0.95", "net-64", "wind-strong", "gravity-moon"):
            with self.subTest(variant=variant):
                self.assertFalse(analysis.is_side_quest(variant))
        self.assertFalse(analysis.is_side_quest(None))

    def test_only_budget_loses_its_delta(self) -> None:
        """预算连配对都不做；评估种子必须留在配对里——σ 就是从它的 Δ 估出来的。"""
        self.assertTrue(analysis.has_no_delta("budget-25k"))
        self.assertFalse(analysis.has_no_delta("evalseed-0"))
        self.assertFalse(analysis.has_no_delta("lr-1e-3"))
        self.assertFalse(analysis.has_no_delta(None))


class TestPairTable(unittest.TestCase):
    def test_pairs_variant_with_same_seed_baseline(self) -> None:
        runs = [
            run("demo_ppo_s42_STAMP", "PPO", 42, 100.0),
            run("demo_ppo_s43_STAMP", "PPO", 43, 200.0),
            run("demo_ppo_lr_s42_STAMP", "PPO", 42, 120.0, variant="lr-1e-3"),
            run("demo_ppo_lr_s43_STAMP", "PPO", 43, 190.0, variant="lr-1e-3"),
        ]
        pairs, stamp = analysis.pair_table(runs)
        self.assertEqual(stamp, "STAMP")
        self.assertEqual(len(pairs), 2)
        by_seed = {pair.seed: pair for pair in pairs}
        self.assertEqual(by_seed[42].delta, 20.0)
        self.assertEqual(by_seed[43].delta, -10.0)

    def test_variant_without_baseline_is_dropped(self) -> None:
        runs = [
            run("demo_ppo_lr_s42_STAMP", "PPO", 42, 120.0, variant="lr-1e-3"),
            run("demo_sac_s42_STAMP", "SAC", 42, 50.0),
        ]
        pairs, _ = analysis.pair_table(runs)
        self.assertEqual(pairs, [])

    def test_cross_batch_baseline_is_not_used(self) -> None:
        """拿另一批的基准当对照，等于把批次差异算进变体效应。"""
        runs = [
            run("demo_ppo_s42_OLD", "PPO", 42, 100.0),
            run("demo_ppo_lr_s42_NEW", "PPO", 42, 120.0, variant="lr-1e-3"),
        ]
        pairs, _ = analysis.pair_table(runs)
        self.assertEqual(pairs, [])

    def test_runs_without_summary_are_skipped(self) -> None:
        runs = [
            run("demo_ppo_s42_STAMP", "PPO", 42, 100.0),
            run("demo_ppo_lr_s42_STAMP", "PPO", 42, 0.0, variant="lr-1e-3", has_summary=False),
        ]
        self.assertEqual(analysis.pair_table(runs)[0], [])


class TestCardModels(unittest.TestCase):
    def test_model_is_taken_after_the_colon(self) -> None:
        with FakeRegistry({"a_STAMP": ("GPU-x", "GeForce RTX 3090")}):
            models = analysis.card_model_of([run("a_STAMP", "PPO", 42, 0.0)])
        self.assertEqual(models["a_STAMP"], "GeForce RTX 3090")

    def test_unknown_runs_are_absent_not_guessed(self) -> None:
        with FakeRegistry({}):
            models = analysis.card_model_of([run("a_STAMP", "PPO", 42, 0.0)])
        self.assertEqual(models, {})

    def test_same_model_compares_card_models(self) -> None:
        with FakeRegistry({
            "base_STAMP": ("GPU-1", "GeForce RTX 3090"),
            "var_STAMP": ("GPU-2", "GeForce RTX 3090"),
        }):
            pairs, _ = analysis.pair_table([
                run("base_STAMP", "PPO", 42, 100.0),
                run("var_STAMP", "PPO", 42, 120.0, variant="lr-1e-3"),
            ])
        (pair,) = pairs
        # 不同物理卡、但同型号 —— 实测逐位相同，所以这不算「跨卡噪声」。
        self.assertIs(pair.same_device, False)
        self.assertIs(pair.same_model, True)

    def test_unknown_card_is_neither_same_nor_cross(self) -> None:
        with FakeRegistry({}):
            pairs, _ = analysis.pair_table([
                run("base_STAMP", "PPO", 42, 100.0),
                run("var_STAMP", "PPO", 42, 120.0, variant="lr-1e-3"),
            ])
        self.assertIsNone(pairs[0].same_model)


class TestNoiseScale(unittest.TestCase):
    def setUp(self) -> None:
        # 安慰剂：两个变体 × 三个种子，Δ 分别是 ±20 与 ±40。
        self.pairs = [
            analysis.Pair("PPO", "evalseed-0", seed, 0.0, delta, baseline_card="3090",
                          value_card="3090")
            for seed, delta in ((42, 20.0), (43, -20.0), (44, 20.0))
        ] + [
            analysis.Pair("PPO", "evalseed-5000", seed, 0.0, delta, baseline_card="3090",
                          value_card="4090")
            for seed, delta in ((42, 40.0), (43, -40.0), (44, 40.0))
        ]

    def test_scale_uses_placebo_deltas(self) -> None:
        scales = analysis.noise_scale(self.pairs)
        self.assertAlmostEqual(scales["PPO"], 32.86, places=2)

    def test_same_model_only_excludes_cross_model_placebo(self) -> None:
        scales = analysis.noise_scale(self.pairs, same_model_only=True)
        # 只剩 evalseed-0 的三个同型号配对（Δ = 20, -20, 20）。
        self.assertAlmostEqual(scales["PPO"], 23.094, places=3)

    def test_non_placebo_variants_never_contribute(self) -> None:
        pairs = self.pairs + [
            analysis.Pair("PPO", "lr-1e-3", 42, 0.0, 999.0),
            analysis.Pair("PPO", "lr-1e-3", 43, 0.0, -999.0),
        ]
        self.assertAlmostEqual(analysis.noise_scale(pairs)["PPO"], 32.86, places=2)

    def test_single_observation_gives_no_scale(self) -> None:
        # 一个样本算不出标准差；宁可缺一档，也不能返回 0 让后面所有 z 变成 inf。
        pairs = [analysis.Pair("PPO", "evalseed-0", 42, 0.0, 20.0)]
        self.assertEqual(analysis.noise_scale(pairs), {})

    def test_placebo_deltas_groups_by_algorithm(self) -> None:
        pairs = self.pairs + [analysis.Pair("DQN", "evalseed-0", 42, 0.0, 5.0)]
        grouped = analysis.placebo_deltas(pairs)
        self.assertEqual(len(grouped["PPO"]), 6)
        self.assertEqual(grouped["DQN"], [5.0])


class TestSignTest(unittest.TestCase):
    def test_all_one_way(self) -> None:
        # 10 个全同向：2 * C(10,10) / 2^10 = 2/1024。
        self.assertAlmostEqual(analysis._sign_p_value(10, 10), 2 / 1024, places=9)

    def test_balanced_is_one(self) -> None:
        self.assertEqual(analysis._sign_p_value(5, 10), 1.0)

    def test_six_of_six(self) -> None:
        self.assertAlmostEqual(analysis._sign_p_value(6, 6), 2 / 64, places=9)

    def test_eight_of_eight(self) -> None:
        self.assertAlmostEqual(analysis._sign_p_value(8, 8), 2 / 256, places=9)

    def test_empty(self) -> None:
        self.assertEqual(analysis._sign_p_value(0, 0), 1.0)

    def test_is_symmetric_in_direction(self) -> None:
        self.assertEqual(analysis._sign_p_value(2, 10), analysis._sign_p_value(8, 10))


class TestEffects(unittest.TestCase):
    def test_z_divides_by_scale_over_sqrt_n(self) -> None:
        pairs = [
            analysis.Pair("PPO", "lr-1e-3", 42, 0.0, 100.0),
            analysis.Pair("PPO", "lr-1e-3", 43, 0.0, 100.0),
            analysis.Pair("PPO", "lr-1e-3", 44, 0.0, 100.0),
        ]
        effect = analysis.effect_of(pairs, "lr-1e-3", "PPO", {"PPO": 50.0})
        self.assertEqual(effect.n, 3)
        self.assertAlmostEqual(effect.mean_delta, 100.0)
        # 100 / (50 / sqrt(3)) = 3.46
        self.assertAlmostEqual(effect.z, 3.4641, places=3)
        self.assertTrue(effect.significant)
        self.assertEqual(effect.verdict, "有利")

    def test_no_scale_means_no_z(self) -> None:
        pairs = [analysis.Pair("PPO", "lr-1e-3", 42, 0.0, 100.0)]
        effect = analysis.effect_of(pairs, "lr-1e-3", "PPO", {})
        self.assertIsNone(effect.z)
        self.assertFalse(effect.significant)
        self.assertEqual(effect.verdict, "—")

    def test_missing_combination_returns_none(self) -> None:
        self.assertIsNone(analysis.effect_of([], "lr-1e-3", "PPO", {}))

    def test_clean_delta_uses_same_model_subset_only(self) -> None:
        pairs = [
            analysis.Pair("PPO", "lr-1e-3", 42, 0.0, 100.0, value_card="3090",
                          baseline_card="3090"),
            analysis.Pair("PPO", "lr-1e-3", 43, 0.0, -100.0, value_card="3090",
                          baseline_card="4090"),
        ]
        effect = analysis.effect_of(pairs, "lr-1e-3", "PPO", {"PPO": 50.0})
        self.assertEqual(effect.mean_delta, 0.0)
        self.assertEqual(effect.cross_model, 1)
        self.assertEqual(effect.clean_deltas, (100.0,))
        self.assertEqual(effect.clean_delta, 100.0)
        self.assertFalse(effect.mixed_cards)

    def test_mixed_cards_flags_all_cross_model(self) -> None:
        pairs = [
            analysis.Pair("PPO", "lr-1e-3", 42, 0.0, 100.0, value_card="3090",
                          baseline_card="4090"),
        ]
        self.assertTrue(analysis.effect_of(pairs, "lr-1e-3", "PPO", {}))


class TestVariantSummary(unittest.TestCase):
    def test_consensus_and_orientation(self) -> None:
        pairs = []
        # 三个算法一致变差（Δ<0），一个变好。
        for algorithm, deltas in (
            ("PPO", (-10.0, -10.0)), ("DQN", (-10.0, -10.0)),
            ("SAC", (-10.0, -10.0)), ("TD3", (10.0, 10.0)),
        ):
            for index, delta in enumerate(deltas):
                pairs.append(analysis.Pair(algorithm, "lr-1e-3", 42 + index, 0.0, delta))

        (summary,) = analysis.summarize_variants(pairs, {})
        self.assertEqual(summary.algorithms, 4)
        self.assertEqual(summary.negative, 3)
        self.assertEqual(summary.positive, 1)
        self.assertEqual(summary.direction, "偏有害")
        self.assertEqual(summary.consensus, 0.75)
        self.assertEqual(summary.sign_p, analysis._sign_p_value(3, 4))

    def test_ties_count_as_no_direction(self) -> None:
        pairs = [
            analysis.Pair("PPO", "x", 42, 0.0, -1.0),
            analysis.Pair("DQN", "x", 42, 0.0, 1.0),
        ]
        (summary,) = analysis.summarize_variants(pairs, {})
        self.assertEqual(summary.direction, "无一致方向")
        self.assertEqual(summary.consensus, 0.5)

    def test_sorted_most_effect_like_first(self) -> None:
        pairs = [
            # 一致有害（4/4）
            *[analysis.Pair(name, "consistent", 42, 0.0, -5.0)
              for name in ("A", "B", "C", "D")],
            # 五五开
            *[analysis.Pair(name, "split", 42, 0.0, value)
              for name, value in (("A", -5.0), ("B", 5.0), ("C", -5.0), ("D", 5.0))],
        ]
        summaries = analysis.summarize_variants(pairs, {})
        self.assertEqual([item.variant for item in summaries], ["consistent", "split"])

    def test_effects_within_a_variant_are_sorted_by_delta(self) -> None:
        pairs = [
            analysis.Pair("A", "x", 42, 0.0, -5.0),
            analysis.Pair("B", "x", 42, 0.0, 5.0),
        ]
        (summary,) = analysis.summarize_variants(pairs, {})
        self.assertEqual([item.algorithm for item in summary.effects], ["B", "A"])


class TestSensitivityTable(unittest.TestCase):
    def test_columns_present(self) -> None:
        pairs = [
            analysis.Pair("PPO", "lr-1e-3", 42, 0.0, -10.0, value_card="3090",
                          baseline_card="3090"),
            analysis.Pair("PPO", "lr-1e-3", 43, 0.0, -20.0, value_card="3090",
                          baseline_card="4090"),
        ]
        (row,) = analysis.sensitivity_table(pairs, {"PPO": 10.0})
        self.assertEqual(row["变体"], "lr-1e-3")
        self.assertEqual(row["维度"], "学习超参")
        self.assertEqual(row["算法"], "PPO")
        self.assertEqual(row["种子数"], 2)
        self.assertEqual(row["Δ均值"], -15.0)
        self.assertEqual(row["Δ均值（同型号）"], -10.0)
        self.assertEqual(row["同型号配对数"], 1)
        self.assertEqual(row["跨型号配对数"], 1)
        self.assertEqual(row["Δ最小"], -20.0)
        self.assertEqual(row["Δ最大"], -10.0)


class TestTwinNull(unittest.TestCase):
    """零分布：配置逐字段相同（``SAC`` 与 ``SB3-SAC`` 是同一实现），只换卡。"""

    def test_identical_twin_gives_zero(self) -> None:
        runs = [
            run("a_s42_STAMP", "SAC", 42, 10.0),
            run("b_s42_STAMP", "SB3-SAC", 42, 10.0),
            run("a_s43_STAMP", "SAC", 43, 20.0),
            run("b_s43_STAMP", "SB3-SAC", 43, 20.0),
        ]
        result = analysis.twin_null(runs)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["median"], 0.0)
        self.assertEqual(result["max"], 0.0)

    def test_cross_model_twin_produces_nonzero_null(self) -> None:
        runs = [
            run("a_s42_STAMP", "SAC", 42, 0.0),
            run("b_s42_STAMP", "SB3-SAC", 42, 100.0),
        ]
        result = analysis.twin_null(runs)
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["max"], 100.0)
        self.assertEqual(result["share_over_50"], 1.0)
        self.assertEqual(result["share_over_100"], 0.0)  # 严格大于

    def test_no_twins(self) -> None:
        self.assertEqual(analysis.twin_null([run("a_s42_STAMP", "DQN", 42, 1.0)]), {"n": 0})

    def test_non_twin_families_are_excluded(self) -> None:
        """``A2C`` / ``PPO`` 的 plain 名是**自研实现**，与 ``SB3-*`` 的差是实现差异。

        把它们算进零分布会让底线虚高（实测 24.4 → 33.0），整张表的判定跟着变松。
        """
        runs = [
            run("a_s42_STAMP", "PPO", 42, 0.0),
            run("b_s42_STAMP", "SB3-PPO", 42, 500.0),
            run("c_s42_STAMP", "A2C", 42, 0.0),
            run("d_s42_STAMP", "SB3-A2C", 42, 500.0),
        ]
        self.assertEqual(analysis.twin_null(runs), {"n": 0})


class TestImplementationPairs(unittest.TestCase):
    def test_only_sac_td3_dqn_are_twins(self) -> None:
        runs = [
            run(f"{name}_s42_STAMP", name, 42, float(index))
            for index, name in enumerate(
                ("A2C", "SB3-A2C", "PPO", "SB3-PPO", "DQN", "SB3-DQN", "SAC", "SB3-SAC",
                 "TD3", "SB3-TD3")
            )
        ]
        by_algorithm = {row["算法"]: row for row in analysis.implementation_pairs(runs)}
        self.assertFalse(by_algorithm["A2C"]["同名同实现"])
        self.assertFalse(by_algorithm["PPO"]["同名同实现"])
        self.assertTrue(by_algorithm["DQN"]["同名同实现"])
        self.assertTrue(by_algorithm["SAC"]["同名同实现"])
        self.assertTrue(by_algorithm["TD3"]["同名同实现"])

    def test_same_model_pair_counts_as_identical(self) -> None:
        runs = [
            run("a_s42_STAMP", "SAC", 42, 5.0),
            run("b_s42_STAMP", "SB3-SAC", 42, 5.0),
        ]
        with FakeRegistry({
            "a_s42_STAMP": ("GPU-1", "GeForce RTX 3090"),
            "b_s42_STAMP": ("GPU-2", "GeForce RTX 3090"),
        }):
            (row,) = [
                item for item in analysis.implementation_pairs(runs)
                if item["算法"] == "SAC"
            ]
        self.assertEqual(row["同型号配对"], 1)
        self.assertEqual(row["同型号逐位相同"], 1)
        self.assertEqual(row["跨型号配对"], 0)

    def test_different_model_reports_divergence(self) -> None:
        runs = [
            run("a_s42_STAMP", "SAC", 42, 5.0),
            run("b_s42_STAMP", "SB3-SAC", 42, 7.0),
        ]
        with FakeRegistry({
            "a_s42_STAMP": ("GPU-1", "GeForce RTX 3090"),
            "b_s42_STAMP": ("GPU-2", "GeForce RTX 4090 D"),
        }):
            (row,) = [
                item for item in analysis.implementation_pairs(runs)
                if item["算法"] == "SAC"
            ]
        self.assertEqual(row["同型号配对"], 0)
        self.assertEqual(row["跨型号配对"], 1)
        self.assertEqual(row["跨型号逐位相同"], 0)


class TestBudget(unittest.TestCase):
    def test_budget_variants_are_not_paired_against_the_baseline(self) -> None:
        """预算变体回答「多少步够用」，把它们塞进敏感性 Δ 会把训练不足读成参数有害。"""
        runs = [
            run("base_s42_STAMP", "PPO", 42, 300.0),
            run("budget_s42_STAMP", "PPO", 42, 100.0, variant="budget-25k", timesteps=25_000),
        ]
        pairs, _ = analysis.pair_table(runs)
        self.assertEqual(pairs, [])

    def test_pivot_folds_the_baseline_in_as_the_top_rung(self) -> None:
        runs = [
            run("base_s42_STAMP", "PPO", 42, 300.0),
            run("budget_s42_STAMP", "PPO", 42, 100.0, variant="budget-25k", timesteps=25_000),
            run("budget_s43_STAMP", "PPO", 43, 200.0, variant="budget-25k", timesteps=25_000),
        ]
        pivot = analysis.budget_pivot(runs)
        self.assertEqual(pivot["PPO"][25_000], 150.0)
        self.assertEqual(pivot["PPO"][200_000], 300.0)


class TestSeedSpread(unittest.TestCase):
    def test_only_baselines_count(self) -> None:
        runs = [
            run("a_s42_STAMP", "PPO", 42, 10.0),
            run("a_s43_STAMP", "PPO", 43, 20.0),
            run("v_s44_STAMP", "PPO", 44, 999.0, variant="lr-1e-3"),
        ]
        rows = analysis.seed_spread(runs)
        (row,) = [item for item in rows if item["算法"] == "PPO"]
        self.assertEqual(row["种子数"], 2)
        self.assertEqual(row["均值"], 15.0)
        self.assertEqual(row["最高"], 20.0)


class TestReproducibility(unittest.TestCase):
    def test_only_placebo_pairs_measure_the_noise_floor(self) -> None:
        """真实效应绝不能混进「噪声底线」——那会把 lr-1e-3 的差异算成抖动。"""
        pairs = [
            analysis.Pair("PPO", "evalseed-0", 42, 0.0, 10.0, value_card="3090",
                          baseline_card="3090"),
            analysis.Pair("PPO", "evalseed-0", 43, 0.0, -30.0, value_card="3090",
                          baseline_card="4090"),
            analysis.Pair("PPO", "evalseed-0", 44, 0.0, 0.0),
            # 真实变体：效应再大也不该进同一型号/跨型号那两桶。
            analysis.Pair("PPO", "lr-1e-3", 42, 0.0, -999.0, value_card="3090",
                          baseline_card="3090"),
        ]
        result = analysis.reproducibility(pairs)
        self.assertEqual(result["pairs"], 4)
        self.assertEqual(result["placebo_same_model"]["n"], 1)
        self.assertEqual(result["placebo_same_model"]["median"], 10.0)
        self.assertEqual(result["placebo_cross_model"]["n"], 1)
        self.assertEqual(result["placebo_cross_model"]["median"], 30.0)
        self.assertEqual(result["unknown_model"], 1)
        self.assertEqual(result["placebo"]["n"], 3)
        # 跨型号占比算的是**全部**配对，不是安慰剂——它说明偏置影响面有多大。
        self.assertEqual(result["cross_model_pairs"], 1)
        self.assertEqual(result["cross_model_share"], 0.25)


class TestAnalyse(unittest.TestCase):
    def test_smoke_on_a_small_batch(self) -> None:
        runs = []
        for algorithm in ("PPO", "DQN"):
            for seed in (42, 43, 44):
                runs.append(run(f"base_{algorithm}_s{seed}_STAMP", algorithm, seed, 100.0))
                runs.append(run(f"lr_{algorithm}_s{seed}_STAMP", algorithm, seed, 90.0,
                                variant="lr-1e-3"))
                runs.append(run(f"placebo_{algorithm}_s{seed}_STAMP", algorithm, seed, 101.0,
                                variant="evalseed-0"))
        result = analysis.analyse(runs)
        self.assertEqual(result["stamp"], "STAMP")
        self.assertEqual(len(result["pairs"]), 12)
        self.assertEqual({item["变体"] for item in result["table"]}, {"lr-1e-3", "evalseed-0"})
        self.assertEqual(len(result["summaries"]), 2)
        # lr-1e-3 在两个算法上一致变差。
        (summary,) = [item for item in result["summaries"] if item.variant == "lr-1e-3"]
        self.assertEqual(summary.direction, "偏有害")
        self.assertEqual(summary.algorithms, 2)


if __name__ == "__main__":
    unittest.main()
