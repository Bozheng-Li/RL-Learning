"""``webui.views.compare`` 里「取舍」函数的测试。

这一页在变体批次跑完之后面对的是 639 个运行、21 个变体名。两道筛选都不是「随手排个
序」：**按改动类别切开**（一类里同一个算法只有二到五档），**默认选中按族轮流取**
（否则跑得最多的那个族会把整屏占满）。这类错误在界面上表现为「图里少了个族」或
「明明选了却在图上找不到」，不报错、不抛异常，肉眼要读很久才发现，所以用单测钉住。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from webui import data
from webui.views import compare


def run(
    algorithm: str,
    *,
    variant: str | None = None,
    seed: int = 42,
    family: str | None = None,
) -> data.RunInfo:
    """造一个最小的 ``RunInfo``；只填这几个取舍函数真正会读的字段。"""
    name = f"lunarlander_{algorithm.lower()}_s{seed}"
    if variant:
        name += f"_{variant}"
    return data.RunInfo(
        name=name,
        path=Path("/nonexistent") / name,
        status="completed",
        mtime=0.0,
        environment="LunarLander-v3",
        algorithm=algorithm,
        seed=seed,
        summary={"mean_reward": 1.0},
        family=family or data.algorithm_family(algorithm),
        variant=variant,
    )


def seeds(algorithm: str, count: int, **kwargs: object) -> list[data.RunInfo]:
    return [run(algorithm, seed=42 + index, **kwargs) for index in range(count)]  # type: ignore[arg-type]


class TestFamilyBuckets(unittest.TestCase):
    def test_families_follow_learning_order(self) -> None:
        runs = [run("TD3"), run("PPO"), run("REINFORCE"), run("A2C")]
        self.assertEqual(
            [family for family, _groups in compare._family_buckets(runs)],
            ["REINFORCE", "A2C", "PPO", "TD3"],
        )

    def test_each_algorithm_variant_pair_is_its_own_group(self) -> None:
        runs = seeds("PPO", 2) + seeds("PPO", 2, variant="lr-1e-3")
        _family, groups = compare._family_buckets(runs)[0]
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(group) for group in groups), [2, 2])

    def test_unknown_algorithm_lands_in_other(self) -> None:
        runs = [run("MYSTERY-ALGO", family="OTHER")]
        self.assertEqual(compare._family_buckets(runs)[0][0], "OTHER")


class TestDimensionsAvailable(unittest.TestCase):
    def test_baseline_only_environment_offers_no_empty_doors(self) -> None:
        """一个还没跑过变体的环境不该出现「训练预算」这种点进去是空的档位。"""
        runs = seeds("PPO", 3) + seeds("DQN", 3)
        self.assertEqual(compare._dimensions_available(runs), ["基准", "全部"])

    def test_dimensions_follow_the_analysis_order(self) -> None:
        runs = seeds("PPO", 3)
        for variant in ("net-64", "wind-strong", "budget-50k", "evalseed-0"):
            runs.extend(seeds("PPO", 2, variant=variant))
        self.assertEqual(
            compare._dimensions_available(runs),
            ["基准", "学习超参", "环境扰动", "训练预算", "评估种子", "全部"],
        )

    def test_unknown_prefix_falls_into_its_own_bucket(self) -> None:
        runs = seeds("PPO", 2, variant="mystery-1")
        self.assertEqual(
            compare._dimensions_available(runs), ["基准", "其它", "全部"]
        )


class TestDimensionRuns(unittest.TestCase):
    def test_baseline_bucket_keeps_only_baselines(self) -> None:
        runs = seeds("PPO", 2) + seeds("PPO", 2, variant="lr-1e-3")
        picked = compare._dimension_runs(runs, "基准")
        self.assertTrue(all(info.variant is None for info in picked))
        self.assertEqual(len(picked), 2)

    def test_variant_bucket_brings_the_matching_baseline_along(self) -> None:
        """变体必须带上自己算法的基准——脱离基准，`lr-1e-3` 是高是低无从判断。"""
        runs = seeds("PPO", 2) + seeds("PPO", 2, variant="lr-1e-3")
        runs += seeds("DQN", 2)
        picked = compare._dimension_runs(runs, "学习超参")
        self.assertEqual(
            {(info.algorithm, info.variant) for info in picked},
            {("PPO", None), ("PPO", "lr-1e-3")},
        )

    def test_other_dimensions_and_their_baselines_stay_out(self) -> None:
        runs = seeds("PPO", 2) + seeds("PPO", 2, variant="lr-1e-3")
        runs += seeds("PPO", 2, variant="wind-strong")
        runs += seeds("SAC", 2)  # 没有这一维度的变体，基准也不该被拖进来
        picked = compare._dimension_runs(runs, "环境扰动")
        self.assertEqual(
            {(info.algorithm, info.variant) for info in picked},
            {("PPO", None), ("PPO", "wind-strong")},
        )

    def test_all_bucket_passes_everything_through(self) -> None:
        runs = seeds("PPO", 2) + seeds("DQN", 2, variant="wind-mild")
        self.assertEqual(compare._dimension_runs(runs, "全部"), runs)

    def test_empty_environment_returns_nothing_for_a_dimension(self) -> None:
        runs = seeds("PPO", 2)
        self.assertEqual(compare._dimension_runs(runs, "训练预算"), [])


class TestDimensionAlgorithms(unittest.TestCase):
    def test_spreads_across_families(self) -> None:
        """默认选中必须每个族都有代表。

        变体批次里跑得最多的是一个族（PPO 两套实现 + 七个学习超参档），按运行数排序
        取前 24 个会落成「这个环境只有 PPO」。这里钉住按族轮流取的行为。
        """
        runs = []
        for algorithm in ("A2C", "PPO", "DQN", "SAC", "TD3", "REINFORCE"):
            for variant in ("lr-1e-3", "lr-1e-4", "net-64", "net-256", "gamma-0.95"):
                runs.extend(seeds(algorithm, 2, variant=variant))
        picked = compare._dimension_algorithms(runs, 6)
        self.assertEqual(
            picked, ["REINFORCE", "A2C", "PPO", "DQN", "SAC", "TD3"]
        )

    def test_returns_algorithm_names_not_run_names(self) -> None:
        """粒度是算法名，所以「同配置的多个种子」天然整组进出，不可能被截成半个。"""
        runs = seeds("PPO", 5, variant="lr-1e-3")
        picked = compare._dimension_algorithms(runs, 8)
        self.assertEqual(picked, ["PPO"])

    def test_respects_the_limit_and_stays_unique(self) -> None:
        runs = []
        for index in range(40):
            runs.extend(seeds(f"ALGO-{index}", 2))
        picked = compare._dimension_algorithms(runs, 10)
        self.assertLessEqual(len(picked), 10)
        self.assertEqual(len(set(picked)), len(picked))

    def test_within_a_family_the_busiest_algorithm_comes_first(self) -> None:
        runs = seeds("SB3-PPO", 2) + seeds("PPO", 6)
        picked = compare._dimension_algorithms(runs, 1)
        self.assertEqual(picked, ["PPO"])

    def test_unknown_algorithms_still_get_a_representative(self) -> None:
        runs = seeds("MYSTERY-ALGO", 2, family="OTHER") + seeds("PPO", 2)
        picked = compare._dimension_algorithms(runs, 8)
        self.assertEqual(set(picked), {"MYSTERY-ALGO", "PPO"})

    def test_runs_without_an_algorithm_are_skipped(self) -> None:
        runs = seeds("PPO", 2) + [run("PPO", variant="lr-1e-3")]
        runs[-1].algorithm = None
        self.assertEqual(compare._dimension_algorithms(runs, 8), ["PPO"])

    def test_empty_input(self) -> None:
        self.assertEqual(compare._dimension_algorithms([], 8), [])


if __name__ == "__main__":
    unittest.main()
