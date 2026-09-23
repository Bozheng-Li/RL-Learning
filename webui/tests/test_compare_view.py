"""``webui.views.compare`` 里两个「取舍」函数的测试。

这一页在变体批次跑完之后面对的是 639 个运行：多选框的默认值和曲线图的取子集都不再
是「随手排个序」，而是有明确要求的选择——每个算法族都要有代表、同一个 ``(算法, 变体)``
的多个种子不能被拆开。这类错误在界面上表现为「图里少了个族」或「曲线只剩一半种子」，
不报错、不抛异常，肉眼要读很久才发现，所以用单测钉住。
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
    """造一个最小的 ``RunInfo``；只填这两个取舍函数真正会读的字段。"""
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


class TestDefaultSelection(unittest.TestCase):
    def test_baselines_win_when_present(self) -> None:
        runs = seeds("PPO", 40) + [run("PPO", variant="lr-1e-3")]
        picked = compare._default_selection(runs, 24)
        self.assertEqual(len(picked), 40)  # 一条基准都没有被自己的变体顶掉

    def test_variants_with_no_baseline_spread_across_families(self) -> None:
        """「只看变体」范围里没有基准，默认值必须每个族都有代表。

        按传入顺序取前 N 个会落成「目录里排在前面的那几个族」——界面上看起来就像
        这个环境只有两个算法族。这里钉住按族轮流取的行为。
        """
        runs = []
        for algorithm in ("A2C", "PPO", "DQN", "SAC", "TD3", "REINFORCE"):
            for variant in ("lr-1e-3", "lr-1e-4"):
                runs.extend(seeds(algorithm, 2, variant=variant))
        picked = compare._default_selection(runs, 24)
        chosen = [info for info in runs if info.name in picked]
        self.assertEqual(len(picked), 24)
        self.assertEqual(
            {info.family for info in chosen},
            {"A2C", "PPO", "DQN", "SAC", "TD3", "REINFORCE"},
        )

    def test_never_splits_a_seed_group(self) -> None:
        """上限装不下整组时宁可不选，也不能只取半个组的种子。"""
        runs = seeds("PPO", 5, variant="lr-1e-3")
        picked = compare._default_selection(runs, 3)
        self.assertEqual(picked, [])

    def test_respects_the_limit(self) -> None:
        runs = []
        for algorithm in ("A2C", "PPO", "DQN", "SAC", "TD3"):
            for variant in ("a", "b", "c"):
                runs.extend(seeds(algorithm, 2, variant=variant))
        picked = compare._default_selection(runs, 10)
        self.assertLessEqual(len(picked), 10)
        self.assertEqual(len(set(picked)), len(picked))

    def test_group_larger_than_limit_does_not_block_other_families(self) -> None:
        """一个族有个超大组时，不能因为它「第一个」就把其它族全饿死。"""
        runs = seeds("PPO", 30, variant="huge")
        for algorithm in ("A2C", "DQN", "SAC", "TD3"):
            runs.extend(seeds(algorithm, 2, variant="small"))
        picked = compare._default_selection(runs, 10)
        chosen = [info for info in runs if info.name in picked]
        self.assertEqual(
            {info.family for info in chosen},
            {"A2C", "DQN", "SAC", "TD3"},
        )


class TestCurveSets(unittest.TestCase):
    def test_small_selection_passes_through_untouched(self) -> None:
        runs = seeds("PPO", 3) + seeds("SB3-PPO", 3)
        picked, dropped = compare._curve_sets(runs)
        self.assertEqual([info.name for info in picked], [info.name for info in runs])
        self.assertEqual(dropped, 0)

    def test_keeps_whole_seed_groups(self) -> None:
        runs = []
        for index in range(60):
            runs.extend(seeds(f"ALGO-{index}", 3))
        picked, dropped = compare._curve_sets(runs)
        self.assertGreater(dropped, 0)
        counts: dict[str, int] = {}
        for info in picked:
            counts[info.algorithm] = counts.get(info.algorithm, 0) + 1
        self.assertTrue(all(count == 3 for count in counts.values()), counts)

    def test_preserves_the_user_ordering(self) -> None:
        """取子集之后要按用户 multiselect 的顺序排回去，图例才不会每轮乱跳。"""
        runs = []
        for index in range(40):
            runs.extend(seeds(f"ALGO-{index}", 3))
        reversed_runs = list(reversed(runs))
        picked, _dropped = compare._curve_sets(reversed_runs)
        order = {info.name: index for index, info in enumerate(reversed_runs)}
        self.assertEqual(
            [order[info.name] for info in picked],
            sorted(order[info.name] for info in picked),
        )

    def test_every_family_is_represented(self) -> None:
        runs = []
        for algorithm in ("A2C", "PPO", "DQN", "SAC", "TD3", "REINFORCE"):
            for variant in ("a", "b", "c", "d"):
                runs.extend(seeds(algorithm, 2, variant=variant))
        picked, _dropped = compare._curve_sets(runs)
        self.assertEqual(
            {info.family for info in picked},
            {"A2C", "PPO", "DQN", "SAC", "TD3", "REINFORCE"},
        )

    def test_respects_the_group_and_run_caps(self) -> None:
        runs = []
        for index in range(200):
            runs.extend(seeds(f"ALGO-{index}", 4))
        picked, dropped = compare._curve_sets(runs)
        self.assertLessEqual(len({(info.algorithm, info.variant) for info in picked}),
                             compare._CURVE_GROUPS)
        self.assertLessEqual(len(picked), compare._CURVE_LIMIT)
        self.assertEqual(dropped, len(runs) - len(picked))


if __name__ == "__main__":
    unittest.main()
