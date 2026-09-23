"""``webui.views.train`` 的变体模式：把 ``Spec`` 铺成待启动列表的那一步。

这一层不渲染任何控件，但它是「界面上的表」与「真正启动的命令行」之间唯一的接缝。
接缝错了不会报错：界面照样显示 N 个运行，实际启动的命令行却少了一个 ``--set``——
变体跑出来与基准一模一样，排行榜上并排两行相同的数字，看起来像「这个参数没影响」。

所以这里钉住四件事：

- ``--set experiment.name`` 必须出现在变体运行里（它是事后识别变体身份的唯一依据）；
- 目录名与基准不同（否则后启动的静默覆盖先启动的）；
- ``job_label`` 带变体标识（训练监控页靠它区分同一算法的多个变体）；
- 预算变体各自带自己的步数，而不是整批用同一个数。
"""

from __future__ import annotations

import unittest

from webui import data, jobs, variants
from webui.views import train as view


def make_config() -> dict:
    """结构与 ``config/lunarlander.yaml`` 同形的最小配置。

    ``DQN`` 有 profile 但**没有** ``ent_coef``——真配置里就是这样，它正是「同一个变体
    在部分算法上会被跳过」的那个场景。
    """
    return {
        "experiment": {"name": "demo_baseline", "seed": 42},
        "environment": {"id": "Demo-v0", "kwargs": {"gravity": -10.0, "enable_wind": False}},
        "training": {"total_timesteps": 1000},
        "output": {"directory": "outputs/demo", "timestamped": False},
        "algorithm": {
            "name": "PPO",
            "device": "cpu",
            "profiles": {
                "PPO": {"kwargs": {"learning_rate": 0.0003, "ent_coef": 0.01}},
                "DQN": {"kwargs": {"learning_rate": 0.0001}},
            },
        },
    }


class TestPlannedVariants(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()

    def _plan(self, combos: list[variants.Variant | None], *, seeds=(42,), timesteps=1000):
        specs, _skipped = variants.expand(
            self.config, "demo", ["PPO"], list(seeds), combos, stamp="STAMP", timesteps=timesteps
        )
        return view._planned_variants("demo", self.config, specs, choices=[])

    def test_variant_run_carries_its_experiment_name(self) -> None:
        """``experiment.name`` 是变体身份的唯一来源——它没进命令行，这次运行就是基准。"""
        lr = variants.Variant(
            name="lr-1e-3",
            overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001},
        )
        planned = self._plan([None, lr])
        by_variant = {item["variant"]: item for item in planned}
        self.assertIn("experiment.name=demo__lr-1e-3", " ".join(by_variant["lr-1e-3"]["argv"]))
        self.assertNotIn("experiment.name", " ".join(by_variant[None]["argv"]))

    def test_baseline_and_variant_write_to_different_directories(self) -> None:
        lr = variants.Variant(
            name="lr-1e-3",
            overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001},
        )
        planned = self._plan([None, lr])
        directories = [item["directory"] for item in planned]
        self.assertEqual(len(set(directories)), len(directories))
        self.assertIn("lr-1e-3", next(d for d in directories if "lr" in d))

    def test_job_label_identifies_the_variant(self) -> None:
        """监控页与注册表里同算法同种子的多条必须分得开，否则看不出哪个是哪个。"""
        lr = variants.Variant(
            name="lr-1e-3",
            overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001},
        )
        planned = self._plan([None, lr])
        labels = {item["job_label"] for item in planned}
        self.assertEqual(len(labels), len(planned))
        self.assertTrue(any(label.endswith(":lr-1e-3") for label in labels))

    def test_budget_variant_keeps_its_own_timesteps(self) -> None:
        """预算变体走 ``--timesteps``，每条的步数各不相同——整批共用一个数会让
        5 万步的变体在监控页显示成「跑了 1/4」。"""
        budget = variants.Variant(name="budget-25k", overrides={"training.total_timesteps": 25000})
        planned = self._plan([None, budget], timesteps=1000)
        by_variant = {item["variant"]: item for item in planned}
        self.assertEqual(by_variant["budget-25k"]["timesteps"], 25000)
        self.assertIn("25000", " ".join(by_variant["budget-25k"]["argv"]))
        self.assertEqual(by_variant[None]["timesteps"], 1000)

    def test_labels_include_the_algorithm_for_every_row(self) -> None:
        planned = self._plan([None])
        self.assertEqual(planned[0]["label"], "PPO")


class TestVariantNotes(unittest.TestCase):
    """校验说明：哪些组合会被跳过，必须逐条说清原因。"""

    def setUp(self) -> None:
        self.config = make_config()

    def test_reports_algorithms_that_cannot_take_the_variant(self) -> None:
        variant = variants.Variant(
            name="ent-0", overrides={"algorithm.profiles.<ALGO>.kwargs.ent_coef": 0.0}
        )
        notes = view._variant_notes([variant], self.config, ["PPO", "DQN"])
        self.assertTrue(any("DQN" in note and "ent_coef" in note for note in notes))

    def test_says_so_when_the_variant_is_useless_everywhere(self) -> None:
        variant = variants.Variant(
            name="nope", overrides={"algorithm.profiles.<ALGO>.kwargs.nonexistent": 1}
        )
        notes = view._variant_notes([variant], self.config, ["PPO"])
        self.assertTrue(any("都用不了" in note for note in notes))

    def test_notes_are_deduplicated(self) -> None:
        variant = variants.Variant(
            name="ent-0", overrides={"algorithm.profiles.<ALGO>.kwargs.ent_coef": 0.0}
        )
        notes = view._variant_notes([variant], self.config, ["PPO", "DQN"])
        self.assertEqual(len(notes), len(set(notes)))


class TestVariantRows(unittest.TestCase):
    """长表的行：值和 YAML 标量写法往返一次不能变形。"""

    def test_rows_roundtrip_through_variant_from_lines(self) -> None:
        original = variants.Variant(name="net", overrides={
            "algorithm.profiles.<ALGO>.policy_kwargs.net_arch": [64, 64],
            "environment.kwargs.enable_wind": True,
        })
        rows = view._variant_rows([original])
        rendered = [
            {"变体名": row["变体名"], "覆盖键": row["覆盖键"], "值": row["值"]}
            for row in rows
        ]
        restored = variants.variant_from_lines(rendered)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].overrides, original.overrides)

    def test_every_override_gets_its_own_row(self) -> None:
        variant = variants.Variant(name="wind", overrides={
            "environment.kwargs.enable_wind": True,
            "environment.kwargs.wind_power": 15.0,
        })
        rows = view._variant_rows([variant])
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["变体名"] for row in rows}, {"wind"})


class TestPresets(unittest.TestCase):
    """快捷预设填进表里的行必须能通过 ``variant_from_lines`` 还原成原生类型。"""

    def test_every_preset_roundtrips(self) -> None:
        for label, payload in view._VARIANT_PRESETS.items():
            with self.subTest(preset=label):
                rows = view._variant_rows(
                    variants.Variant(name=name, overrides=dict(overrides))
                    for name, overrides in payload.items()
                )
                restored = variants.variant_from_lines(rows)
                self.assertEqual(
                    {variant.name: variant.overrides for variant in restored},
                    {name: dict(overrides) for name, overrides in payload.items()},
                )

    def test_preset_names_are_valid(self) -> None:
        """预设的变体名会进目录名与 ``experiment.name``，必须是合法标识。"""
        for payload in view._VARIANT_PRESETS.values():
            for name in payload:
                with self.subTest(variant=name):
                    self.assertEqual(variants.name_problems(name), [])


class TestLaunchArguments(unittest.TestCase):
    """启动时用的步数：从计划条目上取，而不是整批共用一个。"""

    def test_budget_row_would_pass_its_own_timesteps(self) -> None:
        config = make_config()
        budget = variants.Variant(name="budget-25k", overrides={"training.total_timesteps": 25000})
        specs, _ = variants.expand(
            config, "demo", ["PPO"], [42], [None, budget], stamp="STAMP", timesteps=1000
        )
        planned = view._planned_variants("demo", config, specs, choices=[])
        resolved = {
            item["variant"]: item["timesteps"] or 1000 for item in planned
        }
        self.assertEqual(resolved["budget-25k"], 25000)
        # 目录名与实验名都已经在 argv 里，注册表里的 total_timesteps 只用来画进度条。
        self.assertEqual(
            [item for item in planned if item["variant"] == "budget-25k"][0]["timesteps"],
            25000,
        )


class TestBaselineModeUnchanged(unittest.TestCase):
    """基准模式（没有变体）的行为必须与加变体之前逐字段相同。"""

    def test_planned_runs_keep_the_original_directory_and_label(self) -> None:
        config = make_config()
        planned = view._planned_runs(
            "demo", config, ["PPO"], [42], 1000,
            unique=True, stamp="STAMP", choices=[], overrides={},
        )
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["directory"], jobs.suggest_output_directory(
            "demo", "PPO", 42, unique=True, stamp="STAMP"
        ))
        self.assertEqual(planned[0]["label"], "PPO")
        self.assertEqual(planned[0]["job_label"], "train:demo:PPO:s42")
        self.assertEqual(planned[0]["timesteps"], 1000)


if __name__ == "__main__":
    unittest.main()
