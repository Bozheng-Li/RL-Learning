"""``webui.variants`` 的纯函数测试。

只用标准库 ``unittest``，与 ``test_gpus.py`` 同一套约定——这一层刻意不 import
Streamlit 也不 import torch，就是为了能这样单独验证。

跑法::

    .venv/bin/python -m unittest discover webui/tests -v

这里钉住的每一条都对应一个真实踩过的坑：

- ``slug`` / ``name_problems``：目录名与 ``experiment.name`` 里不能出现斜杠、空格、
  ``__``。变体名是人手写的，直接拼进路径会写出子目录，``compare.discover_runs``
  只扫一层，跑完的实验会从 ``compare.py`` 里消失。
- ``resolve`` 的 ``<ALGO>``：``PPO`` 与 ``SB3-PPO`` 必须展开成**同一个** profile 键，
  否则「同超参、只换实现」这组对照根本不成立。
- ``validate``：REINFORCE 没有 profile、DQN 没有 ``ent_coef``、覆盖写到别的算法的
  profile 段上——这三种都要在启动前拦下，不能等到子进程里报错或静默忽略。
- ``coerce_value``：``yaml.safe_load("1e-3")`` 是**字符串**（YAML 1.1），不兜一层的话
  ``learning_rate`` 会变成一个 str 一路带到 SB3 构造优化器时才炸。
- 撞名回归：同算法同种子、不同变体必须落到不同目录；``tag=None`` 时目录与改动前逐字相同。
"""

from __future__ import annotations

import unittest

import yaml

from webui import data, jobs, variants


def make_config() -> dict:
    """一份最小配置，结构与 ``config/lunarlander.yaml`` 同形但小得多。

    刻意区别于真配置：真配置会随项目演进，把断言挂在它上面等于把测试绑在另一个
    文件的细节上。这里只保留 ``validate`` 要用到的形状。
    """
    return {
        "experiment": {"name": "demo_baseline", "seed": 42},
        "environment": {"id": "Demo-v0", "kwargs": {"gravity": -10.0, "enable_wind": False}},
        "training": {"total_timesteps": 1000},
        "evaluation": {"seed_offset": 10000},
        "output": {"directory": "outputs/demo", "timestamped": False},
        "algorithm": {
            "name": "PPO",
            "device": "cpu",
            "profiles": {
                "PPO": {"kwargs": {"learning_rate": 0.0003, "ent_coef": 0.01}},
                "DQN": {"kwargs": {"learning_rate": 0.0001}},
                # 故意不给 REINFORCE 建 profile —— 真配置里它也是这样，由
                # ``require_profile = False`` 决定。
            },
        },
    }


class TestProfileKey(unittest.TestCase):
    """``profile_key`` 只去 ``SB3-`` 前缀再大写，是与 ``algorithms`` 那边的契约。"""

    def test_strips_sb3_prefix(self) -> None:
        self.assertEqual(variants.profile_key("SB3-PPO"), "PPO")

    def test_plain_name_is_uppercased(self) -> None:
        self.assertEqual(variants.profile_key("sac"), "SAC")

    def test_only_leading_prefix_is_stripped(self) -> None:
        self.assertEqual(variants.profile_key("SB3-SB3-PPO"), "SB3-PPO")

    def test_matches_authoritative_implementation(self) -> None:
        """与 ``algorithms.sb3_wrapper.profile_key`` 必须逐字一致。

        那边是唯一权威，但顶层 import stable_baselines3 要付 1.6 秒，所以
        ``webui/variants.py`` 自己实现了一份。这条用例就是那份复制品的保险。
        """
        try:
            from algorithms.sb3_wrapper import profile_key as authoritative
        except Exception:  # noqa: BLE001  —— 没装 SB3 时跳过，不当失败
            self.skipTest("stable_baselines3 不可用")
        for name in ("PPO", "SB3-PPO", "SB3-SAC", "sac", "TD3"):
            with self.subTest(name=name):
                self.assertEqual(variants.profile_key(name), authoritative(name))


class TestSlugAndExperimentName(unittest.TestCase):
    def test_slug_folds_unsafe_characters(self) -> None:
        self.assertEqual(variants.slug("lr 1e-3"), "lr-1e-3")
        self.assertEqual(variants.slug("net/64"), "net-64")
        self.assertEqual(variants.slug("a//b"), "a-b")

    def test_slug_has_no_path_separator(self) -> None:
        # 目录名里出现 ``/`` 会写出子目录，``compare.discover_runs`` 只扫一层，会看不见。
        self.assertNotIn("/", variants.slug("../../etc/passwd"))

    def test_slug_never_empty(self) -> None:
        self.assertEqual(variants.slug(""), "variant")
        self.assertEqual(variants.slug("///"), "variant")

    def test_slug_truncates(self) -> None:
        self.assertLessEqual(len(variants.slug("x" * 200)), 24)

    def test_experiment_name_roundtrip(self) -> None:
        name = variants.experiment_name("demo", "lr-1e-3")
        self.assertEqual(name, "demo__lr-1e-3")
        self.assertEqual(variants.parse_experiment(name), ("demo", "lr-1e-3"))

    def test_experiment_name_without_variant_is_config_name(self) -> None:
        self.assertEqual(variants.experiment_name("demo", None), "demo")
        self.assertEqual(variants.experiment_name("demo", ""), "demo")

    def test_parse_rejects_ambiguous_names(self) -> None:
        # 历史运行里有 ``lunarlander_baseline``、甚至 ``a__b__c`` 这类写法。
        # 认不出来就返回 (None, None)，绝不能瞎猜——猜错会把基准当成变体。
        for raw in (None, "", "demo", "demo_baseline", "a__b__c", "__x", "x__"):
            with self.subTest(raw=raw):
                self.assertEqual(variants.parse_experiment(raw), (None, None))

    def test_name_problems_flags_separator_and_unsafe_chars(self) -> None:
        self.assertEqual(variants.name_problems("lr-1e-3"), [])
        self.assertEqual(variants.name_problems("Lr_1e.3-net"), [])
        self.assertTrue(variants.name_problems("a__b"))
        self.assertTrue(variants.name_problems("风力强"))
        self.assertTrue(variants.name_problems("net 64"))
        self.assertTrue(variants.name_problems(""))
        self.assertTrue(variants.name_problems("   "))


class TestResolve(unittest.TestCase):
    def test_placeholder_expands_to_same_profile_for_both_implementations(self) -> None:
        variant = variants.Variant(
            name="lr", overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001}
        )
        for algorithm in ("PPO", "SB3-PPO"):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(
                    variants.resolve(variant, algorithm),
                    {"algorithm.profiles.PPO.kwargs.learning_rate": 0.001},
                )

    def test_non_profile_keys_pass_through(self) -> None:
        variant = variants.Variant(name="wind", overrides={"environment.kwargs.enable_wind": True})
        self.assertEqual(
            variants.resolve(variant, "PPO"), {"environment.kwargs.enable_wind": True}
        )


class TestValidate(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()

    def test_clean_variant_passes(self) -> None:
        variant = variants.Variant(
            name="lr", overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001}
        )
        self.assertEqual(variants.validate(variant, "PPO", self.config), [])

    def test_environment_key_applies_to_algorithms_without_profile(self) -> None:
        """环境类覆盖与算法无关，REINFORCE 也该能跑——不能整条跳过。"""
        variant = variants.Variant(name="wind", overrides={"environment.kwargs.enable_wind": True})
        self.assertEqual(variants.validate(variant, "REINFORCE", self.config), [])

    def test_reinforce_without_profile_is_rejected(self) -> None:
        variant = variants.Variant(
            name="lr", overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001}
        )
        problems = variants.validate(variant, "REINFORCE", self.config)
        self.assertTrue(any("没有 profile" in item for item in problems))

    def test_missing_key_is_rejected(self) -> None:
        variant = variants.Variant(
            name="ent", overrides={"algorithm.profiles.<ALGO>.kwargs.ent_coef": 0.0}
        )
        problems = variants.validate(variant, "DQN", self.config)
        self.assertTrue(any("ent_coef" in item for item in problems), problems)

    def test_override_on_another_algorithms_profile_is_rejected(self) -> None:
        """**静默 no-op 的防线**：覆盖写到别的算法的 profile 段上不会报错，但不会生效。"""
        variant = variants.Variant(
            name="cross", overrides={"algorithm.profiles.PPO.kwargs.learning_rate": 0.001}
        )
        problems = variants.validate(variant, "DQN", self.config)
        self.assertTrue(any("静默忽略" in item for item in problems), problems)

    def test_reserved_keys_are_rejected(self) -> None:
        for key in sorted(variants.RESERVED_KEYS):
            with self.subTest(key=key):
                variant = variants.Variant(name="bad", overrides={key: "x"})
                problems = variants.validate(variant, "PPO", self.config)
                self.assertTrue(problems, key)

    def test_bad_name_is_reported_for_any_algorithm(self) -> None:
        variant = variants.Variant(name="风力", overrides={"environment.kwargs.gravity": -3.0})
        self.assertTrue(variants.validate(variant, "PPO", self.config))


class TestCoerceValue(unittest.TestCase):
    def test_yaml_11_returns_string_for_scientific_notation(self) -> None:
        """先钉住这个前提：``1e-3`` 经 ``yaml.safe_load`` 是**字符串**。"""
        self.assertEqual(yaml.safe_load("1e-3"), "1e-3")
        self.assertIsInstance(yaml.safe_load("1e-3"), str)

    def test_string_is_coerced_to_target_float(self) -> None:
        value, warning = variants.coerce_value("1e-3", 0.0003, "x")
        self.assertEqual(value, 0.001)
        self.assertIsNotNone(warning)

    def test_string_is_coerced_to_target_int(self) -> None:
        value, _ = variants.coerce_value("25000", 200000, "x")
        self.assertEqual(value, 25000)
        self.assertIsInstance(value, int)

    def test_bool_target_is_left_alone(self) -> None:
        # bool 是 int 的子类，漏掉这个判断会把 True 当成数字去转。
        value, warning = variants.coerce_value(True, False, "x")
        self.assertIs(value, True)
        self.assertIsNone(warning)

    def test_non_numeric_target_is_left_alone(self) -> None:
        value, warning = variants.coerce_value("cuda:0", "cpu", "x")
        self.assertEqual(value, "cuda:0")
        self.assertIsNone(warning)

    def test_unparseable_string_is_left_for_validate(self) -> None:
        value, warning = variants.coerce_value("abc", 0.5, "x")
        self.assertEqual(value, "abc")
        self.assertIsNone(warning)

    def test_resolve_for_drops_values_equal_to_the_default(self) -> None:
        """覆盖等于默认值 = 白烧一张卡，而且排行榜上会出现两行一样的数字。"""
        config = make_config()
        variant = variants.Variant(
            name="lr", overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.0003}
        )
        resolved, notes = variants.resolve_for(variant, "PPO", config)
        self.assertEqual(resolved, {})
        self.assertTrue(notes)

    def test_resolve_for_keeps_real_changes(self) -> None:
        config = make_config()
        variant = variants.Variant(
            name="lr", overrides={"algorithm.profiles.<ALGO>.kwargs.learning_rate": "1e-3"}
        )
        resolved, _ = variants.resolve_for(variant, "PPO", config)
        self.assertEqual(resolved, {"algorithm.profiles.PPO.kwargs.learning_rate": 0.001})


class TestNaming(unittest.TestCase):
    def test_directory_is_unique_across_variants(self) -> None:
        """同算法同种子、不同变体必须落到不同目录——否则后启动的静默覆盖先启动的。"""
        tag_a = variants.Variant(name="lr-1e-3").tag
        tag_b = variants.Variant(name="lr-1e-4").tag
        directory_a = jobs.suggest_output_directory(
            "demo", "PPO", 42, unique=True, stamp="STAMP", tag=tag_a
        )
        directory_b = jobs.suggest_output_directory(
            "demo", "PPO", 42, unique=True, stamp="STAMP", tag=tag_b
        )
        self.assertNotEqual(directory_a, directory_b)
        self.assertIn("lr-1e-3", directory_a)

    def test_no_tag_keeps_the_original_format(self) -> None:
        """``tag=None`` 时必须与加变体概念之前逐字相同——33 个基线的目录名是历史产物。"""
        without = jobs.suggest_output_directory("demo", "SB3-PPO", 42, unique=True, stamp="STAMP")
        explicit_none = jobs.suggest_output_directory(
            "demo", "SB3-PPO", 42, unique=True, stamp="STAMP", tag=None
        )
        self.assertEqual(without, explicit_none)
        self.assertEqual(without, "outputs/demo_sb3ppo_s42_STAMP")

    def test_not_unique_ignores_the_tag(self) -> None:
        directory = jobs.suggest_output_directory(
            "demo", "PPO", 42, unique=False, tag="lr"
        )
        self.assertEqual(directory, "outputs/demo")


class TestExpand(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()

    def test_baseline_only_matches_old_behaviour(self) -> None:
        specs, skipped = variants.expand(
            self.config, "demo", ["PPO"], [42, 43], [None], stamp="STAMP"
        )
        self.assertEqual(len(specs), 2)
        self.assertEqual(skipped, [])
        for spec in specs:
            self.assertIsNone(spec.variant)
            self.assertEqual(spec.overrides, {})
            self.assertIsNone(spec.experiment)
            self.assertEqual(spec.directory, f"outputs/demo_ppo_s{spec.seed}_STAMP")
            self.assertEqual(spec.label, "PPO")

    def test_cartesian_product(self) -> None:
        lr = variants.Variant(name="lr-1e-3", overrides={
            "algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001,
        })
        wind = variants.Variant(name="wind", overrides={"environment.kwargs.enable_wind": True})
        specs, _ = variants.expand(
            self.config, "demo", ["PPO", "DQN"], [42, 43], [None, lr, wind], stamp="STAMP"
        )
        self.assertEqual(len(specs), 2 * 2 * 3)
        self.assertEqual(len({spec.directory for spec in specs}), len(specs))

    def test_invalid_combination_is_skipped_not_launched(self) -> None:
        lr = variants.Variant(name="lr", overrides={
            "algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001,
        })
        specs, skipped = variants.expand(
            self.config, "demo", ["PPO", "REINFORCE"], [42], [lr], stamp="STAMP"
        )
        self.assertEqual({spec.algorithm for spec in specs}, {"PPO"})
        self.assertTrue(any("没有 profile" in item for item in skipped))

    def test_timesteps_override_is_lifted_to_the_flag(self) -> None:
        budget = variants.Variant(name="budget", overrides={"training.total_timesteps": 25000})
        specs, _ = variants.expand(
            self.config, "demo", ["PPO"], [42], [None, budget], stamp="STAMP"
        )
        by_variant = {spec.variant_name: spec for spec in specs}
        self.assertEqual(by_variant["budget"].timesteps, 25000)
        self.assertNotIn("training.total_timesteps", by_variant["budget"].overrides)
        # 基准沿用批次默认（这里是 None，由调用方/配置决定）。
        self.assertIsNone(by_variant[None].timesteps)

    def test_duplicate_cover_is_deduplicated(self) -> None:
        """名字不同但覆盖与基准完全相同的变体不该白烧一张卡。"""
        same = variants.Variant(name="lr-same", overrides={
            "algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.0003,
        })
        specs, skipped = variants.expand(
            self.config, "demo", ["PPO"], [42], [None, same], stamp="STAMP"
        )
        self.assertEqual(len(specs), 1)
        self.assertTrue(any("已去重" in item for item in skipped))

    def test_experiment_name_is_written_for_variants_only(self) -> None:
        lr = variants.Variant(name="lr-1e-3", overrides={
            "algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001,
        })
        specs, _ = variants.expand(self.config, "demo", ["PPO"], [42], [None, lr], stamp="S")
        self.assertEqual({spec.experiment for spec in specs}, {None, "demo__lr-1e-3"})


class TestVariantSetIO(unittest.TestCase):
    def test_load_missing_file_returns_empty(self) -> None:
        self.assertEqual(variants.load_variant_set("这配置不存在"), {})

    def test_roundtrip_through_yaml(self) -> None:
        items = [
            variants.Variant(name="lr-1e-3", overrides={
                "algorithm.profiles.<ALGO>.kwargs.learning_rate": 0.001,
            }),
            variants.Variant(name="net-64", overrides={
                "algorithm.profiles.<ALGO>.policy_kwargs.net_arch": [64, 64],
            }),
        ]
        text = variants.dump_variant_set(items)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.yaml"
            path.write_text(text, encoding="utf-8")
            loaded = variants.load_variant_set("demo", root=Path(directory))
        self.assertEqual({name: item.overrides for name, item in loaded.items()},
                         {item.name: item.overrides for item in items})
        # list 值必须原样往返，不能被摊成字符串。
        self.assertEqual(loaded["net-64"].overrides[
            "algorithm.profiles.<ALGO>.policy_kwargs.net_arch"], [64, 64])

    def test_variant_from_lines_parses_yaml_values(self) -> None:
        items = variants.variant_from_lines([
            {"变体名": "lr", "覆盖键": "algorithm.profiles.<ALGO>.kwargs.learning_rate", "值": "1e-3"},
            {"变体名": "net", "覆盖键": "algorithm.profiles.<ALGO>.policy_kwargs.net_arch",
             "值": "[64, 64]"},
            {"变体名": "wind", "覆盖键": "environment.kwargs.enable_wind", "值": "true"},
            {"变体名": "", "覆盖键": "x", "值": "1"},  # 没有名字的行丢掉
        ])
        by_name = {item.name: item.overrides for item in items}
        self.assertEqual(by_name["lr"]["algorithm.profiles.<ALGO>.kwargs.learning_rate"], "1e-3")
        self.assertEqual(by_name["net"]["algorithm.profiles.<ALGO>.policy_kwargs.net_arch"], [64, 64])
        self.assertIs(by_name["wind"]["environment.kwargs.enable_wind"], True)
        self.assertNotIn("", by_name)


class TestDiffAgainstBase(unittest.TestCase):
    def test_reports_only_changed_leaves(self) -> None:
        base = {"a": {"b": 1, "c": [1, 2]}, "d": True}
        resolved = {"a": {"b": 2, "c": [1, 2]}, "d": True}
        self.assertEqual(variants.diff_against_base(resolved, base), [("a.b", 1, 2)])

    def test_lists_are_compared_as_values(self) -> None:
        base = {"a": {"arch": [128, 128]}}
        resolved = {"a": {"arch": [64, 64]}}
        self.assertEqual(variants.diff_against_base(resolved, base), [("a.arch", [128, 128], [64, 64])])

    def test_missing_side_is_noted(self) -> None:
        changed = variants.diff_against_base({"a": 1}, {"a": 1, "b": 2})
        self.assertEqual(changed, [("b", 2, variants.MISSING)])

    def test_underscore_keys_are_ignored(self) -> None:
        self.assertEqual(variants.diff_against_base({"_x": 1, "a": 1}, {"a": 1}), [])


class TestVersionAgainstRealConfig(unittest.TestCase):
    """对真实配置跑一遍：确认项目里那份 ``lunarlander`` 变体集全都立得住。

    这条用例的价值在于它会随配置演进自动报警——哪天有人给 DQN 删掉了
    ``learning_rate``，或者改了 ``<ALGO>`` 的展开规则，这里立刻红。
    """

    def test_lunarlander_variants_validate_against_every_algorithm(self) -> None:
        try:
            config = data.load_config_dict("lunarlander")
        except (OSError, ValueError):
            self.skipTest("读不到 lunarlander 配置")
        catalog = variants.load_variant_set("lunarlander")
        if not catalog:
            self.skipTest("没有 lunarlander 变体集")
        algorithms = data.compatible_algorithms(config)
        self.assertIn("PPO", algorithms)

        for name, variant in catalog.items():
            with self.subTest(variant=name):
                self.assertEqual(variants.name_problems(name), [])
                # 每个变体至少要能用在**某个**算法上，否则它在这个环境里毫无意义。
                usable = [
                    algorithm for algorithm in algorithms
                    if not variants.validate(variant, algorithm, config)
                ]
                self.assertTrue(usable, f"{name} 在任何算法上都用不了")

        # 基准永远可跑，且不带任何覆盖。
        specs, skipped = variants.expand(
            config, "lunarlander", ["PPO"], [42], [None], stamp="STAMP"
        )
        self.assertEqual(len(specs), 1)
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
