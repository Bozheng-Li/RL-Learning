"""``webui.charts`` 的结构测试。

这一层是 altair 的构造层，产出的是**一张图该长什么样**，而它出的错都不是异常：
线型分配错了一条线看上去只是「跟别人重了」，y 轴没独立只是「一个算法被压成直线」，
刷子没提升到分面层只是「拖了没反应」。界面上一律不报错，肉眼要读很久才发现，所以
直接断言 ``.to_dict()``——不需要浏览器，毫秒级。

断言的是**结构**而不是像素：分面数、``resolve`` 里 y 是否独立、每个序列有没有各自的
线型与线宽、中文有没有真的进到标题里。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from webui import charts, data


def run(
    algorithm: str,
    *,
    variant: str | None = None,
    seed: int = 42,
    family: str | None = None,
) -> data.RunInfo:
    """造一个最小的 ``RunInfo``；只填样式分配真正会读的字段。"""
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


def frame_for(runs: list[data.RunInfo], styles: list[charts.CurveSeries]) -> pd.DataFrame:
    rows = [
        {
            "算法": info.algorithm,
            "曲线": style.label,
            "步数": float(step),
            "回报": float(step) * 0.5,
        }
        for info, style in zip(runs, styles)
        for step in range(4)
    ]
    return pd.DataFrame(rows)


class TestSeriesStyle(unittest.TestCase):
    def test_same_family_shares_a_color(self) -> None:
        runs = [run("PPO"), run("PPO", variant="lr-1e-3"), run("SB3-PPO")]
        styles = charts.series_style(runs)
        self.assertEqual(len({style.color for style in styles}), 1)

    def test_different_families_get_different_colors(self) -> None:
        styles = charts.series_style([run("PPO"), run("DQN"), run("REINFORCE")])
        self.assertEqual(len({style.color for style in styles}), 3)

    def test_baseline_and_variant_do_not_share_a_line_style(self) -> None:
        """同一个面板里基准与变体撞成同一条实线，两条线就分不开了。"""
        runs = [run("PPO"), run("PPO", variant="lr-1e-3")]
        styles = charts.series_style(runs)
        self.assertNotEqual(styles[0].dash, styles[1].dash)

    def test_baseline_is_thicker_than_variant(self) -> None:
        runs = [run("PPO"), run("PPO", variant="lr-1e-3")]
        baseline, variant = charts.series_style(runs)
        self.assertTrue(baseline.is_baseline)
        self.assertFalse(variant.is_baseline)
        self.assertGreater(baseline.width, variant.width)

    def test_dashes_are_stable_across_input_order(self) -> None:
        """线型编号按 ``(实现, 变体)`` 字典序，不按传入顺序。

        传入顺序取决于目录 mtime——新跑一个运行会让所有线型整体平移，前后两次刷新
        看到的图不一样，就没法对着图说话了。
        """
        runs = [run("PPO", variant="net-64"), run("PPO"), run("PPO", variant="lr-1e-3")]
        first = {style.label: style.dash for style in charts.series_style(runs)}
        second = {style.label: style.dash for style in charts.series_style(list(reversed(runs)))}
        self.assertEqual(first, second)

    def test_labels_carry_the_variant_name(self) -> None:
        styles = charts.series_style([run("PPO"), run("PPO", variant="lr-1e-3")])
        self.assertEqual(styles[0].label, "PPO")
        self.assertEqual(styles[1].label, "PPO · lr-1e-3")

    def test_one_style_per_run_in_input_order(self) -> None:
        runs = [run("PPO", seed=1), run("DQN", seed=2), run("PPO", seed=3)]
        styles = charts.series_style(runs)
        self.assertEqual([style.family for style in styles], ["PPO", "DQN", "PPO"])

    def test_unknown_family_falls_back_to_other(self) -> None:
        styles = charts.series_style([run("MYSTERY", family="OTHER")])
        self.assertEqual(styles[0].family, "OTHER")


class TestComparisonCurves(unittest.TestCase):
    def build(self, runs: list[data.RunInfo]):
        styles = charts.series_style(runs)
        return charts.comparison_curves(frame_for(runs, styles), styles).to_dict(), styles

    def test_facet_field_is_the_algorithm(self) -> None:
        spec, _styles = self.build([run("PPO"), run("DQN"), run("REINFORCE")])
        self.assertEqual(spec["facet"]["field"], "算法")
        self.assertEqual(spec["facet"]["type"], "nominal")

    def test_each_algorithm_gets_a_panel(self) -> None:
        runs = [
            run("PPO", seed=1), run("PPO", seed=2), run("PPO", variant="lr-1e-3"),
            run("DQN", seed=4), run("REINFORCE", seed=5),
        ]
        spec, _styles = self.build(runs)
        # 分面的数据被 altair 存进 ``datasets``（键名是内容哈希，不是 ``data-1``），
        # 取值而不是猜键名。
        rows = next(iter(spec["datasets"].values()))
        self.assertEqual({row["算法"] for row in rows}, {"PPO", "DQN", "REINFORCE"})

    def test_y_axis_is_independent_per_panel(self) -> None:
        """REINFORCE 的回报能到 −600，跟 DQN 共用一根轴会把它压成一条直线。"""
        spec, _styles = self.build([run("PPO"), run("REINFORCE")])
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")

    def test_brush_is_a_shared_interval_over_x(self) -> None:
        spec, _styles = self.build([run("PPO"), run("DQN")])
        brushes = [param for param in spec["params"] if param.get("select")]
        self.assertEqual(len(brushes), 1)
        self.assertEqual(brushes[0]["name"], "compare_brush")
        self.assertEqual(brushes[0]["select"]["type"], "interval")
        self.assertEqual(brushes[0]["select"]["encodings"], ["x"])

    def test_dim_colors_are_a_scale_not_a_fixed_value(self) -> None:
        spec, _styles = self.build([run("PPO"), run("DQN")])
        spec_encoding = spec["spec"]["encoding"] if "spec" in spec else spec["encoding"]
        self.assertEqual(
            spec_encoding["opacity"]["condition"]["value"], charts._FOCUS_OPACITY
        )
        self.assertEqual(spec_encoding["opacity"]["value"], charts._DIM_OPACITY)

    def test_line_style_scales_are_given_explicit_domains(self) -> None:
        """显式 domain：中文变体名交给 altair 自己排序，图例顺序会在刷新之间跳动。"""
        runs = [run("PPO"), run("PPO", variant="lr-1e-3"), run("DQN")]
        spec, styles = self.build(runs)
        spec_encoding = spec["spec"]["encoding"]
        labels = [style.label for style in styles]
        for channel in ("color", "strokeDash", "strokeWidth"):
            self.assertEqual(spec_encoding[channel]["scale"]["domain"], labels)

    def test_axis_titles_are_chinese(self) -> None:
        spec, _styles = self.build([run("PPO")])
        spec_encoding = spec["spec"]["encoding"]
        self.assertEqual(spec_encoding["x"]["title"], "训练步数")


class TestJourneyLines(unittest.TestCase):
    def panels(self):
        table = pd.DataFrame({"步数": [0.0, 1.0], "值": [0.0, 1.0]})
        return [("回合回报", table), ("回合长度", table), ("熵", table)]

    def test_every_panel_is_kept_in_order(self) -> None:
        spec = charts.journey_lines(self.panels()).to_dict()
        self.assertEqual(len(spec["vconcat"]), 3)

    def test_panels_share_one_brush(self) -> None:
        spec = charts.journey_lines(self.panels()).to_dict()
        names = {param["name"] for param in spec.get("params", [])}
        self.assertEqual(names, {"journey_brush"})

    def test_y_axis_is_independent_between_panels(self) -> None:
        spec = charts.journey_lines(self.panels()).to_dict()
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")

    def test_only_the_last_panel_names_the_x_axis(self) -> None:
        """重复三遍的轴名只占地方；刻度标签仍然保留，对读时间靠的是它们。"""
        spec = charts.journey_lines(self.panels()).to_dict()
        titles = [panel["encoding"]["x"]["title"] for panel in spec["vconcat"]]
        self.assertEqual(titles, ["", "", "训练步数"])

    def test_panel_titles_are_chinese(self) -> None:
        spec = charts.journey_lines(self.panels()).to_dict()
        first = spec["vconcat"][0]["title"]
        self.assertEqual(first["text"], "回合回报")
        self.assertEqual(first["anchor"], "start")

    def test_empty_panels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            charts.journey_lines([])


class TestBars(unittest.TestCase):
    def test_baseline_rule_is_drawn_on_top_of_the_bars(self) -> None:
        frame = pd.DataFrame({"回合": ["第 1 回", "第 2 回"], "回报": [1.0, 3.0]})
        spec = charts.bars(frame, x_field="回合", y_field="回报", baseline=2.0).to_dict()
        # 两层叠加：柱 + 参考线
        self.assertIn("layer", spec)
        self.assertEqual(len(spec["layer"]), 2)
        self.assertIn("rule", spec["layer"][1]["mark"]["type"])

    def test_baseline_goes_into_its_own_dataset(self) -> None:
        frame = pd.DataFrame({"回合": ["第 1 回"], "回报": [1.0]})
        spec = charts.bars(frame, x_field="回合", y_field="回报", baseline=5.0).to_dict()
        datasets = spec["datasets"]
        self.assertIn(5.0, [row["回报"] for row in datasets[list(datasets)[-1]]])


class TestActionAreas(unittest.TestCase):
    def test_area_is_stacked_so_the_band_sums_to_one(self) -> None:
        frame = pd.DataFrame(
            {
                "步数": [0.0, 0.0, 1.0, 1.0],
                "动作": ["0", "1", "0", "1"],
                "占比": [0.4, 0.6, 0.2, 0.8],
            }
        )
        spec = charts.action_areas(frame).to_dict()
        self.assertEqual(spec["encoding"]["y"]["stack"], "zero")
        self.assertEqual(spec["encoding"]["color"]["field"], "动作")

    def test_discrete_actions_get_a_percent_axis(self) -> None:
        frame = pd.DataFrame({"步数": [0.0], "动作": ["0"], "占比": [1.0]})
        spec = charts.action_areas(frame).to_dict()
        self.assertEqual(spec["encoding"]["y"]["axis"]["format"], "%")


class TestLineWithBand(unittest.TestCase):
    def test_three_layers_mean_band_and_floor(self) -> None:
        frame = pd.DataFrame(
            {"步数": [0.0, 1.0], "均值": [1.0, 2.0], "上界": [2.0, 3.0], "下界": [0.0, 1.0]}
        )
        spec = charts.line_with_band(
            frame, lower="下界", upper="上界", middle="均值"
        ).to_dict()
        self.assertEqual(len(spec["layer"]), 3)

    def test_brush_dims_all_three_layers_together(self) -> None:
        """线暗了带还亮着，视觉上就成了「这一段只有区间有效」，是错的。"""
        import altair as alt

        frame = pd.DataFrame(
            {"步数": [0.0], "均值": [1.0], "上界": [2.0], "下界": [0.0]}
        )
        brush = alt.selection_interval(encodings=["x"], name="b")
        spec = charts.line_with_band(
            frame, lower="下界", upper="上界", middle="均值", brush=brush
        ).to_dict()
        for layer in spec["layer"]:
            self.assertIn("opacity", layer["encoding"])


if __name__ == "__main__":
    unittest.main()
