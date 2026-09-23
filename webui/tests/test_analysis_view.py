"""``webui.views.analysis`` 的标记层测试——只查 HTML 结构，不跑整页。

整页的渲染由 ``AppTest`` 覆盖（慢，20 秒量级），这里钉住的是**能靠读字符串就查出来**
的错误。写这个文件是因为真踩过一次：``_variant_card`` 里写了

    "".join(f'<div class="rl-algo-row">{row}</div>' for row in rows)

而 ``rows`` 已经是一个拼好的字符串，于是 ``for row in rows`` 按**字符**迭代，
13 行小卡片变成近十万字符的垃圾标记，页面上什么都不显示——但 Streamlit 不报错、
``AppTest.exception`` 也是 0。断言「卡片里没有单字符的行」就能在 0.1 秒内抓住它。
"""

from __future__ import annotations

import re
import unittest

from webui import analysis
from webui.views import analysis as view

ROW = re.compile(r'<div class="rl-algo-row">(.*?)</div>', re.S)


def effect(algorithm: str, mean_delta: float, **kwargs: object) -> analysis.Effect:
    return analysis.Effect(
        variant="lr-1e-3", algorithm=algorithm, n=3,
        mean_delta=mean_delta, sd_delta=1.0, z=kwargs.pop("z", 5.0), **kwargs,  # type: ignore[arg-type]
    )


class TestDeltaRow(unittest.TestCase):
    def test_returns_exactly_one_row(self) -> None:
        html = view._delta_row(effect("PPO", -20.0), {"PPO": 10.0})
        self.assertTrue(html.startswith('<div class="rl-algo-row">'))
        self.assertTrue(html.endswith("</div>"))
        self.assertEqual(len(ROW.findall(html)), 1)

    def test_algorithm_name_is_escaped(self) -> None:
        html = view._delta_row(effect("SB3-PPO", 5.0), {"SB3-PPO": 10.0})
        self.assertIn('<span class="rl-algo-name">SB3-PPO</span>', html)

    def test_significant_effect_is_colored(self) -> None:
        html = view._delta_row(effect("PPO", -20.0), {"PPO": 10.0})
        self.assertIn("background:#bf3c48", html)  # theme.DANGER

    def test_insignificant_effect_is_muted(self) -> None:
        html = view._delta_row(effect("PPO", -1.0, z=0.2), {"PPO": 10.0})
        self.assertIn(f"background:{view.theme.MUTED}", html)

    def test_missing_scale_still_renders(self) -> None:
        """σ 估不出来（该算法没有安慰剂配对）时不能整行消失。"""
        html = view._delta_row(effect("PPO", -20.0, z=None), {})
        self.assertEqual(len(ROW.findall(html)), 1)

    def test_partial_cross_model_shows_the_count(self) -> None:
        html = view._delta_row(effect("PPO", -20.0, cross_model=1), {"PPO": 10.0})
        self.assertIn("1/3 跨型号", html)

    def test_all_cross_model_warns_that_the_magnitude_is_untrustworthy(self) -> None:
        """全部配对都跨型号时，Δ 的绝对值不能信——只有方向还能看，必须显式提示。"""
        html = view._delta_row(effect("PPO", -20.0, cross_model=3), {"PPO": 10.0})
        self.assertIn("⚠跨型号", html)


class TestVariantCard(unittest.TestCase):
    def make_summary(self, count: int = 3) -> analysis.VariantSummary:
        items = [
            effect(f"ALGO{index}", -float(index + 1))
            for index in range(count)
        ]
        summary = analysis.VariantSummary(
            variant="lr-1e-3", dimension="学习超参",
            algorithms=count, negative=count, significant=count, harmful=count,
            mean_delta=-2.0, sign_p=0.031, effects=items,
        )
        return summary

    def render(self, summary: analysis.VariantSummary) -> str:
        """拦下 ``st.markdown`` 的调用，把 HTML 拿回来。"""
        captured: list[str] = []
        original = view.st.markdown
        view.st.markdown = lambda body, **kwargs: captured.append(body)  # type: ignore[assignment]
        try:
            view._variant_card(summary, {item.algorithm: 1.0 for item in summary.effects})
        finally:
            view.st.markdown = original  # type: ignore[assignment]
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_one_row_per_effect(self) -> None:
        html = self.render(self.make_summary(13))
        self.assertEqual(len(ROW.findall(html)), 13)

    def test_rows_are_not_iterated_character_by_character(self) -> None:
        """回归：把已拼好的字符串再 ``for ... in`` 一遍会产生上千个单字符行。"""
        html = self.render(self.make_summary(13))
        bodies = ROW.findall(html)
        self.assertTrue(all(len(body) > 10 for body in bodies),
                        [body for body in bodies if len(body) <= 10][:5])
        self.assertLess(len(html), 20_000)

    def test_verdict_pill_is_present(self) -> None:
        html = self.render(self.make_summary())
        self.assertIn("一致有害", html)
        self.assertIn("lr-1e-3", html)


if __name__ == "__main__":
    unittest.main()
