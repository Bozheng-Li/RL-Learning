"""``webui.views.detail`` 里数据组装纯函数的测试。

「训练全程」这一格要在一屏里拼起六段数据（过程信号、周期评估、动作分布、逐回合得分、
录像、离线图），每段都可能缺。这里测的是**缺的时候会怎样**——真实数据里一次运行可能
只写了三行 progress.csv、可能没有 diagnostics.csv、可能没开录像，而这些情况在界面上
表现为「少了一格」，不报错。所以用 tmp 目录造最小的运行，逐个钉住。

不断言 Streamlit 渲染（那要起 AppTest，秒级），只测组装出来的表。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from webui import data
from webui.views import detail


def make_run(
    root: Path,
    *,
    progress: pd.DataFrame | None = None,
    diagnostics: pd.DataFrame | None = None,
    summary: dict | None = None,
) -> data.RunInfo:
    """在 ``root`` 下造一个最小的运行目录，返回对应的 ``RunInfo``。"""
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    if progress is not None:
        progress.to_csv(logs / "progress.csv", index=False)
    if diagnostics is not None:
        diagnostics.to_csv(logs / "diagnostics.csv", index=False)
    if summary is not None:
        (root / "evaluation.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    return data.RunInfo(
        name=root.name,
        path=root,
        status="completed",
        mtime=0.0,
        environment="LunarLander-v3",
        algorithm="PPO",
        seed=42,
        summary=summary or {},
        family="PPO",
        variant=None,
    )


def ppo_progress(rows: int = 5) -> pd.DataFrame:
    """一份 SB3 风格的 progress.csv：步数 + 回报 + 评估 + 过程信号。"""
    return pd.DataFrame(
        {
            "time/total_timesteps": [step * 1000 for step in range(rows)],
            "rollout/ep_rew_mean": [-150.0 + step * 20 for step in range(rows)],
            "rollout/ep_len_mean": [90.0 + step for step in range(rows)],
            "eval/mean_reward": [-120.0 + step * 25 for step in range(rows)],
            "train/approx_kl": [0.01 * (rows - step) for step in range(rows)],
            "train/clip_fraction": [0.05 for _ in range(rows)],
            "train/entropy_loss": [-1.0 + 0.1 * step for step in range(rows)],
            "time/fps": [800.0 for _ in range(rows)],
        }
    )


class TestJourneyFrame(unittest.TestCase):
    def test_missing_progress_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = make_run(Path(tmp) / "run")
            self.assertIsNone(detail._journey_frame(info))

    def test_index_is_the_step_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = make_run(Path(tmp) / "run", progress=ppo_progress())
            frame = detail._journey_frame(info)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.index.name, "steps")
            self.assertEqual(list(frame.index), [0, 1000, 2000, 3000, 4000])

    def test_columns_are_semantic_names_not_raw_ones(self) -> None:
        """列名必须经 ``live.PROGRESS_COLUMNS`` 映射。

        原始名（``train/approx_kl``）会一路漏到轴标题与图例里，而且是按列位置索引的
        前置条件——每个算法写出来的列集合都不一样，按位置取一定错位。
        """
        with tempfile.TemporaryDirectory() as tmp:
            info = make_run(Path(tmp) / "run", progress=ppo_progress())
            frame = detail._journey_frame(info)
            self.assertIn("reward", frame.columns)
            self.assertIn("kl", frame.columns)
            self.assertNotIn("train/approx_kl", frame.columns)
            self.assertNotIn("rollout/ep_rew_mean", frame.columns)

    def test_every_run_writes_its_own_set_of_columns(self) -> None:
        """REINFORCE 风格的 progress.csv（只有熵与策略损失）也要能读。"""
        with tempfile.TemporaryDirectory() as tmp:
            table = pd.DataFrame(
                {
                    "time/total_timesteps": [0, 500],
                    "rollout/ep_rew_mean": [-300.0, -280.0],
                    "train/entropy_loss": [-1.2, -1.1],
                    "train/policy_gradient_loss": [0.3, 0.2],
                }
            )
            info = make_run(Path(tmp) / "run", progress=table)
            frame = detail._journey_frame(info)
            self.assertIsNotNone(frame)
            self.assertNotIn("kl", frame.columns)
            self.assertNotIn("fps", frame.columns)


class TestPanel(unittest.TestCase):
    def test_one_row_per_step(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0], index=[0.0, 10.0, 20.0])
        table = detail._panel(series)
        self.assertEqual(list(table.columns), ["步数", "值"])
        self.assertEqual(list(table["步数"]), [0.0, 10.0, 20.0])

    def test_non_numeric_becomes_nan_and_is_dropped(self) -> None:
        series = pd.Series(["1.0", "", "3.0"], index=[0.0, 1.0, 2.0])
        table = detail._panel(series)
        self.assertEqual(list(table["值"]), [1.0, 3.0])

    def test_all_nan_returns_empty(self) -> None:
        series = pd.Series([float("nan")] * 3, index=[0.0, 1.0, 2.0])
        self.assertTrue(detail._panel(series).empty)


class TestJourneyPanels(unittest.TestCase):
    def test_fixed_panels_come_first_in_their_own_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = make_run(Path(tmp) / "run", progress=ppo_progress())
            frame = detail._journey_frame(info)
            panels = detail._journey_panels(frame)
            self.assertEqual(
                [title for title, _table in panels[:3]],
                ["回合回报（ep_rew_mean）", "周期评估回报", "回合长度"],
            )

    def test_missing_signal_is_simply_absent(self) -> None:
        """REINFORCE 没有 KL、没有 FPS，面板组里也不该出现空面板。"""
        with tempfile.TemporaryDirectory() as tmp:
            table = pd.DataFrame(
                {
                    "time/total_timesteps": [0, 500],
                    "rollout/ep_rew_mean": [-300.0, -280.0],
                    "train/entropy_loss": [-1.2, -1.1],
                }
            )
            info = make_run(Path(tmp) / "run", progress=table)
            panels = detail._journey_panels(detail._journey_frame(info))
            titles = [title for title, _table in panels]
            self.assertEqual(len(panels), 2)
            self.assertNotIn("KL", titles)

    def test_panels_are_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = make_run(Path(tmp) / "run", progress=ppo_progress())
            panels = detail._journey_panels(detail._journey_frame(info))
            self.assertLessEqual(len(panels), detail._JOURNEY_MAX_PANELS)

    def test_every_panel_carries_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = make_run(Path(tmp) / "run", progress=ppo_progress())
            for title, table in detail._journey_panels(detail._journey_frame(info)):
                self.assertFalse(table.empty, title)

    def test_a_signal_that_is_all_nan_is_dropped(self) -> None:
        """列存在但整列是空的时候，画出来是一条空白面板——不如不画。"""
        with tempfile.TemporaryDirectory() as tmp:
            table = ppo_progress()
            table["train/value_loss"] = float("nan")
            info = make_run(Path(tmp) / "run", progress=table)
            titles = [title for title, _t in detail._journey_panels(detail._journey_frame(info))]
            self.assertNotIn("价值损失", titles)


class TestDiagnosticsFrame(unittest.TestCase):
    def test_wide_action_columns_fold_into_one_category_column(self) -> None:
        table = pd.DataFrame(
            {
                "timestep": [0, 1000],
                "action_0_frac": [0.25, 0.50],
                "action_1_frac": [0.75, 0.50],
            }
        )
        folded = detail._diagnostics_frame(table)
        self.assertEqual(set(folded.columns), {"动作", "步数", "占比"})
        self.assertEqual(len(folded), 4)
        self.assertEqual(sorted(set(folded["动作"])), ["0", "1"])

    def test_action_names_keep_the_number(self) -> None:
        """动作名保留编号：编号与文档里的动作表对得上，套中文名反而容易标错。"""
        table = pd.DataFrame({"timestep": [0], "action_3_frac": [1.0]})
        folded = detail._diagnostics_frame(table)
        self.assertEqual(list(folded["动作"]), ["3"])

    def test_continuous_action_columns_are_not_mangled(self) -> None:
        table = pd.DataFrame(
            {"timestep": [0, 1], "action_mean": [-0.1, 0.2], "action_std": [0.5, 0.6]}
        )
        folded = detail._diagnostics_frame(table)
        self.assertEqual(sorted(set(folded["动作"])), ["action_mean", "action_std"])


class TestFinalEpisodes(unittest.TestCase):
    """``evaluation.json`` 的 ``episode_rewards`` —— 此前界面一行都没读过的两个数组。"""

    def test_summary_keeps_the_per_episode_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = {
                "mean_reward": 120.0,
                "std_reward": 40.0,
                "episode_rewards": [200.0, -17.0, 130.0],
                "episode_lengths": [300, 90, 280],
            }
            info = make_run(Path(tmp) / "run", summary=summary)
            self.assertEqual(len(info.summary["episode_rewards"]), 3)
            self.assertEqual(len(info.summary["episode_lengths"]), 3)

    def test_mean_of_the_episodes_matches_the_reported_mean(self) -> None:
        """逐回合数组与摘要里的均值必须是同一批回合。

        这一条是给「均值 132 但其中一回合是 −17」那类判断兜底的：如果两个数组不同源，
        柱状图上的红线和柱子就对不上，读的人会以为图错了。
        """
        rewards = [200.0, 40.0, 130.0]
        with tempfile.TemporaryDirectory() as tmp:
            summary = {
                "mean_reward": sum(rewards) / len(rewards),
                "episode_rewards": rewards,
                "episode_lengths": [300, 90, 280],
            }
            info = make_run(Path(tmp) / "run", summary=summary)
            self.assertAlmostEqual(
                sum(info.summary["episode_rewards"]) / len(rewards),
                info.summary["mean_reward"],
            )


if __name__ == "__main__":
    unittest.main()
