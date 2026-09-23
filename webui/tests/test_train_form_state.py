"""「发起训练」页换环境配置时的表单状态——用 ``AppTest`` 跑真页面。

这个文件存在的唯一理由是一次实测出来的 bug：从 acrobot（10 万步）切到 lunarlander
（20 万步）之后，服务端当次算的、右栏预览显示的**都是 20 万**，但页面上的输入框里
还写着 10 万；下一次任何原因的 rerun，前端把这个旧值原样发回来，服务端又按 10 万算。

原因是 Streamlit 的双向绑定：把 ``train_timesteps`` 从 ``session_state`` 里删掉只能
清掉**服务端**的值，前端控件自己还记着，只要它还在原地、还带着同一个 key，重新渲染
时就会把旧值发回来。修法是让「跟着配置走」的控件 key 带配置名后缀——新的配置是新的
控件，前端没有旧值可发。这个行为只有真的跑一遍页面才看得见，所以这里的测试用
``AppTest`` 而不是纯函数（每个用例两三秒）。
"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

APP = "/home/libaizheng/Reinforce/webui/app.py"


def open_train_page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["page"] = "发起训练"
    at.run()
    assert not at.exception, at.exception  # 页面上任何一处未捕获的异常都不该出现
    return at


def timesteps_for(at: AppTest, config_name: str) -> int:
    return int(at.number_input(key=f"train_timesteps::{config_name}").value)


class TestConfigSwitchResetsTheForm(unittest.TestCase):
    def test_step_count_follows_the_new_config(self) -> None:
        """换环境后步数必须是新配置自己的，而不是上一个配置留在框里的。"""
        at = open_train_page()
        self.assertEqual(at.selectbox(key="train_config").value, "acrobot")
        self.assertEqual(timesteps_for(at, "acrobot"), 100_000)

        at.selectbox(key="train_config").set_value("lunarlander")
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(timesteps_for(at, "lunarlander"), 200_000)

    def test_the_old_widget_is_gone_not_rebound(self) -> None:
        """不是把旧值改成新值，而是换了一个控件——这正是挡住前端回发的那一步。"""
        at = open_train_page()
        at.selectbox(key="train_config").set_value("lunarlander")
        at.run()
        with self.assertRaises(KeyError):
            at.number_input(key="train_timesteps::acrobot")  # type: ignore[attr-defined]

    def test_the_other_configs_edits_do_not_leak(self) -> None:
        """在 acrobot 上改过步数之后切到 lunarlander，看到的必须是 20 万。

        这里不断言「切回去还能看到 12,345」：Streamlit 会把这一轮没有渲染的控件状态
        清掉，而这是好事——每个环境回到自己的配置默认值，比带着另一个环境的手填值
        更安全。要钉住的是**不串味**。
        """
        at = open_train_page()
        at.number_input(key="train_timesteps::acrobot").set_value(12_345)
        at.run()
        at.selectbox(key="train_config").set_value("lunarlander")
        at.run()
        self.assertEqual(timesteps_for(at, "lunarlander"), 200_000)


class TestComparePagePreset(unittest.TestCase):
    """「补齐对照」把缺口算法预填进训练页，键也跟着配置走。"""

    def test_preset_lands_on_the_new_key(self) -> None:
        at = AppTest.from_file(APP, default_timeout=240)
        at.session_state["page"] = "发起训练"
        at.session_state["_nav_presets"] = {
            "train_config": "lunarlander",
            "train_algorithms::lunarlander": ["TD3", "SAC"],
        }
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.selectbox(key="train_config").value, "lunarlander")
        self.assertEqual(
            list(at.multiselect(key="train_algorithms::lunarlander").value), ["TD3", "SAC"]
        )


if __name__ == "__main__":
    unittest.main()
