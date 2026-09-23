"""跨页跳转与跨页预填的唯一登记入口。

页面之间互相跳转看起来是「写一下 ``st.session_state["page"]`` 就行」，实际不行：
侧边栏的导航 radio 用的就是 ``key="page"``，而它在每个页面的 ``render()`` **之前**
就已经实例化。Streamlit 对已实例化的控件键是只读的，再写会直接抛
``StreamlitWidgetAlreadyInstantiatedError``——页面白屏，只剩下一段 traceback。

所以跳转分两步：页面里的按钮调 ``goto()`` 登记意图（此时不碰 ``page``），
``app.main()`` 在下一次重跑的最开头、radio 实例化之前 ``consume()`` 掉它。
调用方登记完要自己 ``st.rerun()``，否则要等下一次交互才生效。

``preset()`` / ``take_preset()`` 解决同一类问题的另一半：想把「对比页算出来的缺口
算法」预填进发起训练页的表单。直接写 ``st.session_state["train_algorithms::lunarlander"]``
会撞上两条规则——控件已有值时 ``default`` 会被忽略（只在日志里留一句 warning），而
``default`` 与 session_state 同时给还会再报一次。这里的做法是：预填值单独存一份，
表单页取走它，取走时顺手把控件键删掉，这样 ``default`` 才真正生效。

预填值的键就是控件的 ``key``，所以「发起训练」页带配置名后缀的那几个键在这里也带
后缀（``train_algorithms::<配置名>``）——预填进哪个配置的表单，得写对是哪一个。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

#: 待处理跳转的登记键。带下划线前缀，避免与页面自己的 session_state 键撞名。
NAV_KEY = "_nav_target"

#: 待消费的表单预填值。结构是 ``{控件键: 值}``。
PRESET_KEY = "_nav_presets"


def goto(page: str) -> None:
    """登记「下一次重跑时切到 ``page``」。调用方随后要 ``st.rerun()``。"""
    st.session_state[NAV_KEY] = page


def consume(valid_pages: object) -> None:
    """把登记好的跳转落到 ``page`` 上。必须在导航控件实例化**之前**调用。

    ``valid_pages`` 传可迭代的合法页名：页面名写错时宁可什么都不做，也不要让
    侧栏 radio 收到一个不在选项里的值。
    """
    target = st.session_state.pop(NAV_KEY, None)
    if target is not None and target in valid_pages:
        st.session_state["page"] = target


def preset(**values: Any) -> None:
    """登记若干「下一次进这个页面时用它当默认值」的表单值，键是控件的 ``key``。"""
    pending = dict(st.session_state.get(PRESET_KEY) or {})
    pending.update(values)
    st.session_state[PRESET_KEY] = pending


def take_preset(key: str) -> Any | None:
    """取走针对 ``key`` 的预填值（取一次就没了），并清掉该控件的旧值。

    清旧值是关键：控件键一旦有值，``default`` / ``index`` 会被忽略，预填就白填了。
    这里删的是**尚未实例化**的键，在 Streamlit 的规则内是允许的。
    """
    pending = dict(st.session_state.get(PRESET_KEY) or {})
    value = pending.pop(key, None)
    if pending:
        st.session_state[PRESET_KEY] = pending
    else:
        st.session_state.pop(PRESET_KEY, None)
    if value is not None:
        st.session_state.pop(key, None)
    return value
