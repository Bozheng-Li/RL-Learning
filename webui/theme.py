"""视觉层：色板、全局样式与可复用的展示组件。

设计取舍
--------

**为什么是浅色**：页面里嵌的训练曲线、轨迹图、诊断图都是 matplotlib 产出的
白底 PNG。深色外壳会让它们变成一块块刺眼的亮矩形，观感远比"不够酷"更糟。

**为什么用 CSS 而不是纯 Streamlit 主题**：``[theme]`` 只能调五个颜色，做不出
卡片、徽章、间距节奏这些东西。但**只用 ``data-testid`` 与元素选择器**——那些
``.st-emotion-cache-xxxx`` 的类名每次发版都会变，写死它们等于给自己埋雷。

**为什么图内文字仍然是英文**：matplotlib 的默认字体不含 CJK。界面文字走浏览器
渲染，中文没问题；图里的文字改成中文会变成方块。这个分界在项目里是一贯的。
"""

from __future__ import annotations

import html
from typing import Iterable, Literal

import streamlit as st

# --------------------------------------------------------------------------- #
# 色板
# --------------------------------------------------------------------------- #

ACCENT = "#4f46e5"          # 主色：靛蓝
ACCENT_HOVER = "#4338ca"
ACCENT_SOFT = "#eef2ff"
SUCCESS = "#059669"
SUCCESS_SOFT = "#ecfdf5"
WARNING = "#b45309"
WARNING_SOFT = "#fffbeb"
DANGER = "#dc2626"
DANGER_SOFT = "#fef2f2"
MUTED = "#64748b"
BORDER = "#e2e8f0"
SURFACE = "#ffffff"
CANVAS = "#f7f8fa"

#: 状态 -> (前景色, 背景色)，供 ``pill()`` 使用。
STATUS_TONES: dict[str, tuple[str, str]] = {
    "completed": (SUCCESS, SUCCESS_SOFT),
    "running": (ACCENT, ACCENT_SOFT),
    "failed": (DANGER, DANGER_SOFT),
    "not_started": (MUTED, "#f1f5f9"),
}

#: 字体栈。全部是系统字体——服务器可能没有外网，而且中文需要本地字体才能真正
#: 生效。按各平台的默认中文屏显字体排序。
_FONT_SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", '
    '"Hiragino Sans GB", "Microsoft YaHei", "Source Han Sans SC", '
    '"Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif'
)
_FONT_MONO = (
    '"JetBrains Mono", "SF Mono", "Cascadia Code", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)


def _css() -> str:
    return f"""
<style>
:root {{
  --rl-accent: {ACCENT};
  --rl-accent-hover: {ACCENT_HOVER};
  --rl-accent-soft: {ACCENT_SOFT};
  --rl-border: {BORDER};
  --rl-surface: {SURFACE};
  --rl-canvas: {CANVAS};
  --rl-muted: {MUTED};
  --rl-radius: 12px;
  --rl-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 1px 3px rgba(15, 23, 42, .06);
  --rl-shadow-lg: 0 4px 12px rgba(15, 23, 42, .06), 0 2px 4px rgba(15, 23, 42, .04);
}}

/* ---------- 画布与排版 ---------- */
html, body, [class*="css"] {{ font-family: {_FONT_SANS}; }}

[data-testid="stAppViewContainer"] {{ background: var(--rl-canvas); }}

[data-testid="stHeader"] {{ background: transparent; }}

[data-testid="stMainBlockContainer"] {{
  padding-top: 2.4rem;
  padding-bottom: 4rem;
  max-width: 1400px;
}}

h1, h2, h3, h4 {{
  font-family: {_FONT_SANS};
  font-weight: 650;
  letter-spacing: -.015em;
  color: #0f172a;
}}
h1 {{ font-size: 1.9rem; }}
h2 {{ font-size: 1.28rem; margin-top: .4rem; }}
h3 {{ font-size: 1.06rem; }}

a {{ color: var(--rl-accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

code, pre, [data-testid="stCode"] code {{ font-family: {_FONT_MONO}; }}

/* ---------- 侧边栏 ---------- */
[data-testid="stSidebar"] {{
  background: var(--rl-surface);
  border-right: 1px solid var(--rl-border);
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .45rem; }}
[data-testid="stSidebar"] h1 {{
  font-size: 1.12rem;
  margin-bottom: .2rem;
  display: flex;
  align-items: center;
  gap: .5rem;
}}
[data-testid="stSidebar"] h1::before {{
  content: "";
  width: 9px; height: 9px;
  border-radius: 3px;
  background: var(--rl-accent);
  box-shadow: 0 0 0 4px var(--rl-accent-soft);
}}
[data-testid="stSidebar"] hr {{ margin: .7rem 0; border-color: var(--rl-border); }}

/* 侧边栏导航：把 radio 做成导航条目。
   选择器基于 data-testid 与 data-selected —— 这两个是 Streamlit 明确维护的接口；
   那些 st-emotion-cache-xxxx 的哈希类名每次发版都变，绝不能依赖。 */
[data-testid="stSidebar"] [data-testid="stRadioOption"] {{
  padding: .45rem .65rem;
  border-radius: 8px;
  transition: background .13s ease, color .13s ease;
  cursor: pointer;
  width: 100%;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: #f1f5f9; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{
  background: var(--rl-accent-soft);
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{
  color: var(--rl-accent) !important;
  font-weight: 620;
}}
/* 隐藏圆点：它是文字容器的兄弟节点，且本身没有 data-testid。
   导航靠底色与字色区分选中态，比小圆点更清楚。 */
[data-testid="stSidebar"] [data-testid="stRadioOption"]
  div:has(> [data-testid="stMarkdownContainer"])
  > div:not([data-testid="stMarkdownContainer"]) {{
  display: none;
}}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: .12rem; }}

/* ---------- 卡片 ---------- */
.rl-card {{
  background: var(--rl-surface);
  border: 1px solid var(--rl-border);
  border-radius: var(--rl-radius);
  box-shadow: var(--rl-shadow);
  padding: 1rem 1.15rem;
}}

.rl-stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .75rem;
  margin: .2rem 0 1.1rem;
}}
.rl-stat {{
  background: var(--rl-surface);
  border: 1px solid var(--rl-border);
  border-radius: var(--rl-radius);
  box-shadow: var(--rl-shadow);
  padding: .8rem 1rem .9rem;
  position: relative;
  overflow: hidden;
}}
.rl-stat::after {{
  content: "";
  position: absolute; inset: 0 auto 0 0;
  width: 3px;
  background: var(--rl-accent);
  opacity: .85;
}}
.rl-stat-label {{
  font-size: .74rem;
  font-weight: 600;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--rl-muted);
}}
.rl-stat-value {{
  font-size: 1.65rem;
  font-weight: 680;
  line-height: 1.25;
  color: #0f172a;
  font-variant-numeric: tabular-nums;
}}
.rl-stat-hint {{ font-size: .72rem; color: var(--rl-muted); }}

/* ---------- 页头 ---------- */
.rl-header {{ margin-bottom: 1.1rem; }}
.rl-header-title {{
  font-size: 1.6rem;
  font-weight: 680;
  letter-spacing: -.02em;
  color: #0f172a;
  line-height: 1.3;
}}
.rl-header-sub {{ font-size: .86rem; color: var(--rl-muted); margin-top: .18rem; }}

/* ---------- 徽章 ---------- */
.rl-pill {{
  display: inline-flex;
  align-items: center;
  gap: .3rem;
  padding: .16rem .55rem;
  border-radius: 999px;
  font-size: .74rem;
  font-weight: 600;
  line-height: 1.5;
  white-space: nowrap;
}}
.rl-pill-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
.rl-meta {{ display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; }}
.rl-meta-key {{ color: var(--rl-muted); font-size: .78rem; }}

/* ---------- 表格 ---------- */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--rl-border);
  border-radius: var(--rl-radius);
  overflow: hidden;
  box-shadow: var(--rl-shadow);
}}
[data-testid="stDataFrame"] [role="columnheader"] {{
  background: #f8fafc !important;
  font-weight: 600 !important;
}}

/* ---------- 图表容器 ---------- */
[data-testid="stImage"] img {{ border-radius: 10px; }}
[data-testid="stVegaLiteChart"] {{
  background: var(--rl-surface);
  border: 1px solid var(--rl-border);
  border-radius: var(--rl-radius);
  padding: .55rem .3rem;
  box-shadow: var(--rl-shadow);
}}

/* ---------- 控件 ---------- */
.stButton > button, .stDownloadButton > button {{
  border-radius: 9px;
  border: 1px solid var(--rl-border);
  font-weight: 550;
  transition: all .13s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--rl-accent);
  color: var(--rl-accent);
}}
.stButton > button[kind="primary"] {{
  background: var(--rl-accent);
  border-color: var(--rl-accent);
  box-shadow: var(--rl-shadow);
}}
.stButton > button[kind="primary"]:hover {{
  background: var(--rl-accent-hover);
  border-color: var(--rl-accent-hover);
  color: #fff;
}}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
  border-radius: 9px;
}}

/* ---------- 标签页 ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: .15rem;
  border-bottom: 1px solid var(--rl-border);
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
  border-radius: 8px 8px 0 0;
  padding: .42rem .85rem;
  font-weight: 550;
}}
[data-testid="stTabs"] [aria-selected="true"] {{ color: var(--rl-accent); }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--rl-accent); }}

/* ---------- 提示条 ---------- */
[data-testid="stAlert"] {{
  border-radius: 10px;
  border: 1px solid var(--rl-border);
}}

/* ---------- 进度条 ---------- */
[data-testid="stProgress"] > div > div > div > div {{
  background: linear-gradient(90deg, {ACCENT}, #818cf8);
}}

/* ---------- 分隔线 ---------- */
hr {{ border-color: var(--rl-border); margin: 1.4rem 0; }}

/* ---------- 收起 Streamlit 自带页脚 ---------- */
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
</style>
"""


def inject() -> None:
    """注入全局样式。每次脚本重跑调一次即可（元素会随重跑被替换，不会累积）。"""
    st.markdown(_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 展示组件
# --------------------------------------------------------------------------- #


def page_header(title: str, subtitle: str = "") -> None:
    """统一的页头。"""
    parts = [f'<div class="rl-header-title">{html.escape(title)}</div>']
    if subtitle:
        parts.append(f'<div class="rl-header-sub">{html.escape(subtitle)}</div>')
    st.markdown(f'<div class="rl-header">{"".join(parts)}</div>', unsafe_allow_html=True)


def stat_row(items: Iterable[tuple[str, object, str]]) -> None:
    """一排统计卡片。

    Args:
        items: ``(标签, 数值, 补充说明)`` 三元组；补充说明可为空串。
    """
    cards = []
    for label, value, hint in items:
        hint_html = f'<div class="rl-stat-hint">{html.escape(str(hint))}</div>' if hint else ""
        cards.append(
            '<div class="rl-stat">'
            f'<div class="rl-stat-label">{html.escape(str(label))}</div>'
            f'<div class="rl-stat-value">{html.escape(str(value))}</div>'
            f"{hint_html}</div>"
        )
    st.markdown(f'<div class="rl-stats">{"".join(cards)}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    """返回一个徽章的 HTML 片段（供拼接进 ``markdown`` 使用，本身不渲染）。

    ``tone`` 可以是 :data:`STATUS_TONES` 的键（``completed`` / ``running`` /
    ``failed`` / ``not_started``），也可以是 ``neutral``。
    """
    if tone in STATUS_TONES:
        foreground, background = STATUS_TONES[tone]
    else:
        foreground, background = MUTED, "#f1f5f9"
    return (
        f'<span class="rl-pill" style="color:{foreground};background:{background}">'
        f'<span class="rl-pill-dot"></span>{html.escape(text)}</span>'
    )


def meta_row(pairs: Iterable[tuple[str, str]]) -> None:
    """一行元信息：键用浅色小字，值用徽章。"""
    chunks = []
    for key, value in pairs:
        chunks.append(f'<span class="rl-meta-key">{html.escape(key)}</span>')
        chunks.append(
            '<span class="rl-pill" style="color:#334155;background:#f1f5f9">'
            f"{html.escape(value)}</span>"
        )
    st.markdown(f'<div class="rl-meta">{"".join(chunks)}</div>', unsafe_allow_html=True)


def card(body_html: str) -> None:
    """把一段 HTML 包进卡片容器。"""
    st.markdown(f'<div class="rl-card">{body_html}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    """比 ``st.caption`` 略大、颜色更柔的说明文字，用于段落级注解。"""
    st.markdown(
        f'<div style="color:{MUTED};font-size:.82rem;line-height:1.65;'
        f'margin:.35rem 0 .7rem">{text}</div>',
        unsafe_allow_html=True,
    )


def status_tone(status: str) -> Literal["completed", "running", "failed", "not_started"]:
    """把运行状态映射到徽章色调。"""
    if status in STATUS_TONES:
        return status  # type: ignore[return-value]
    return "not_started"


def fit_height(rows: int, *, max_height: int = 620, row: int = 35, header: int = 42) -> int:
    """给 ``st.dataframe`` 算一个刚好放下内容的高度。

    不给高度时 Streamlit 会用一个默认值，行数一多就从中间截断，看起来像渲染坏了。
    """
    return min(max_height, header + row * max(rows, 1))
