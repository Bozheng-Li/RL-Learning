"""Reinforce WebUI 视觉系统。

设计约束：
- 浅色外壳。训练曲线与轨迹图都是白底 PNG，深色外壳会把它们变成一块块刺眼的亮矩形。
- 图内文字只用英文（matplotlib 默认字体不含 CJK），中文只出现在 Streamlit 的标题、
  标签与说明里。
- 所有 HTML 片段都走 ``html.escape``，调用方传进来的运行名、环境 id 都不可信。
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Literal

import streamlit as st

ACCENT = "#e4572e"
ACCENT_HOVER = "#c9421d"
ACCENT_SOFT = "#fff1eb"
INK = "#182126"
MUTED = "#69757b"
SUBTLE = "#eef1f0"
BORDER = "#dfe5e2"
SURFACE = "#ffffff"
CANVAS = "#f4f6f3"
SUCCESS = "#16806a"
SUCCESS_SOFT = "#e7f5ef"
WARNING = "#a66a16"
WARNING_SOFT = "#fff6df"
DANGER = "#bf3c48"
DANGER_SOFT = "#fff0f1"

#: 算法族色。对比图、环境卡上的算法条都用同一套，保证跨页面认得出「这是同一个算法」。
FAMILY_COLORS: dict[str, str] = {
    "REINFORCE": "#e4572e",
    "A2C": "#d97706",
    "PPO": "#2563eb",
    "DQN": "#7c3aed",
    "SAC": "#0f9d8a",
    "TD3": "#db2777",
    "OTHER": "#69757b",
}

STATUS_TONES: dict[str, tuple[str, str]] = {
    "completed": (SUCCESS, SUCCESS_SOFT),
    "running": (ACCENT, ACCENT_SOFT),
    "failed": (DANGER, DANGER_SOFT),
    "not_started": (MUTED, SUBTLE),
}

_FONT_SANS = ('-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", '
              '"Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif')
_FONT_MONO = '"JetBrains Mono", "SF Mono", "Cascadia Code", Menlo, Consolas, monospace'


def family_color(family: str) -> str:
    """算法族对应的色值；未知族退回中性灰。"""
    return FAMILY_COLORS.get(family, FAMILY_COLORS["OTHER"])


def _css() -> str:
    return f"""
<style>
:root {{ --rl-accent:{ACCENT}; --rl-accent-hover:{ACCENT_HOVER}; --rl-accent-soft:{ACCENT_SOFT};
  --rl-ink:{INK}; --rl-muted:{MUTED}; --rl-subtle:{SUBTLE}; --rl-border:{BORDER};
  --rl-surface:{SURFACE}; --rl-canvas:{CANVAS}; --rl-radius:10px;
  --rl-shadow:0 1px 2px rgba(24,33,38,.04),0 6px 20px rgba(24,33,38,.045); }}
html,body,[class*="css"] {{ font-family:{_FONT_SANS}; color:var(--rl-ink); }}
[data-testid="stAppViewContainer"] {{ background:var(--rl-canvas); }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stMainBlockContainer"] {{ max-width:1560px; padding:2rem 3.2rem 4.5rem; }}
h1,h2,h3,h4 {{ color:var(--rl-ink); font-weight:690; letter-spacing:-.02em; }}
h1 {{ font-size:2rem; line-height:1.12; }} h2 {{ font-size:1.2rem; }} h3 {{ font-size:1rem; }}
code,pre,[data-testid="stCode"] code {{ font-family:{_FONT_MONO}; }}

/* ---- 侧边栏 ---- */
[data-testid="stSidebar"] {{ background:#fbfcfa; border-right:1px solid var(--rl-border); }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap:.42rem; }}
[data-testid="stSidebar"] hr {{ border-color:var(--rl-border); margin:.85rem 0; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ padding:.5rem .68rem; border-radius:8px; cursor:pointer; width:100%; border:1px solid transparent; transition:background .15s ease,border .15s ease; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background:#f1f4f2; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background:var(--rl-accent-soft); border-color:#ffd9cb; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color:var(--rl-accent)!important; font-weight:680; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] div:has(>[data-testid="stMarkdownContainer"])>div:not([data-testid="stMarkdownContainer"]) {{ display:none; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:.14rem; }}
.rl-brand {{ padding:.4rem .1rem .7rem; }}
.rl-brand-mark {{ display:flex; align-items:center; gap:.55rem; color:var(--rl-ink); font-weight:760; font-size:1.05rem; letter-spacing:-.02em; }}
.rl-brand-mark::before {{ content:""; display:block; width:11px; height:11px; border-radius:3px; background:var(--rl-accent); box-shadow:6px 0 0 #f4a261; }}
.rl-brand-sub {{ color:var(--rl-muted); font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; margin:.4rem 0 0 1.05rem; }}
.rl-nav-label {{ color:#98a29f; font-size:.64rem; font-weight:760; letter-spacing:.14em; text-transform:uppercase; margin:.7rem .3rem .05rem; }}
.rl-sidebar-note {{ border:1px solid var(--rl-border); background:var(--rl-surface); border-radius:8px; padding:.7rem .78rem; margin-top:.2rem; }}
.rl-sidebar-note strong {{ display:block; color:var(--rl-ink); font-size:.8rem; margin-bottom:.16rem; }} .rl-sidebar-note span {{ display:block; color:var(--rl-muted); font-size:.72rem; line-height:1.45; }}

/* ---- 页头 ---- */
.rl-topbar {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin:.05rem 0 1.5rem; }}
.rl-eyebrow {{ color:var(--rl-accent); font-size:.68rem; font-weight:770; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.42rem; }}
.rl-header-title {{ color:var(--rl-ink); font-size:1.85rem; font-weight:750; letter-spacing:-.035em; line-height:1.08; }}
.rl-header-sub {{ color:var(--rl-muted); font-size:.85rem; line-height:1.55; margin-top:.4rem; max-width:46rem; }}
.rl-header-mark {{ width:38px; height:38px; border:1px solid #ffd7c8; border-radius:9px; background:var(--rl-accent-soft); position:relative; flex:0 0 auto; }}
.rl-header-mark::before,.rl-header-mark::after {{ content:""; position:absolute; background:var(--rl-accent); border-radius:2px; }} .rl-header-mark::before {{ width:16px; height:3px; left:10px; top:12px; box-shadow:0 7px 0 var(--rl-accent),0 14px 0 #f4a261; }} .rl-header-mark::after {{ width:3px; height:22px; left:10px; top:9px; opacity:.18; }}

/* ---- KPI ---- */
.rl-stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; margin:0 0 1.4rem; }}
.rl-stat {{ background:var(--rl-surface); border:1px solid var(--rl-border); border-radius:var(--rl-radius); padding:.85rem 1rem .95rem; box-shadow:var(--rl-shadow); position:relative; overflow:hidden; }} .rl-stat::after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:2px; background:var(--rl-accent); opacity:.85; }}
.rl-stat-label {{ color:var(--rl-muted); font-size:.68rem; font-weight:760; letter-spacing:.1em; text-transform:uppercase; }} .rl-stat-value {{ color:var(--rl-ink); font-size:1.65rem; font-weight:760; line-height:1.15; margin:.3rem 0 .18rem; font-variant-numeric:tabular-nums; letter-spacing:-.02em; white-space:nowrap; }} .rl-stat-hint {{ color:var(--rl-muted); font-size:.72rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

/* ---- 分段 ---- */
.rl-section-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:1rem; margin:1.5rem 0 .55rem; }} .rl-section-title {{ color:var(--rl-ink); font-size:1.02rem; font-weight:720; letter-spacing:-.01em; }} .rl-section-note {{ color:var(--rl-muted); font-size:.74rem; }}

/* ---- 卡片 ---- */
.rl-card {{ background:var(--rl-surface); border:1px solid var(--rl-border); border-radius:var(--rl-radius); box-shadow:var(--rl-shadow); padding:1rem 1.15rem; }}
.rl-empty {{ background:var(--rl-surface); border:1px dashed #cdd5d1; border-radius:var(--rl-radius); padding:1.6rem 1.4rem; text-align:center; }}
.rl-empty strong {{ display:block; color:var(--rl-ink); font-size:.95rem; margin-bottom:.3rem; }} .rl-empty span {{ color:var(--rl-muted); font-size:.82rem; line-height:1.55; }}

/* ---- 环境卡 ---- */
.rl-env-card {{ background:var(--rl-surface); border:1px solid var(--rl-border); border-radius:var(--rl-radius); box-shadow:var(--rl-shadow); padding:.95rem 1.05rem 1.05rem; height:100%; display:flex; flex-direction:column; }}
.rl-env-head {{ display:flex; align-items:center; justify-content:space-between; gap:.6rem; margin-bottom:.15rem; }}
.rl-env-name {{ color:var(--rl-ink); font-weight:720; font-size:.98rem; letter-spacing:-.015em; }}
.rl-env-sub {{ color:var(--rl-muted); font-size:.72rem; margin:.1rem 0 .7rem; }}
.rl-algo-row {{ display:flex; align-items:center; gap:.5rem; margin:.3rem 0; }}
.rl-algo-name {{ width:6.4rem; flex:0 0 auto; font-size:.76rem; font-weight:650; color:var(--rl-ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.rl-algo-track {{ flex:1 1 auto; height:7px; background:#eef1f0; border-radius:99px; overflow:hidden; position:relative; }}
.rl-algo-bar {{ height:100%; border-radius:99px; position:absolute; left:0; top:0; }}
/* 多种子时的离散度：画一条半透明的范围带，压在实心条上。 */
.rl-algo-range {{ position:absolute; top:0; height:100%; background:currentColor; opacity:.28; border-radius:99px; }}
.rl-algo-value {{ width:4.6rem; flex:0 0 auto; text-align:right; font-size:.74rem; font-variant-numeric:tabular-nums; color:#36434a; font-weight:650; }}
.rl-algo-row--todo .rl-algo-name {{ color:#9aa4a0; font-weight:600; }}
.rl-algo-row--todo .rl-algo-value {{ color:#9aa4a0; font-weight:600; }}
.rl-env-foot {{ margin-top:auto; padding-top:.55rem; color:var(--rl-muted); font-size:.7rem; }}

/* ---- 显卡卡 ---- */
.rl-gpu-card {{ background:var(--rl-surface); border:1px solid var(--rl-border); border-radius:var(--rl-radius); box-shadow:var(--rl-shadow); padding:.8rem .9rem .85rem; }}
.rl-gpu-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:.5rem; }}
.rl-gpu-name {{ color:var(--rl-ink); font-weight:700; font-size:.86rem; letter-spacing:-.01em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.rl-gpu-sub {{ color:var(--rl-muted); font-size:.68rem; white-space:nowrap; font-variant-numeric:tabular-nums; }}
.rl-gpu-track {{ height:8px; background:#eef1f0; border-radius:99px; overflow:hidden; margin:.45rem 0 .4rem; }}
.rl-gpu-fill {{ height:100%; border-radius:99px; }}
.rl-gpu-proc {{ color:var(--rl-muted); font-size:.68rem; line-height:1.5; word-break:break-all; }}
.rl-gpu-proc b {{ color:#36434a; font-weight:650; }}

/* ---- 徽章 ---- */
.rl-meta {{ display:flex; flex-wrap:wrap; gap:.38rem .45rem; align-items:center; margin:.3rem 0 .7rem; }} .rl-meta-key {{ color:var(--rl-muted); font-size:.74rem; }}
.rl-pill {{ display:inline-flex; align-items:center; gap:.34rem; padding:.18rem .55rem; border-radius:999px; font-size:.7rem; font-weight:680; line-height:1.4; white-space:nowrap; }} .rl-pill-dot {{ width:6px; height:6px; border-radius:50%; background:currentColor; }}
.rl-legend {{ display:flex; flex-wrap:wrap; gap:.4rem .7rem; margin:.2rem 0 .6rem; }} .rl-legend-item {{ display:inline-flex; align-items:center; gap:.38rem; font-size:.74rem; color:#36434a; }} .rl-swatch {{ width:10px; height:10px; border-radius:3px; }}

/* ---- 原生组件微调 ---- */
[data-testid="stDataFrame"] {{ border:1px solid var(--rl-border); border-radius:var(--rl-radius); overflow:hidden; box-shadow:var(--rl-shadow); background:var(--rl-surface); }} [data-testid="stDataFrame"] [role="columnheader"] {{ background:#f7f9f7!important; color:#5e6965!important; font-weight:700!important; }}
[data-testid="stImage"] img {{ border-radius:8px; }}
[data-testid="stVegaLiteChart"] {{ background:var(--rl-surface); border:1px solid var(--rl-border); border-radius:var(--rl-radius); padding:.5rem .25rem; box-shadow:var(--rl-shadow); }}
.stButton>button,.stDownloadButton>button {{ border-radius:8px; border:1px solid var(--rl-border); font-weight:650; min-height:2.3rem; transition:all .14s ease; }} .stButton>button:hover,.stDownloadButton>button:hover {{ border-color:#f0aa92; color:var(--rl-accent); }} .stButton>button[kind="primary"] {{ background:var(--rl-accent); border-color:var(--rl-accent); color:#fff; box-shadow:0 3px 8px rgba(228,87,46,.18); }} .stButton>button[kind="primary"]:hover {{ background:var(--rl-accent-hover); border-color:var(--rl-accent-hover); color:#fff; }}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div {{ border-radius:8px; border-color:var(--rl-border); }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:.1rem; border-bottom:1px solid var(--rl-border); }} [data-testid="stTabs"] [data-baseweb="tab"] {{ border-radius:6px 6px 0 0; padding:.46rem .76rem; font-size:.82rem; font-weight:650; }} [data-testid="stTabs"] [aria-selected="true"] {{ color:var(--rl-accent); }} [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background:var(--rl-accent); }}
[data-testid="stAlert"] {{ border-radius:8px; border:1px solid var(--rl-border); }} [data-testid="stProgress"]>div>div>div>div {{ background:var(--rl-accent); }} hr {{ border-color:var(--rl-border); margin:1.2rem 0; }} footer,#MainMenu {{ visibility:hidden; }}
@media(max-width:900px) {{ [data-testid="stMainBlockContainer"] {{ padding-left:1.1rem; padding-right:1.1rem; }} .rl-stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }} @media(max-width:560px) {{ .rl-stat-value {{ font-size:1.3rem; }} .rl-header-title {{ font-size:1.5rem; }} }}
</style>
"""


def inject() -> None:
    st.markdown(_css(), unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        '<div class="rl-brand"><div class="rl-brand-mark">Reinforce</div>'
        '<div class="rl-brand-sub">experiment console</div></div>',
        unsafe_allow_html=True,
    )


def sidebar_note(title: str, body: str) -> None:
    st.markdown(
        f'<div class="rl-sidebar-note"><strong>{html.escape(title)}</strong>'
        f'<span>{html.escape(body)}</span></div>',
        unsafe_allow_html=True,
    )


def section_head(title: str, note: str = "") -> None:
    note_html = f'<span class="rl-section-note">{html.escape(note)}</span>' if note else ""
    st.markdown(
        f'<div class="rl-section-head"><span class="rl-section-title">{html.escape(title)}</span>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", *, eyebrow: str = "Reinforce / workspace") -> None:
    st.markdown(
        '<div class="rl-topbar"><div><div class="rl-eyebrow">' + html.escape(eyebrow)
        + '</div><div class="rl-header-title">' + html.escape(title)
        + '</div><div class="rl-header-sub">' + html.escape(subtitle)
        + '</div></div><div class="rl-header-mark"></div></div>',
        unsafe_allow_html=True,
    )


def stat_row(items: Iterable[tuple[str, object, str]]) -> None:
    cards = []
    for label, value, hint in items:
        hint_html = f'<div class="rl-stat-hint">{html.escape(str(hint))}</div>' if hint else ""
        cards.append(
            '<div class="rl-stat"><div class="rl-stat-label">' + html.escape(str(label))
            + '</div><div class="rl-stat-value">' + html.escape(str(value)) + '</div>'
            + hint_html + '</div>'
        )
    st.markdown(f'<div class="rl-stats">{"".join(cards)}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "neutral") -> str:
    foreground, background = STATUS_TONES.get(tone, (MUTED, SUBTLE))
    return (
        f'<span class="rl-pill" style="color:{foreground};background:{background}">'
        f'<span class="rl-pill-dot"></span>{html.escape(text)}</span>'
    )


def meta_row(pairs: Iterable[tuple[str, str]]) -> None:
    chunks = []
    for key, value in pairs:
        chunks.append(
            f'<span class="rl-meta-key">{html.escape(key)}</span>'
            f'<span class="rl-pill" style="color:#36434a;background:{SUBTLE}">'
            f'{html.escape(value)}</span>'
        )
    st.markdown(f'<div class="rl-meta">{"".join(chunks)}</div>', unsafe_allow_html=True)


def card(body_html: str) -> None:
    st.markdown(f'<div class="rl-card">{body_html}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    """空状态：虚线框 + 一句下一步该做什么。"""
    st.markdown(
        f'<div class="rl-empty"><strong>{html.escape(title)}</strong>'
        f'<span>{html.escape(body)}</span></div>',
        unsafe_allow_html=True,
    )


def legend(items: Iterable[tuple[str, str]]) -> None:
    """色块图例。``items`` 是 ``(标签, 色值)``。"""
    chunks = [
        f'<span class="rl-legend-item"><span class="rl-swatch" style="background:{color}"></span>'
        f'{html.escape(label)}</span>'
        for label, color in items
    ]
    if chunks:
        st.markdown(f'<div class="rl-legend">{"".join(chunks)}</div>', unsafe_allow_html=True)


def algo_bars(
    entries: Iterable[dict[str, Any]],
    *,
    scale: float,
    placeholders: Iterable[str] = (),
) -> str:
    """排行榜的条形组 HTML。``entries`` 是 ``environment_leaderboard()`` 的行。

    每行三样东西：

    - 左侧名字用 **算法全名**（``SB3-PPO`` 与 ``PPO`` 必须分得开，同族不同实现正是本项目
      的核心对照维度；族名只用来取色，不再折叠显示）。
    - 中间的条。宽度按 ``mean`` 归一化；有多种子时条上再压一条半透明的范围带
      （``mean ± std``），只有一根条加一个数字时看不出「这个均值稳不稳」。
    - 右侧数值，多种子时写成 ``mean ± std``。

    ``placeholders`` 是还没跑过的算法：画成灰色空条并标「未跑」，让缺口一眼可见。
    """
    rows = []
    for entry in entries:
        mean = float(entry.get("mean", 0.0))
        spread = entry.get("spread")
        width = 0.0 if scale <= 0 else max(2.0, min(100.0, abs(mean) / scale * 100))
        color = entry.get("color") or family_color(str(entry.get("family", "OTHER")))
        label = entry.get("label") or entry.get("algorithm") or entry.get("family", "?")
        bar_extra = ""
        if spread:
            low, high = entry.get("low"), entry.get("high")
            if low is not None and high is not None and scale > 0:
                left = max(0.0, min(100.0, abs(low) / scale * 100))
                right = max(0.0, min(100.0, abs(high) / scale * 100))
                bar_extra = (
                    f'<span class="rl-algo-range" style="left:{left:.1f}%;'
                    f'width:{max(0.6, right - left):.1f}%;color:{color}"></span>'
                )
        value = f"{mean:.2f}" + (f" ± {spread:.2f}" if spread else "")
        rows.append(
            '<div class="rl-algo-row">'
            f'<span class="rl-algo-name">{html.escape(str(label))}</span>'
            '<span class="rl-algo-track">'
            f'<span class="rl-algo-bar" style="width:{width:.1f}%;background:{color}"></span>'
            f'{bar_extra}</span>'
            f'<span class="rl-algo-value">{value}</span></div>'
        )
    for label in placeholders:
        # 条宽给 0，只留轨道本身的浅灰底，读作「这一格是空的」而不是「这一格是 0 分」。
        rows.append(
            '<div class="rl-algo-row rl-algo-row--todo">'
            f'<span class="rl-algo-name">{html.escape(str(label))}</span>'
            '<span class="rl-algo-track"></span>'
            '<span class="rl-algo-value">未跑</span></div>'
        )
    return "".join(rows)


def env_card(
    environment: str,
    *,
    runs: int,
    algorithms: int,
    best_name: str | None,
    best_value: float | None,
    entries: Iterable[dict[str, Any]],
    placeholders: Iterable[str] = (),
    note: str = "",
) -> str:
    """一张环境卡的 HTML。

    ``entries`` 是 ``data.environment_leaderboard()`` 的行——每个算法一根条，条宽按
    **这张卡内**最大的回报绝对值归一化。不同环境的奖励尺度完全不同，不能跨卡比长度。
    ``placeholders`` 是「还没跑过的算法」，画成灰色空条：只列跑过的算法时看不出缺什么。
    """
    rows = list(entries)
    scale = max((abs(float(row.get("mean", 0.0))) for row in rows), default=0.0)
    body = algo_bars(rows, scale=scale, placeholders=placeholders)
    best_html = (
        f'<div class="rl-env-foot">当前最好 · {html.escape(best_name or "-")} · '
        f'{best_value:.2f}</div>'
        if best_value is not None else '<div class="rl-env-foot">还没有完成的评估</div>'
    )
    note_html = f' · {html.escape(note)}' if note else ""
    return (
        '<div class="rl-env-card">'
        '<div class="rl-env-head">'
        f'<span class="rl-env-name">{html.escape(environment)}</span>'
        f'{pill(f"{runs} 次运行", "completed" if runs else "not_started")}'
        '</div>'
        f'<div class="rl-env-sub">{algorithms} 个算法{note_html}</div>'
        + body + best_html + '</div>'
    )


def gpu_card(
    title: str,
    *,
    subtitle: str,
    used_mb: int | None,
    total_mb: int | None,
    utilization_pct: int | None,
    processes: Iterable[str] = (),
    tone: str = "neutral",
    note: str = "",
) -> str:
    """一张显卡的 HTML 卡片：显存条 + 利用率 + 占用进程。

    ``tone`` 只影响显存条的颜色：``busy`` 用告警色，``free`` 用成功色，``neutral``
    用中性灰——读不到数据的卡不该被染成「正常」。
    """
    fill_color = {"busy": DANGER, "free": SUCCESS, "neutral": MUTED}.get(tone, MUTED)
    if used_mb is not None and total_mb:
        fraction = max(0.0, min(100.0, used_mb / total_mb * 100))
        memory_text = f"{used_mb / 1024:.1f} / {total_mb / 1024:.1f} GB"
    else:
        fraction = 0.0
        memory_text = "显存未知"
    util_text = f"利用率 {utilization_pct}%" if utilization_pct is not None else "利用率未知"
    proc_html = "".join(f'<div class="rl-gpu-proc">{html.escape(str(item))}</div>' for item in processes)
    note_html = f'<div class="rl-gpu-proc">{html.escape(note)}</div>' if note else ""
    return (
        '<div class="rl-gpu-card">'
        '<div class="rl-gpu-head">'
        f'<span class="rl-gpu-name">{html.escape(title)}</span>'
        f'<span class="rl-gpu-sub">{html.escape(subtitle)}</span>'
        "</div>"
        f'<div class="rl-gpu-track"><span class="rl-gpu-fill" style="width:{fraction:.1f}%;'
        f'background:{fill_color}"></span></div>'
        f'<div class="rl-gpu-proc"><b>{html.escape(memory_text)}</b> · {html.escape(util_text)}</div>'
        + proc_html + note_html + "</div>"
    )


def caption(text: str) -> None:
    st.markdown(
        f'<div style="color:{MUTED};font-size:.78rem;line-height:1.6;margin:.3rem 0 .65rem">{text}</div>',
        unsafe_allow_html=True,
    )


def status_tone(status: str) -> Literal["completed", "running", "failed", "not_started"]:
    return status if status in STATUS_TONES else "not_started"  # type: ignore[return-value]


def fit_height(rows: int, *, max_height: int = 620, row: int = 35, header: int = 42) -> int:
    return min(max_height, header + row * max(rows, 1))
