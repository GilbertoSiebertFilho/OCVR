# -*- coding: utf-8 -*-
"""OCVR — Olds College Variable-Rate economics: the brand of the app in one place (copied from OCCD, same palette and
the same look) (name, version, palette, logos, page icon, the CSS of the page, the map colours).

    import brand                      # from app/app.py (Streamlit puts app/ on sys.path)
    from app import brand             # from the project root (app/ is a namespace package: no __init__.py)

    st.set_page_config(page_title=brand.NAME, page_icon=brand.page_icon(), layout="wide")
    brand.inject()                    # once per run, right after set_page_config
    st.logo(brand.LOGO_H, icon_image=brand.LOGO_SHIELD, size="large")
    brand.brand_block()               # name, version and tagline of the title bar
    brand.step_header(1, "Trial", "Project, files and the tested inputs")    # the numbered heading of a step card

Palette (assets/brand/BRAND.md, the Olds College site stylesheet): navy is the primary colour (buttons, headings,
active states), light blue the accent (icons, outlines, selected states), green = kept / success, orange = removed /
warnings. Contrast on white (WCAG): navy 11.1:1, text 14.7:1, LIGHT_BLUE_TEXT 4.9:1 (AA), GREEN_TEXT 5.5:1,
ORANGE_TEXT 5.4:1. LIGHT_BLUE (3.1:1), GREEN (2.7:1) and ORANGE (2.0:1) fail AA as body text on white: use them for
fills, icons, outlines and large labels only, and the *_TEXT shades when a word must carry the colour.

The theme itself is .streamlit/config.toml (read when the app is started from the project folder: run_app.bat; the
sidebar tint is [theme.sidebar] there). inject() adds what a theme cannot set, as light-dark() pairs that follow the
rendered theme (the OCCD 2.6 look): a navy title bar (the app_header container: white logo, name, version pill,
tagline, the open project), the six steps as pill tabs (the selected one navy) whose content is a white card on a
light canvas with a navy top edge and a numbered badge (step_header(); the ocvr_step_1..6 containers), the settings
that decide a step in tinted, edged panels (containers keyed panel_*), the column mappings and the crop history as
edged expanders (containers keyed colmap_*), the blocks nested in a step as white cards with a blue edge (containers
keyed block_*), primary buttons navy with a lift on hover, the main action of each step larger (MAIN_ACTIONS) and the
main downloads green, metrics as small cards, the step checklist of the sidebar (steps_html()), the captions at full
opacity (Streamlit draws them at 60 %, 4.0:1 on white; here MUTED, 7.6:1, links LIGHT_BLUE_TEXT, 4.9:1) and the
.ocvr-rule class of rule(). Only stable hooks are used (data-testid, role / aria attributes, the st-key-* classes of
keyed elements). No external font or CDN: the app runs offline on a field laptop.
"""
from __future__ import annotations

import functools
import os

NAME = "OCVR"
LONG_NAME = "OCVR — Olds College Variable-Rate economics"
TAGLINE = "Trial plots × yield × multi-year NDVI zones × prices: variable rate vs flat rate"
ORG = "Olds College Smart Farm (OFPE)"
VERSION = "1.1.0"

# ---- palette (hex strings; BRAND.md)
NAVY = "#00358E"            # primary: buttons, headings, tree group titles, active states, lines and fits in figures
LIGHT_BLUE = "#009CDE"      # accent: icons, outlines (AOI on maps), selected / active switch; not body text on white
GREEN = "#00B373"           # kept points / success fills
DARK_GREEN = "#006747"      # secondary accent (sparingly)
ORANGE = "#F5A623"          # removed points / warning highlight fills
TEXT = "#1F2937"            # body text
MUTED = "#4B5563"           # captions (7.6:1 on white)
BG2 = "#F3F7FC"             # secondary background (widgets, expanders, sidebar)
RULE = "#D7E3F2"            # borders and rules
WHITE = "#FFFFFF"
# AA-safe shades for words that must carry the colour (4.5:1 or better on white and on BG2)
LIGHT_BLUE_TEXT = "#0077B6"
GREEN_TEXT = "#00784D"
ORANGE_TEXT = "#9A5B00"

# ---- logos (absolute paths built from this file: the app can be started from any folder)
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "brand")
LOGO_H = os.path.join(ASSETS, "oc_logo_horizontal.png")               # 500 x 125, white background: header, sidebar
LOGO_H_REV = os.path.join(ASSETS, "oc_logo_horizontal_reversed.png")  # white wordmark for dark backgrounds
LOGO_SHIELD = os.path.join(ASSETS, "oc_logo_shield.png")              # shield + wordmark: favicon, collapsed sidebar

# pydeck colours of the app's maps ([r, g, b]; add the alpha in the layer)
map_colors = {"kept": [0, 179, 115], "removed": [245, 166, 35], "aoi": [0, 156, 222]}


@functools.lru_cache(maxsize=1)
def page_icon():
    """The page icon for st.set_page_config: the shield mark of oc_logo_shield.png as a square PIL image (the
    wordmark under the shield is unreadable at favicon size, so the rows above the first transparent gap are kept and
    padded to a square). The file's path when PIL cannot open it (st.set_page_config accepts both)."""
    try:
        from PIL import Image
        im = Image.open(LOGO_SHIELD).convert("RGBA")
        w, h = im.size
        alpha = im.getchannel("A")
        cut = h
        for y in range(h):                                    # first fully transparent row = the gap under the shield
            if alpha.crop((0, y, w, y + 1)).getbbox() is None:
                cut = y
                break
        mark = im.crop((0, 0, w, cut))
        box = mark.getbbox()
        if box:
            mark = mark.crop(box)
        side = max(mark.size) + 8
        icon = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        icon.paste(mark, ((side - mark.width) // 2, (side - mark.height) // 2), mark)
        return icon
    except Exception:                                         # pragma: no cover - a missing/odd file: the path works too
        return LOGO_SHIELD


# The keys (inside the project prefix of app.k()) of the main button of each step: drawn larger, as OCCD's Run cleaning.
# Streamlit turns a widget key into the class st-key-<key> with every character outside [A-Za-z0-9_-] made '-', so
# the key '<project id>:btn_an' is the class 'st-key-<project id>-btn_an' and the CSS hooks the '-btn_an' end.
MAIN_ACTIONS = ("btn_design", "btn_zones", "btn_prices", "btn_join", "btn_an", "btn_pdf")


def _css() -> str:
    """The style block: stable hooks only (data-testid attributes, role / aria attributes, the .stApp root, the
    st-key-* classes of the keyed containers and widgets, our own .ocvr-* classes); no emotion class names. Every colour
    is a CSS light-dark() pair resolved by the browser against the color-scheme that Streamlit sets on .stApp for the
    rendered theme, so the block is right on the very first run too (dark mode is not a target, but stays legible).
    Python-side detection through st.context.theme.type is not used: on the first run after a connection it reports
    the OS preference, not the configured theme.

    The look (OCCD 2.6): a navy title bar (app_header), the steps as pill tabs (the selected one navy) whose content is
    a white card on a light canvas with a navy top edge and a numbered badge (step_header; keys ocvr_step_1..6), the
    settings that decide a step in tinted panels (keys panel_*), the column mappings as tinted, edged expanders (keys
    colmap_*), nested blocks as white cards with a blue edge (keys block_*), primary buttons navy with a lift on hover,
    the main action of each step (MAIN_ACTIONS) larger and the main downloads green, metrics as small cards. The
    captions keep full opacity (WCAG AA: 7.6:1 on white)."""
    navy, blue = NAVY, LIGHT_BLUE
    heading = f"light-dark({NAVY}, {LIGHT_BLUE})"
    rule = f"light-dark({RULE}, rgba(255,255,255,0.14))"
    muted = f"light-dark({MUTED}, rgba(255,255,255,0.68))"
    link = f"light-dark({LIGHT_BLUE_TEXT}, {LIGHT_BLUE})"
    canvas = "light-dark(#F3F6FA, #0E1117)"
    card = "light-dark(#FFFFFF, #161B22)"
    tint = "light-dark(#EAF2FC, rgba(0,156,222,0.12))"
    tint2 = "light-dark(#F5F9FE, rgba(255,255,255,0.03))"
    edge = "light-dark(#C9DCF3, rgba(0,156,222,0.35))"
    pill_txt = f"light-dark({NAVY}, #CFE3FA)"
    ok = f"light-dark({GREEN_TEXT}, {GREEN})"
    shadow = "0 1px 2px rgba(16,24,40,0.05), 0 6px 22px rgba(0,53,142,0.07)"
    main_btn = ", ".join(f'.stApp [class*="-{a}"] [data-testid="stBaseButton-primary"]' for a in MAIN_ACTIONS)
    main_btn_p = ", ".join(f'.stApp [class*="-{a}"] [data-testid="stBaseButton-primary"] p' for a in MAIN_ACTIONS)
    return f"""<style>
/* ---- OCVR brand (app/brand.py) ---- */
.stApp [data-testid="stMain"], .stApp [data-testid="stHeader"] {{ background: {canvas}; }}
.stApp h1, .stApp h2, .stApp h3 {{ color: {heading}; }}
.stApp [data-testid="stMainBlockContainer"] {{ padding-top: 4.4rem; }}
/* captions: full opacity, muted text (7.6:1 on white, 7.0:1 on the sidebar), AA links */
div[data-testid="stCaptionContainer"] {{ opacity: 1; }}
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] li {{ color: {muted}; }}
div[data-testid="stCaptionContainer"] a {{ color: {link}; }}

/* ---- title bar */
.stApp .st-key-app_header {{
  background: linear-gradient(115deg, #00266B 0%, {navy} 42%, #0060C0 82%, {blue} 125%);
  border-radius: 16px; padding: 1.0rem 1.5rem !important; box-shadow: 0 10px 28px rgba(0,53,142,0.28);
  margin-bottom: 0.4rem;
}}
.ocvr-brand {{ display: flex; flex-direction: column; gap: 0.15rem; }}
.ocvr-brand-name {{ font-size: 2.05rem; font-weight: 800; letter-spacing: 0.03em; color: #FFFFFF; line-height: 1.1;
  display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }}
.ocvr-brand-long {{ font-size: 1.02rem; font-weight: 600; color: #FFFFFF; }}
.ocvr-brand-tag {{ font-size: 0.92rem; color: rgba(255,255,255,0.88); }}
.ocvr-pill {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; padding: 0.18rem 0.55rem;
  border-radius: 999px; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.45); color: #FFFFFF; }}
/* the open project, at the right of the bar */
.ocvr-proj {{ display: flex; flex-direction: column; align-items: flex-end; gap: 0.35rem; text-align: right; }}
.ocvr-proj-label {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  color: rgba(255,255,255,0.78); }}
.ocvr-proj-name {{ font-size: 1.05rem; font-weight: 700; color: #FFFFFF; line-height: 1.2; overflow-wrap: anywhere; }}
.ocvr-proj-meta {{ font-size: 0.85rem; color: rgba(255,255,255,0.88); }}
.ocvr-proj-chip {{ font-size: 0.78rem; font-weight: 700; padding: 0.2rem 0.65rem; border-radius: 999px;
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.45); color: #FFFFFF; white-space: nowrap; }}
.ocvr-proj-chip.done {{ background: rgba(0,179,115,0.28); border-color: rgba(120,230,180,0.75); }}
.stApp .st-key-app_header [data-testid="stCaptionContainer"],
.stApp .st-key-app_header [data-testid="stCaptionContainer"] p {{ color: rgba(255,255,255,0.88) !important; }}
@media (max-width: 640px) {{ .ocvr-proj {{ align-items: flex-start; text-align: left; }} }}

/* ---- the steps: pill tabs, the content a card with a navy top edge and a numbered badge */
.stApp [data-testid="stMain"] [data-testid="stTabs"] [role="tablist"] {{
  gap: 0.4rem; flex-wrap: wrap; padding: 0.45rem; background: {tint2}; border: 1px solid {rule}; border-radius: 12px;
  box-shadow: none; }}
.stApp [data-testid="stMain"] [data-testid="stTab"] {{
  height: auto; margin: 0; padding: 0.42rem 0.95rem; border-radius: 999px; background: {card}; border: 1px solid {edge};
  transition: background 0.12s ease, border-color 0.12s ease, box-shadow 0.12s ease; }}
.stApp [data-testid="stMain"] [data-testid="stTab"] p {{ font-weight: 600; font-size: 0.95rem; color: {pill_txt}; }}
.stApp [data-testid="stMain"] [data-testid="stTab"] [data-testid="stIconMaterial"] {{ color: {pill_txt}; }}
.stApp [data-testid="stMain"] [data-testid="stTab"]:hover {{ background: {tint}; border-color: {blue}; }}
.stApp [data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] {{
  background: {navy}; border-color: {navy}; box-shadow: 0 2px 8px rgba(0,53,142,0.28); }}
.stApp [data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] p,
.stApp [data-testid="stMain"] [data-testid="stTab"][aria-selected="true"] [data-testid="stIconMaterial"] {{
  color: #FFFFFF; }}
.stApp [data-testid="stMain"] [data-testid="stTabs"] .react-aria-SelectionIndicator {{ display: none; }}
.stApp [data-testid="stMain"] [data-testid="stTabPanel"] {{ padding-top: 0.6rem; }}
.stApp [class*="st-key-ocvr_step_"] {{
  background: {card}; border: 1px solid {rule} !important; border-top: 4px solid {navy} !important;
  border-radius: 14px !important; box-shadow: {shadow}; padding: 1.15rem 1.45rem 1.4rem !important;
}}
.stApp [class*="st-key-ocvr_step_"] h3 {{ font-size: 1.3rem; }}
.ocvr-step {{ display: flex; align-items: center; gap: 0.85rem; margin: 0.05rem 0 0.25rem; }}
.ocvr-step-num {{ flex: none; width: 2.35rem; height: 2.35rem; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-weight: 800; font-size: 1.15rem; color: #FFFFFF;
  background: linear-gradient(135deg, {navy} 0%, #0060C0 100%); box-shadow: 0 3px 8px rgba(0,53,142,0.30); }}
.ocvr-step-num.done {{ background: linear-gradient(135deg, #00A86B 0%, #00875A 100%);
  box-shadow: 0 3px 8px rgba(0,135,90,0.30); }}
.ocvr-step-title {{ font-size: 1.45rem; font-weight: 800; color: {heading}; line-height: 1.15; display: flex;
  align-items: center; gap: 0.55rem; flex-wrap: wrap; }}
.ocvr-step-sub {{ font-size: 0.92rem; color: {muted}; margin-top: 0.1rem; }}
.ocvr-done {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  padding: 0.16rem 0.55rem; border-radius: 999px; color: {ok}; border: 1px solid currentColor; }}

/* ---- the settings that decide a step: tinted panels */
.stApp [class*="st-key-panel_"] {{
  background: {tint}; border: 1px solid {edge}; border-left: 5px solid {blue}; border-radius: 12px;
  padding: 0.85rem 1.1rem 0.95rem !important;
}}
.stApp [class*="st-key-panel_"] h3 {{ font-size: 1.2rem; padding: 0.1rem 0 0.35rem; }}
.stApp [class*="-in_mode"] [data-testid="stWidgetLabel"] p,
.stApp [class*="-ndvi_src"] [data-testid="stWidgetLabel"] p {{ font-weight: 700; font-size: 1.02rem; color: {heading}; }}
/* the column mappings and the crop history: tinted, edged expanders */
.stApp [class*="st-key-colmap_"] [data-testid="stExpander"] details {{
  border: 1px solid {edge}; border-left: 5px solid {navy}; border-radius: 12px; background: {card}; overflow: hidden; }}
.stApp [class*="st-key-colmap_"] [data-testid="stExpander"] summary {{ background: {tint}; padding-top: 0.8rem;
  padding-bottom: 0.8rem; }}
.stApp [class*="st-key-colmap_"] [data-testid="stExpander"] summary p {{ font-weight: 700; font-size: 1.02rem;
  color: {heading}; }}
.stApp [class*="st-key-colmap_"] [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{ color: {heading}; }}
/* other expanders of the main area: a soft hover */
.stApp [data-testid="stMain"] [data-testid="stExpander"] details {{ border-radius: 12px; }}
.stApp [data-testid="stMain"] [data-testid="stExpander"] summary:hover {{ background: {tint2}; }}
.stApp [data-testid="stMain"] [data-testid="stExpander"] summary p {{ font-weight: 600; }}

/* ---- blocks nested in a step (files, input prices, the result of one input): white cards, a smaller heading */
.stApp [class*="st-key-block_"] {{
  background: {card}; border-radius: 12px !important; border: 1px solid {rule} !important;
  border-left: 4px solid {blue} !important; }}
.stApp [class*="st-key-block_"] h3 {{ font-size: 1.2rem; padding: 0.2rem 0 0.45rem; }}

/* ---- buttons: primary navy with a lift, secondary outlined, the main action larger, the main downloads green */
.stApp [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(180deg, #0A4FB8 0%, {navy} 100%); border: 1px solid #00307F; color: #FFFFFF;
  font-weight: 600; box-shadow: 0 1px 2px rgba(16,24,40,0.10), 0 4px 12px rgba(0,53,142,0.22);
  transition: transform 0.08s ease, box-shadow 0.15s ease, filter 0.15s ease; }}
.stApp [data-testid="stBaseButton-primary"]:hover {{
  filter: brightness(1.08); color: #FFFFFF; border-color: #00307F; transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(0,53,142,0.32); }}
.stApp [data-testid="stBaseButton-primary"]:disabled {{ filter: grayscale(0.6) opacity(0.55); box-shadow: none;
  transform: none; }}
.stApp [data-testid="stMain"] [data-testid="stBaseButton-secondary"] {{ border-color: {edge}; color: {pill_txt};
  font-weight: 500; }}
.stApp [data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover {{ background: {tint}; border-color: {navy};
  color: {pill_txt}; }}
{main_btn} {{
  min-height: 3.3rem; padding: 0.75rem 2.3rem; border-radius: 12px; font-size: 1.12rem; font-weight: 700;
  background: linear-gradient(135deg, {navy} 0%, #0060C0 70%, {blue} 130%); letter-spacing: 0.01em; }}
{main_btn_p} {{ font-size: 1.12rem; font-weight: 700; }}
.stApp [data-testid="stDownloadButton"] [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(180deg, #00A86B 0%, #00875A 100%); border-color: #007A50;
  box-shadow: 0 1px 2px rgba(16,24,40,0.10), 0 4px 12px rgba(0,135,90,0.25); }}
.stApp [data-testid="stDownloadButton"] [data-testid="stBaseButton-primary"]:hover {{
  border-color: #007A50; box-shadow: 0 6px 18px rgba(0,135,90,0.32); }}

/* ---- metrics as small cards */
.stApp [data-testid="stMain"] [data-testid="stMetric"] {{ background: {card}; border: 1px solid {rule};
  border-radius: 12px; padding: 0.7rem 0.95rem; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{ font-weight: 600; }}
.stApp [data-testid="stMain"] [data-testid="stMetricLabel"] p {{ text-transform: uppercase; letter-spacing: 0.05em;
  font-size: 0.78rem; color: {muted}; }}
[data-testid="stMetricValue"] {{ color: {heading}; font-weight: 700; }}
/* the cards are narrower than the bare metrics: a smaller value, and long labels / values ('+17.39 $/ac',
   'VR vs flat (cross-fitted)') wrap at their spaces instead of ending in '...' */
.stApp [data-testid="stMain"] [data-testid="stMetricValue"] {{ font-size: 1.75rem; line-height: 1.2; }}
.stApp [data-testid="stMain"] [data-testid="stMetricValue"] *,
.stApp [data-testid="stMain"] [data-testid="stMetricLabel"] * {{ white-space: normal; text-overflow: clip; }}

/* ---- alerts, tables, images */
.stApp [data-testid="stAlert"] > div {{ border-radius: 10px; }}
.stApp [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
.stApp [data-testid="stMain"] [data-testid="stImage"] img {{ border-radius: 10px; }}

/* ---- sidebar: section labels, the step checklist */
[data-testid="stSidebar"] h3 {{ font-size: 0.82rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.09em;
  color: {heading}; }}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {{ color: {heading}; font-weight: 700; }}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{ background: {card}; border-radius: 10px;
  border-color: {rule}; }}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{ font-weight: 600; }}
.ocvr-steps {{ display: flex; flex-direction: column; gap: 0.3rem; margin: 0.1rem 0 0.2rem; }}
.ocvr-steps-row {{ display: flex; align-items: center; gap: 0.6rem; padding: 0.38rem 0.6rem; border-radius: 10px;
  background: {card}; border: 1px solid {rule}; }}
.ocvr-steps-num {{ flex: none; width: 1.55rem; height: 1.55rem; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 0.8rem; font-weight: 800; color: {heading}; border: 2px solid {edge}; }}
.ocvr-steps-row.done .ocvr-steps-num {{ background: {GREEN}; border-color: {GREEN}; color: #FFFFFF; }}
.ocvr-steps-label {{ font-size: 0.92rem; font-weight: 600; color: {muted}; }}
.ocvr-steps-row.done .ocvr-steps-label {{ color: {heading}; }}
.ocvr-steps-state {{ margin-left: auto; font-size: 0.72rem; font-weight: 700; color: {ok}; }}
.ocvr-rule {{ height: 1px; background: {rule}; border: 0; margin: 0.6rem 0 1.0rem; }}
</style>"""


def _esc(x) -> str:
    import html
    return html.escape(str(x))


def step_header(num: int, title: str, subtitle: str = "", done: bool = False) -> None:
    """The heading of a step card: a numbered navy badge (green with a check when the step is done), the title, a
    'done' tag and a one-line subtitle (the .ocvr-step classes of inject())."""
    import streamlit as st
    badge = "✓" if done else str(int(num))
    st.html(f'<div class="ocvr-step"><div class="ocvr-step-num{" done" if done else ""}" title="Step {int(num)}">'
            f'{badge}</div><div><div class="ocvr-step-title">{_esc(title)}'
            + ('<span class="ocvr-done">done</span>' if done else "") + "</div>"
            + (f'<div class="ocvr-step-sub">{_esc(subtitle)}</div>' if subtitle else "") + "</div></div>")


def brand_block() -> None:
    """The name, version, long name and tagline of the title bar (white on navy)."""
    import streamlit as st
    long_ = LONG_NAME.split(" — ", 1)[-1]
    st.html(f'<div class="ocvr-brand"><div class="ocvr-brand-name">{_esc(NAME)}<span class="ocvr-pill">'
            f'v{_esc(VERSION)}</span></div><div class="ocvr-brand-long">{_esc(long_)}</div>'
            f'<div class="ocvr-brand-tag">{_esc(TAGLINE)}</div></div>')


def project_html(name: str, meta: str, n_done: int, n_steps: int) -> str:
    """The open project at the right of the title bar: its name, crop / field / year and how many steps are done."""
    chip = "done" if n_done >= n_steps else ""
    return (f'<div class="ocvr-proj"><div class="ocvr-proj-label">Project</div>'
            f'<div class="ocvr-proj-name">{_esc(name)}</div>'
            + (f'<div class="ocvr-proj-meta">{_esc(meta)}</div>' if meta else "")
            + f'<div class="ocvr-proj-chip {chip}">{int(n_done)} of {int(n_steps)} steps done</div></div>')


def steps_html(steps) -> str:
    """The step checklist of the sidebar: one row per (number, label, done), a green check on the done ones."""
    rows = []
    for num, label, ok in steps:
        rows.append(f'<div class="ocvr-steps-row{" done" if ok else ""}"><div class="ocvr-steps-num">'
                    f'{"✓" if ok else _esc(num)}</div><div class="ocvr-steps-label">{_esc(label)}</div>'
                    + ('<div class="ocvr-steps-state">done</div>' if ok else "") + "</div>")
    return '<div class="ocvr-steps">' + "".join(rows) + "</div>"


def inject():
    """Write the style block into the page (st.html; once per run, after st.set_page_config)."""
    import streamlit as st
    st.html(_css())


def rule():
    """A thin light rule between the steps (the .ocvr-rule class of inject())."""
    import streamlit as st
    st.html('<div class="ocvr-rule"></div>')


def footer_caption() -> str:
    """The sidebar footer: 'OCVR 1.1.0 · Olds College Smart Farm (OFPE)'."""
    return f"{NAME} {VERSION} · {ORG}"
