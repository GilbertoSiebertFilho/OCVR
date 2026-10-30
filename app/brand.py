# -*- coding: utf-8 -*-
"""OCVR — Olds College Variable-Rate economics: the brand of the app in one place (copied from OCCD, same palette) (name, version, palette, logos, page icon,
the small CSS of the page, the map colours).

    import brand                      # from app/app.py (Streamlit puts app/ on sys.path)
    from app import brand             # from the project root (app/ is a namespace package: no __init__.py)

    st.set_page_config(page_title=brand.NAME, page_icon=brand.page_icon(), layout="wide")
    brand.inject()                    # once per run, right after set_page_config
    st.logo(brand.LOGO_H, icon_image=brand.LOGO_SHIELD, size="large")

Palette (assets/brand/BRAND.md, the Olds College site stylesheet): navy is the primary colour (buttons, headings,
tree group titles, active states), light blue the accent (icons, outlines, selected states), green = kept / success,
orange = removed / warnings. Contrast on white (WCAG): navy 11.1:1, text 14.7:1, LIGHT_BLUE_TEXT 4.9:1 (AA), GREEN_TEXT
5.5:1, ORANGE_TEXT 5.4:1. LIGHT_BLUE (3.1:1), GREEN (2.7:1) and ORANGE (2.0:1) fail AA as body text on white: use them
for fills, icons, outlines and large labels only, and the *_TEXT shades when a word must carry the colour.

The theme itself is .streamlit/config.toml (and the same flags in the "occd" launch configuration, because the config
file is only read when the app is started from the project folder: the sidebar tint is [theme.sidebar] there);
inject() adds the few things a theme cannot set, as light-dark() pairs that follow the rendered theme: the heading and
metric colours, the metric label weight, the captions (Streamlit draws them at 60 % opacity, 4.0:1 on white and 2.4:1
for the links inside them; here they are MUTED at full opacity, 7.6:1, and their links LIGHT_BLUE_TEXT, 4.9:1), the
smaller headings of the blocks nested in a step (the field of interest in step 1, the filter boxes of step 3, hooked
by the containers' keys) and the .occd-rule class of rule(). No external font or CDN: the app runs offline on a field
laptop.
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


def _css() -> str:
    """The style block: stable hooks only (data-testid attributes, the .stApp root, the st-key-* classes of the keyed
    containers, our own .occd-rule class); no emotion class names. Every colour is a CSS light-dark() pair resolved
    by the browser against the color-scheme that Streamlit sets on .stApp for the rendered theme, so the block is
    right on the very first run too (dark mode is not a target, but if the user switches to it the headings turn
    light blue instead of vanishing). Python-side detection through st.context.theme.type is not used: on the first
    run after a connection it reports the OS preference, not the configured theme, and a light page would get the
    dark variant until the next rerun. The caption rules undo Streamlit's 60 % opacity (WCAG AA for the most used
    secondary text of the page); the nested-heading rules hook the keys of the containers app.py gives the field-of-
    interest block (aoi_block) and the filter boxes (filter_box_<F>)."""
    heading = f"light-dark({NAVY}, {LIGHT_BLUE})"
    rule = f"light-dark({RULE}, rgba(255,255,255,0.18))"
    muted = f"light-dark({MUTED}, rgba(255,255,255,0.65))"
    link = f"light-dark({LIGHT_BLUE_TEXT}, {LIGHT_BLUE})"
    return f"""<style>
/* OCVR brand (app/brand.py) */
.stApp h1, .stApp h2, .stApp h3 {{ color: {heading}; }}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {{ color: {heading}; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{ font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {heading}; }}
/* captions: full opacity, muted text (7.6:1 on white, 7.0:1 on the sidebar), AA links */
div[data-testid="stCaptionContainer"] {{ opacity: 1; }}
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] li {{ color: {muted}; }}
div[data-testid="stCaptionContainer"] a {{ color: {link}; }}
/* a block nested in a step (field of interest, one filter): a smaller heading than the step's own */
.stApp .st-key-aoi_block h3, .stApp [class*="st-key-filter_box_"] h3 {{ font-size: 1.25rem; padding: 0.25rem 0 0.5rem; }}
.occd-rule {{ height: 1px; background: {rule}; border: 0; margin: 0.6rem 0 1.0rem; }}
</style>"""


def inject():
    """Write the style block into the page (st.html; once per run, after st.set_page_config)."""
    import streamlit as st
    st.html(_css())


def rule():
    """A thin light rule between the steps (the .occd-rule class of inject())."""
    import streamlit as st
    st.html('<div class="occd-rule"></div>')


def footer_caption() -> str:
    """The sidebar footer: 'OCCD 2.2.0 · Olds College Smart Farm (OFPE)'."""
    return f"{NAME} {VERSION} · {ORG}"
