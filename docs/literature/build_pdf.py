"""Build a house-style PDF from a Markdown report (Olds College OFPE / OCVR).

Usage:
    python build_pdf.py [input.md] [output.pdf] [--date "9 October 2026"]

Defaults: OCVR_method_and_references.md next to this script, PDF next to it.

No external Markdown package: a small parser handles headings (# ## ###), paragraphs,
bullet and numbered lists, **bold**, *italic*, `code`, [links](url), bare URLs,
pipe tables (with \\| escapes) and horizontal rules.

House style (imitates engine/report/make_report.py): US letter, 2 cm margins, body
10.5 pt justified, green (#2e6b34) headings, tables with light grey header, 8.5 pt
wrapped cells, repeated header rows, title page, page numbers in the footer.

Fonts: Arial (Helvetica metrics, full Latin/Greek coverage) when present, else DejaVu
Sans (matplotlib copy or system), else Helvetica. Any character the body font lacks
is drawn with DejaVu Sans as a per-glyph fallback; if none can draw it, a plain-ASCII
substitute is used.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MD = os.path.join(HERE, "OCVR_method_and_references.md")

GREEN = colors.HexColor("#2e6b34")
GREY = colors.HexColor("#555555")
HEAD_BG = colors.HexColor("#e6e6e6")
GRID = colors.HexColor("#b0b0b0")
ZEBRA = colors.HexColor("#f7f7f7")
CODE_COLOR = colors.HexColor("#333333")

# cover and footer strings per language (--lang)
LANG = {
    "en": dict(org="Olds College \u2014 OFPE / OCVR variable-rate economics",
               note="Prepared with Claude (Anthropic) under the supervision of Gilberto Siebert Filho", page="Page %d"),
    "pt": dict(org="Olds College \u2014 OFPE / OCVR economia da taxa vari\u00e1vel",
               note="Preparado com Claude (Anthropic) sob a supervis\u00e3o de Gilberto Siebert Filho", page="P\u00e1gina %d"),
}
ORG_LINE = LANG["en"]["org"]
NOTE_LINE = LANG["en"]["note"]


# ----------------------------------------------------------------------------- fonts
def _font_dirs():
    dirs = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/Library/Fonts"]
    try:
        import matplotlib
        dirs.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"))
    except Exception:
        pass
    return [d for d in dirs if os.path.isdir(d)]


def _find(name):
    for d in _font_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _register_family(family, files):
    paths = [_find(f) for f in files]
    if not all(paths):
        return None
    from fontTools.ttLib import TTFont as _FT  # reportlab depends on nothing here; fontTools optional
    names = [family, family + "-Bold", family + "-Italic", family + "-BoldItalic"]
    for n, p in zip(names, paths):
        pdfmetrics.registerFont(TTFont(n, p))
    pdfmetrics.registerFontFamily(family, normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
    try:
        cmap = set(_FT(paths[0]).getBestCmap())
    except Exception:
        cmap = None
    return cmap


def _register_fonts():
    fonts = {}
    try:
        cm_dv = _register_family("DejaVu", ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
                                            "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"])
    except Exception:
        cm_dv = None
    try:
        cm_ar = _register_family("Body", ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"])
    except Exception:
        cm_ar = None
    if cm_ar is not None:
        fonts.update(body="Body", cmap=cm_ar)
    elif cm_dv is not None:
        fonts.update(body="DejaVu", cmap=cm_dv)
    else:
        fonts.update(body="Helvetica", cmap=None)
    fonts["fallback"] = "DejaVu" if cm_dv is not None else None
    fonts["fallback_cmap"] = cm_dv
    mono = None
    for fam, files in [("Mono", ["consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"]),
                       ("Mono", ["DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf",
                                 "DejaVuSansMono-Oblique.ttf", "DejaVuSansMono-BoldOblique.ttf"])]:
        try:
            if _register_family(fam, files) is not None:
                mono = fam
                break
        except Exception:
            pass
    fonts["mono"] = mono or "Courier"
    return fonts


FONTS = _register_fonts()
BODY_FONT = FONTS["body"]
BOLD_FONT = BODY_FONT + "-Bold" if BODY_FONT != "Helvetica" else "Helvetica-Bold"
ITAL_FONT = BODY_FONT + "-Italic" if BODY_FONT != "Helvetica" else "Helvetica-Oblique"

_ASCII = {"\u2265": ">=", "\u2264": "<=", "\u2212": "-", "\u2192": "->", "\u2190": "<-", "\u2248": "~",
          "\u2260": "!=", "\u2208": "in", "\u0394": "Delta", "\u03c3": "sigma", "\u03c1": "rho",
          "\u03c7": "chi", "\u2026": "...", "\u2009": " ", "\u202f": " ", "\u2020": "+", "\u2021": "++"}
MISSING_GLYPHS = set()


def glyphs(text_escaped):
    """Wrap characters the body font cannot draw in a fallback <font> tag (text already XML-escaped)."""
    cmap = FONTS["cmap"]
    if cmap is None:  # Helvetica: Windows-1252 only
        out = []
        for ch in text_escaped:
            try:
                ch.encode("cp1252")
                out.append(ch)
            except UnicodeEncodeError:
                MISSING_GLYPHS.add(ch)
                out.append(escape(_ASCII.get(ch, "?")))
        return "".join(out)
    out = []
    for ch in text_escaped:
        if ord(ch) < 128 or ord(ch) in cmap:
            out.append(ch)
        elif FONTS["fallback_cmap"] and ord(ch) in FONTS["fallback_cmap"]:
            out.append('<font name="%s">%s</font>' % (FONTS["fallback"], ch))
        else:
            MISSING_GLYPHS.add(ch)
            out.append(escape(_ASCII.get(ch, "?")))
    return "".join(out)


# ----------------------------------------------------------------------------- styles
def S(name, **kw):
    base = dict(fontName=BODY_FONT, fontSize=10.5, leading=14.5)
    base.update(kw)
    return ParagraphStyle(name, **base)


H1 = S("H1", fontName=BOLD_FONT, fontSize=15, leading=19, textColor=GREEN, spaceBefore=10, spaceAfter=8)
H2 = S("H2", fontName=BOLD_FONT, fontSize=12, leading=15, textColor=GREEN, spaceBefore=10, spaceAfter=5)
H3 = S("H3", fontName=BOLD_FONT, fontSize=10.5, leading=14, textColor=GREEN, spaceBefore=7, spaceAfter=3)
BODY = S("BODY", alignment=TA_JUSTIFY, spaceAfter=7)
BODYL = S("BODYL", alignment=TA_LEFT, spaceAfter=7)
LISTP = S("LISTP", alignment=TA_JUSTIFY, spaceAfter=3)
REF = S("REF", fontSize=9.5, leading=12.5, alignment=TA_LEFT, leftIndent=14, firstLineIndent=-14, spaceAfter=4)
CELL = S("CELL", fontSize=8.5, leading=10.6, alignment=TA_LEFT, embeddedHyphenation=1, splitLongWords=1)
CELLH = S("CELLH", fontName=BOLD_FONT, fontSize=8.5, leading=10.6, alignment=TA_LEFT, embeddedHyphenation=1,
          splitLongWords=1)
TITLE = S("TITLE", fontName=BOLD_FONT, fontSize=22, leading=27, textColor=GREEN, spaceAfter=10)
SUBT = S("SUBT", fontSize=12, leading=16, textColor=colors.HexColor("#444444"), spaceAfter=6)
NOTE = S("NOTE", fontSize=8.5, leading=11, textColor=GREY)


# ----------------------------------------------------------------------------- inline markdown
_TOKEN = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<link>\[(?P<ltext>[^\]]+)\]\((?P<lurl>[^)\s]+)\))"
    r"|(?P<url>https?://[^\s<>()]+[^\s<>().,;:])"
    r"|(?P<esc>\\[\\`*_|\[\]()#+\-.!])"
    r"|(?P<bold>\*\*)"
    r"|(?P<ital>\*)"
)


def inline(md):
    """Convert one block of inline Markdown to reportlab mini-HTML."""
    out, pos, bold, ital = [], 0, False, False

    def text(s):
        return glyphs(escape(s))

    for m in _TOKEN.finditer(md):
        out.append(text(md[pos:m.start()]))
        pos = m.end()
        if m.group("code"):
            out.append('<font name="%s" color="#333333">%s</font>' % (FONTS["mono"], text(m.group("code")[1:-1])))
        elif m.group("link"):
            url = m.group("lurl")
            out.append('<link href="%s" color="#1f4e8c">%s</link>' % (escape(url, {'"': "&quot;"}), inline(m.group("ltext"))))
        elif m.group("url"):
            url = m.group("url")
            out.append('<link href="%s" color="#1f4e8c">%s</link>' % (escape(url, {'"': "&quot;"}), text(url)))
        elif m.group("esc"):
            out.append(text(m.group("esc")[1]))
        elif m.group("bold"):
            out.append("</b>" if bold else "<b>")
            bold = not bold
        elif m.group("ital"):
            out.append("</i>" if ital else "<i>")
            ital = not ital
    out.append(text(md[pos:]))
    if ital:
        out.append("</i>")
    if bold:
        out.append("</b>")
    return "".join(out)


# ----------------------------------------------------------------------------- tables
def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, cur, i = [], [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur.append("\\|")
            i += 2
            continue
        if line[i] == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(line[i])
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _plain(s):
    return re.sub(r"[*`\\]", "", s)


def col_widths(rows, avail):
    """Widths proportional to a blend of the longest word and the total text per column."""
    n = max(len(r) for r in rows)
    minw, want = [], []
    for j in range(n):
        texts = [_plain(r[j]) if j < len(r) else "" for r in rows]
        # unbreakable tokens: whitespace-separated, and split after embedded hyphens (embeddedHyphenation)
        toks = [w for t in texts for w in re.split(r"\s+|(?<=-)", t) if w]
        longest_word = max((pdfmetrics.stringWidth(w, BOLD_FONT, 8.5) for w in toks), default=10)
        total = max(pdfmetrics.stringWidth(t, BODY_FONT, 8.5) for t in texts)
        avg = sum(pdfmetrics.stringWidth(t, BODY_FONT, 8.5) for t in texts) / max(len(texts), 1)
        minw.append(min(longest_word + 8, avail * 0.35))
        want.append(max(longest_word + 8, (total ** 0.5) * 6 + avg * 0.35))
    widths = [max(m, w) for m, w in zip(minw, want)]
    tot = sum(widths)
    if tot <= avail:
        return [w * avail / tot for w in widths]
    # shrink the flexible part only, keeping each column at least its minimum
    extra = [w - m for w, m in zip(widths, minw)]
    room = avail - sum(minw)
    if room <= 0:
        return [m * avail / sum(minw) for m in minw]
    k = room / sum(extra) if sum(extra) else 0
    return [m + e * k for m, e in zip(minw, extra)]


def make_table(lines, avail):
    rows = [split_row(l) for l in lines]
    header, body = rows[0], [r for r in rows[2:]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in body]
    widths = col_widths([header] + body, avail)
    data = [[Paragraph(inline(c), CELLH) for c in header]]
    data += [[Paragraph(inline(c), CELL) for c in r] for r in body]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
             ("GRID", (0, 0), (-1, -1), 0.4, GRID),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


# ----------------------------------------------------------------------------- block parser
RE_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
RE_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
RE_BUL = re.compile(r"^(\s*)[-*+]\s+(.*)$")
RE_NUM = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
RE_TSEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def parse(md_text, avail):
    lines = md_text.replace("\r\n", "\n").split("\n")
    story, title, i, in_refs = [], None, 0, False
    para = []

    def flush():
        nonlocal para
        if para:
            txt = " ".join(s.strip() for s in para)
            story.append(Paragraph(inline(txt), REF if in_refs else BODY))
            para = []

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            flush(); i += 1; continue
        m = RE_HEAD.match(ln)
        if m:
            flush()
            level, txt = len(m.group(1)), m.group(2)
            if level == 1 and title is None:
                title = txt
            else:
                in_refs = level <= 2 and re.match(r"^(\d+\.\s*)?(references|referências)\b", txt, re.I) is not None
                story.append(Paragraph(inline(txt), {1: H1, 2: H1, 3: H2}.get(level, H3)))
                # in_refs is reset by the next level-1/2 heading only
            i += 1
            continue
        if RE_HR.match(ln) and not ln.lstrip().startswith("|"):
            flush()
            story.append(HRFlowable(width="100%", thickness=0.6, color=GRID, spaceBefore=4, spaceAfter=8))
            i += 1
            continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and RE_TSEP.match(lines[i + 1]):
            flush()
            tl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tl.append(lines[i]); i += 1
            story.append(Spacer(1, 2))
            story.append(make_table(tl, avail))
            story.append(Spacer(1, 8))
            continue
        if RE_BUL.match(ln) or RE_NUM.match(ln):
            flush()
            items, ordered = [], bool(RE_NUM.match(ln))
            start = int(RE_NUM.match(ln).group(2)) if ordered else 1
            while i < len(lines):
                l2 = lines[i]
                mb, mn = RE_BUL.match(l2), RE_NUM.match(l2)
                if (mn if ordered else mb):
                    items.append([(mn.group(3) if ordered else mb.group(2))])
                    i += 1
                elif l2.strip() and items and (l2.startswith("   ") or l2.startswith("\t")) and not (mb or mn):
                    items[-1].append(l2.strip()); i += 1  # continuation line
                elif l2.strip() and items and not (mb or mn) and not RE_HEAD.match(l2) \
                        and not l2.lstrip().startswith("|") and not RE_HR.match(l2):
                    items[-1].append(l2.strip()); i += 1  # lazy continuation
                else:
                    break
            flow = [ListItem(Paragraph(inline(" ".join(it)), LISTP), leftIndent=18) for it in items]
            kw = dict(bulletType="1", start=start, bulletFormat="%s.") if ordered else dict(bulletType="bullet", start="\u2022")
            story.append(ListFlowable(flow, leftIndent=18, bulletFontName=BODY_FONT, bulletFontSize=10.5, bulletOffsetY=-1, **kw))
            story.append(Spacer(1, 5))
            continue
        para.append(ln)
        i += 1
    flush()
    return title, story


# ----------------------------------------------------------------------------- document
def build(md_path, pdf_path, date_str, lang="en"):
    L = LANG.get(lang, LANG["en"])
    with open(md_path, encoding="utf-8") as f:
        md = f.read()
    left = right = top = bottom = 2 * cm
    pad = 6  # SimpleDocTemplate frame padding; margins are offset so content edges sit exactly at 2 cm
    avail = letter[0] - left - right
    title, body = parse(md, avail)
    title = title or os.path.splitext(os.path.basename(md_path))[0].replace("_", " ")
    short = re.split(r":\s", title)[0]

    cover = [Spacer(1, 6 * cm),
             Paragraph(inline(title), TITLE),
             HRFlowable(width="100%", thickness=1.4, color=GREEN, spaceBefore=4, spaceAfter=14),
             Paragraph(glyphs(escape(L["org"])), SUBT),
             Paragraph(glyphs(escape(date_str)), SUBT),
             Spacer(1, 9 * cm),
             Paragraph(glyphs(escape(L["note"])), NOTE),
             PageBreak()]

    def on_page(canv, doc):
        canv.saveState()
        canv.setFont(BODY_FONT, 8)
        canv.setFillColor(GREY)
        w = letter[0]
        if doc.page > 1:
            canv.drawString(left, bottom - 0.9 * cm, short[:110])
        canv.drawRightString(w - right, bottom - 0.9 * cm, L["page"] % doc.page)
        canv.setStrokeColor(GRID)
        canv.setLineWidth(0.4)
        canv.line(left, bottom - 0.55 * cm, w - right, bottom - 0.55 * cm)
        canv.restoreState()

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=left - pad, rightMargin=right - pad,
                            topMargin=top - pad, bottomMargin=bottom - pad, title=title,
                            author="Olds College OFPE team", subject=L["org"],
                            creator="build_pdf.py (reportlab)")
    doc.build(cover + body, onFirstPage=on_page, onLaterPages=on_page)
    return doc.page


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("md", nargs="?", default=DEFAULT_MD)
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--date", default="29 October 2026")
    ap.add_argument("--lang", default="en", choices=sorted(LANG))
    a = ap.parse_args(argv)
    pdf = a.pdf or os.path.splitext(a.md)[0] + ".pdf"
    pages = build(a.md, pdf, a.date, a.lang)
    print("wrote %s (%d pages); body font %s, mono %s" % (pdf, pages, BODY_FONT, FONTS["mono"]))
    if MISSING_GLYPHS:
        print("WARNING: characters replaced by ASCII:", sorted(MISSING_GLYPHS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
