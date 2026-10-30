# -*- coding: utf-8 -*-
"""Assemble OCVR_method_and_references.md from the section files, check every citation against references.csv,
write rule_table.md, run update_refs.py (PDF reference page + docs/LITERATURE.md) and build the PDF.

    python assemble.py [<sections_dir>] [--pt]

--pt builds the Portuguese version (sections_pt/ -> OCVR_metodo_e_referencias.md / .pdf) and leaves the app's
reference page and docs/LITERATURE.md alone.

Section files, in order: front.md, zones.md, join.md, response.md, economics1.md, economics2.md, back.md. The
reference list replaces REFLIST_PLACEHOLDER in back.md. A citation is any 'Key YYYY[a-z]' whose key is in
references.csv; the check also reports words that look like citations (Capitalised word + 4-digit year) but match no
key, and references no section cites. Exit 1 when an unknown citation is found.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["front.md", "zones.md", "join.md", "response.md", "economics1.md", "economics2.md", "back.md"]


PT = {"verified": "verificado", "full text": "texto completo", "abstract": "resumo", "not read": "não lido"}


def main(sections_dir, lang="en"):
    pt = lang == "pt"
    with open(os.path.join(HERE, "references.csv"), encoding="utf-8-sig", newline="") as fh:
        refs = {r["key"]: r for r in csv.DictReader(fh)}
    parts = []
    for name in ORDER:
        p = os.path.join(sections_dir, name)
        if not os.path.exists(p):
            print("MISSING", p)
            return 2
        parts.append(open(p, encoding="utf-8").read().strip() + "\n")
    text = "\n\n".join(parts)
    # the rule table: every '| Rule | Support | ...' table row of the sections, in order
    rows = [ln for ln in text.splitlines() if re.match(r"^\|\s*[A-Z]\d{1,2}\s*\|", ln)]
    table = "| Rule | Support | Verified references (keys) | House choices |\n|---|---|---|---|\n" + "\n".join(rows)
    if not pt:
        open(os.path.join(HERE, "rule_table.md"), "w", encoding="utf-8", newline="").write(table + "\n")
    # citation check
    keys = sorted(refs, key=len, reverse=True)
    cited = {}
    for k in keys:
        n = len(re.findall(re.escape(k) + r"(?![a-z0-9])", text))
        if n:
            cited[k] = n
    looks = set(re.findall(r"\b([A-Z][A-Za-z'’\-]+(?: et al\.)?(?:,? (?:&|and) [A-Z][A-Za-z'’\-]+)? (?:19|20)\d{2}[a-z]?)\b", text))
    unknown = sorted(x for x in looks if not any(x.startswith(k.split()[0]) and x.endswith(k.split()[-1]) for k in refs)
                     and not re.match(r"^(Trial|Section|Figure|Table|October|June|August|May|In|On|For|The|Version|Prepared|Appendix|"
                                      r"Seção|Tabela|Figura|Em|Na|No|Versão|Preparado|Apêndice) ", x))
    uncited = sorted(k for k in refs if k not in cited)
    print(f"{len(cited)} of {len(refs)} references cited; {sum(cited.values())} citations")
    if unknown:
        print("UNKNOWN citation-like strings:", unknown)
    if uncited:
        print("not cited anywhere:", uncited)
    # reference list
    sys.path.insert(0, HERE)
    import update_refs as U
    refs_, log, ev = U.load()
    acc = U.access_levels(ev, refs_)
    lines = []
    for k in sorted(refs_, key=lambda s: s.lower()):
        r = refs_[k]
        st = log.get(k, {}).get("status", "?")
        a_ = acc.get(k, "not read")
        if pt:
            used = " (não citada neste documento)" if k not in cited else ""
            lines.append(f"- **{k}** — {r['authors']}. {r['year']}. {r['title']}. {r['venue']}. DOI {r['doi']} — "
                         f"{PT.get(st, st)}; lido: {PT.get(a_, a_)}.{used}")
        else:
            used = " (not cited in this document)" if k not in cited else ""
            lines.append(f"- **{k}** — {r['authors']}. {r['year']}. {r['title']}. {r['venue']}. DOI {r['doi']} — {st}; "
                         f"read: {a_}.{used}")
    text = text.replace("REFLIST_PLACEHOLDER", "\n".join(lines))
    base = "OCVR_metodo_e_referencias" if pt else "OCVR_method_and_references"
    out = os.path.join(HERE, base + ".md")
    open(out, "w", encoding="utf-8", newline="").write(text)
    print("wrote", out, len(text), "chars")
    if not pt:
        # the English build also rewrites the PDF reference page of the app's report and docs/LITERATURE.md
        U.write_report_refs(refs_)
        U.write_literature_md(refs_, log, ev, table)
    r = subprocess.run([sys.executable, os.path.join(HERE, "build_pdf.py"), out, os.path.join(HERE, base + ".pdf"),
                        "--date", "29 de outubro de 2026" if pt else "29 October 2026", "--lang", lang],
                       capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-2000:], r.stderr[-2000:])
    return 1 if unknown else 0


if __name__ == "__main__":
    lang_ = "pt" if "--pt" in sys.argv else "en"
    args_ = [a for a in sys.argv[1:] if a != "--pt"]
    raise SystemExit(main(args_[0] if args_ else os.path.join(HERE, "sections_pt" if lang_ == "pt" else "sections"), lang_))
