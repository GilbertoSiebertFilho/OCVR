# -*- coding: utf-8 -*-
"""Rewrite the PDF's reference list (engine/report.py REFS) and the verified list of docs/LITERATURE.md from
references.csv + verification_log.json + evidence_by_rule.json.

    python update_refs.py

REFS gets the references the PDF's text cites (REPORT_KEYS below), in the short house form. LITERATURE.md section C
gets every reference of references.csv with its verification status and how it was read in the cite-check (full
text / abstract / not read), derived from evidence_by_rule.json (the best access level any judge reached).
"""
from __future__ import annotations

import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

REPORT_KEYS = ["Blackmore 2000", "Lenoir 2024", "Basso 2019", "Georgi 2018", "Mkhabela 2011", "Taylor 2007",
               "Bullock 2019", "dos Santos 2025", "Chung 2002", "Trevisan 2021", "Cerrato 1990", "Bullock 1994",
               "Miguez 2022", "Cameron 2015", "Abadie 2023", "Anselin 2004", "Scharf 2006", "Swinton 1998",
               "Bullock 2002", "Mieno 2024", "Chernozhukov 2018", "Athey 2016", "Hernandez 2008", "Jaynes 2011",
               "Pannell 2006", "Pannell 2019", "Rajsic 2008", "Babcock 1992", "Puntel 2024", "Tukey 1949",
               "Khakbazan 2013"]


DEFAULTS_MD = """
| Parameter | Default | Support (keys; see section A) |
|---|---|---|
| NDVI seasons | all clear seasons before the trial year from 2017; same crop when ≥ 3 known, else the crop group, else all | Lenoir 2024 (≥ 3 seasons); Georgi 2018; Joernsgaard 2003 |
| NDVI per season | median of the 3 clearest dates with the highest field-mean NDVI, 15 Jun – 15 Aug, relative to the field mean | house choice; window: Mkhabela 2011 |
| Zones | 3 (2–5), equal-area quantiles (k-means optional), 3 × 3 smoothing, patches < 0.5 ha merged | house choice; analogues Ping 2003, Javadi 2022 |
| End trim | 10 m at each plot end (side trim 0) | Bullock 2019; Trevisan 2021 |
| Footprint share | ≥ 95 % of the record's swath × distance rectangle inside one trimmed plot | house choice; dos Santos 2025 for footprints |
| Analysis unit | plot × zone, ≥ 10 records and ≥ 150 m² | house choice |
| Response form | quadratic; quadratic-plateau always reported; straight line with 2 design rates | Cerrato 1990; Bullock 1994; Miguez 2022 |
| Inference | area-weighted LS, SE clustered by plot, Wald tests against F(h, plots − 1) | Cameron 2015; Abadie 2023 |
| Zone merge | < 20 units or < 3 design rates → merged into the NDVI-nearest zone | house choice |
| Cross-fit | 2-fold split of plots stratified by design rate, 30 repeats, judged on the common rate support | Chernozhukov 2018; Athey 2016; Mieno 2024 |
| Bootstrap | 500 plot resamples for new projects (1,000 to publish) | Hernandez 2008; Jaynes 2011; Field 2007 |
| VR verdict | mean cross-fitted gain ≥ $3/ac after the VR cost, ≥ 70 % of repeats positive, zone × rate p < 0.10 | house calibration; Pannell 2006, 2019 |
| Flat rate vs status quo | same gate; a higher rate also needs pooled response p < 0.10 | house rule; Babcock 1992; Rajsic 2008 |
| Price grid | crop and input price ± 30 % | house choice; Pannell 2006 |
| Status quo outside the range | linear extension with the edge slope; not priced beyond 25 % of the range | house choice; Scharf 2005 for optima kept in range |
| Treatment strips | RCBD on strip means, Tukey HSD at α = 0.05 | Tukey 1949; Kyveryga 2018 |
| Units | lb/ac in the input's basis; kg/ha × 0.892179; urea 46-0-0 | Khakbazan 2013 |
"""


def load():
    with open(os.path.join(HERE, "references.csv"), encoding="utf-8-sig", newline="") as fh:
        refs = {r["key"]: r for r in csv.DictReader(fh)}
    with open(os.path.join(HERE, "verification_log.json"), encoding="utf-8") as fh:
        log = json.load(fh)["log"]
    with open(os.path.join(HERE, "evidence_by_rule.json"), encoding="utf-8") as fh:
        ev = json.load(fh)
    return refs, log, ev


def short_authors(a):
    parts = [p.strip() for p in a.split(",") if p.strip()]
    if len(parts) <= 3:
        return ", ".join(parts)
    return parts[0] + " et al."


def report_line(r):
    a = short_authors(r['authors'])
    return f"{a}{'' if a.endswith('.') else '.'} {r['year']}. {r['title']}. {r['venue']}. doi:{r['doi']}"


def access_levels(ev, refs):
    """key -> best access level reached by any judge ('full text' > 'abstract' > 'not read'), from the pairs (by
    LITERATURE.md number) and the new_refs (by DOI)."""
    rank = {"full text": 3, "abstract": 2, "not read": 1}
    by_num = {}
    for r in refs.values():
        m = re.search(r"\[(\d+)\]", r.get("source") or "")
        if m:
            by_num[int(m.group(1))] = r["key"]
    by_doi = {r["doi"].lower(): r["key"] for r in refs.values()}
    out = {}

    def level(txt):
        t = (txt or "").lower()
        if "full text" in t or "full-text" in t or "read in full" in t:
            return "full text"
        if "abstract" in t or "snippet" in t or "introduction" in t:
            return "abstract"
        return "not read"

    def put(key, lv):
        if key and rank[lv] > rank.get(out.get(key, "not read"), 0):
            out[key] = lv
    for j in ev["judgements"]:
        for p in j["pairs"]:
            lv = level(p.get("how_read")) if p.get("support") != "unread" else level(p.get("how_read"))
            if p.get("support") == "unread" and lv == "full text":
                lv = "abstract"
            put(by_num.get(int(p["ref"])), lv)
        for n in j["new_refs"]:
            key = by_doi.get(n["doi"].lower().replace("https://doi.org/", ""))
            t = (n.get("supports_what") or "").lower()
            put(key, "full text" if ("full text" in t or "read in full" in t or "full-text" in t) else "abstract")
    return out


def write_report_refs(refs):
    p = os.path.join(ROOT, "engine", "report.py")
    s = open(p, encoding="utf-8", newline="").read()
    i = s.index("REFS = [")
    j = s.index("]", i) + 1
    lines = [f"    {json.dumps(report_line(refs[k]), ensure_ascii=False)}," for k in REPORT_KEYS]
    block = ("REFS = [\n" + "\n".join(lines) +
             '\n    "Full list, with how each paper was read and checked: docs/literature/OCVR_method_and_references.pdf",\n]')
    s = s[:i] + block + s[j:]
    open(p, "w", encoding="utf-8", newline="").write(s)
    print("report.py REFS:", len(REPORT_KEYS))


def write_literature_md(refs, log, ev, rule_table_md):
    acc = access_levels(ev, refs)
    keys = sorted(refs, key=lambda k: k.lower())
    lines = ["# OCVR literature base: rule-to-reference table and verified reference list",
             "",
             "The full document, with what each paper says and how each rule is supported, is "
             "`docs/literature/OCVR_method_and_references.pdf` (Markdown beside it). This file is generated from "
             "`references.csv`, `verification_log.json` and `evidence_by_rule.json` by `update_refs.py`; edit those, "
             "not this.",
             "",
             "## A. Rules and the references that support them",
             "",
             rule_table_md.strip(),
             "",
             "## B. Defaults the app ships with (engine/project.py default_config, 29 October 2026)",
             "",
             DEFAULTS_MD.strip(),
             "",
             "## C. Verified reference list",
             "",
             f"{len(keys)} references. Every DOI was resolved on 2026-10-29 on Crossref (OpenAlex as the second source) "
             "and the returned title, first author and year compared with the citation (`verify_dois.py`). 'Read' says "
             "how far the cite-check could read the paper: full text, abstract (or publisher preview), or not read.",
             ""]
    for k in keys:
        r = refs[k]
        st = log.get(k, {}).get("status", "?")
        note = log.get(k, {}).get("note") or ""
        lines.append(f"- **{k}** — {r['authors']}. {r['year']}. {r['title']}. {r['venue']}. "
                     f"DOI {r['doi']} — {st}{(' (' + note + ')') if note else ''}; read: {acc.get(k, 'not read')}.")
    p = os.path.join(ROOT, "docs", "LITERATURE.md")
    open(p, "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")
    print("LITERATURE.md:", len(keys), "references")


if __name__ == "__main__":
    import sys
    refs, log, ev = load()
    missing = [k for k in REPORT_KEYS if k not in refs]
    assert not missing, missing
    write_report_refs(refs)
    tbl = open(os.path.join(HERE, "rule_table.md"), encoding="utf-8").read() if os.path.exists(os.path.join(HERE, "rule_table.md")) else "(table pending)"
    write_literature_md(refs, log, ev, tbl)
