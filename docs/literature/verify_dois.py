# -*- coding: utf-8 -*-
"""Verify every reference of references.csv against Crossref (and OpenAlex as a second source).

    python verify_dois.py [references.csv] [verification_log.json]

references.csv columns: key, authors, year, title, venue, volume, issue, pages, doi. For each row the DOI is resolved
on Crossref; the returned title, year, container and first author are compared with the row (title similarity by
token overlap, year equal, first-author surname present). Rows that fail on Crossref are tried on OpenAlex. The log
holds, per key: status (verified / mismatch / not_found), the record found and the fields compared. Exit 1 when any
row is not verified, so the build can refuse an unverified list.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time

import requests

UA = {"User-Agent": "OCVR-literature-check/1.0 (mailto:ocvr@oldscollege.ca)"}


def tokens(s):
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if len(t) > 2}


def sim(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def crossref(doi):
    r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=30)
    if r.status_code != 200:
        return None
    m = r.json()["message"]
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        dp = (m.get(k) or {}).get("date-parts") or []
        if dp and dp[0] and dp[0][0]:
            year = int(dp[0][0])
            break
    auth = m.get("author") or []
    return dict(title=(m.get("title") or [""])[0], year=year, container=(m.get("container-title") or [""])[0],
                first_author=(auth[0].get("family") if auth else None), volume=m.get("volume"), pages=m.get("page"),
                source="crossref")


def openalex(doi):
    r = requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}", headers=UA, timeout=30)
    if r.status_code != 200:
        return None
    m = r.json()
    auth = m.get("authorships") or []
    fa = (auth[0].get("author") or {}).get("display_name") if auth else None
    return dict(title=m.get("title") or "", year=m.get("publication_year"),
                container=((m.get("primary_location") or {}).get("source") or {}).get("display_name"),
                first_author=(fa.split()[-1] if fa else None), volume=(m.get("biblio") or {}).get("volume"),
                pages=(m.get("biblio") or {}).get("first_page"), source="openalex")


def check(row, rec):
    if rec is None:
        return "not_found", {}
    comp = dict(title_sim=round(sim(row["title"], rec["title"]), 2),
                year_ok=(str(rec.get("year")) == str(row["year"]).strip()[:4]) if rec.get("year") else None,
                author_ok=bool(rec.get("first_author")) and rec["first_author"].lower().split()[-1] in row["authors"].lower())
    ok = comp["title_sim"] >= 0.6 and comp["author_ok"] and comp["year_ok"] is not False
    # a one-year difference between online-first and print is accepted when title and author match well
    if not ok and comp["title_sim"] >= 0.8 and comp["author_ok"] and rec.get("year") and abs(int(rec["year"]) - int(str(row["year"])[:4])) <= 1:
        ok = True
        comp["year_note"] = "online-first vs print year"
    return ("verified" if ok else "mismatch"), comp


def main(argv=None):
    a = argv or sys.argv[1:]
    src = a[0] if a else "references.csv"
    out = a[1] if len(a) > 1 else "verification_log.json"
    log, bad = {}, 0
    with open(src, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        doi = (row.get("doi") or "").strip().replace("https://doi.org/", "")
        key = row["key"]
        if not doi:
            log[key] = dict(status="no_doi", comparison={}, record=None)
            bad += 1
            print(f"{key}: no DOI")
            continue
        rec = None
        for fn in (crossref, openalex):
            try:
                rec = fn(doi)
            except Exception as exc:
                rec = None
                print(f"{key}: {fn.__name__} error {exc}")
            status, comp = check(row, rec)
            if status == "verified":
                break
            time.sleep(0.3)
        log[key] = dict(status=status, comparison=comp, record=rec, doi=doi)
        if status != "verified":
            bad += 1
        print(f"{key}: {status} {comp}")
        time.sleep(0.2)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(checked=len(rows), verified=len(rows) - bad, log=log), fh, indent=1, ensure_ascii=False)
    print(f"{len(rows) - bad} of {len(rows)} verified -> {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
