# -*- coding: utf-8 -*-
"""Build a portable copy of OCVR to open on another computer.

    python tools\\make_portable.py <dest_dir> [--zip <file.zip>] [--no-workspace]

The copy holds the app (engine, app, tests, docs, assets, README, requirements, setup.bat, run_app.bat), a copy of the
NDVI Map Viewer (NDVIMAP\\, which the NDVI fetch imports) WITHOUT its data\\ folder (Copernicus credentials, cache and
downloads stay on this computer: enter the client id and secret once in the viewer on the new computer), and the saved
projects of workspace\\. Input files a project reads from outside its own folder (a design or yield file left in the
grower's folder) are copied into <project>\\inputs\\external\\ with their shapefile sidecars, and project.json is
rewritten to point at them, so every project opens and runs on the new computer. Paths inside the project are
relocated by engine.project.load when it opens. Nothing on this computer is changed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_ITEMS = ["engine", "app", "tests", "docs", "assets", "tools", "README.md", "requirements.txt", "pytest.ini",
             "setup.bat", "run_app.bat"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.tmp")
SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".sbn", ".sbx", ".shp.xml")


def ndvimap_src():
    sys.path.insert(0, ROOT)
    from engine import ndvi_fetch as NF
    return NF.NDVIMAP_DIR


def copy_with_sidecars(src, dst_dir):
    """Copy a vector file (with its shapefile sidecars) or a folder into dst_dir; return the new path."""
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.isdir(src):
        dst = os.path.join(dst_dir, os.path.basename(src.rstrip("\\/")))
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=SKIP)
        return dst
    base, ext = os.path.splitext(src)
    if ext.lower() == ".shp":
        for e in SIDECARS:
            for cand in (base + e, base + e.upper()):
                if os.path.exists(cand):
                    shutil.copy2(cand, os.path.join(dst_dir, os.path.basename(cand)))
    else:
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
    return os.path.join(dst_dir, os.path.basename(src))


def pack_project(src_dir, dst_dir, log):
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True, ignore=SKIP)
    pj = os.path.join(dst_dir, "project.json")
    if not os.path.exists(pj):
        return
    with open(pj, encoding="utf-8") as fh:
        rec = json.load(fh)
    src_abs = os.path.abspath(src_dir).replace("\\", "/").lower()
    copied = {}

    def fix(p):
        if not p or not isinstance(p, str) or not os.path.exists(p):
            return p
        q = os.path.abspath(p).replace("\\", "/")
        if q.lower().startswith(src_abs + "/"):
            return os.path.join(dst_dir, os.path.relpath(p, src_dir))        # inside the project: same place
        if q in copied:
            return copied[q]
        tag = hashlib.sha1(q.encode()).hexdigest()[:8]
        new = copy_with_sidecars(p, os.path.join(dst_dir, "inputs", "external", tag))
        copied[q] = new
        log(f"   external input copied into the project: {p}")
        return new
    sys.path.insert(0, ROOT)
    from engine import project as P
    P._walk_paths(rec["config"].get("files") or {}, fix)
    rec["workdir"] = os.path.abspath(dst_dir)
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dest")
    ap.add_argument("--zip")
    ap.add_argument("--no-workspace", action="store_true")
    a = ap.parse_args(argv)
    dest = os.path.abspath(a.dest)
    if os.path.exists(dest) and os.listdir(dest):
        print(f"{dest} is not empty: choose a new folder")
        return 2
    os.makedirs(dest, exist_ok=True)
    for it in APP_ITEMS:
        s = os.path.join(ROOT, it)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dest, it), ignore=SKIP)
        elif os.path.exists(s):
            shutil.copy2(s, os.path.join(dest, it))
    nm = ndvimap_src()
    if os.path.isdir(os.path.join(nm, "server")):
        shutil.copytree(nm, os.path.join(dest, "NDVIMAP"), ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc"))
        print(f"NDVI Map Viewer copied from {nm} (without data\\: credentials, cache and downloads)")
    else:
        print(f"WARNING: NDVI Map Viewer not found at {nm}; NDVI fetch will not work in the copy")
    if not a.no_workspace:
        for d in sorted(glob.glob(os.path.join(ROOT, "workspace", "*"))):
            if os.path.isdir(d):
                print("project", os.path.basename(d))
                pack_project(d, os.path.join(dest, "workspace", os.path.basename(d)), print)
    if a.zip:
        zp = os.path.abspath(a.zip)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(dest):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join("OCVR", os.path.relpath(full, dest)))
        print(f"zip: {zp} ({os.path.getsize(zp) / 1e6:.1f} MB)")
    print(f"portable copy: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
