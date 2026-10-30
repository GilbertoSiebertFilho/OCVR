# -*- coding: utf-8 -*-
"""Build the Olds College apps bundle for a new computer: OCVR + OCCD + NDVI Map Viewer with ONE installer.

    python tools\\make_bundle.py <empty dest dir> [--zip <file.zip>] [--occd <OCCD folder>]

Layout of <dest> (unzip it to C:\\OC_Apps on the new computer):
    INSTALAR.bat        one click: installs Python 3.14.3 if missing, builds .venv with the tested versions, shortcuts
    OCVR.bat, OCCD.bat, NDVI.bat   launchers (they run INSTALAR.bat first when .venv is missing)
    LEIA-ME.txt         the steps, in Portuguese
    requirements_all.txt, constraints.txt   both apps' packages; the exact versions of this computer (pip freeze)
    OCVR\\              tools\\make_portable.py output (projects with their inputs), without its own NDVIMAP copy
    OCCD\\              the cleaning app's code and docs (no workspace\\, _archive\\ or examples\\)
    NDVIMAP\\           the NDVI Map Viewer without data\\ (no Copernicus credentials)
    tools\\make_shortcuts.ps1, tools\\oc.ico
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OCVR_ROOT = os.path.dirname(HERE)
OCCD_DEFAULT = os.path.join(os.path.dirname(OCVR_ROOT), "App limpeza DIFM")
OCCD_ITEMS = ["app", "engine", "assets", "docs", "tools", "tests", "README.md", "INSTALAR.txt", "requirements.txt",
              "constraints.txt", "pytest.ini", "run_app.bat", "install_occd.bat"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.tmp")


def req_names(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            out.append(line)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dest")
    ap.add_argument("--zip")
    ap.add_argument("--occd", default=OCCD_DEFAULT)
    a = ap.parse_args(argv)
    dest = os.path.abspath(a.dest)
    if os.path.exists(dest) and os.listdir(dest):
        print(f"{dest} is not empty: choose a new folder")
        return 2
    os.makedirs(dest, exist_ok=True)
    # OCVR (with its projects), then its NDVIMAP copy moved to the top so both apps share one viewer
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_portable.py"), os.path.join(dest, "OCVR")])
    if r.returncode:
        return r.returncode
    if os.path.isdir(os.path.join(dest, "OCVR", "NDVIMAP")):
        shutil.move(os.path.join(dest, "OCVR", "NDVIMAP"), os.path.join(dest, "NDVIMAP"))
    # OCCD code
    for it in OCCD_ITEMS:
        s = os.path.join(a.occd, it)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dest, "OCCD", it), ignore=SKIP)
        elif os.path.exists(s):
            os.makedirs(os.path.join(dest, "OCCD"), exist_ok=True)
            shutil.copy2(s, os.path.join(dest, "OCCD", it))
    # installer, launchers, readme
    for fn in ("INSTALAR.bat", "OCVR.bat", "OCCD.bat", "NDVI.bat", "LEIA-ME.txt"):
        shutil.copy2(os.path.join(HERE, "bundle", fn), os.path.join(dest, fn))
    os.makedirs(os.path.join(dest, "tools"), exist_ok=True)
    shutil.copy2(os.path.join(HERE, "bundle", "make_shortcuts.ps1"), os.path.join(dest, "tools", "make_shortcuts.ps1"))
    try:
        from PIL import Image
        Image.open(os.path.join(OCVR_ROOT, "assets", "brand", "oc_logo_shield.png")).convert("RGBA").save(
            os.path.join(dest, "tools", "oc.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    except Exception as exc:
        print("icon not made:", exc)
    # requirements of both apps, and the exact versions tested on this computer
    reqs = []
    for p in (os.path.join(OCVR_ROOT, "requirements.txt"), os.path.join(a.occd, "requirements.txt")):
        for line in req_names(p):
            name = re.split(r"[<>=!~ ]", line)[0].lower()
            if name not in [re.split(r"[<>=!~ ]", x)[0].lower() for x in reqs]:
                reqs.append(line)
    with open(os.path.join(dest, "requirements_all.txt"), "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("# OCVR + OCCD packages (INSTALAR.bat installs them within constraints.txt)\n" + "\n".join(reqs) + "\n")
    frz = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    frz = [l for l in frz if "==" in l and "@" not in l and not l.startswith("-e")]
    with open(os.path.join(dest, "constraints.txt"), "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(f"# exact versions tested on the development computer (Python {sys.version.split()[0]}, "
                 f"pip freeze); pip -c applies only to the packages being installed\n" + "\n".join(frz) + "\n")
    # batch files need CRLF line endings for cmd.exe
    for fn in ("INSTALAR.bat", "OCVR.bat", "OCCD.bat", "NDVI.bat", "LEIA-ME.txt"):
        p = os.path.join(dest, fn)
        t = open(p, encoding="utf-8").read().replace("\r\n", "\n").replace("\n", "\r\n")
        open(p, "w", encoding="utf-8", newline="").write(t)
    if a.zip:
        zp = os.path.abspath(a.zip)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(dest):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join("OC_Apps", os.path.relpath(full, dest)))
        print(f"zip: {zp} ({os.path.getsize(zp) / 1e6:.1f} MB)")
    print(f"bundle: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
