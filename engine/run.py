# -*- coding: utf-8 -*-
r"""Command line: run a whole project from its config.

    python -m engine.run <project.json | config.json> [--out WORKDIR] [--no-fetch] [--B 100] [--R 30]

<project.json> is the file the app saves in every project folder ({"config": {...}}); a bare config dict (the keys of
engine.project.default_config()) works too. Without --out the project's own folder is used, except that a run with
--B or --R (fewer resamples than the project's settings) goes to <project>\quick_B<B>_R<R>, so the saved project and
its outputs are not replaced by a quick run. Exit codes: 0 done, 2 a problem with the inputs (the message says
what), 3 an unexpected error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from . import project as P


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m engine.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out")
    ap.add_argument("--no-fetch", action="store_true", help="do not download NDVI (use files.ndvi_dir as it is)")
    ap.add_argument("--B", type=int, help="bootstrap resamples")
    ap.add_argument("--R", type=int, help="cross-fit repeats")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        rec = json.load(f)
    cfg = rec.get("config", rec)
    if a.B is not None:
        cfg.setdefault("model", {})["B"] = a.B
    if a.R is not None:
        cfg.setdefault("model", {})["R"] = a.R
    out = a.out or os.path.dirname(os.path.abspath(a.config))
    if (a.B is not None or a.R is not None) and not a.out:
        # write_outputs saves the config it ran with into project.json of the output folder: a quick run with fewer
        # resamples used to replace the project's own settings and results, silently. It goes to a folder of its own.
        out = os.path.join(out, f"quick_B{cfg['model'].get('B')}_R{cfg['model'].get('R')}")
        print(f"--B/--R override the project's settings: this run is written to {out}, the project is untouched")
    try:
        st = P.run(cfg, out, log=print, progress=lambda f, m: None, fetch=not a.no_fetch)
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        # a missing vector file raises FileNotFoundError (io.read_vector); rasterio's errors are OSError subclasses
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:                                                 # pragma: no cover
        traceback.print_exc()
        return 3
    for name, R in st.res.get("analysis", {}).items():
        if name.startswith("_"):
            continue
        # an input analyse() skipped (fewer than 2 distinct design rates) has no verdict, only the reason; reading
        # R['verdict'] crashed the command after the PDF had already been written
        if R.get("skipped"):
            print(f"{name}: not analysed - {R['skipped']}")
            continue
        v = R.get("verdict") or {}
        print(f"{name}: {v.get('text', '-')}")
        for r in v.get("reasons") or []:
            print(f"   - {r}")
    T = st.res.get("analysis", {}).get("_treatments")
    if T:
        for var in ("yield", "net_return"):
            p = (T.get(var) or {}).get("p_treatment")
            if p is not None:
                print(f"treatments ({var}): p = {p:.3f}")
    print(f"report: {getattr(st, 'report_path', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
