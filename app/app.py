# -*- coding: utf-8 -*-
"""OCVR - Olds College Variable-Rate economics (Streamlit).

    python -m streamlit run app/app.py        (or run_app.bat)

Six steps, one tab each; every step writes into the project folder workspace/<stamp>_<name>/ and saves project.json, so
a project can be closed and opened again from the sidebar:
    1 Trial          name, crop, files (design, cleaned yield, boundary, as-applied), tested inputs and status quo
    2 NDVI & zones   multi-year NDVI (fetched from Copernicus, GeoTIFFs, or a ready map) or a zone map -> zones
    3 Prices         crop sale price, input purchase prices, VR cost, sensitivity ranges
    4 Join           plots x yield x zones -> analysis units
    5 Economics      response by zone / NDVI, optima, VR vs flat, cross-fit, bootstrap, prices, realised trial
    6 Report         PDF, prescription shapefile, tables
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import brand  # noqa: E402
from engine import project as P, report as RP, io as eio, prices as EP, ndvi_fetch as NF  # noqa: E402

WORKSPACE = os.environ.get("OCVR_WORKSPACE", os.path.join(ROOT, "workspace"))
os.makedirs(WORKSPACE, exist_ok=True)
CROPS = ["wheat", "durum", "barley", "oats", "canola", "peas", "lentils", "flax", "corn", "soybeans", "other"]
# An input name becomes part of column names (target_<name>, rate_<name>) and of model terms, so it must be a plain
# identifier: a name such as 'urea N' or 'N-rate' could not be parsed in the response model of the other input and
# the economics crashed. Its length is free: the prescription shapefile's 10-character field names are made unique
# by the engine (_shp_field_names) and zones_rx_fields.csv, shipped in the prescription zip, maps them to inputs.
INPUT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# Names the engine already uses as column names of the plots and units. An input after the first gets the plot column
# design_<name>, so an input named 'rate' wrote its rates into design_rate, the first input's column: the seed
# economics then ran on the urea design without an error (or the trial failed to load on a duplicated column). The
# other names collide the same way with columns of the units (yield, zone, ndvi, target, plot_id, ...). Compared
# case-blind, because a Windows disk and the column lookups of the engine are case-blind too.
RESERVED_INPUT_NAMES = ("rate", "design_rate", "target", "yield", "zone", "ndvi", "plot_id", "treatment", "block")

st.set_page_config(page_title=brand.NAME, page_icon=brand.page_icon(), layout="wide")
brand.inject()
try:
    st.logo(brand.LOGO_H, icon_image=brand.LOGO_SHIELD, size="large")
except Exception:
    pass


# ================================================================================================ session
def ss():
    return st.session_state


def new_project():
    for k in list(ss().keys()):
        del ss()[k]
    ss()["cfg"] = P.default_config()
    ss()["pid"] = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ss()["workdir"] = None
    ss()["state"] = None
    ss()["done"] = {}


if "cfg" not in ss():
    new_project()
cfg = ss()["cfg"]
PID = ss()["pid"]


def k(name):
    """Widget key scoped to the open project, so opening another project resets the widgets."""
    return f"{PID}:{name}"


def workdir():
    if not ss().get("workdir"):
        nm = re.sub(r"[^A-Za-z0-9_-]+", "_", cfg.get("name") or "field").strip("_")[:40] or "field"
        ss()["workdir"] = os.path.join(WORKSPACE, f"{ss()['pid']}_{nm}")
        os.makedirs(ss()["workdir"], exist_ok=True)
    return ss()["workdir"]


def state():
    s = ss().get("state")
    if s is None or s.workdir != workdir():
        s = P.State(cfg, workdir(), log=lambda m: None)
        ss()["state"] = s
    s.cfg = cfg
    return s


def save():
    P.save(cfg, workdir(), summary=state().summary() if ss().get("done", {}).get("analysis") else {})


def done(step):
    return bool(ss().get("done", {}).get(step))


def mark(step, val=True):
    ss().setdefault("done", {})[step] = val


# ------------------------------------------------------------------------------------------------ staleness
# Results are only valid for the settings they were computed with. Each step that produces results records a
# fingerprint of the settings it used; when the settings on the page no longer match, that step and every step after
# it are marked not done, so stale numbers are neither shown as current nor written into the PDF.
def _norm(o):
    """Floats rounded to 10 significant digits, so the app's fingerprints below do not change on float noise (the
    economics stamp rounds its prices itself: engine/project.py analysis_stamp)."""
    if isinstance(o, float):
        return float(f"{o:.10g}")
    if isinstance(o, dict):
        return {str(kk): _norm(vv) for kk, vv in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm(vv) for vv in o]
    return o


def _sha(obj):
    return hashlib.sha1(json.dumps(_norm(obj), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def analysis_stamp(c):
    """The engine's fingerprint of what the economics depends on beyond the units (engine/project.py analysis_stamp:
    the prices without the report-only source / as_of / currency and with floats rounded, so a $/bu -> $/t -> $/bu
    round trip or typing the source of the prices does not throw the economics away; the input definitions; the
    model and join settings). The app uses the engine's recipe only, so the stamp stored with each result and the
    stamp of the page settings are computed the same way."""
    return P.analysis_stamp(c)


def analysis_stale(s_):
    """True when the economics on screen was computed with other prices / inputs / model / join settings than the
    ones now on the page: an input's stored stamp differs from the current stamp. A result that carries no stamp (an
    input skipped by the engine, or an older results file) is current only while the page settings equal the ones
    this session ran the economics with (ss()['an_stamp'], recorded right after the run)."""
    cur = analysis_stamp(cfg)
    rec = ss().get("an_stamp") or {}
    for name, R in ((s_.res or {}).get("analysis") or {}).items():
        if name.startswith("_") or not isinstance(R, dict):
            continue
        st_ = R.get("stamp")
        if st_ is not None:
            if st_ == cur:
                continue
            return True
        if rec.get("app") == cur:
            continue
        return True
    return False


SHP_SIDECARS = (".shx", ".dbf", ".prj", ".cpg")


def _sig_parts(p):
    """The files whose content decides what is read from `p`: the file itself and, for a shapefile, every sidecar
    that exists (.shx index, .dbf attributes, .prj coordinate system, .cpg encoding). Editing an attribute in QGIS
    rewrites only the .dbf, and correcting a wrong coordinate system replaces only the .prj, so signing the .shp
    alone left the steps done on the old attributes or the old geometry."""
    parts = [p]
    if p.lower().endswith(".shp"):
        base = os.path.splitext(p)[0]
        for ext in SHP_SIDECARS:
            parts += [q for q in (base + ext, base + ext.upper()) if os.path.exists(q)][:1]
    return parts


def _file_sig(path):
    """Path, modification time and size of a referenced file and of the files read with it (_sig_parts). A file
    replaced in place, or uploaded again under the same name, then changes the fingerprints; with the path alone it
    did not, and steps 4-6 stayed done on the old data.

    A folder (Path mode accepts one; read_vector reads its largest layer) is signed by every vector file under it
    and their sidecars: on Windows a folder's own modification time does not change when a file inside it is
    rewritten, so a layer edited in place in that folder was not noticed."""
    if not path:
        return None
    p = str(path)
    if os.path.isdir(p):
        parts = []
        for root, _, files in os.walk(p):
            for fn in sorted(files):
                low = fn.lower()
                if low.endswith(eio.VECTOR_EXT) and not low.endswith(".shp.xml"):
                    parts += _sig_parts(os.path.join(root, fn))
        parts = sorted(set(parts)) or [p]
    else:
        parts = _sig_parts(p)
    out = []
    for q in parts:
        try:
            out.append([q, os.path.getmtime(q), os.path.getsize(q)])
        except OSError:
            out.append([q, None, None])
    return out


def join_payload(c):
    """What the units were built from: the input names (units carry target_/rate_ columns per name), the as-applied
    file (path, mtime, size) and rate column of each input, the yield file (path, mtime, size) and its columns, and the
    join settings."""
    names = [i.get("name") for i in (c["prices"].get("inputs") or [])]
    cols = c.get("columns") or {}
    return {"inputs": names,
            "asapplied": {n: _file_sig((c["files"].get("asapplied") or {}).get(n)) for n in names},
            "applied_cols": {n: (cols.get("applied") or {}).get(n) for n in names},
            "yield": _file_sig(c["files"].get("yield")),
            "yield_cols": {r: cols.get(r) for r in ("yield", "heading", "distance", "swath", "machine")},
            # the units and bases the join converts with (project._rate_conversion, yield_polygons)
            "length_units": {r: cols.get(f"{r}_unit") for r in ("distance", "swath")},
            "rate_units": {i.get("name"): [i.get("rate_unit"), i.get("rate_basis"), (cols.get("applied_basis") or {}).get(i.get("name"))]
                           for i in (c["prices"].get("inputs") or [])},
            "join": c.get("join")}


def join_fingerprint(c):
    return _sha(join_payload(c))


DESIGN_ROLES = ("design_rate", "plot_id", "type", "treatment", "block")


def design_payload(c):
    """What 'Load the trial' (State.load_design) used: the design, boundary and extra design files
    (path, mtime, size), the design columns, the rate column of every input after the first, and the input names
    (the plots carry design_<name> for those inputs). It is stored when the trial is loaded; when it no longer
    matches, the trial must be loaded again: re-running the join alone built target_<name> from a design_<name> the
    plots did not have (all NaN) and the economics failed. None and "" (auto-detect / explicit none) count as the
    same here, because load_design turns a None it cannot detect into no column, as it does with ""."""
    f, cols = c["files"], c.get("columns") or {}
    names = [i.get("name") for i in (c["prices"].get("inputs") or [])]
    # an input whose design rates are read from its as-applied map (FROM_APPLIED): that map and its rate column
    # decide the plots' rates, so they belong to the design fingerprint too
    fa = {}
    for j_, n in enumerate(names):
        src = cols.get("design_rate") if j_ == 0 else (cols.get("extra_design_rate") or {}).get(n)
        if src == P.FROM_APPLIED:
            fa[n] = [_file_sig((f.get("asapplied") or {}).get(n)), (cols.get("applied") or {}).get(n)]
    # the plot medians those levels come from use the join's trims (load_design trims the plots with them) and the
    # design-level settings, so a changed trim must reload the trial, not only the join
    jn = c.get("join") or {}
    fa_settings = ({k_: jn.get(k_) for k_ in ("end_trim_m", "side_trim_m", "min_pts_design", "max_levels_design")}
                   if fa else None)
    return {"inputs": names,
            "design": _file_sig(f.get("design")), "design_layer": f.get("design_layer"),
            "boundary": _file_sig(f.get("boundary")),
            "extra_designs": {n: _file_sig((f.get("extra_designs") or {}).get(n)) for n in names[1:]},
            "extra_design_rate": {n: (cols.get("extra_design_rate") or {}).get(n) or None for n in names[1:]},
            "applied_for_design": fa, "applied_for_design_settings": fa_settings,
            # the rate unit of each input converts the design rates (and the as-applied levels) on loading
            "rate_units": {i.get("name"): [i.get("rate_unit"), i.get("rate_basis"), (cols.get("applied_basis") or {}).get(i.get("name"))]
                           for i in (c["prices"].get("inputs") or [])},
            "design_cols": {r: cols.get(r) or None for r in DESIGN_ROLES}}


def design_fingerprint(c):
    return _sha(design_payload(c))


def _raster_sigs(path):
    """_file_sig of every raster load_ndvi would read from an NDVI folder (or of the zip / file itself). A GeoTIFF
    added to or removed from a folder given by path (a cloudy date taken out) changes what the zones are built from,
    but not the folder's path, so the path alone left step 2 done on the old images. A zip is signed as one file:
    eio.list_rasters would extract it on every rerun of the page just to list it."""
    if not path:
        return []
    p = str(path)
    # A folder that does not answer (a network-share hiccup) is signed as one unreadable entry, not as an empty
    # folder: [] would look like "every image was removed" and throw the zones away (see fp_changed).
    if os.path.isfile(p) or not os.path.exists(p):
        return [_file_sig(p)]
    out = []
    for root, _, files in os.walk(p):
        for fn in sorted(files):
            if fn.lower().endswith(eio.RASTER_EXT):
                out.append(_file_sig(os.path.join(root, fn)))
    return sorted(out, key=lambda x: x[0][0])


def zones_payload(c):
    """What 'Build zones' (load_ndvi + make_zones) used: the NDVI and zone settings, the NDVI / zone
    files (path, mtime, size; every raster of the NDVI folder, the fetched list), the zone and NDVI columns, and the
    trial crop and year (load_ndvi picks the seasons by the trial crop and leaves out the trial year, so correcting
    either changes the seasons the zones come from). When it changes after the zones were built, the join and the
    economics would still be built on the old zones."""
    f, cols = c["files"], c.get("columns") or {}
    paths = f.get("ndvi_paths") or []
    return {"ndvi": c.get("ndvi"), "zones": c.get("zones"), "crop": c.get("crop"), "year": c.get("year"),
            "ndvi_dir": f.get("ndvi_dir"), "ndvi_dir_rasters": _raster_sigs(f.get("ndvi_dir")),
            "ndvi_paths": [_file_sig(p) for p in paths],
            "ndvi_map": _file_sig(f.get("ndvi_map")), "zone_map": _file_sig(f.get("zones")),
            "zone_col": cols.get("zone"), "ndvi_col": cols.get("ndvi")}


def zones_fingerprint(c):
    return _sha(zones_payload(c))


# The payload entries that hold file signatures (_file_sig / _raster_sigs, or a dict of them per input).
FILE_SIG_KEYS = ("design", "boundary", "extra_designs", "asapplied", "yield", "ndvi_dir_rasters", "ndvi_paths",
                 "ndvi_map", "zone_map", "applied_for_design")


def _unreadable(sig):
    """True when a file signature holds a part whose time and size could not be read this run ([path, None, None])."""
    if isinstance(sig, (list, tuple)):
        if len(sig) == 3 and isinstance(sig[0], str) and sig[1] is None and sig[2] is None:
            return True
        return any(_unreadable(x) for x in sig)
    return False


def _settle(new, old):
    """`new` with every unreadable file signature replaced by the one stored when the step was run (`old`)."""
    if isinstance(new, dict):
        old = old if isinstance(old, dict) else {}
        return {kk: _settle(vv, old.get(kk)) for kk, vv in new.items()}
    if old is not None and _unreadable(new):
        return old
    return new


def fp_store(name, payload):
    """Record the fingerprint of a step that just ran, and its payload (for fp_changed)."""
    ss()[name] = _sha(payload)
    ss()[name + "_payload"] = json.loads(json.dumps(_norm(payload), sort_keys=True, default=str))


def fp_changed(name, payload):
    """True when the settings and files of a finished step no longer match the ones it was run with.

    A file that cannot be read this run (a network share that does not answer for one rerun, a file locked or being
    copied) is not a change: its signature is taken from the stored payload. Before, the unreadable file changed the
    fingerprint, every later step was marked not done, and nothing restored them one rerun later when the file
    answered again with the same content, so the trial, the zones, the join, the bootstrap and the PDF had to be run
    again. A real change (the file readable with another time or size, a column, a setting) still counts, also while
    another file is unreadable. Without a stored payload (an older session) the hashes alone are compared."""
    old = ss().get(name)
    if not old:
        return False
    prev = ss().get(name + "_payload")
    if isinstance(prev, dict):
        payload = dict(payload)
        for key_ in FILE_SIG_KEYS:
            if key_ in payload:
                payload[key_] = _settle(payload[key_], prev.get(key_))
    return _sha(payload) != old


# The fields make_pdf prints but no other fingerprint covers (the economics stamp leaves the price source, date and
# currency out on purpose, so typing them does not throw the economics away). Their values when the PDF was built are
# kept; when they change, the PDF on offer shows the old title / producer / currency, so step 6 is marked not done.
def report_fingerprint(c):
    pr = c.get("prices") or {}
    return _sha({"name": c.get("name"), "producer": c.get("producer"), "field": c.get("field"), "crop": c.get("crop"),
                 "year": c.get("year"), "currency": pr.get("currency"), "source": pr.get("source"),
                 "as_of": pr.get("as_of")})


def invalidate(steps, reason=None):
    """Mark `steps` not done. A report built from older results is forgotten as well (its path is cleared), so the old
    PDF is not offered next to newly written outputs. `reason` is kept to tell the user why the step must be re-run."""
    was_done = any(done(x) for x in steps)
    for x in steps:
        mark(x, False)
    if "report" in steps:
        s_ = ss().get("state")
        if s_ is not None:
            s_.report_path = None
    if reason and was_done:
        ss().setdefault("stale", {})[steps[0]] = reason


def show_stale(step):
    """Show why `step` has to be run again (until it is run again; the step's button clears the message)."""
    msg = (ss().get("stale") or {}).get(step)
    if msg:
        st.warning(msg)


# ================================================================================================ files
def _folder_name(key):
    """Folder (under inputs/) for an uploaded file. Keys such as 'asapplied:seed' carry a ':', which Windows does not
    allow in a folder name (os.makedirs raised and the rest of the page was not drawn), so the key is reduced to
    letters, digits, '_' and '-'. When that changed it, a short hash of the original key is appended so two keys never
    share a folder (e.g. inputs named 'seed' and 'Seed' on a case-blind disk)."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", key).strip("_") or "file"
    if safe != key:
        safe += "_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]
    return safe


def _save_upload(u, p):
    """Write an uploaded file to `p` only when it is new. Streamlit hands the same uploads back on every rerun;
    rewriting them each time changed their modification time and so defeated the cache of peek_columns (the whole
    yield map was read again on every click)."""
    fid = getattr(u, "file_id", None) or f"{u.name}:{u.size}"
    rec = ss().setdefault("_uploads", {})
    if rec.get(p) == fid and os.path.exists(p) and os.path.getsize(p) == u.size:
        return
    with open(p, "wb") as f:
        f.write(u.getbuffer())
    rec[p] = fid


def _in_upload_dir(path, key):
    """True when `path` is (inside) the folder file_field saves the uploads of `key` into."""
    if not path:
        return False
    root = os.path.normcase(os.path.abspath(os.path.join(workdir(), "inputs", _folder_name(key))))
    return os.path.normcase(os.path.abspath(str(path))).startswith(root + os.sep)


def _same_path(a, b):
    try:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
    except (TypeError, ValueError):
        return False


def file_field(label, key, types=None, help=None, multiple_ok=True, names_key=None):
    """A file given either as an upload (saved into the project's inputs/ folder; a shapefile's parts together or a
    zip) or as a path on this computer (read in place). Returns the path or None.

    Upload mode keeps the file saved earlier when the uploader is simply empty (a reopened project, or the uploader
    drawn again after being hidden), but when the user removes the upload with the uploader's x in this session the
    file is no longer used (before, the old file silently stayed in the analysis).

    names_key: cfg['files'] key that keeps the original names of the uploaded files. Several files are saved into a
    folder named by a hash of the upload, and that hash is all the path says: the PDF named it as 'the user's folder'.
    The names are kept while the path is still that upload (a reopened project shows it in Path mode) and cleared
    when the user gives another path or removes the upload."""
    cur = cfg_get_file(key)
    c1, c2 = st.columns([1, 3])
    mode = c1.radio(label, ["Path", "Upload"], key=k(f"mode_{key}"), horizontal=True, help=help,
                    index=0)
    path = None
    upk, hadk = k(f"up_{key}"), k(f"uphad_{key}")
    if mode == "Path":
        ss()[hadk] = False
        val = c2.text_input("path", value=cur or "", key=k(f"path_{key}"), label_visibility="hidden",
                            placeholder=r"C:\...\file.shp, a .zip, or a folder")
        val = val.strip().strip('"')
        if val:
            if os.path.exists(val):
                path = val
            elif cur and _same_path(val, cur):
                # The stored path (it existed when it was given) does not answer this run: a network-share hiccup,
                # the file mid-copy or locked. Writing None into cfg changed the fingerprints, threw every finished
                # step away, and nothing restored them one rerun later when the file answered again. The path is
                # kept, and fp_changed does not count a file it cannot read as a change.
                path = cur
                c2.warning("Not reachable right now (network share, or the file is locked or being copied): the "
                           "stored path is kept and the finished steps stay done. If the file was moved or deleted, "
                           "give its new path.")
            else:
                c2.error("Not found on this computer.")
        if names_key and not _in_upload_dir(path, key):
            cfg["files"][names_key] = None      # a path of the user's own: its name says where the files are
    else:
        # A widget's state lives in session_state only while it is drawn on every run, so an uploader that is not
        # there yet is being drawn for the first time (or again after it was hidden): empty then is not a removal.
        if upk not in ss():
            ss()[hadk] = False
        ups = c2.file_uploader("files", type=types, accept_multiple_files=multiple_ok, key=upk,
                               label_visibility="hidden")
        if ups:
            lst = ups if isinstance(ups, list) else [ups]
            # The current set of files goes into its own folder, named by a hash of the upload ids: a file removed
            # from a multi-file upload (an NDVI GeoTIFF of a cloudy date), or an earlier set with other names, is
            # then not in the folder that is read (load_ndvi reads every raster of the folder).
            fids = sorted(getattr(u, "file_id", None) or f"{u.name}:{u.size}" for u in lst)
            d = os.path.join(workdir(), "inputs", _folder_name(key),
                             hashlib.sha1("|".join(fids).encode("utf-8")).hexdigest()[:10])
            try:
                os.makedirs(d, exist_ok=True)
                names = []
                for u in lst:
                    p = os.path.join(d, u.name)
                    _save_upload(u, p)
                    names.append(p)
                shp = [p for p in names if p.lower().endswith(".shp")]
                path = shp[0] if shp else (names[0] if len(names) == 1 else d)
                ss()[hadk] = True
                if names_key:
                    cfg["files"][names_key] = sorted(u.name for u in lst)
            except OSError as exc:
                c2.error(f"Could not save the upload into {d}: {exc}")
                path = cur
        elif ss().get(hadk):
            ss()[hadk] = False          # the user removed the upload in this session: stop using the file
            path = None
            if names_key:
                cfg["files"][names_key] = None
        elif cur:
            path = cur
    cfg_set_file(key, path)
    up_names = cfg["files"].get(names_key) if names_key else None
    if path and up_names and _in_upload_dir(path, key):
        shown = ", ".join(up_names[:6]) + (f", ... (+{len(up_names) - 6})" if len(up_names) > 6 else "")
        c2.caption(f"Using {len(up_names)} file{'s' if len(up_names) != 1 else ''} uploaded in the app: {shown}")
    elif path:
        c2.caption(f"Using `{path}`")
    return path


def cfg_get_file(key):
    if key.startswith("asapplied:"):
        return (cfg["files"].get("asapplied") or {}).get(key.split(":", 1)[1])
    if key.startswith("extra:"):
        return (cfg["files"].get("extra_designs") or {}).get(key.split(":", 1)[1])
    return cfg["files"].get(key)


def cfg_set_file(key, path):
    if key.startswith("asapplied:"):
        cfg["files"].setdefault("asapplied", {})[key.split(":", 1)[1]] = path
    elif key.startswith("extra:"):
        cfg["files"].setdefault("extra_designs", {})[key.split(":", 1)[1]] = path
    else:
        cfg["files"][key] = path


@st.cache_data(show_spinner=False, max_entries=16)
def _peek(path, sig):
    """`sig` is only the cache key (see peek_columns)."""
    g = eio.read_vector(path)
    return pd.DataFrame(g.drop(columns="geometry").head(5000))


def peek_columns(path):
    """Columns and a 5,000-row sample of a vector file, cached per file signature (_file_sig: the file and every
    sidecar, or every vector file of a folder), so the page does not reread a 35,000-point yield map on every click.
    The key was the .shp's modification time alone: a column added or renamed in QGIS rewrites only the .dbf, so the
    fingerprints saw the change and asked for the step again while every column picker kept offering the old
    columns until the app was restarted."""
    try:
        df = _peek(path, json.dumps(_file_sig(path), default=str))
        return list(df.columns), df
    except Exception as exc:
        st.warning(f"Could not read {os.path.basename(str(path))}: {exc}")
        return [], None


def _layer_id(src):
    """What identifies the layer `src` for the column pickers: its path, except for a file uploaded in the app.

    file_field saves every upload set into inputs/<key>/<hash of the upload ids>/, so the same layer uploaded again
    (a .prj added, the corrected file uploaded after a fix) gets a new path. Keyed by the path, an explicit '(none)'
    (Swath, Distance, Heading, Treatment, Block...) was no longer recognised and find_col's guess came back. An upload
    is therefore identified by its key folder and its file name, without the hash: 'upload:<key>/<file>'."""
    if not src:
        return None
    p = str(src)
    if p.startswith("upload:"):
        return p
    try:
        root = os.path.normcase(os.path.abspath(os.path.join(workdir(), "inputs")))
        ap = os.path.normcase(os.path.abspath(p))
    except (TypeError, ValueError):
        return p
    if not ap.startswith(root + os.sep):
        return p
    parts = os.path.relpath(ap, root).split(os.sep)
    if len(parts) >= 3:         # <key folder>/<hash>/<file...>
        return "upload:" + "/".join([parts[0]] + parts[2:])
    return p


# 'What was tested?' in step 1: ready rows for seed and urea, or the free editor
INPUT_MODES = ["Seed", "Urea", "Seed and urea", "Other inputs"]
INPUT_PRESETS = {
    "seed": dict(name="seed", product="seed", rate_basis="product", rate_unit="lb/ac", status_quo=None,
                 price_per_lb_product=0.22),
    "urea": dict(name="urea", product="urea 46-0-0", rate_basis="product", rate_unit="lb/ac", status_quo=None,
                 price_per_lb_product=0.352),      # CAD/lb product, the price given for Trial 22 (2026-10-28)
}
NONE_COL = ""      # an explicit "(none)": stored as "" so the engine does not auto-detect the column again
CHOOSE = "(choose)"     # placeholder of a picker that needs a column: returned as None and the step is blocked
# The rate picker's option that reads an input's design rates from its as-applied map (engine.project.FROM_APPLIED
# is what the project stores): for a trial whose design file has no column for that input, e.g. seed rates in the
# design and the urea rates only on the drill's as-applied map, one rate per plot (Trial 22).
FROM_APPLIED_LABEL = "(from the as-applied map)"


def col_select(label, cols, role, current, df=None, key=None, none_ok=True, src=None, guess=True, guess_col=None,
               choose_first=False, exclude=(), extra_opts=None):
    """Column picker for one role of the file `src`. Returns the column, NONE_COL ("") for an explicit "(none)", or
    None while a picker without "(none)" (none_ok=False) shows "(choose)": the caller then blocks its step instead of
    using the file's first column (an Id column taken as the yield or as the zones).

    choose_first=True keeps "(choose)" as the first option even when "(none)" is allowed, and makes it the fallback:
    "(none)" is then shown only when the user picked it for this file. The design rate needs this: "(none)" there
    means a treatment-only trial (no rate response, no optimum, no prescription), so it must be the user's choice.
    Before, the picker fell back to opts[0] = "(none)" as soon as a treatment column appeared, and the first input
    was silently analysed as treatment-only.

    The picker is (re)initialised when it is drawn for the first time, for another file than last time, or when it
    holds an option the current file does not have. It then shows, in this order:
      - the stored column `current`, when the current file has it;
      - "(none)", when `current` is an explicit "(none)" chosen for this same file (kept in cfg['ui']['none_cols'],
        so it survives closing and opening the project; "same file" is _layer_id, so the same layer uploaded again
        under the same name keeps it). A "(none)" or a column name left over from another file says
        nothing about this file, so the column is guessed again (before, a name of the previous file became an
        explicit "(none)" and the engine never detected the column of the new file: no swath, no distance);
      - `guess_col`, a guess the caller made with more context (the first input's rate column, guessed from the
        input's name), when the file has it;
      - eio.find_col's guess, unless guess=False (the zone column of a zone map: the user must pick it, a guessed
        OBJECTID or Name column gave one zone per polygon or zones in alphabetical order). Columns in `exclude` are
        never guessed: the first input's rate picker excludes the columns named after a later input, because
        find_col ranks tgt_urea / tgt_n before Tgt_Rate, so with inputs [seed, urea] the seed picker took tgt_urea
        and the urea picker (which drops the main rate column) could no longer offer its own column;
      - else "(choose)" with choose_first, "(none)" (none_ok) or "(choose)".
    Otherwise it keeps what the user selected.

    A file that could not be read this run (peek_columns gave no columns and no sample: a file mid-copy, a .dbf
    locked while QGIS rewrites it, a network-share hiccup) keeps the stored choice untouched. Re-initialising there
    fell to opts[0] = "(none)", recorded it as the user's explicit "(none)" for this file, and the next run (file
    readable again, same path) kept it: Heading, Distance, Swath, Treatment or Block were silently dropped and saved
    with the project.

    `extra_opts` = {label: stored value}: choices that are not columns of the file, drawn after "(choose)" /
    "(none)" (the first input's Rate picker offers "(from the as-applied map)" -> engine.project.FROM_APPLIED when
    the input has an as-applied map). A stored value among them is shown by its label and returned as the value."""
    nk = key or f"col_{role}"
    wkey = k(nk)
    ex = dict(extra_opts or {})
    ex_label = {v: lab for lab, v in ex.items()}
    if not cols and df is None and src:
        shown = (ex_label.get(current) or (FROM_APPLIED_LABEL if current == P.FROM_APPLIED else None)
                 or (current if current else ("(none)" if current == NONE_COL and none_ok else CHOOSE)))
        # a disabled stand-in under its own key: the real widget's state and cfg['ui']['none_cols'] are not touched
        st.selectbox(label, [shown], key=wkey + "::unreadable", disabled=True,
                     help="The file could not be read just now; the stored choice is kept.")
        return current
    if choose_first:
        opts = [CHOOSE] + (["(none)"] if none_ok else []) + list(ex) + list(cols)
    else:
        opts = (["(none)"] if none_ok else [CHOOSE]) + list(ex) + list(cols)
    srcs = ss().setdefault("_colsrc", {})
    none_for = cfg.setdefault("ui", {}).setdefault("none_cols", {})
    sig = _layer_id(src) if src else None
    if wkey not in ss() or ss()[wkey] not in opts or srcs.get(wkey) != sig:
        if current and current in cols:
            init = current
        elif current in ex_label:
            init = ex_label[current]
        elif current == NONE_COL and none_ok and sig is not None and _layer_id(none_for.get(nk)) == sig:
            init = "(none)"
        elif guess_col and guess_col in cols:
            init = guess_col
        else:
            g = eio.find_col(df, role, exclude=tuple(exclude)) if (guess and df is not None) else None
            init = g if g in cols else opts[0]
        # set through session_state (before the widget is drawn): a keyed selectbox ignores `index` once it has a
        # state, so a changed file would otherwise keep showing the option chosen for the previous file
        ss()[wkey] = init
    srcs[wkey] = sig
    v = st.selectbox(label, opts, key=wkey)
    if v == "(none)":
        none_for[nk] = sig
        return NONE_COL
    none_for.pop(nk, None)
    if v in ex:
        return ex[v]
    return None if v == CHOOSE else v


def _tokens(s):
    """Lower-case word tokens of a column or input name: 'Tgt_Urea', 'TgtUrea' and 'TGT-UREA' all give {tgt, urea}."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(s))
    return {t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t}


RATE_TOKENS = {"tgt", "target", "rate", "rx", "dose"}


def guess_input_rate_col(df, cols, name):
    """The column of `cols` holding the design rates of input `name`, or None when there is no confident match.

    Whole tokens, not substrings: a short input name such as 'N' is a substring of Plot_No, Name or Notes, and those
    plot numbers were used as N rates without a warning. Only numeric columns count; when several match, the ones
    that also look like a design rate (tgt / target / rate / rx / dose) are kept, and if that is still ambiguous the
    user chooses."""
    want = _tokens(name)
    if df is None or not want:
        return None
    hits = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and want <= _tokens(c)]
    if len(hits) > 1:
        hits = [c for c in hits if _tokens(c) & RATE_TOKENS] or hits
    return hits[0] if len(hits) == 1 else None


def name_matches(col, name):
    """True when column `col` is named after input `name` by whole tokens (tgt_urea for 'urea', Tgt_N for 'N', not
    Plot_No for 'N')."""
    want = _tokens(name)
    return bool(want) and want <= _tokens(col)


def _code(x):
    """A no-data code typed by the user: a number when it reads as one (0, -9999), else the text label."""
    x = x.strip()
    try:
        v = float(x)
    except ValueError:
        return x
    return int(v) if v.is_integer() else v


def main_form(R):
    """Functional form of the main model of an input's results (as report._main_form): the form the economics and
    the status-quo yield were computed with, after any straight-line fallback."""
    M = R.get("models") or {}
    kind = R.get("model_kind") or ("zone" if "zone" in M else "pooled")
    return (M.get(kind) or M.get("pooled") or {}).get("form") or "quadratic"


def sq_extrap_text(R, d):
    """How the engine valued a status quo outside the trial range (economics.bounded_curve), worded for the fitted
    form (economics.bounded_curve: the one-sided slope at the nearest edge; a plateau reached inside the range is
    flat above it, and below the range its rising segment's slope is used)."""
    sq, grid = R.get("status_quo"), list(d.get("grid") or [])
    lo, hi = (float(grid[0]), float(grid[-1])) if grid else (None, None)
    plateau = main_form(R) == "qplateau"
    if plateau and sq is not None and lo is not None and float(sq) < lo:
        return (f"extended from the lowest tested rate ({lo:.0f}) with the slope of the plateau curve's rising segment "
                "there (quadratic-plateau fit; never below 0)")
    if plateau and sq is not None and hi is not None and float(sq) > hi:
        return (f"extended from the highest tested rate ({hi:.0f}) with the curve's slope there: flat when the "
                "plateau is reached inside the trial range, rising otherwise (quadratic-plateau fit; never below 0)")
    return "extended linearly with the slope of the curve at the nearest edge of the tested rates (never below 0)"


def show_fig(fig):
    st.pyplot(fig, clear_figure=True)
    import matplotlib.pyplot as plt
    plt.close("all")


# ================================================================================================ sidebar
def draw_steps():
    """The step checklist of the sidebar (drawn at the start of the run and again at its end, when this run's
    buttons have acted)."""
    with steps_box.container():
        for key_, lab in (("design", "1 Trial"), ("zones", "2 NDVI & zones"), ("prices", "3 Prices"), ("join", "4 Join"),
                          ("analysis", "5 Economics"), ("report", "6 Report")):
            st.markdown(f"{'✅' if done(key_) else '⬜'} {lab}")
    if ss().get("workdir"):
        folder_box.caption(f"Folder: `{ss()['workdir']}`")


with st.sidebar:
    st.markdown(f"**{brand.LONG_NAME}**")
    st.caption(brand.TAGLINE)
    if st.button("New project", width="stretch"):
        new_project()
        st.rerun()
    projs = P.list_projects(WORKSPACE)
    if projs:
        st.markdown("**Saved projects**")
        labels = [f"{p['name']} · {str(p['saved'])[:10]}" for p in projs]
        i = st.selectbox("Project", range(len(projs)), format_func=lambda j: labels[j], label_visibility="collapsed")
        if st.button("Open", width="stretch"):
            rec = P.load(projs[i]["dir"])
            new_project()
            ss()["cfg"] = rec["config"]
            ss()["workdir"] = projs[i]["dir"]
            ss()["pid"] = os.path.basename(projs[i]["dir"])
            st.rerun()
    st.markdown("**Steps**")
    steps_box = st.empty()          # drawn now, and drawn again at the end of the run, after this run's buttons acted
    folder_box = st.empty()

    st.caption(brand.footer_caption())
draw_steps()

st.title(brand.LONG_NAME)
st.caption("Cleaned yield maps + trial design + as-applied + multi-year NDVI zones + prices → does variable rate pay "
           "against a flat rate?")
tabs = st.tabs(["1 Trial", "2 NDVI & zones", "3 Prices", "4 Join", "5 Economics", "6 Report"])

# ================================================================================================ 1 trial
with tabs[0]:
    c = st.columns([2, 2, 2, 1.2, 1])
    cfg["name"] = c[0].text_input("Project name", cfg.get("name") or "", key=k("name"), placeholder="Trial 29")
    cfg["producer"] = c[1].text_input("Producer", cfg.get("producer") or "", key=k("producer"))
    cfg["field"] = c[2].text_input("Field", cfg.get("field") or "", key=k("field"))
    crop0 = cfg.get("crop") or "wheat"
    cfg["crop"] = c[3].selectbox("Crop", CROPS, index=CROPS.index(crop0) if crop0 in CROPS else len(CROPS) - 1, key=k("crop"))
    cfg["year"] = int(c[4].number_input("Year", 2000, 2100, int(cfg.get("year") or dt.date.today().year), key=k("year")))
    cfg["prices"]["crop"] = cfg["crop"]

    st.subheader("Files")
    st.caption("The yield map must already be cleaned (OCCD or the yield-map-cleaning pipeline). Shapefiles: give the .shp "
               "path, a .zip, or upload the .shp/.shx/.dbf/.prj together.")
    design = file_field("Trial design (plots + border) *", "design", ["shp", "shx", "dbf", "prj", "cpg", "zip", "geojson", "gpkg", "kml", "kmz"])
    yfile = file_field("Cleaned yield points *", "yield", ["shp", "shx", "dbf", "prj", "cpg", "zip", "gpkg", "parquet"])
    file_field("Field boundary (optional; else the trial area)", "boundary", ["shp", "shx", "dbf", "prj", "zip", "geojson", "gpkg", "kml", "kmz"])

    st.subheader("Tested inputs")
    st.caption("Choose what the trial tested. **Status quo** = the grower's usual flat rate (ask; never infer). Rate basis "
               "*product* = the trial rates are lb of product (seed, urea as applied); *nutrient* = lb of N. An input "
               "whose rates are not in any design file (the urea of a seed x urea trial logged only by the drill) can "
               "take its design rates **from its as-applied map**: give the map, then choose "
               f"'{FROM_APPLIED_LABEL}' as its rate column.")
    cur_inputs = cfg["prices"]["inputs"] or [dict(INPUT_PRESETS["seed"])]
    stash = cfg.setdefault("ui", {}).setdefault("input_stash", {})
    cur_names = [i.get("name") for i in cur_inputs]
    n_prods = [p_ for p_ in EP.PRODUCTS if p_ not in ("seed", "other")]
    ready_ok = all((i.get("name") == "seed" and i.get("product") == "seed")
                   or (i.get("name") == "urea" and i.get("product") in n_prods) for i in cur_inputs)
    if not ready_ok:
        # e.g. a seed input stored with product 'urea 46-0-0', or a urea input with product 'other': the ready rows
        # would silently change the product (and the price per lb N), so the free editor shows the project as it is
        mode0 = "Other inputs"
    elif cur_names == ["seed"]:
        mode0 = "Seed"
    elif cur_names == ["urea"]:
        mode0 = "Urea"
    elif sorted(cur_names) == ["seed", "urea"]:
        mode0 = "Seed and urea"
    else:
        mode0 = "Other inputs"
    mode = st.radio("What was tested?", INPUT_MODES, index=INPUT_MODES.index(mode0), horizontal=True,
                    key=k("in_mode"),
                    help="Seed and urea are analysed the same way, each with its own price, status quo and as-applied "
                         "map; with both, each is analysed with the other as a control. 'Other inputs' opens the "
                         "free editor (any input name, up to 3 inputs).")
    new_inputs = []
    if mode != "Other inputs":
        # the ready rows: seed and urea keep their names (the as-applied maps, columns and results are keyed by
        # them); their price, status quo, product and units come from the project, from the stash of an input taken
        # out earlier, or from the defaults
        want = {"Seed": ["seed"], "Urea": ["urea"], "Seed and urea": ["seed", "urea"]}[mode]
        if mode == "Seed and urea" and cur_names == ["urea", "seed"]:
            want = ["urea", "seed"]          # keep the order of a project that had urea first (its main design column)
        for nm in want:
            old = (next((i for i in cur_inputs if i.get("name") == nm), None) or (stash.get(nm) or {}).get("input")
                   or dict(INPUT_PRESETS[nm]))
            c = st.columns([1.3, 1.6, 1.1, 1.0, 1.2])
            c[0].markdown(f"**{nm.capitalize()}**")
            c[0].caption("tested input")
            if nm == "seed":
                prod = "seed"
                c[1].text_input("Product", "seed", disabled=True, key=k("pre_seed_prod"))
            else:
                prods = [p_ for p_ in EP.PRODUCTS if p_ not in ("seed", "other")]
                p0 = old.get("product") if old.get("product") in prods else "urea 46-0-0"
                prod = c[1].selectbox("Product", prods, index=prods.index(p0), key=k(f"pre_{nm}_prod"),
                                      help="The nitrogen product applied; urea 46-0-0 by default.")
            basis = c[2].selectbox("Rate basis", ["product", "nutrient"],
                                   index=["product", "nutrient"].index(old.get("rate_basis", "product")),
                                   key=k(f"pre_{nm}_basis"),
                                   help="product = the rates are lb of product per acre; nutrient = lb of N per acre.")
            unit = c[3].selectbox("Rate unit", list(EP.RATE_UNITS),
                                  index=list(EP.RATE_UNITS).index(old.get("rate_unit")) if old.get("rate_unit") in EP.RATE_UNITS else 0,
                                  key=k(f"pre_{nm}_unit"),
                                  help="The unit of the design rates, the as-applied map and the status quo; kg/ha is "
                                       "converted to lb/ac (x 0.8922) on loading, and the results are in lb/ac.")
            sqv = c[4].number_input("Status quo rate", 0.0, 5000.0,
                                    value=None if old.get("status_quo") is None else float(old["status_quo"]),
                                    step=1.0, key=k(f"pre_{nm}_sq"), placeholder="ask the grower")
            new_inputs.append(dict(name=nm, product=prod, rate_basis=basis, rate_unit=unit,
                                   status_quo=None if sqv is None else float(sqv),
                                   price_per_lb_product=float(old.get("price_per_lb_product") or 0.0)))
    n_in = len(new_inputs) if mode != "Other inputs" else int(
        st.number_input("How many inputs were tested?", 1, 3, len(cur_inputs), key=k("n_inputs")))
    for j in ([] if mode != "Other inputs" else range(n_in)):
        old = cur_inputs[j] if j < len(cur_inputs) else dict(name=f"input{j + 1}", product="urea 46-0-0", rate_basis="product",
                                                             rate_unit="lb/ac", status_quo=None, price_per_lb_product=0.49)
        c = st.columns([1.3, 1.6, 1.1, 1.0, 1.2])
        nm = c[0].text_input("Input", old["name"], key=k(f"in_name_{j}")).strip() or f"input{j + 1}"
        prods = list(EP.PRODUCTS)
        prod = c[1].selectbox("Product", prods, index=prods.index(old["product"]) if old["product"] in prods else len(prods) - 1,
                              key=k(f"in_prod_{j}"))
        basis = c[2].selectbox("Rate basis", ["product", "nutrient"], index=["product", "nutrient"].index(old.get("rate_basis", "product")),
                               key=k(f"in_basis_{j}"))
        unit = c[3].selectbox("Rate unit", list(EP.RATE_UNITS),
                              index=list(EP.RATE_UNITS).index(old.get("rate_unit")) if old.get("rate_unit") in EP.RATE_UNITS else 0,
                              key=k(f"in_unit_{j}"),
                              help="The unit of the design rates, the as-applied map and the status quo; kg/ha is "
                                   "converted to lb/ac (x 0.8922) on loading, and the results are in lb/ac.")
        sqv = c[4].number_input("Status quo rate", 0.0, 5000.0, value=None if old.get("status_quo") is None else float(old["status_quo"]),
                                step=1.0, key=k(f"in_sq_{j}"), placeholder="ask the grower")
        new_inputs.append(dict(name=nm, product=prod, rate_basis=basis, rate_unit=unit,
                               status_quo=None if sqv is None else float(sqv),
                               price_per_lb_product=float(old.get("price_per_lb_product") or 0.0)))
    # Input names key the widgets, the as-applied files and columns, and the units' target_/rate_ columns, so two
    # inputs with the same name (case-blind, as on a Windows disk) are refused, and so are names that are not plain
    # identifiers (see INPUT_NAME_RE). Names sharing their first 10 characters are allowed: that rule refused valid
    # pairs (seed_rate_1 / seed_rate_2) and let clashing field names through (Flat_input2 / Flat_input3), while the
    # engine already makes the shapefile's field names unique and writes the key (zones_rx_fields.csv). Until the
    # names are fixed the last valid list is kept and 'Load the trial' waits.
    low = [i["name"].lower() for i in new_inputs]
    dup_names = sorted({i["name"] for i in new_inputs if low.count(i["name"].lower()) > 1})
    bad_names = [i["name"] for i in new_inputs if not INPUT_NAME_RE.match(i["name"])]
    reserved_names = [i["name"] for i in new_inputs if i["name"].lower() in RESERVED_INPUT_NAMES]
    name_errors = []
    if dup_names:
        name_errors.append(f"Two inputs have the same name ({', '.join(dup_names)}). Give each input its own name.")
    if bad_names:
        name_errors.append(f"Input names must start with a letter and hold only letters, digits and '_' (no spaces or "
                           f"hyphens): {', '.join(repr(x) for x in bad_names)}. Use e.g. 'urea_N' or 'N_rate'.")
    if reserved_names:
        name_errors.append(f"{', '.join(repr(x) for x in reserved_names)} cannot name an input: the app already uses "
                           f"{'it' if len(reserved_names) == 1 else 'them'} for its own columns "
                           f"({', '.join(RESERVED_INPUT_NAMES)}, in any case). Name the input by what was tested, "
                           "e.g. 'seed', 'urea' or 'N_rate'.")
    for msg in name_errors:
        st.error(msg + " Until then the previous list of inputs is kept and 'Load the trial' waits.")
    if not name_errors:
        # A rename carries the input's as-applied file and columns and its extra design along (they are keyed by
        # name); otherwise the renamed input would silently lose its applied map and fall back to the target rates.
        ren = ({cur_inputs[j]["name"]: new_inputs[j]["name"] for j in range(min(len(cur_inputs), n_in))
                if cur_inputs[j]["name"] != new_inputs[j]["name"]} if mode == "Other inputs" else {})
        if ren:
            for dct in (cfg["files"].setdefault("asapplied", {}), cfg["files"].setdefault("extra_designs", {}),
                        cfg["columns"].setdefault("applied", {}), cfg["columns"].setdefault("extra_design_rate", {}),
                    cfg["columns"].setdefault("applied_basis", {})):
                moved = {new: dct[old] for old, new in ren.items() if old in dct}
                for old in ren:
                    dct.pop(old, None)
                for new in ren.values():
                    if new not in moved:
                        dct.pop(new, None)
                dct.update(moved)
        # Entries of inputs that are no longer in the list (the number of inputs was reduced) are dropped: nothing on
        # the page shows them any more, so they could not be cleared, yet 'Load the trial' still read the removed
        # input's design file (and failed on it) and its as-applied map stayed in the project.
        keep = {i["name"] for i in new_inputs}
        # An input taken out (Seed and urea -> Seed) is stashed in cfg['ui'] with its as-applied map, design file,
        # columns and row (price, status quo), so choosing it again brings everything back instead of asking for
        # the files again; the engine never reads cfg['ui'], so a stashed input cannot reach 'Load the trial'.
        STASH_KEYS = (("files", "asapplied"), ("files", "extra_designs"), ("columns", "applied"),
                      ("columns", "extra_design_rate"))
        for i_ in cur_inputs:
            nm_ = i_.get("name")
            if nm_ and nm_ not in keep:
                prev = stash.get(nm_) or {}
                live = {f"{a}.{b}": (cfg[a].get(b) or {}).get(nm_) for a, b in STASH_KEYS}
                # what was set aside earlier (its own design file while it was the first input) is kept when the
                # live config has nothing in its place
                stash[nm_] = {**prev, "input": dict(i_), **{kk: vv for kk, vv in live.items() if vv is not None}}
        # Rate sources move with their input. The first input reads its rates from the main design's Rate column
        # (cfg columns.design_rate); a later input from a design file of its own (files.extra_designs) or a column of
        # the main design (columns.extra_design_rate). When the ready rows change which input is first:
        #   * the old first input, if it stays in the list, takes its Rate column into extra_design_rate (it is a
        #     main-design column or '(from the as-applied map)', both valid for a later input); if it is taken out,
        #     the column is stashed with it;
        #   * the new first input takes its own source only when the first slot can use it: a main-design column or
        #     '(from the as-applied map)'. A design file of its own or an explicit '(none)' cannot be read by the
        #     first slot (load_design reads the first input from the main design only): they are set aside in the
        #     stash (and come back when the input is a later input again), and the Rate picker shows '(choose)'
        #     (it guesses only by the input's name, tgt_urea for urea), so the seed rates are never taken for urea.
        new_first = new_inputs[0]["name"] if new_inputs else None
        old_first = cur_names[0] if cur_names else None
        xdr = cfg["columns"].setdefault("extra_design_rate", {})
        xdf = cfg["files"].setdefault("extra_designs", {})
        swap_note = None
        if mode != "Other inputs" and new_first and old_first and new_first != old_first:
            old_col = cfg["columns"].get("design_rate")
            if old_first in keep:
                ent0 = stash.get(old_first) or {}
                if ent0.get("aside"):
                    # its own design file or '(none)' was set aside when it became first: it gets them back
                    if ent0.get("files.extra_designs"):
                        xdf[old_first] = ent0["files.extra_designs"]
                    if ent0.get("columns.extra_design_rate") is not None:
                        xdr[old_first] = ent0["columns.extra_design_rate"]
                elif old_col is not None and not xdf.get(old_first):
                    xdr[old_first] = old_col
                stash.pop(old_first, None)
            else:
                stash.setdefault(old_first, {"input": dict(cur_inputs[0])})["columns.design_rate"] = old_col
            if new_first in cur_names:                  # a later input becomes the first: its live source
                own_file, own_col = xdf.get(new_first), xdr.get(new_first)
            else:                                       # an input coming back: its stashed sources
                st_ = stash.get(new_first) or {}
                own_file = st_.get("files.extra_designs")
                own_col = st_.get("columns.design_rate")
                if own_col is None and not own_file:
                    own_col = st_.get("columns.extra_design_rate")
            back = own_col if (own_col and not own_file) else None
            if own_file or own_col == NONE_COL:
                # set aside for when this input is a later input again
                ent = stash.setdefault(new_first, {"input": dict(next((i for i in cur_inputs if i.get("name") == new_first),
                                                                      None) or new_inputs[0])})
                ent["files.extra_designs"] = own_file
                ent["columns.extra_design_rate"] = own_col
                ent["aside"] = True
                swap_note = (f"{new_first.capitalize()} is now the first input, whose rates are read from the main "
                             "design (or its as-applied map): its own "
                             + ("design file" if own_file else "'(none)' choice")
                             + " is kept aside and comes back when it is tested with another input. Choose its "
                             "rate column under 'Design columns'.")
            xdf.pop(new_first, None)
            xdr.pop(new_first, None)
            cfg["columns"]["design_rate"] = back
            cfg["ui"]["first_rate_name_only"] = back is None
            ss().pop(k("col_design_rate"), None)
        # inputs coming back from the stash get their files and columns again (a later input: its own design file
        # or main-design column; an input that was first when stashed and is now later: its Rate column)
        for nm_ in keep:
            if nm_ in stash and nm_ not in cur_names:
                ent = stash[nm_]
                for a, b in STASH_KEYS:
                    v_ = ent.get(f"{a}.{b}")
                    if v_ is not None and (cfg[a].setdefault(b, {}).get(nm_) is None):
                        if nm_ == new_first and (a, b) in (("files", "extra_designs"), ("columns", "extra_design_rate")):
                            continue            # the first slot cannot use them (kept in the stash above)
                        cfg[a][b][nm_] = v_
                if nm_ != new_first and xdr.get(nm_) is None and not xdf.get(nm_) and ent.get("columns.design_rate"):
                    xdr[nm_] = ent["columns.design_rate"]
                if nm_ != new_first or not ent.get("aside"):
                    stash.pop(nm_, None)
        # an input still in the list keeps no stash entry: an old entry would later override its live choice
        for nm_ in list(stash):
            if nm_ in keep and nm_ in cur_names and not (stash[nm_] or {}).get("aside"):
                stash.pop(nm_, None)
        if swap_note:
            cfg["ui"]["swap_note"] = swap_note
        for dct in (cfg["files"].setdefault("asapplied", {}), cfg["files"].setdefault("extra_designs", {}),
                    cfg["columns"].setdefault("applied", {}), cfg["columns"].setdefault("extra_design_rate", {}),
                    cfg["columns"].setdefault("applied_basis", {})):
            for key_ in [kk for kk in dct if kk not in keep]:
                dct.pop(key_, None)
        cfg["prices"]["inputs"] = new_inputs
    if (cfg.get("ui") or {}).get("swap_note"):
        st.info(cfg["ui"].pop("swap_note"))
    if any(i["status_quo"] is None for i in new_inputs):
        st.caption("Without a status quo the report compares VR with the best flat rate only (no grower comparison).")
    xrate_missing = []          # inputs after the first with no design-rate column yet: 'Load the trial' waits
    # Inputs after the first declared with an explicit '(none)' design rate (stored as NONE_COL): a treatment-only
    # strip trial whose urea rates exist only in its as-applied map. load_design ignores the empty column, analyse
    # skips the input's rate response with the reason, and treatments.compare still charges its applied rate. It is
    # allowed only with a treatment column or an as-applied map of that input (checked after the design columns are
    # drawn, so the Treatment picker of this same run counts); without either the input would carry nothing.
    xrate_none = []
    xhas_ap = set()             # inputs with an as-applied map given
    ap_nocol = set()            # inputs with an as-applied map whose rate column is not chosen yet
    xrate_applied = []          # inputs whose design rates are read from their as-applied map (FROM_APPLIED)
    # (input, design file, rate column) of every input after the first that reads its rates from a column: two inputs
    # on the same column of the same file gave design_<a> == design_<b>, each became a control in the other's model
    # with an identical regressor, and the pseudo-inverse split the rate effect between them without an error.
    xr_used = []
    for j, inp in enumerate([] if name_errors else cfg["prices"]["inputs"]):
        # a later input's box is open until the trial is loaded or while its rate column is not chosen (None; an
        # explicit '(none)' is a choice), so that choice is not hidden in a collapsed box
        need_rate = bool(j) and (not done("design")
                                 or (cfg["columns"].get("extra_design_rate") or {}).get(inp["name"]) is None)
        with st.expander(f"As-applied map for **{inp['name']}**" + (" and its design" if j else ""), expanded=need_rate):
            ap = file_field(f"As-applied ({inp['name']})", f"asapplied:{inp['name']}",
                            ["shp", "shx", "dbf", "prj", "cpg", "zip", "gpkg", "parquet"])
            if ap:
                xhas_ap.add(inp["name"])
                cols, df = peek_columns(ap)
                # no "(none)" here: a given as-applied map needs its rate column (to not use the map, clear the file)
                apc = col_select("Applied-rate column", cols, "applied_rate", cfg["columns"].get("applied", {}).get(inp["name"]),
                                 df, key=f"apcol_{inp['name']}", none_ok=False, src=ap)
                cfg["columns"].setdefault("applied", {})[inp["name"]] = apc
                if apc is None:
                    ap_nocol.add(inp["name"])
                    st.warning(f"Choose the applied-rate column of the {inp['name']} as-applied map (the join waits for it).")
                if not (inp.get("rate_basis") == "nutrient" and inp.get("product") not in ("seed", "other")):
                    # the choice applies to a nutrient-basis input only: a stale value must not convert the map
                    (cfg["columns"].get("applied_basis") or {}).pop(inp["name"], None)
                if inp.get("rate_basis") == "nutrient" and inp.get("product") not in ("seed", "other"):
                    # the rates of this input are lb of N, but most controllers log lb of product: the map's own
                    # basis is asked and the engine converts (project._rate_conversion); before this the product
                    # rates were costed at the price per lb N
                    abd = cfg["columns"].setdefault("applied_basis", {})
                    cur_ab = abd.get(inp["name"]) or "nutrient"
                    abd[inp["name"]] = st.selectbox(f"What the {inp['name']} map logs", ["product", "nutrient"],
                                                    index=["product", "nutrient"].index(cur_ab), key=k(f"apbasis_{inp['name']}"),
                                                    help="The rates of this input are lb of nutrient. If the as-applied map "
                                                         "logs lb of product (most controllers do), choose product: the map "
                                                         "is converted with the product's nutrient fraction. The join warns "
                                                         "when the map and the design look like different bases.")
            if j:
                st.caption(f"The rates of {inp['name']}: a separate design file on the same plots, or leave the file empty "
                           "and choose the column of the main design that holds them. Load the trial again after changing it.")
                xd = file_field(f"Design for {inp['name']}", f"extra:{inp['name']}", ["shp", "shx", "dbf", "prj", "zip", "geojson", "gpkg"])
                xr = cfg["columns"].setdefault("extra_design_rate", {})
                src_path = xd or design
                if src_path:
                    xcols, xdf = peek_columns(src_path)
                    cur_x = xr.get(inp["name"])
                    xlabel = f"Rate column for {inp['name']} " + ("(in its design file)" if xd else "(in the main design)")
                    # '(from the as-applied map)' only without a design file of its own (that file's column wins
                    # in load_design) and with an as-applied map given: the plot medians of that map are the rates
                    # A stored '(from the as-applied map)' is kept while the map is absent for a run (a mistyped
                    # path, a transient input-name error): the option stays offered so the picker does not fall
                    # back to a guessed design column, and 'Load the trial' waits with the reason (no map).
                    applied_x = (not xd) and (inp["name"] in xhas_ap or cur_x == P.FROM_APPLIED)
                    if not xcols and xdf is None:
                        # the file could not be read this run (see col_select): keep the stored choice, do not guess
                        shown_x = (FROM_APPLIED_LABEL if cur_x == P.FROM_APPLIED else
                                   cur_x if cur_x else ("(none)" if cur_x == NONE_COL and not xd else CHOOSE))
                        st.selectbox(xlabel, [shown_x], disabled=True, key=k(f"xrate_unreadable_{inp['name']}"))
                        if cur_x is None or (xd and not cur_x):
                            xrate_missing.append(inp["name"])
                        elif cur_x == NONE_COL:
                            xrate_none.append(inp["name"])
                        elif cur_x == P.FROM_APPLIED:
                            xrate_applied.append(inp["name"])
                        else:
                            xr_used.append((inp["name"], src_path, cur_x))
                    else:
                        main_rate = cfg["columns"].get("design_rate")
                        if not xd:          # the main design: its own rate column belongs to the first input
                            xcols = [c_ for c_ in xcols if c_ != main_rate]
                        # '(none)' only in the main design: a separate design file given for the input says it has
                        # design rates (load_design would auto-detect its column); to drop them, clear that file
                        none_x = not xd
                        if cur_x not in xcols and not (none_x and cur_x == NONE_COL) and not (applied_x and cur_x == P.FROM_APPLIED):
                            # guess: a numeric column named after the input by whole tokens (tgt_urea for 'urea',
                            # Tgt_N for 'N' but not Plot_No); in a separate file, the usual design-rate names. Never
                            # the main design's rate column (that is the first input's). No match: '(choose)'.
                            cur_x = guess_input_rate_col(xdf, xcols, inp["name"])
                            if cur_x is None and xd and xdf is not None:
                                cur_x = eio.find_col(xdf, "design_rate")
                                if cur_x == main_rate:      # e.g. the main design given again: tgt_seed is not urea
                                    cur_x = None
                        opts = [CHOOSE] + (["(none)"] if none_x else []) + ([FROM_APPLIED_LABEL] if applied_x else []) + xcols
                        init = ("(none)" if (none_x and cur_x == NONE_COL) else
                                FROM_APPLIED_LABEL if (applied_x and cur_x == P.FROM_APPLIED) else
                                cur_x if cur_x in xcols else CHOOSE)
                        wk = k(f"xrate_{'file' if xd else 'main'}_{inp['name']}")
                        # set through session_state, as col_select does: a keyed selectbox ignores `index` once it
                        # has a state, and a state that is no longer an option must be re-initialised
                        if wk not in ss() or ss()[wk] not in opts:
                            ss()[wk] = init
                        v = st.selectbox(xlabel, opts, key=wk)
                        if v == CHOOSE:
                            xr[inp["name"]] = None
                            xrate_missing.append(inp["name"])
                        elif v == FROM_APPLIED_LABEL:
                            xr[inp["name"]] = P.FROM_APPLIED
                            xrate_applied.append(inp["name"])
                            st.caption(f"'{FROM_APPLIED_LABEL}': each plot takes the median {inp['name']} rate of the "
                                       "as-applied points inside its trimmed interior, and the plot medians are "
                                       "grouped into the levels the drill delivered; those levels are the design "
                                       "rates of this input (the units still use their own applied medians). Plots "
                                       "with mixed rates or too few points get no design rate and are listed after "
                                       "loading.")
                        elif v == "(none)":
                            xr[inp["name"]] = NONE_COL
                            xrate_none.append(inp["name"])
                            st.caption(f"'(none)': {inp['name']} has no design rates, so it gets no rate response, "
                                       "optimum or prescription; its applied rate (as-applied map) is still costed in "
                                       "the treatment comparison. Allowed with a treatment column or an as-applied "
                                       f"map of {inp['name']}.")
                        else:
                            xr[inp["name"]] = v
                            xr_used.append((inp["name"], src_path, v))
    for nm in xrate_missing:        # outside the expander, so it is seen
        st.warning(f"Choose the column with the {nm} rates (or give its design file, or '(none)' for a treatment-only "
                   f"input) under 'As-applied map for {nm} and its design'. 'Load the trial' waits for it.")

    if design:
        cols, ddf = peek_columns(design)
        with st.expander("Design columns", expanded=not done("design")):
            cc = st.columns(5)
            # The Treatment picker is drawn (into its own column) before the Rate picker: the Rate picker offers
            # '(none)' only when a treatment column is set, and reading the treatment of the previous run gave a
            # Rate picker without '(none)' on the first draw and with it on the next rerun.
            with cc[3]:
                cfg["columns"]["treatment"] = col_select("Treatment", cols, "treatment", cfg["columns"].get("treatment"), ddf, src=design)
            with cc[0]:
                # '(none)' only for a treatment-only trial (a treatment column is set): the engine then skips the
                # rate response and still compares the treatments. It is never a fallback (choose_first): a rate
                # column that is not recognised shows '(choose)' and blocks loading, so a declared input is not
                # silently analysed as treatment-only. The guess is name-aware, as for the later inputs: find_col
                # alone always took tgt_seed first, so with inputs [urea, seed] urea got the seed rates and the seed
                # input could no longer pick its own column.
                first_in = ((cfg["prices"]["inputs"] or [{}])[0] or {}).get("name")
                g_first = guess_input_rate_col(ddf, cols, first_in) if first_in else None
                # find_col's fallback must not hand the first input a later input's column: its list ranks
                # tgt_urea / tgt_n before Tgt_Rate, so [seed, urea] with Tgt_Rate + tgt_urea pre-filled seed with
                # tgt_urea. Excluded: columns named after a later input (unless also named after the first) and the
                # main-design columns the later inputs already hold.
                xr_now = cfg["columns"].get("extra_design_rate") or {}
                xfiles = cfg["files"].get("extra_designs") or {}
                not_first = set()
                for inp_ in (cfg["prices"]["inputs"] or [])[1:]:
                    nm_ = (inp_ or {}).get("name")
                    if not nm_:
                        continue
                    not_first |= {c_ for c_ in cols if name_matches(c_, nm_)
                                  and not (first_in and name_matches(c_, first_in))}
                    if not xfiles.get(nm_) and xr_now.get(nm_):
                        not_first.add(xr_now[nm_])
                first_ap = bool(first_in) and (first_in in xhas_ap or cfg["columns"].get("design_rate") == P.FROM_APPLIED)
                name_only = bool(cfg.get("ui", {}).get("first_rate_name_only"))
                cfg["columns"]["design_rate"] = col_select("Rate", cols, "design_rate", cfg["columns"].get("design_rate"), ddf,
                                                           none_ok=bool(cfg["columns"].get("treatment")), src=design,
                                                           guess_col=g_first, choose_first=True, guess=not name_only,
                                                           exclude=sorted(not_first),
                                                           extra_opts={FROM_APPLIED_LABEL: P.FROM_APPLIED} if first_ap else None)
                if cfg["columns"]["design_rate"] is not None:
                    cfg.setdefault("ui", {}).pop("first_rate_name_only", None)
                if cfg["columns"]["design_rate"] == P.FROM_APPLIED:
                    st.caption(f"'{FROM_APPLIED_LABEL}': the {first_in} rates of the plots are read from the "
                               f"{first_in} as-applied map (plot medians grouped into levels); the design file gives "
                               "the plots only.")
            with cc[1]:
                cfg["columns"]["plot_id"] = col_select("Plot id", cols, "plot_id", cfg["columns"].get("plot_id"), ddf, src=design)
            with cc[2]:
                cfg["columns"]["type"] = col_select("Type (border)", cols, "type", cfg["columns"].get("type"), ddf, src=design)
            with cc[4]:
                cfg["columns"]["block"] = col_select("Block", cols, "block", cfg["columns"].get("block"), ddf, src=design)
    if yfile:
        cols, ydf = peek_columns(yfile)
        with st.expander("Yield columns", expanded=not done("design")):
            cc = st.columns(5)
            for i, (role, lab) in enumerate((("yield", "Yield (dry, bu/ac)"), ("heading", "Heading"), ("distance", "Distance"),
                                             ("swath", "Swath width"), ("machine", "Machine"))):
                with cc[i]:
                    cfg["columns"][role] = col_select(lab, cols, role, cfg["columns"].get(role), ydf, none_ok=role != "yield",
                                                      src=yfile)
                    if role in ("distance", "swath") and cfg["columns"].get(role):
                        # the unit of a length column was read from its name alone ('_m' = metres, else feet), and a
                        # metric SMS export truncates 'Swth Wdth(m)' to the same name as the imperial one; the user
                        # chooses, the name only sets the default, and the join checks the swath against the pass
                        # spacing of the yield points
                        ukey = f"{role}_unit"
                        col_ = cfg["columns"][role]
                        # the unit chosen for ANOTHER column is not carried over: the new column's name sets it
                        cur_u = cfg["columns"].get(ukey) if cfg["columns"].get(f"{ukey}_col") == col_ else None
                        cur_u = cur_u or eio.length_unit_from_name(col_)
                        cur_u = "m" if str(cur_u).lower().startswith("m") else "ft"
                        cfg["columns"][f"{ukey}_col"] = col_
                        cfg["columns"][ukey] = st.selectbox("Unit", ["ft", "m"], index=["ft", "m"].index(cur_u),
                                                            key=k(f"{ukey}_{col_}"),
                                                            help="SMS and Ops Center exports are in feet unless the "
                                                                 "export was set to metric.")
            if ydf is not None and cfg["columns"].get("yield"):
                v = pd.to_numeric(ydf[cfg["columns"]["yield"]], errors="coerce")
                st.caption(f"Sample of {len(ydf):,} points: yield mean {v.mean():.1f}, 5-95 % {v.quantile(.05):.1f} to {v.quantile(.95):.1f}")

    load_block = []
    if design and cfg["columns"].get("design_rate") is None:
        load_block.append("Choose the rate column of the design" + (
            " (or '(none)' for a treatment-only trial: no rate response, optimum or prescription)."
            if cfg["columns"].get("treatment") else " (or set a treatment column for a treatment-only trial)."))
    if yfile and cfg["columns"].get("yield") is None:
        load_block.append("Choose the yield column of the yield map.")
    # A later input's '(none)' needs a treatment column (treatment-only trial) or that input's as-applied map:
    # with neither, the input would have no rates at all and nothing to cost or compare.
    for nm in xrate_none:
        if not (cfg["columns"].get("treatment") or nm in xhas_ap):
            load_block.append(f"'(none)' for the {nm} rates needs a treatment column in the design or an as-applied "
                              f"map of {nm}: choose its rate column, or set one of those.")
    # The first input's rate column named after a later input (tgt_urea for seed) loads silently with the two
    # inputs' designs swapped (economics and prescription columns swapped too), so it needs a confirmation.
    dr_ = cfg["columns"].get("design_rate")
    first_ = ((cfg["prices"]["inputs"] or [{}])[0] or {}).get("name")
    if design and dr_ and dr_ != P.FROM_APPLIED and not name_errors:
        clash = [i.get("name") for i in (cfg["prices"]["inputs"] or [])[1:]
                 if i and i.get("name") and name_matches(dr_, i["name"]) and not (first_ and name_matches(dr_, first_))]
        if clash and not st.checkbox(f"Yes, '{dr_}' holds the {first_} rates (not the {clash[0]} rates)",
                                     key=k(f"confirm_first_rate_{dr_}")):
            load_block.append(f"The Rate column '{dr_}' of the first input ({first_}) is named after {clash[0]}: choose "
                              f"the {first_} column under 'Design columns', or confirm that '{dr_}' holds the {first_} rates.")
    # Two inputs on the same rate column of the same file (the first input's Rate column included) would be analysed
    # as one input split in two: each other's control with identical rates, each response roughly halved.
    if not name_errors:
        same_col = {}
        used_ = ([(first_, design, dr_)] if (design and dr_ and dr_ != P.FROM_APPLIED and first_) else []) + xr_used
        for nm_, p_, col_ in used_:
            same_col.setdefault((os.path.normcase(os.path.abspath(str(p_))), col_), []).append(nm_)
        for (p_, col_), nms_ in same_col.items():
            if len(nms_) > 1:
                where_ = f"under 'As-applied map for {nms_[-1]} and its design'" + (
                    f" (or another Rate column for {first_} under 'Design columns')" if nms_[0] == first_ else "")
                load_block.append(f"{' and '.join(nms_)} take the same design column '{col_}': each input needs its "
                                  f"own rate column. Choose another column for {nms_[-1]} {where_}.")
    first_fa = design and cfg["columns"].get("design_rate") == P.FROM_APPLIED
    for nm in ([first_] if (first_fa and first_) else []) + xrate_applied:
        if nm not in xhas_ap:
            load_block.append(f"The {nm} rates are to be read from the {nm} as-applied map, but no map is given: give "
                              f"it under 'As-applied map for {nm}', or choose a design column.")
        elif nm in ap_nocol:
            load_block.append(f"The {nm} rates are to be read from the {nm} as-applied map: choose that map's "
                              "applied-rate column first.")
    for msg in load_block:
        st.warning(msg)
    # The plots, their design_<input> columns and the field area come from these files and columns: when they change
    # after the trial was loaded, every later step is built on the old plots (re-running the join alone gave target
    # rates of NaN for an input added or renamed later), so the trial and everything after it must be redone.
    if done("design") and fp_changed("design_fp", design_payload(cfg)):
        invalidate(["design", "zones", "join", "analysis", "report"],
                   "The design, boundary or extra design files, the design columns, the rates of an input or the list "
                   "of inputs changed after the trial was loaded: load the trial again (then zones, join and economics).")
    stale_box = st.empty()
    if st.button("Load the trial", type="primary",
                 disabled=not (design and yfile) or bool(load_block or xrate_missing or name_errors),
                 key=k("btn_design")):
        try:
            s = state()
            r = s.load_design()
            mark("design")
            fp_store("design_fp", design_payload(cfg))
            (ss().get("stale") or {}).pop("design", None)
            invalidate(["zones", "join", "analysis", "report"])
            save()
        except Exception as exc:
            st.exception(exc)
    with stale_box.container():
        show_stale("design")
    if done("design"):
        s = state()
        D = s.res["design"]
        c = st.columns(4)
        c[0].metric("Plots", D["n_plots"])
        c[1].metric("Plot area", f"{D['plot_area_ac']:.1f} ac")
        c[2].metric("Field area", f"{D['boundary_ac']:.1f} ac")
        c[3].metric("Rates", ", ".join(f"{r:g}" for r in D["rates"]) or "-",
                    help="Of the first input; the rates of every input are listed below.")
        st.caption(f"Field area from the {D['boundary_source']}.")
        for nm_, cv_ in (D.get("rate_conversion") or {}).items():
            if cv_.get("note"):
                st.info(f"Rates of {nm_}: {cv_['note']}. Every rate below, in the PDF and in the prescription is lb/ac.")
        for nm_, rb in (D.get("rates_by_input") or {}).items():
            fa_ = rb.get("from_applied") or {}
            if str(rb.get("source", "")).startswith("none"):
                st.caption(f"**{nm_}**: no design rates (" + str(rb["source"]).split(": ", 1)[-1] + ")")
                continue
            line = (f"**{nm_}**: rates " + (", ".join(f"{r:g}" for r in rb["rates"]) or "none")
                    + f" ({rb['source']}); plots per rate: "
                    + (", ".join(f"{float(k_):g}: {v}" for k_, v in rb["plots_per_rate"].items()) or "-"))
            if fa_:
                n_no = int(fa_.get("n_few_points") or 0) + int(fa_.get("n_mixed") or 0) + int(fa_.get("n_off_level") or 0)
                line += (f"; levels read from column '{fa_.get('column')}': {fa_.get('n_with_level')} of "
                         f"{fa_.get('n_plots')} plots at a level")
                if n_no:
                    line += (f", {fa_.get('n_few_points')} with fewer than {fa_.get('min_pts')} points, "
                             f"{fa_.get('n_mixed')} with mixed rates, {fa_.get('n_off_level')} off every level "
                             "(those plots have no design rate"
                             + (" and enter the response with their applied rate)"
                                if cfg["join"].get("rate_source", "applied") == "applied"
                                else " and are left out of the response: the models use the design rate)"))
            elif rb.get("n_without"):
                line += f"; {rb['n_without']} plot(s) without a design rate"
            st.caption(line)
            if fa_.get("warning"):
                st.warning(fa_["warning"][0].upper() + fa_["warning"][1:] + ".")
        if not D.get("rates_by_input"):
            st.caption("Plots per rate: " + ", ".join(f"{k_:g}: {v}" for k_, v in D["plots_per_rate"].items()))
        show_fig(RP.fig_design(s))

# ================================================================================================ 2 NDVI & zones
with tabs[1]:
    if not done("design"):
        st.info("Load the trial in step 1 first.")
    else:
        s = state()
        n, z = cfg["ndvi"], cfg["zones"]
        SRC = {"fetch": "Fetch peak-season NDVI from Copernicus (Sentinel-2, 2017 on)",
               "files": "NDVI GeoTIFFs I already have (one per date)",
               "map": "A multi-year NDVI map (raster or polygons with an NDVI column)",
               "zonemap": "A zone map I already have"}
        cur = "zonemap" if z.get("source") == "map" else n.get("source", "fetch")
        src = st.radio("Source of the zones", list(SRC), format_func=SRC.get, index=list(SRC).index(cur), key=k("ndvi_src"))
        zone_block = None       # why "Build zones" cannot run now
        if src == "zonemap":
            z["source"] = "map"
            zp = file_field("Zone map", "zones", ["shp", "shx", "dbf", "prj", "zip", "geojson", "gpkg", "kml", "kmz"])
            # Without a readable zone map and its zone column, make_zones would quietly fall back to NDVI zones (or
            # one zone) that the user did not choose, so the button is disabled until both are given.
            if not zp:
                zone_block = "Give the zone map (a path that exists or an upload) to build the zones from it."
            else:
                cols, zdf = peek_columns(zp)
                if not cols:
                    zone_block = "The zone map could not be read; give a valid file."
                else:
                    # never guessed: an OBJECTID or Name column taken by default gave one zone per polygon, or zones
                    # in alphabetical order, so the button waits until the user picks the column
                    cfg["columns"]["zone"] = col_select("Zone column", cols, "zone", cfg["columns"].get("zone"), zdf,
                                                        none_ok=False, src=zp, guess=False)
                    if not cfg["columns"]["zone"]:
                        zone_block = "Choose the zone column of the zone map."
                    nd = st.text_input("No-data codes (optional)", ", ".join(str(x) for x in (z.get("nodata_codes") or [])),
                                       key=k("znodata"), placeholder="e.g. 0, -9999, outside",
                                       help="Codes of the zone column that mean 'no zone' (their cells go to the "
                                            "neighbouring zones). Negative numbers are always no-data. 0 is read as a "
                                            "zone unless it is listed here, or its polygons cover under 5 % of the field "
                                            "and at least 2 zones remain without it; the zone note says which reading "
                                            "was taken.")
                    z["nodata_codes"] = [_code(x) for x in nd.split(",") if x.strip()] or None
            st.caption("Optional: also give NDVI (below) to use it as a continuous covariate; the zones come from the map.")
            n["source"] = st.selectbox("NDVI for the covariate", ["none", "fetch", "files", "map"], key=k("ndvi_cov"),
                                       index=["none", "fetch", "files", "map"].index(n.get("source", "none")) if n.get("source") in ("none", "fetch", "files", "map") else 0)
        else:
            z["source"] = "ndvi"
            n["source"] = src
        # 'files' and 'fetch' used to share files.ndvi_dir. After a fetch it pointed at the project's fetch cache
        # (workdir/ndvi_images, which keeps the images of every earlier boundary, window and dates-per-season
        # setting), so switching the source to 'files' pre-filled that folder and load_ndvi stacked all of it. The
        # user's own folder is now kept under files.ndvi_user_dir, and ndvi_dir (what load_ndvi reads) is set from
        # the folder that belongs to the source in use.
        fl = cfg["files"]
        fetch_dir = os.path.join(workdir(), "ndvi_images")
        if "ndvi_user_dir" not in fl:
            # a project saved before the split: its ndvi_dir was the user's folder unless it is the fetch cache
            nd0 = fl.get("ndvi_dir")
            is_cache = bool(nd0) and os.path.normcase(os.path.abspath(str(nd0))) == os.path.normcase(os.path.abspath(fetch_dir))
            fl["ndvi_user_dir"] = None if (not nd0 or is_cache or n["source"] == "fetch") else nd0
        if n["source"] == "fetch":
            if os.path.isdir(fetch_dir):
                fl["ndvi_dir"] = fetch_dir
            elif fl.get("ndvi_dir") and fl.get("ndvi_dir") == fl.get("ndvi_user_dir"):
                fl["ndvi_dir"] = None       # the user's folder is not a fetch
        if n["source"] == "fetch":
            ok = NF.available()
            if not ok:
                st.error(f"The NDVI Map Viewer was not found at {NF.NDVIMAP_DIR} (set OCVR_NDVIMAP_DIR).")
            yrs = list(range(2017, int(cfg["year"]) + 1))
            c = st.columns([3, 1, 1, 1])
            n["years"] = c[0].multiselect("Seasons", yrs, default=[y for y in n.get("years", yrs) if y in yrs and y != int(cfg["year"])],
                                          key=k("years"))
            w0 = c[1].text_input("Window from (MM-DD)", n["window"][0], key=k("w0"))
            w1 = c[2].text_input("to (MM-DD)", n["window"][1], key=k("w1"))
            n["window"] = [w0, w1]
            n["dates_per_year"] = int(c[3].number_input("Dates per season", 1, 5, int(n.get("dates_per_year", 3)), key=k("dpy")))
            st.caption("Each season = the per-pixel median of its clearest dates (≥ 95 % of the field usable after the cloud "
                       "mask) with the highest field-mean NDVI inside the window. The trial year is left out. Credentials "
                       "are the ones saved in the NDVI Map Viewer.")
            if st.button("Fetch NDVI", disabled=not ok, key=k("btn_fetch")):
                bar = st.progress(0.0, "starting")
                try:
                    log = s.fetch_ndvi(progress=lambda f, m: bar.progress(min(1.0, f), m))
                    if (ss().get("stale") or {}).get("zones"):
                        # the 'fetch again' advice is done: what is left is building the zones
                        ss()["stale"]["zones"] = ("The NDVI was fetched again: build the zones again (then the join "
                                                  "and the economics).")
                    save()
                    st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)
                except Exception as exc:
                    st.exception(exc)
            if cfg["files"].get("ndvi_dir") or cfg["files"].get("ndvi_paths"):
                # the images load_ndvi uses (fetched or reused for this boundary and these settings), not every file
                # of the shared cache folder
                st.caption(f"{len(cfg['files'].get('ndvi_paths') or [])} image(s) for this boundary and these settings, "
                           f"in `{cfg['files'].get('ndvi_dir') or fetch_dir}`")
        elif n["source"] == "files":
            # the original names of uploaded GeoTIFFs are kept for the PDF ('N files uploaded in the app'): the
            # folder they are saved in is named by a hash of the upload
            file_field("Folder or zip of NDVI GeoTIFFs", "ndvi_user_dir", ["tif", "tiff", "zip"],
                       names_key="ndvi_upload_names")
            fl["ndvi_dir"] = fl.get("ndvi_user_dir")
            st.caption("Band 1 = NDVI (float), band 2 (optional) = validity; the date is read from the file name "
                       "(the NDVI Map Viewer's 'GeoTIFF - index values' downloads are named that way).")
        elif n["source"] == "map":
            mp = file_field("Multi-year NDVI map", "ndvi_map", ["tif", "tiff", "shp", "shx", "dbf", "prj", "zip", "gpkg", "geojson"])
            if mp and not mp.lower().endswith((".tif", ".tiff")):
                cols, mdf = peek_columns(mp)
                cfg["columns"]["ndvi"] = col_select("NDVI column", cols, "ndvi", cfg["columns"].get("ndvi"), mdf, none_ok=False,
                                                    src=mp)
                if not cfg["columns"]["ndvi"] and not zone_block:
                    zone_block = "Choose the NDVI column of the NDVI map."

        # NDVI zones with no NDVI given: load_ndvi then returned 'no NDVI' and make_zones quietly built one zone over
        # the whole field, which the verdict and the PDF read as a finding about the field. One zone has to be an
        # explicit choice (a zone map with a single zone), so the button waits for the NDVI.
        if z["source"] == "ndvi" and not zone_block:
            if n["source"] == "files" and not fl.get("ndvi_dir"):
                zone_block = "Give the folder or zip of NDVI GeoTIFFs (a path that exists or an upload) to build the zones."
            elif n["source"] == "map" and not fl.get("ndvi_map"):
                zone_block = "Give the multi-year NDVI map (a path that exists or an upload) to build the zones."
            elif n["source"] == "fetch" and not (fl.get("ndvi_paths") or fl.get("ndvi_dir")):
                zone_block = "Fetch the NDVI first (button above) to build the zones from it."
            elif n["source"] == "fetch" and isinstance(fl.get("ndvi_paths"), list) and not fl["ndvi_paths"]:
                zone_block = ("The last fetch found no usable image for this boundary: widen the window or the seasons "
                              "and fetch again.")
        # The engine refuses fetched images made with another window / dates per season / boundary (load_ndvi raises
        # 'fetch again'), and a season added after the fetch has no image yet: say so here, before the click. The
        # engine's own comparison (State.ndvi_fetch_changes, which also checks the boundary) is used when it can run;
        # otherwise the page compares the window and the dates per season itself.
        fetch_changed = []
        fs = fl.get("ndvi_fetch_settings") if n["source"] == "fetch" else None
        if isinstance(fs, dict) and fs:
            eng = None
            if getattr(s, "boundary", None) is not None and hasattr(s, "ndvi_fetch_changes"):
                try:
                    eng = list(s.ndvi_fetch_changes())
                except Exception:
                    eng = None
            if eng is not None:
                lab = {"window": "window", "dates_per_year": "dates per season", "boundary": "boundary"}
                fetch_changed += [lab.get(x, str(x)) for x in eng]
            else:
                if fs.get("window") is not None and [str(x) for x in fs["window"]] != [str(x) for x in (n.get("window") or [])]:
                    fetch_changed.append("window")
                try:
                    if fs.get("dates_per_year") is not None and int(fs["dates_per_year"]) != int(n.get("dates_per_year", 3)):
                        fetch_changed.append("dates per season")
                except (TypeError, ValueError):
                    pass
            want = {int(y) for y in (n.get("years") or [])}
            if n.get("exclude_trial_year", True):
                want.discard(int(cfg["year"]))
            # fetch_ndvi records in 'years' only the seasons it fetched (or found without a clear date) and in
            # 'failed' the ones the network failed on. A failed season is not a setting the user changed, so it does
            # not block the zones (they can be built from the other seasons, and the PDF names it), but the user is
            # told to fetch again: the next fetch tries those seasons again.
            failed = sorted(want & {int(y) for y in (fs.get("failed") or [])})
            partial = sorted(set(failed) & {int(y) for y in (fs.get("partial") or [])})
            lost = [y for y in failed if y not in partial]
            if partial:
                st.warning(f"Season{'s' if len(partial) > 1 else ''} {', '.join(map(str, partial))} lost some dates in "
                           "the last fetch (network or server error): the zones use the dates that did download, with "
                           "fewer dates in the season median. Fetch again to complete them.")
            if lost:
                st.warning(f"Season{'s' if len(lost) > 1 else ''} {', '.join(map(str, lost))} could not be fetched "
                           "in the last fetch (network or server error): fetch again to add "
                           f"{'them' if len(lost) > 1 else 'it'}. The zones can be built without "
                           f"{'them' if len(lost) > 1 else 'it'}; the report then names the missing "
                           f"season{'s' if len(lost) > 1 else ''}.")
            if fs.get("years") is not None:
                new_years = sorted(want - {int(y) for y in fs["years"]} - set(failed))
                if new_years:
                    fetch_changed.append("seasons (" + ", ".join(str(y) for y in new_years) + " not fetched)")
        if fetch_changed and not zone_block:
            zone_block = (f"The NDVI {', '.join(fetch_changed)} changed since the last fetch: fetch again, then build "
                          "the zones.")
        # Images fetched before the fetch settings were recorded: load_ndvi cannot tell which window, dates per
        # season or boundary they belong to, and refuses them (it used to stack the whole cache folder). Say so here.
        older_fetch = (n["source"] == "fetch" and bool(fl.get("ndvi_paths") or fl.get("ndvi_dir"))
                       and not fl.get("ndvi_fetch_settings"))
        if older_fetch and not zone_block:
            zone_block = ("This project was fetched with an older version: fetch again (the 'Fetch NDVI' button "
                          "above), then build the zones.")

        if n["source"] in ("fetch", "files"):
            with st.expander("Crop history (which seasons count)", expanded=True):
                yrs = sorted(set(int(y) for y in (n.get("years") or [])) | {int(y) for y in (n.get("crop_history") or {})})
                ch = pd.DataFrame({"year": yrs, "crop": [(n.get("crop_history") or {}).get(str(y)) or (n.get("crop_history") or {}).get(y) or "" for y in yrs]})
                ch = st.data_editor(ch, key=k("croph"), hide_index=True, width="stretch",
                                    column_config={"year": st.column_config.NumberColumn(disabled=True, format="%d"),
                                                   "crop": st.column_config.SelectboxColumn(options=[""] + CROPS)})
                n["crop_history"] = {str(int(r.year)): str(r.crop).strip() for _, r in ch.iterrows() if str(r.crop).strip()}
                POL = {"same_crop": f"Same crop as the trial ({cfg['crop']}), else the crop group, else all",
                       "crop_group": "Crop group (" + ", ".join(n.get("crop_group") or []) + ")", "all": "All clear seasons"}
                n["policy"] = st.radio("Seasons used", list(POL), format_func=POL.get, index=list(POL).index(n.get("policy", "same_crop")),
                                       key=k("policy"), horizontal=True)
                n["crop_group"] = [x.strip() for x in st.text_input("Crop group", ", ".join(n.get("crop_group") or []),
                                                                     key=k("cgroup")).split(",") if x.strip()]
                st.caption("NDVI is scaled differently by crop and stage; zones are best built from seasons of the same crop "
                           "(at least 3; Blackmore 2000, Georgi et al. 2018).")
        if z["source"] == "ndvi":
            c = st.columns(4)
            z["k"] = int(c[0].slider("Zones", 2, 5, int(z.get("k", 3)), key=k("zk")))
            z["method"] = c[1].selectbox("Breaks", ["quantile", "kmeans"], index=["quantile", "kmeans"].index(z.get("method", "quantile")),
                                         key=k("zm"), help="quantile = equal-area classes (default); kmeans = natural breaks")
            z["min_patch_ha"] = float(c[2].number_input("Smallest patch (ha)", 0.0, 10.0, float(z.get("min_patch_ha", 0.5)), 0.1, key=k("zp")))
            z["smooth"] = int(c[3].selectbox("Smoothing (pixels)", [1, 3, 5], index=[1, 3, 5].index(int(z.get("smooth", 3))), key=k("zs")))
        if zone_block:
            st.warning(zone_block)
        if done("zones") and fp_changed("zones_fp", zones_payload(cfg)):
            invalidate(["zones", "join", "analysis", "report"],
                       ("The NDVI fetch settings changed after the zones were built: fetch again, then build the zones "
                        "(then the join and the economics).") if fetch_changed else
                       ("The NDVI or zone settings or files, or the trial crop or year, changed after the zones were "
                        "built: build the zones again (then the join and the economics)."))
        # drawn into a placeholder filled after the button, as in the other steps: drawn before it, the warning the
        # button clears stayed on screen next to the rebuilt zones until the next click
        stale_box = st.empty()
        if st.button("Build zones", type="primary", key=k("btn_zones"), disabled=bool(zone_block)):
            try:
                with st.spinner("Reading the NDVI and classifying"):
                    s.load_ndvi()
                    s.make_zones()
                    RP.figures(s, parts=("ndvi",))
                mark("zones")
                fp_store("zones_fp", zones_payload(cfg))
                (ss().get("stale") or {}).pop("zones", None)
                invalidate(["join", "analysis", "report"])
                save()
            except ValueError as exc:
                if str(exc).startswith("This project was fetched with an older version"):
                    # not a bug: the images predate the recorded fetch settings, so the fix is a new fetch
                    st.warning(f"{exc} with the 'Fetch NDVI' button above, then build the zones.")
                elif "fetch again" in str(exc):      # fetched images made with other settings: a user action, not a bug
                    st.error(f"{exc}, then build the zones.")
                else:
                    st.exception(exc)
            except Exception as exc:
                st.exception(exc)
        with stale_box.container():
            show_stale("zones")
        if done("zones"):
            Z, N = s.res["zones"], s.res.get("ndvi", {})
            if N.get("note"):
                st.info(f"NDVI: {N.get('used')} season(s) used — {N['note']}")
            if Z.get("note"):       # zone map: sieved zones, shifted codes, no-data codes, label order
                st.info(f"Zones: {Z['note']}")
            c = st.columns(len(Z.get("field_acres", {})) or 1)
            for i, (zz, a) in enumerate(sorted(Z.get("field_acres", {}).items())):
                c[i].metric(f"Zone {zz}", f"{a:.1f} ac", f"NDVI {((Z.get('ndvi_means') or {}).get(zz) or float('nan')):.3f}", delta_color="off")
            for key_ in ("ndvi_summary", "ndvi_seasons"):
                p = (getattr(s, "figs", {}) or {}).get(key_)
                if p and os.path.exists(p):
                    st.image(p)
            if N.get("layers"):
                st.dataframe(pd.DataFrame(N["layers"]), width="stretch", hide_index=True)

# ================================================================================================ 3 prices
with tabs[2]:
    pr = cfg["prices"]
    c = st.columns(4)
    unit = c[0].radio("Crop price in", ["$/bu", "$/t"], horizontal=True, key=k("punit"))
    if unit == "$/bu":
        pr["crop_price_bu"] = float(c[1].number_input("Crop sale price ($/bu)", 0.0, 100.0, float(pr.get("crop_price_bu", 9.25)), 0.05, key=k("pc")))
    else:
        t = c[1].number_input("Crop sale price ($/t)", 0.0, 3000.0, float(EP.per_bu_to_per_tonne(pr.get("crop_price_bu", 9.25), cfg["crop"])), 1.0, key=k("pct"))
        pr["crop_price_bu"] = EP.per_tonne_to_per_bu(t, cfg["crop"])
        c[1].caption(f"= {pr['crop_price_bu']:.2f} $/bu at {EP.bushel_lb(cfg['crop']):.0f} lb/bu")
    pr["currency"] = c[2].text_input("Currency", pr.get("currency", "CAD"), key=k("cur"))
    pr["vr_cost_ac"] = float(c[3].number_input("VR cost ($/ac)", 0.0, 100.0, float(pr.get("vr_cost_ac", 0.0)), 0.5, key=k("vrc"),
                                               help="Extra cost of running VR (prescription, imagery, agronomy); subtracted from VR gains only."))
    st.subheader("Input purchase prices")
    for inp in pr["inputs"]:
        c = st.columns([2, 2, 3])
        inp["price_per_lb_product"] = float(c[0].number_input(f"{inp['name']} ($/lb of {inp['product']})", 0.0, 50.0,
                                                              float(inp.get("price_per_lb_product") or 0.0), 0.005, format="%.3f",
                                                              key=k(f"pi_{inp['name']}")))
        ip = EP.InputPrice(**{kk: inp.get(kk) for kk in EP.InputPrice.__dataclass_fields__ if kk in inp})
        # the price is per lb (of product, or of nutrient) whatever the unit the rates were typed in: a kg/ha design
        # is converted to lb/ac on loading, so '$/kg' here was wrong by the unit factor
        c[1].metric("per rate unit", f"{ip.price_per_rate_unit:.3f} $/lb" + (" N" if inp.get("rate_basis") == "nutrient" else "")
                    + (" (rates in kg/ha are converted to lb/ac)" if inp.get("rate_unit") == "kg/ha" else ""))
        if inp["product"] != "seed":
            c[2].caption(f"{EP.PRODUCTS[inp['product']]['label']}: {100 * ip.fraction:.0f} % nutrient. Rate basis "
                         f"*{inp['rate_basis']}* → {'price per lb nutrient = price per lb product / fraction' if inp['rate_basis'] == 'nutrient' else 'price per lb of product'}.")
    # Minimum 5 %: a 0 % range makes every cell of the price grid the same price pair, and the sensitivity heat map
    # then cannot be built (pivot on duplicate entries), which failed the whole "Run the economics".
    rng0 = int(abs(float((pr.get("input_price_range_pct") or [-30, 30])[1])))
    rng = st.slider("Sensitivity grid: crop and input price ± %", 5, 60, min(60, max(5, rng0)), 5, key=k("rng"))
    pr["input_price_range_pct"] = [-float(rng), float(rng)]
    pr["source"] = st.text_input("Source of the prices", pr.get("source", ""), key=k("psrc"),
                                 placeholder="e.g. given by the grower, elevator bid 2026-10-01")
    pr["as_of"] = st.text_input("As of", pr.get("as_of", "") or dt.date.today().isoformat(), key=k("pasof"))
    st.caption("Reference: central-Alberta 2026 prices (CWRS wheat C$8.90/bu, certified seed C$0.225/lb) in "
               "`GIT/OFPE_DIFM/reference/prices_alberta_2026.md`. Net return = crop price × yield − input price × rate "
               "(partial budget).")
    if st.button("Save prices", key=k("btn_prices")):
        mark("prices")
        save()
        st.success("Saved." + (" The economics was run with other prices: run it again in step 5."
                               if done("analysis") and analysis_stamp(cfg) != (ss().get("an_stamp") or {}).get("app") else ""))

# ================================================================================================ 4 join
with tabs[3]:
    if not done("zones"):
        st.info("Build the zones in step 2 first (or use one zone: choose a zone map with a single zone).")
    else:
        s = state()
        j = cfg["join"]
        c = st.columns(6)
        j["end_trim_m"] = float(c[0].number_input("Trim plot ends (m)", 0.0, 50.0, float(j["end_trim_m"]), 1.0, key=k("jend")))
        j["side_trim_m"] = float(c[1].number_input("Trim plot sides (m)", 0.0, 20.0, float(j["side_trim_m"]), 0.5, key=k("jside")))
        j["min_share"] = float(c[2].number_input("Min share in one plot", 0.0, 1.0, float(j["min_share"]), 0.05, key=k("jshare"),
                                                 help="0.95 = DIFM rule; 0 = clip and weight by area"))
        j["min_obs"] = int(c[3].number_input("Min observations / unit", 1, 200, int(j["min_obs"]), key=k("jobs")))
        j["min_area_m2"] = float(c[4].number_input("Min unit area (m²)", 0.0, 5000.0, float(j["min_area_m2"]), 10.0, key=k("jarea")))
        j["rate_source"] = c[5].selectbox("Rate in the models", ["applied", "target"], index=["applied", "target"].index(j["rate_source"]), key=k("jrate"))
        j["unit"] = st.radio("Analysis unit", ["plot_zone", "plot"], format_func={"plot_zone": "plot × zone (default)", "plot": "whole plot"}.get,
                             index=["plot_zone", "plot"].index(j.get("unit", "plot_zone")), horizontal=True, key=k("junit"))
        # The units hold target_/rate_ columns per input name and the applied rates of the as-applied maps: when the
        # inputs, an as-applied map, the yield file/columns or these settings change after the join, the units (and
        # everything computed from them) are out of date, so the join must be run again.
        if done("join") and fp_changed("join_fp", join_payload(cfg)):
            invalidate(["join", "analysis", "report"],
                       "The inputs, an as-applied map, the yield columns or the join settings changed after the join: "
                       "run the join again (and then the economics).")
        ap_missing = [i["name"] for i in cfg["prices"]["inputs"] if (cfg["files"].get("asapplied") or {}).get(i["name"])
                      and not (cfg["columns"].get("applied") or {}).get(i["name"])]
        if ap_missing:
            st.warning(f"Choose the applied-rate column of the as-applied map of {', '.join(ap_missing)} in step 1 (or "
                       "clear that file).")
        # '(choose)' in the yield picker is None. The engine would then detect a yield column itself and write it
        # into cfg, so the join ran on a column the user did not choose, and the next rerun (the picker still on
        # '(choose)') changed the fingerprint back and threw the join and everything after it away again.
        yield_missing = cfg["columns"].get("yield") is None
        if yield_missing:
            st.warning("Choose the yield column of the yield map under 'Yield columns' in step 1: the join waits for it.")
        stale_box = st.empty()
        if st.button("Join yield, plots and zones", type="primary", key=k("btn_join"),
                     disabled=bool(ap_missing) or yield_missing):
            try:
                with st.spinner("Building yield polygons and units"):
                    s.join()
                    RP.figures(s, parts=("units",))
                mark("join")
                fp_store("join_fp", join_payload(cfg))
                (ss().get("stale") or {}).pop("join", None)
                invalidate(["analysis", "report"])
                save()
            except Exception as exc:
                st.exception(exc)
        with stale_box.container():
            show_stale("join")
        if done("join"):
            J = s.res["join"]
            a = J["attrition"]
            c = st.columns(5)
            c[0].metric("Observations", f"{a['n_obs']:,}")
            c[1].metric("In one plot", f"{a['kept']:,}", f"{100 * a['kept'] / max(1, a['n_obs']):.0f} %", delta_color="off")
            c[2].metric("Units", J["units"]["n_units"])
            c[3].metric("Plots with units", J["units"]["n_plots"])
            c[4].metric("Analysed area", f"{J['analysed_ac']:.1f} ac" if J.get("analysed_ac") is not None else "n/a",
                        help=None if J.get("analysed_ac") is not None else
                        f"no {J.get('area_missing') or 'swath width'} in the yield file: records weigh 1 each")
            st.caption(f"Not used: {a.get('outside_trimmed_plots', 0):,} outside the trimmed plots (border, headlands, plot ends), "
                       f"{a.get('straddling', 0):,} straddling two plots, {a.get('partly_in_transition_or_edge', 0):,} partly in a "
                       f"trimmed end or edge. Units dropped: {J['units']['dropped_few_obs']} with < {J['units']['min_obs']} "
                       f"observations, {J['units']['dropped_small']} too small. Yield mean by zone: "
                       + ", ".join(f"z{kk} {v:.1f}" for kk, v in (J.get("yield_means_by_zone") or {}).items())
                       + (f" (zones explain {100 * J['yield_r2_by_zone']:.0f} % of the unit yield variance)" if J.get("yield_r2_by_zone") else ""))
            U = J["units"]
            if U.get("dropped_no_zone_obs") or U.get("filled_nearest_zone_pieces"):
                st.caption(f"Zones: {U.get('dropped_no_zone_obs', 0):,} observations dropped with no zone under them; "
                           f"{U.get('filled_nearest_zone_pieces', 0):,} yield pieces given the nearest zone.")
            for nm_, w_ in (J.get("applied_warnings") or {}).items():
                st.warning(f"As-applied map of {nm_}: {w_}")
            for nm_, ck_ in (J.get("rate_checks") or {}).items():
                if ck_.get("warning"):
                    st.warning(ck_["warning"][0].upper() + ck_["warning"][1:] + ".")
            if J.get("swath_check", {}).get("warning"):
                st.warning(J["swath_check"]["warning"])
            p = (getattr(s, "figs", {}) or {}).get("units")
            if p and os.path.exists(p):
                st.image(p)
            st.dataframe(s.units.round(3), width="stretch", hide_index=True, height=260)

# ================================================================================================ 5 economics
with tabs[4]:
    if not done("join"):
        st.info("Run the join in step 4 first.")
    else:
        s = state()
        m = cfg["model"]
        c = st.columns(6)
        m["form"] = c[0].selectbox("Response form", ["quadratic", "qplateau", "linear"],
                                   index=["quadratic", "qplateau", "linear"].index(m.get("form", "quadratic")), key=k("mform"),
                                   help="The other form is always reported as a sensitivity.")
        m["machine"] = c[1].checkbox("Combine effect", bool(m.get("machine", True)), key=k("mmach"))
        m["spatial"] = c[2].checkbox("Trend surface", bool(m.get("spatial", True)), key=k("msp"))
        m["B"] = int(c[3].number_input("Bootstrap", 0, 2000, int(m.get("B", 500)), 50, key=k("mB"),
                                       help="Plot resamples for the intervals of the optima and gains; 500 by default, 1,000 for a report to publish "
                                            "(Hernandez & Mulla 2008; Jaynes 2011)."))
        m["R"] = int(c[4].number_input("Cross-fit repeats", 0, 500, int(m.get("R", 30)), 10, key=k("mR")))
        m["min_units_zone"] = int(c[5].number_input("Min units per zone", 5, 300, int(m.get("min_units_zone", 20)), 5, key=k("mminu")))
        m["min_gain_ac"] = float(st.number_input("Minimum worthwhile gain ($/ac, cross-fitted: VR vs flat after the VR cost, "
                                                 "and best flat rate vs status quo)", 0.0, 100.0,
                                                 float(m.get("min_gain_ac", 3.0)), 0.5, key=k("mgain"),
                                                 help="Below this the recommendation stays flat rate, or at the status quo: "
                                                      "payoffs are flat near the optimum (Pannell 2006) and small gains "
                                                      "appear by chance. The mean cross-fitted gain must reach it and be "
                                                      "positive in at least 70 % of the repeated splits."))
        missing = [i["name"] for i in cfg["prices"]["inputs"] if not i.get("price_per_lb_product")]
        if missing:
            st.warning(f"Enter the price of {', '.join(missing)} in step 3.")
        # Results computed with other prices, status quo, model or join settings than the ones now on the page are not
        # shown as current: the step is marked not done (the report too) until the economics is run again.
        if done("analysis") and analysis_stale(s):
            invalidate(["analysis", "report"],
                       "Prices, inputs (status quo), model or join settings changed after the economics was run: the "
                       "results shown before no longer match them. Run the economics again.")
        stale_box = st.empty()
        if st.button("Run the economics", type="primary", key=k("btn_an"), disabled=bool(missing)):
            bar = st.progress(0.0, "fitting")
            # a PDF built from the previous results must not be offered next to the new outputs
            invalidate(["analysis", "report"])
            try:
                s.analyse(progress=lambda f, mm: bar.progress(min(1.0, f), mm))
                bar.progress(1.0, "figures")
                RP.figures(s, parts=("analysis",))
                s.write_outputs()
                mark("analysis")
                # what this run was computed with, to tell later whether the page settings still match it
                ss()["an_stamp"] = {"app": analysis_stamp(cfg),
                                    "res": {nm: (R_.get("stamp") if isinstance(R_, dict) else None)
                                            for nm, R_ in (s.res.get("analysis") or {}).items() if not nm.startswith("_")}}
                (ss().get("stale") or {}).pop("analysis", None)
                save()
                bar.empty()
            except Exception as exc:
                st.exception(exc)
        with stale_box.container():
            show_stale("analysis")
        if done("analysis"):
            for name, R in s.res["analysis"].items():
                if name.startswith("_") or not isinstance(R, dict):
                    continue
                st.subheader(f"{name.capitalize()}")
                if R.get("skipped"):        # e.g. a treatment-only design: no rate response, the comparison still runs
                    why_ = R["skipped"].split(": ", 1)[1] if R["skipped"].lower().startswith(f"{name.lower()}: ") else R["skipped"]
                    st.info(f"No rate response or economics for {name}: {why_}")
                    continue
                v = R["verdict"]
                box = {"vr": st.success, "lean_vr": st.warning, "flat": st.info}[v["level"]]
                box(f"**{v['text']}**\n\n" + "\n".join(f"- {r}" for r in v["reasons"]))
                if v.get("borderline"):
                    # A cross-fitted gain sits within 1 $/ac of the minimum worthwhile gain, so that call could flip
                    # with a slightly different price or threshold. verdict['borderline_on'] says WHICH comparison:
                    # a generic "either is defensible" under a clear VR verdict read as if VR vs flat were a toss-up
                    # when only best flat vs status quo was close. Results saved before borderline_on existed are
                    # taken as the VR comparison (the only one flagged then).
                    on = v.get("borderline_on")
                    on = list(on) if on is not None else ["VR vs flat"]
                    notes_ = []
                    if "VR vs flat" in on:
                        notes_.append("VR vs the best flat rate: the cross-fitted gain of VR is within 1 $/ac of the "
                                      "minimum worthwhile gain, so VR and the flat rate are economically equivalent; "
                                      "either is defensible")
                    if "best flat vs status quo" in on:
                        notes_.append("best flat rate vs the status quo: the cross-fitted gain of moving to the best "
                                      "flat rate is within 1 $/ac of the minimum worthwhile gain, so keeping the status "
                                      "quo is about as good" + ("; the VR call above is not borderline"
                                                                if "VR vs flat" not in on and v["level"] != "flat" else ""))
                    if notes_:
                        st.caption("Borderline call on " + "; and on ".join(notes_) + ".")
                d, pl = R["decision"], R.get("pooled") or R["decision"]
                cf = (R.get("crossfit") or {}).get("gain_vr_vs_uniform")
                c = st.columns(5)
                if R.get("status_quo") is not None:
                    c[0].metric("Status quo", f"{R['status_quo']:g}")
                # the gain over the status quo on the summary basis (main model, field zone acres), the number the
                # verdict and the PDF summary print; the single-curve model's own gain (plot zone mix) could differ
                # in size and even in sign from the verdict next to it
                bf = (R.get("summary_basis") or {}).get("best_flat") or {}
                bf_rate = bf.get("rate", pl["r_uniform"])
                bf_gain = bf.get("gain_vs_sq")
                c[1].metric("Best flat rate", f"{bf_rate:.0f}",
                            f"{bf_gain:+.2f} $/ac vs SQ" if R.get("status_quo") is not None and bf_gain is not None else None,
                            help="Valued on the main model over the field's zone acres, as in the verdict and the PDF.")
                c[2].metric("Analysis zones", len(d["r_zone"]))
                if R.get("single_zone"):
                    c[3].metric("VR vs flat (in-sample)", "n/a", help="One analysis zone: there is no VR strategy.")
                else:
                    c[3].metric("VR vs flat (in-sample)", f"{d['gain_vr_vs_uniform']:+.2f} $/ac")
                if cf and not R.get("single_zone"):
                    c[4].metric("VR vs flat (cross-fitted)", f"{cf['mean']:+.2f} $/ac", f"{100 * cf['share_positive']:.0f} % splits > 0", delta_color="off")
                def edge(z):
                    return " (edge of the trial range)" if (d.get("corner") or {}).get(z) else ""
                if R.get("single_zone") or len(d["r_zone"]) <= 1:
                    # one analysis zone: the pooled fit is keyed None, which printed 'zone None: 128'
                    z1, r1 = next(iter(d["r_zone"].items()), (None, d.get("r_uniform")))
                    if r1 is not None:
                        st.caption(f"Single analysis zone: optimum {r1:.0f}" + edge(z1))
                else:
                    st.caption("Zone optima: " + " · ".join(f"zone {z}: {r:.0f}" + edge(z) for z, r in d["r_zone"].items()))
                sq_txt = f"{R['status_quo']:g} " if R.get("status_quo") is not None else ""
                if d.get("sq_too_far"):
                    st.caption(f"Status quo: {sq_txt}lies more than 25 % of the trial range outside the "
                               "tested rates, so no gain over it is computed (the curve cannot be trusted that far).")
                elif d.get("sq_extrapolated"):
                    st.caption(f"Status quo: {sq_txt}lies outside the tested rates; its yield is "
                               + sq_extrap_text(R, d) + ", so gains over it are less certain.")
                for mn in R.get("model_notes") or []:
                    st.caption("Model: " + mn)
                for zn in R.get("zone_notes") or []:
                    st.caption("Zone support: " + zn)
                F = getattr(s, "figs", {}) or {}
                cA, cB = st.columns(2)
                for col_, key_ in ((cA, f"response_{name}"), (cB, f"economics_{name}")):
                    if F.get(key_) and os.path.exists(F[key_]):
                        col_.image(F[key_])
                with st.expander("Per zone, model tests, functional form, prices, NDVI prescription, realised trial"):
                    st.dataframe(pd.DataFrame(d["per_zone"]).round(2), width="stretch", hide_index=True)
                    st.dataframe(pd.DataFrame(R["table"]).round(2), width="stretch", hide_index=True)
                    st.json({k_: v_["tests"] for k_, v_ in R["models"].items()}, expanded=False)
                    if R.get("forms"):
                        st.dataframe(pd.DataFrame({f: {kk: vv for kk, vv in v_.items() if kk in ("basis", "r_flat", "gain_flat_vs_sq", "r_zone", "gain_vr_vs_flat")}
                                                   for f, v_ in R["forms"].items()}).T, width="stretch")
                    for key_ in (f"prices_{name}", f"ndvi_rx_{name}", f"rx_{name}"):
                        if F.get(key_) and os.path.exists(F[key_]):
                            st.image(F[key_])
                    st.markdown("**Realised trial economics (same units)**")
                    st.json(R["realised"]["summary"], expanded=False)
            T = s.res["analysis"].get("_treatments")
            if T:
                st.subheader("Treatment comparison")
                st.dataframe(pd.DataFrame(T["table"]).round(2), width="stretch", hide_index=True)
                for var in ("yield", "net_return"):
                    if T.get(var) and "p_treatment" in T[var]:
                        st.caption(f"{var}: treatment p = {T[var]['p_treatment']:.3f}")

# ================================================================================================ 6 report
with tabs[5]:
    if not done("analysis"):
        st.info("Run the economics in step 5 first.")
    else:
        s = state()
        if done("report") and ss().get("report_fp") and ss()["report_fp"] != report_fingerprint(cfg):
            invalidate(["report"], "The project name, producer, field, crop, year, currency or price source / date "
                                   "changed after the PDF was built: build the PDF again.")
        stale_box = st.empty()
        if st.button("Build the PDF report", type="primary", key=k("btn_pdf")):
            with st.spinner("Writing the report"):
                try:
                    RP.figures(s)
                    s.report_path = RP.make_pdf(s)
                    mark("report")
                    ss()["report_fp"] = report_fingerprint(cfg)
                    (ss().get("stale") or {}).pop("report", None)
                    save()
                except Exception as exc:
                    st.exception(exc)
        with stale_box.container():
            show_stale("report")
        out = s.out
        c = st.columns(5)
        # only a report built from the current results (the economics re-run clears report_path and step 6)
        rp = getattr(s, "report_path", None) if done("report") else None
        if rp and os.path.exists(rp):
            with open(rp, "rb") as f:
                c[0].download_button("Report (PDF)", f.read(), os.path.basename(rp), "application/pdf")
        # the shapefile parts (not the .gpkg) plus zones_rx_fields.csv: dBASE cuts field names to 10 characters and
        # the engine makes them unique with index suffixes (Flat_input, Flat_inpu1), so without that key in the zip
        # nobody could tell which Flat_ field belongs to which input and a wrong flat rate could reach the controller
        shp = [os.path.join(out, f) for f in os.listdir(out)
               if (f.startswith("zones_rx.") and not f.endswith(".gpkg")) or f == "zones_rx_fields.csv"]
        if any(p.endswith(".shp") for p in shp):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z_:
                for p in shp:
                    z_.write(p, os.path.basename(p))
            c[1].download_button("Prescription (zones_rx.zip)", buf.getvalue(), f"{cfg.get('name') or 'field'}_zones_rx.zip", "application/zip",
                                 help="Zone polygons (WGS 84) with, per input, Rx_ = the zone optimum, Flat_ = the "
                                      "recommended flat rate and Zan_ = the analysis zone. Shapefile field names hold "
                                      "at most 10 characters, so they are cut and, when two cut names clash, "
                                      "numbered (Flat_input, Flat_inpu1): zones_rx_fields.csv in the zip says which "
                                      "field belongs to which input and what it holds.")
        for i, (fn, lab, mime) in enumerate((("units.gpkg", "Units (GeoPackage)", "application/geopackage+sqlite3"),
                                              ("units.csv", "Units (CSV)", "text/csv"), ("results.json", "Results (JSON)", "application/json"))):
            p = os.path.join(out, fn)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    c[2 + i].download_button(lab, f.read(), fn, mime)
        st.caption(f"Everything is in `{out}`.")


# ================================================================================================ sidebar status (last)
draw_steps()
