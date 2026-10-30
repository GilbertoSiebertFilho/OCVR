# -*- coding: utf-8 -*-
"""The project: one trial field, its files, its settings and its results, kept in <workdir>/project.json.

    from engine import project as P
    cfg = P.default_config()            # every setting with its house default
    st  = P.State(cfg, workdir)
    st.load_design(); st.load_ndvi(); st.make_zones(); st.join(); st.analyse(); st.write_outputs()
    # or: P.run(cfg, workdir)

Each step stores what the next one needs on the State object, so the app can run them one at a time and show the
result of each (and rerun only the step whose settings changed).
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from . import io as eio, ndvi as en, zones as ez, join as ej, response as er, economics as ee, treatments as et
from . import prices as pr
from .prices import Prices, InputPrice

SCHEMA = 1


# The rate column of an input set to this value means: read the input's design rates from its as-applied map
# (join.rates_from_asapplied) instead of a column of a design file. The app stores it in cfg['columns']['design_rate']
# (first input) or cfg['columns']['extra_design_rate'][<input>] (the others).
FROM_APPLIED = "(as-applied)"


def default_config():
    return {
        "schema": SCHEMA, "name": "", "producer": "", "field": "", "crop": "wheat", "year": dt.date.today().year,
        "files": {"design": None, "design_layer": None, "yield": None, "boundary": None, "zones": None,
                  "ndvi_dir": None, "ndvi_paths": None, "ndvi_map": None, "asapplied": {}, "extra_designs": {}},
        "columns": {"design_rate": None, "plot_id": None, "type": None, "treatment": None, "block": None,
                    "yield": None, "heading": None, "distance": None, "swath": None, "machine": None,
                    "zone": None, "ndvi": None, "applied": {}, "extra_design_rate": {}},
        "prices": Prices().to_dict(),
        "ndvi": {"source": "fetch", "years": list(range(2017, dt.date.today().year)), "crop_history": {},
                 "policy": "same_crop", "crop_group": ["wheat", "barley", "durum", "oats"], "min_years_same_crop": 3,
                 "window": ["06-15", "08-15"], "min_valid": 0.90, "min_valid_pct_fetch": 95.0, "norm": "relative",
                 "res": 10.0, "exclude_trial_year": True, "dates_per_year": 3},
        "zones": {"source": "ndvi", "k": 3, "method": "quantile", "min_patch_ha": 0.5, "smooth": 3},
        "join": {"end_trim_m": 10.0, "side_trim_m": 0.0, "min_share": 0.95, "min_obs": 10, "min_area_m2": 150.0,
                 "rate_source": "applied", "heading_is": "true", "unit": "plot_zone"},
        "model": {"form": "quadratic", "machine": True, "spatial": True, "B": 500, "R": 30, "n_price": 7,
                  "min_units_zone": 20, "min_gain_ac": 3.0},
    }


def merge(base, upd):
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def _walk_paths(obj, fn):
    """Apply fn to every path string in cfg['files'] (strings, dict values, list items), in place."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = fn(v)
            elif isinstance(v, (dict, list)):
                _walk_paths(v, fn)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = fn(v)
            elif isinstance(v, (dict, list)):
                _walk_paths(v, fn)


def relocate(cfg, workdir, old_workdir=None):
    """Make a project usable after its folder moved (another computer, another drive, a restored zip, a copy). Every
    path under cfg['files'] that lies inside the project's old folder is pointed at the same place in `workdir` when
    the file is there, whether or not the old one still exists: a copy of a project on the same computer must read its
    own inputs, not silently the original's. The old folder is the one recorded by save() ('workdir'), or, for
    projects saved before that, the '<...>/workspace/<this folder's name>/' segment of the path. Paths outside the
    project (e.g. a design file left in the grower's folder) are left alone: the app then asks for them again when
    they are missing. Returns the number of paths moved."""
    new = os.path.abspath(workdir)
    name = os.path.basename(new.rstrip("\\/"))
    olds = [os.path.abspath(old_workdir)] if old_workdir else []
    new_ = new.replace("\\", "/").rstrip("/").lower() + "/"
    moved = [0]

    def fix(p):
        if not p:
            return p
        q = p.replace("\\", "/")
        if q.lower().startswith(new_):
            return p                                            # already inside this project folder
        for o in olds:
            o_ = o.replace("\\", "/").rstrip("/") + "/"
            if q.lower().startswith(o_.lower()):
                cand = os.path.join(new, q[len(o_):])
                if os.path.exists(cand):
                    moved[0] += 1
                    return os.path.normpath(cand)
        seg = f"/workspace/{name}/"
        i = q.lower().find(seg.lower())
        if i >= 0:
            cand = os.path.join(new, q[i + len(seg):])
            if os.path.exists(cand):
                moved[0] += 1
                return os.path.normpath(cand)
        return p
    _walk_paths(cfg.get("files") or {}, fix)
    return moved[0]


def save(cfg, workdir, summary=None):
    os.makedirs(workdir, exist_ok=True)
    rec = dict(config=cfg, saved=dt.datetime.now().isoformat(timespec="seconds"), summary=summary or {},
               workdir=os.path.abspath(workdir))
    tmp = os.path.join(workdir, "project.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, default=_json_default)
    os.replace(tmp, os.path.join(workdir, "project.json"))


def load(workdir):
    with open(os.path.join(workdir, "project.json"), encoding="utf-8") as f:
        rec = json.load(f)
    rec["config"] = merge(default_config(), rec.get("config", {}))
    # a project copied to another folder or computer keeps working: its own files are found in their new place
    rec["relocated"] = relocate(rec["config"], workdir, rec.get("workdir"))
    return rec


def list_projects(workspace):
    out = []
    if not os.path.isdir(workspace):
        return out
    for d in sorted(os.listdir(workspace), reverse=True):
        p = os.path.join(workspace, d, "project.json")
        if os.path.exists(p):
            try:
                r = load(os.path.join(workspace, d))
                out.append(dict(dir=os.path.join(workspace, d), name=r["config"].get("name") or d,
                                saved=r.get("saved"), summary=r.get("summary", {})))
            except Exception:
                continue
    return out


# price fields that only label the report (who quoted the prices, when, in which currency): editing them after the
# run must not throw away minutes of bootstrap and cross-fit, so they are left out of the analysis stamp
_STAMP_IGNORED_PRICE_KEYS = ("source", "as_of", "currency")


def _stamp_norm(o):
    """A json-ready copy of `o` in which every number is a float rounded to 6 significant digits. Why: the app's
    $/bu <-> $/t display round trip is not exact (wheat 14.20 comes back as 14.199999999999998) and a status quo may
    come back as 140 or 140.0; neither is a change of the economics, so neither may change the stamp."""
    if isinstance(o, dict):
        return {str(k): _stamp_norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_stamp_norm(v) for v in o]
    if isinstance(o, (bool, np.bool_)) or o is None:
        return bool(o) if o is not None else None
    if isinstance(o, (int, float, np.integer, np.floating)):
        v = float(o)
        return float(f"{v:.6g}") if np.isfinite(v) else None
    return o


def analysis_stamp(cfg):
    """sha1 of everything step 5 depends on besides the data: the prices (without the report-only keys source, as_of
    and currency), the input definitions, cfg['model'] and cfg['join'], numbers rounded to 6 significant digits
    (_stamp_norm), json.dumps with sort_keys. Stored as res['analysis'][input]['stamp']; the app recomputes it from
    the current settings and, when they differ, knows the results on screen are stale."""
    pr = {k: v for k, v in ((cfg or {}).get("prices") or {}).items() if k not in _STAMP_IGNORED_PRICE_KEYS}
    key = _stamp_norm(dict(prices=pr, inputs=pr.get("inputs"), model=(cfg or {}).get("model"),
                           join=(cfg or {}).get("join")))
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return str(o)


class State:
    def __init__(self, cfg, workdir, log=print):
        self.cfg = merge(default_config(), cfg)
        self.workdir = workdir
        self.out = os.path.join(workdir, "output")
        os.makedirs(self.out, exist_ok=True)
        self.log = log
        self.res = {}

    @property
    def prices(self):
        return Prices.from_dict(self.cfg["prices"])

    @property
    def inputs(self):
        return [i.name for i in self.prices.inputs]

    # ------------------------------------------------------------------------------------------ 1 design
    def load_design(self):
        c, f = self.cfg["columns"], self.cfg["files"]
        d = eio.read_vector(f["design"], f.get("design_layer"))
        for role in ("design_rate", "plot_id", "type", "treatment", "block"):
            if c.get(role) is None:
                c[role] = eio.find_col(d, role)
        # A second tested input (e.g. urea next to seed) often has its design rate in another column of the SAME
        # design file, not in a design file of its own. For every input after the first with no extra design file,
        # the configured column of the main design becomes design_<input> on the plots (join.prepare_design).
        extra_cols = {}
        for name in self.inputs[1:]:
            if (f.get("extra_designs") or {}).get(name):
                continue
            col = (c.get("extra_design_rate") or {}).get(name)
            if col and col != FROM_APPLIED:
                # a configured column that is not in the file (a re-exported design, a hand-edited project.json) was
                # dropped here in silence: the input was then skipped for 'no design rate' while the source text
                # still named the column
                if col not in d.columns:
                    raise ValueError(f"Design rate column '{col}' of {name} is not in the design file (columns: "
                                     f"{', '.join(map(str, d.columns))}): choose it in step 1")
                extra_cols[name] = col
        first_from_applied = c.get("design_rate") == FROM_APPLIED
        if c.get("design_rate") and not first_from_applied and c["design_rate"] not in d.columns:
            raise ValueError(f"Design rate column '{c['design_rate']}' is not in the design file (columns: "
                             f"{', '.join(map(str, d.columns))}): choose it in step 1")
        self.plots, self.border, self.epsg = ej.prepare_design(d, None if first_from_applied else c["design_rate"],
                                                               c["plot_id"], c["type"], c["treatment"], c["block"],
                                                               extra_rate_cols=extra_cols or None)
        # Only the inputs of the CURRENT trial: when the user reduces the number of inputs, the extra design file of
        # a removed input may still sit in cfg (with no rate column chosen, or a path that no longer exists), and
        # reading it made 'Load the trial' fail for an input that is not in the trial any more.
        if not isinstance(c.get("extra_design_rate"), dict):
            c["extra_design_rate"] = {}
        for name in self.inputs[1:]:
            path = (f.get("extra_designs") or {}).get(name)
            if not path:
                continue
            od = eio.read_vector(path)
            rc = c["extra_design_rate"].get(name) or eio.find_col(od, "design_rate")
            if rc == FROM_APPLIED:
                rc = eio.find_col(od, "design_rate")
            if rc is None or rc not in od.columns:
                raise ValueError(f"No design rate column recognised in the design file of {name}: choose it in step 1")
            c["extra_design_rate"][name] = rc
            self.plots = ej.attach_rate(self.plots, od, rc, name)
        # Inputs whose design rates are read from their as-applied map (FROM_APPLIED): the median rate of each
        # trimmed plot, grouped into the levels the drill delivered (join.rates_from_asapplied). The trims are the
        # join's, so the controller ramps at the plot ends stay out of the plot medians.
        # Every rate inside the engine is lb/ac in the input's own basis (product, or nutrient when the input is
        # priced per lb N). A design given in kg/ha, or an as-applied map that logs lb of product for an input priced
        # per lb N, is converted once, here and in join(), and the conversion is recorded (rate_conversion) for the
        # app and the PDF. Before this the unit and the basis were labels only, and a metric design was costed at the
        # $/lb price (12 % too dear) while a product map under a nutrient price cost 2.2 x too much per unit.
        self.rate_conversion = {name: self._rate_conversion(name) for name in self.inputs}
        for k_, name in enumerate(self.inputs):
            col = "design_rate" if k_ == 0 else f"design_{name}"
            fac = self.rate_conversion[name]["design_factor"]
            if col in self.plots and fac != 1.0:
                self.plots[col] = pd.to_numeric(self.plots[col], errors="coerce") * fac
        self.inner = ej.trim_plots(self.plots, self.cfg["join"]["end_trim_m"], self.cfg["join"]["side_trim_m"])
        from_applied = {}
        for k_, name in enumerate(self.inputs):
            src = c.get("design_rate") if k_ == 0 else (c.get("extra_design_rate") or {}).get(name)
            if src != FROM_APPLIED:
                continue
            if k_ and (f.get("extra_designs") or {}).get(name):
                continue        # a design file of its own wins (its column was attached above)
            path = (f.get("asapplied") or {}).get(name)
            if not path:
                raise ValueError(f"The rates of {name} are to be read from its as-applied map, but no as-applied map "
                                 f"is given for {name}: give it in step 1 (or choose a design column)")
            a = eio.read_vector(path).to_crs(self.epsg)
            rc = (c.get("applied") or {}).get(name) or eio.find_col(a, "applied_rate")
            c.setdefault("applied", {})[name] = rc
            a = self._convert_applied(a, rc, name)
            level, info = ej.rates_from_asapplied(self.plots, self.inner, a, rc, name=name,
                                                  min_pts=int(self.cfg["join"].get("min_pts_design", 5)),
                                                  max_levels=int(self.cfg["join"].get("max_levels_design", 15)))
            self.plots["design_rate" if k_ == 0 else f"design_{name}"] = level.values
            from_applied[name] = info
            if info.get("warning"):
                self.log("WARNING: " + info["warning"])
            self.log(f"design rates of {name} from its as-applied map: levels {', '.join(f'{x:g}' for x in info['levels'])} "
                     f"({info['n_with_level']} of {info['n_plots']} plots at a level; {info['n_few_points']} with too "
                     f"few points, {info['n_mixed']} with mixed rates, {info['n_off_level']} off every level)")
        if f.get("boundary"):
            b = eio.read_vector(f["boundary"]).to_crs(self.epsg)
            self.boundary = gpd.GeoDataFrame(geometry=[b.geometry.union_all()], crs=self.epsg)
            self.boundary_source = "boundary file"
        else:
            u = pd.concat([self.plots.geometry, self.border.geometry]).union_all().buffer(2).buffer(-2)
            self.boundary = gpd.GeoDataFrame(geometry=[u], crs=self.epsg)
            self.boundary_source = "trial design (plots + border); give the field boundary for whole-field economics"
        # the rates of every input on the plots (design_rate = first input, design_<name> = the others), with where
        # each came from, so the app, the PDF and results.json describe a second input's design as well as the first's
        rates_by_input = {}
        for k_, name in enumerate(self.inputs):
            col = "design_rate" if k_ == 0 else f"design_{name}"
            if name in from_applied:
                src = "as-applied map (levels read from the plot medians)"
            elif k_ == 0:
                src = (f"design column '{c.get('design_rate')}'" if c.get("design_rate") else "none: treatment-only design")
            elif (f.get("extra_designs") or {}).get(name):
                src = f"design file of {name}, column '{c['extra_design_rate'].get(name)}'"
            elif (c.get("extra_design_rate") or {}).get(name):
                src = f"design column '{c['extra_design_rate'][name]}'"
            else:
                src = "none: no design column or file"
            v = pd.to_numeric(self.plots[col], errors="coerce") if col in self.plots else pd.Series(dtype=float)
            rates_by_input[name] = dict(source=src, rates=sorted(v.dropna().unique().tolist()),
                                        plots_per_rate={float(a): int(b) for a, b in v.value_counts().sort_index().items()},
                                        n_without=int(v.isna().sum()) if col in self.plots else int(len(self.plots)),
                                        from_applied=from_applied.get(name))
        for name, cv in self.rate_conversion.items():
            if cv.get("note"):
                self.log(f"rates of {name}: {cv['note']}")
        self.res["design"] = dict(n_plots=len(self.plots), n_border=len(self.border), epsg=self.epsg,
                                  rate_conversion=self.rate_conversion,
                                  rates=sorted(self.plots.design_rate.dropna().unique().tolist()),
                                  plots_per_rate=self.plots.design_rate.value_counts().sort_index().to_dict(),
                                  rates_by_input=rates_by_input,
                                  plot_area_ac=float(self.plots.area_m2.sum() / eio.M2_PER_AC),
                                  boundary_ac=float(self.boundary.area.sum() / eio.M2_PER_AC),
                                  boundary_source=self.boundary_source,
                                  treatments=self.plots.treatment.dropna().unique().tolist() if self.plots.treatment.notna().any() else [],
                                  columns={k: c.get(k) for k in ("design_rate", "plot_id", "type", "treatment", "block")})
        self.grid = en.make_grid(self.boundary, self.cfg["ndvi"]["res"])
        self.log(f"design: {len(self.plots)} plots, {len(self.border)} border polygon(s), rates {self.res['design']['rates']}")
        return self.res["design"]

    # ------------------------------------------------------------------------------------------ rate units
    def _rate_conversion(self, name):
        """How the rates of `name` are brought to the engine's unit (lb/ac in the input's basis): the factor for
        the design rates and the typed status quo (the input's rate unit), the factor for the as-applied map (the
        unit, and the map's basis when it differs from the input's: cfg columns.applied_basis[name], 'product' or
        'nutrient', default the input's own basis), and a note for the app and the PDF."""
        ip = self.prices.input(name)
        unit = ip.rate_unit or "lb/ac"
        fu = pr.rate_unit_factor(unit)
        basis = ip.rate_basis or "product"
        # the map's basis is asked only for an input priced per lb of nutrient (the one case where a controller's
        # lb-of-product log differs from the rates); a stale 'applied_basis' left from an earlier choice must not
        # convert a product-basis input
        ab = basis
        if basis == "nutrient":
            ab = (self.cfg["columns"].get("applied_basis") or {}).get(name) or basis
        fb = 1.0
        if ab == "product" and basis == "nutrient" and 0 < ip.fraction < 1.0:
            fb = ip.fraction
        parts = []
        if fu != 1.0:
            parts.append(f"given in {unit}, converted to lb/ac (x {fu:.6f})")
        if fb != 1.0:
            parts.append(f"the as-applied map logs lb of {ab}, the rates are lb of {basis}: its rates x {fb:.4f}")
        return dict(unit=unit, basis=basis, applied_basis=ab, design_factor=fu, applied_factor=fu * fb,
                    note="; ".join(parts) or None)

    def _convert_applied(self, a, rc, name):
        """The as-applied frame of `name` with its rate column in the engine's unit (see _rate_conversion)."""
        cv = getattr(self, "rate_conversion", None) or {}
        fac = (cv.get(name) or self._rate_conversion(name))["applied_factor"]
        if fac != 1.0 and rc in a.columns:
            a = a.copy()
            a[rc] = pd.to_numeric(a[rc], errors="coerce") * fac
        return a

    def status_quo(self, name):
        """The grower's status quo of `name` in the engine's unit (typed in the input's unit and basis)."""
        ip = self.prices.input(name)
        if ip.status_quo is None:
            return None
        cv = getattr(self, "rate_conversion", None) or {}
        return float(ip.status_quo) * (cv.get(name) or self._rate_conversion(name))["design_factor"]

    # ------------------------------------------------------------------------------------------ 2 NDVI
    def _fetch_years(self):
        n = self.cfg["ndvi"]
        years = [int(y) for y in n["years"]]
        if n.get("exclude_trial_year", True):
            years = [y for y in years if y != int(self.cfg["year"])]
        return years

    def ndvi_fetch_settings(self):
        """The settings a fetch depends on: the seasons, the window, the dates per season and the boundary (sha1).
        fetch_ndvi stores them in cfg['files']['ndvi_fetch_settings'] next to the image list it wrote."""
        n = self.cfg["ndvi"]
        return dict(years=self._fetch_years(), window=[str(w) for w in n["window"]],
                    dates_per_year=int(n.get("dates_per_year", 3)), boundary=_boundary_sha1(self.boundary))

    def ndvi_fetch_changes(self):
        """The fetch settings (window, dates_per_year, boundary) that differ between the last fetch and the current
        settings; [] when they agree or when no fetch settings were stored (a project fetched before they were:
        load_ndvi refuses such a project with OLD_FETCH_MSG, and run() fetches it again).
        The seasons are not compared: load_ndvi keeps only the fetched seasons that are still selected, so
        deselecting a season needs no new download."""
        old = self.cfg["files"].get("ndvi_fetch_settings")
        if not old:
            return []
        cur = self.ndvi_fetch_settings()
        norm = dict(window=lambda v: [str(w) for w in (v or [])], dates_per_year=lambda v: None if v is None else int(v),
                    boundary=lambda v: v)
        return [k for k, fn in norm.items() if fn(old.get(k)) != fn(cur[k])]

    def fetch_ndvi(self, progress=None):
        from . import ndvi_fetch as nf
        n = self.cfg["ndvi"]
        years = self._fetch_years()
        d = os.path.join(self.workdir, "ndvi_images")
        paths, log = nf.fetch_years(self.boundary, years, d, field_name=self.cfg.get("field") or self.cfg.get("name") or "field",
                                    window=tuple(n["window"]), min_valid_pct=n["min_valid_pct_fetch"],
                                    n_per_year=int(n.get("dates_per_year", 3)), progress=progress)
        self.cfg["files"]["ndvi_dir"] = d
        # the folder is shared by every fetch of this project (another boundary, other years): remember exactly the
        # images fetched or reused for THIS boundary, so load_ndvi does not stack stale or foreign rasters
        self.cfg["files"]["ndvi_paths"] = [str(p) for p in paths]
        # what this image list was fetched with: load_ndvi refuses a list fetched with another window, number of
        # dates per season or boundary (the zones would silently be built from the old images)
        # A season whose statistics or download failed on the network (a timeout, a server error) was stored as
        # fetched: run() and the app's 'not fetched' check compare the selection with this list, so that season was
        # never fetched again and silently left the zones (with the same_crop policy it could even switch the zones
        # to all seasons). Only the seasons that really completed are recorded ('ok', or 'no clear date', which a new
        # fetch would not change); the failed ones go to 'failed', so run() fetches them again (images already on
        # disk are reused) and the app and the PDF name them.
        # A season that lost only SOME of its dates ('partial') stays in 'failed', so it is fetched again, but its
        # downloaded dates are in `paths` and load_ndvi uses them: the log, the app and the PDF must not call it
        # missing from the zones.
        st = fetch_status(log, self._fetch_years())
        fs = self.ndvi_fetch_settings()
        fs["years"], fs["failed"], fs["partial"] = st["fetched"], st["failed"], st["partial"]
        self.cfg["files"]["ndvi_fetch_settings"] = fs
        whole = [y for y in st["failed"] if y not in st["partial"]]
        if whole:
            self.log(f"NDVI: season(s) {', '.join(map(str, whole))} failed on the network: fetch again")
        if st["partial"]:
            self.log(f"NDVI: season(s) {', '.join(map(str, st['partial']))} lost some of their dates on the network: "
                     "they are used with fewer dates and fetched again next time")
        self.res["ndvi_fetch"] = log
        return log

    def load_ndvi(self):
        n, f = self.cfg["ndvi"], self.cfg["files"]
        crop_hist = {int(k): v for k, v in (n.get("crop_history") or {}).items() if v}
        not_fetched = []        # seasons selected after the last fetch (source 'fetch'): they need a new fetch
        fetch_failed = []       # seasons whose last fetch failed on the network: they need a new fetch too
        fetch_partial = []      # failed seasons with some dates on disk: used with fewer dates, fetched again later
        # A source whose input is missing used to fall through to 'no NDVI' (one zone, no covariate) in silence, and
        # the verdict then described the fallback as a finding about the field ('the field has one zone'). Only the
        # source 'none' means no NDVI; the app keeps its buttons disabled in these states, the CLI needs the check.
        if n["source"] == "map" and not f.get("ndvi_map"):
            raise ValueError("The NDVI is to come from a ready multi-year map, but none is given: give it in step 2")
        if n["source"] == "files" and not f.get("ndvi_dir"):
            raise ValueError("The NDVI is to come from a folder of GeoTIFFs, but none is given: give it in step 2")
        if n["source"] == "fetch" and not (f.get("ndvi_dir") or f.get("ndvi_paths")):
            if (self.cfg.get("zones") or {}).get("source") == "map":
                # a grower zone map needs no NDVI: 'fetch' is the default source of every new project, and the
                # zones must build without a fetch; the covariate is then left out and said so
                self.log("NDVI: source 'fetch' but nothing fetched; the zones come from the zone map, no NDVI covariate")
                self.stack = None
                self.res["ndvi"] = dict(note="no NDVI (nothing fetched; the zones come from the zone map)", used=0,
                                        source="none")
                return self.res["ndvi"]
            raise ValueError("The NDVI is to be fetched, but nothing has been fetched yet: fetch it in step 2 "
                             "(or run without --no-fetch)")
        if n["source"] == "map" and f.get("ndvi_map"):
            p = f["ndvi_map"]
            if p.lower().endswith((".tif", ".tiff")):
                self.stack = en.single_map_stack(p, self.grid)
            else:
                g = eio.read_vector(p).to_crs(self.epsg)
                col = self.cfg["columns"].get("ndvi") or eio.find_col(g, "ndvi") or eio.numeric_cols(g)[0]
                self.stack = en.vector_map_stack(g, col, self.grid)
            note = "multi-year NDVI map given as is"
        elif n["source"] in ("fetch", "files") and (f.get("ndvi_dir") or (n["source"] == "fetch" and f.get("ndvi_paths"))):
            # after a fetch, only the images fetched for this boundary (fetch_ndvi records them); a folder given by
            # the user ("files") is read whole. The fetched list is ignored once the source is no longer "fetch".
            listed = f.get("ndvi_paths") if n["source"] == "fetch" else None
            if n["source"] == "fetch":
                # A project fetched with the old code has no image list and no fetch settings: its ndvi_dir is the
                # shared cache workdir/ndvi_images, which can hold images of an earlier boundary, window or number of
                # dates per season with no hash in their names. Stacking that whole folder built the zones from a
                # silent mix of old and new images (and nothing could say which settings they came from), so such a
                # project must be fetched again; the app shows this message with its Fetch button.
                if not f.get("ndvi_fetch_settings") or listed is None:
                    self.log("NDVI: " + OLD_FETCH_MSG)
                    raise ValueError(OLD_FETCH_MSG)
                changed = self.ndvi_fetch_changes()
                if changed:
                    self.log(f"NDVI: {', '.join(changed)} changed since the last fetch")
                    raise ValueError("The NDVI settings changed since the last fetch: fetch again")
            if listed is not None:
                paths = [p for p in listed if os.path.exists(p)]
            else:
                paths = eio.list_rasters(f["ndvi_dir"])
            layers = en.layers_from_paths(paths, crop_hist)
            if n["source"] == "fetch":
                # the seasons the user still selects: a season deselected after the fetch must leave the zones (and
                # the report's season table), not only the fetch that comes next
                want = {int(y) for y in (n.get("years") or [])}
                layers = [L for L in layers if L.year is not None and int(L.year) in want]
                fs_ = f.get("ndvi_fetch_settings") or {}
                fetched = fs_.get("years")
                if fetched is not None:
                    sel = set(self._fetch_years())
                    fetch_failed = sorted(sel & {int(y) for y in (fs_.get("failed") or [])})
                    not_fetched = sorted(sel - {int(y) for y in fetched} - set(fetch_failed))
                    # A failed season whose other dates were downloaded still has images here, and build_stack makes
                    # them a season (the median of fewer dates): it is partly fetched, not missing. The images
                    # actually loaded decide, so projects fetched before 'partial' was recorded are described right.
                    have = {int(L.year) for L in layers}
                    fetch_partial = [y for y in fetch_failed if y in have]
                    fetch_failed = [y for y in fetch_failed if y not in have]
                    if not_fetched:
                        self.log(f"NDVI: season(s) {', '.join(map(str, not_fetched))} selected but not fetched: "
                                 "fetch again to add them")
                    if fetch_failed:
                        self.log(f"NDVI: season(s) {', '.join(map(str, fetch_failed))} failed on the network at the "
                                 "last fetch and are missing from the zones: fetch again to add them")
                    if fetch_partial:
                        self.log(f"NDVI: season(s) {', '.join(map(str, fetch_partial))} lost some dates on the network "
                                 "at the last fetch: they are in the zones with fewer dates, fetched again next time")
            if n.get("exclude_trial_year", True):
                layers = [L for L in layers if L.year != int(self.cfg["year"])]
            self.stack = en.build_stack(layers, self.grid)
            used, note = en.select_layers(self.stack, self.cfg["crop"], n["policy"], n["min_valid"],
                                          n["min_years_same_crop"], n.get("crop_group"))
            en.summarise(self.stack, used, n["norm"])
        else:
            self.stack = None
            # 'source' tells the report which NDVI (if any) the zones and covariates came from (report section 3)
            self.res["ndvi"] = dict(note="no NDVI", used=0, source="none")
            return self.res["ndvi"]
        en.save_stack(self.stack, os.path.join(self.out, "ndvi"))
        self.res["ndvi"] = dict(note=note, used=len(self.stack.used), norm=self.stack.norm, source=n["source"],
                                layers=self.stack.summary_table(), years_not_fetched=not_fetched,
                                years_fetch_failed=fetch_failed, years_fetch_partial=fetch_partial,
                                cv_median=float(np.nanmedian(self.stack.cv[self.grid.mask])) if self.stack.cv is not None else None)
        self.log(f"NDVI: {len(self.stack.used)} season(s) used ({note})")
        return self.res["ndvi"]

    # ------------------------------------------------------------------------------------------ 3 zones
    def make_zones(self):
        zc, f = self.cfg["zones"], self.cfg["files"]
        if zc["source"] == "map":
            # no silent fallback to NDVI zones (or one zone) when the zone map is missing: the PDF would then say
            # the zones came from the NDVI, or the verdict that the field has one zone
            p = f.get("zones")
            if not p:
                raise ValueError("The zones are to come from a zone map, but none is given: give it in step 2")
            if not os.path.exists(p):
                raise ValueError(f"The zone map {p} is missing: give it again in step 2")
            g = eio.read_vector(p).to_crs(self.epsg)
            col = self.cfg["columns"].get("zone") or eio.find_col(g, "zone") or eio.numeric_cols(g)[0]
            self.cfg["columns"]["zone"] = col
            # the NDVI mean (when there is one) orders text labels by NDVI (low = zone 1) instead of alphabetically,
            # and the configured no-data codes (e.g. -9999 or 0 in a 1..k map) are left to the neighbouring zones
            surf = self.stack.mean if self.stack is not None else None
            self.zone_r, info = ez.zones_from_vector(g, col, self.grid, surface=surf,
                                                     nodata_codes=zc.get("nodata_codes"))
            self.zone_surface = surf
        elif self.stack is not None:
            self.zone_r, self.zone_surface, info = ez.zones_from_stack(self.stack, zc["k"], zc["method"], zc["smooth"],
                                                                       zc["min_patch_ha"])
        else:
            self.zone_r = self.grid.mask.astype("int32")
            self.zone_surface = None
            info = dict(source="none (one zone)", k=1)
        # The engine renumbers a grower's zone map (0-based codes shifted, text labels put in NDVI order, codes made
        # contiguous 1..k), so the 'zone' code of zones_rx.shp need not be the grower's own value: without the label
        # next to it, the grower read the Rx rates against the wrong zones of their own map. to_polygons adds a
        # 'label' column from info['labels'] (the code as text when there are none, e.g. NDVI zones).
        self.zones = ez.to_polygons(self.zone_r, self.grid, self.boundary, self.zone_surface,
                                    labels=info.get("labels"))
        # the field acres of each zone are those of its polygons clipped to the boundary (zones_rx), so the PDF zone
        # table sums to the field area printed beside it and agrees with the Rx file; the count of 10 m cell centres
        # (zone_field_areas) differed from both by up to 0.1 ac per zone and is kept as the fallback
        if len(self.zones) and "area_ac" in self.zones.columns and self.zones.area_ac.sum() > 0:
            self.zone_ac = {int(z): float(a) for z, a in zip(self.zones.zone, self.zones.area_ac)}
        else:
            self.zone_ac = ej.zone_field_areas(self.zone_r, self.grid)
        q = ez.zone_quality(self.zone_r, self.stack.mean, self.grid.mask) if self.stack is not None else {}
        info.update(field_acres=self.zone_ac, ndvi_r2=q.get("r2"), ndvi_means=q.get("means"))
        self.res["zones"] = info
        en.write_tif(os.path.join(self.out, "zones.tif"), self.zone_r, self.grid, nodata=0, dtype="int32")
        self.log(f"zones: {info.get('k')} from {info.get('source')}, field acres {self.zone_ac}")
        return info

    # ------------------------------------------------------------------------------------------ 4 join
    def join(self):
        c, f, j = self.cfg["columns"], self.cfg["files"], self.cfg["join"]
        y = eio.read_vector(f["yield"]).to_crs(self.epsg)
        y = y[y.geom_type == "Point"].copy()
        for role in ("yield", "heading", "distance", "swath", "machine"):
            if c.get(role) is None:
                c[role] = eio.find_col(y, role)
        if c["yield"] is None:
            raise ValueError("No yield column recognised: choose it in step 1")
        self.inner = ej.trim_plots(self.plots, j["end_trim_m"], j["side_trim_m"])
        ypoly, mode = ej.yield_polygons(y, c["yield"], c["heading"], c["distance"], c["swath"], j["heading_is"],
                                        swath_unit=c.get("swath_unit"), dist_unit=c.get("distance_unit"))
        swath_check = self._swath_check(ypoly, mode)
        pieces, att = ej.assign_yield(ypoly, self.inner, mode, j["min_share"])
        # An empty join used to "succeed" with 0 units: step 5 then skipped every input with 'no design rate', which
        # sent the user to the design-rate column while the real cause was the yield join (and a treatment design
        # crashed in the comparison). The cause is named here instead.
        if int(att.get("kept", 0)) == 0:
            raise ValueError(
                f"No yield observation lands inside a trimmed plot ({int(att.get('n_obs', 0)):,} observation(s) read, "
                f"{int(att.get('outside_trimmed_plots', att.get('n_obs', 0))):,} outside every trimmed plot): the yield "
                "file may be from another field or year, its coordinate system may be wrong, or the plot trims "
                f"(ends {j['end_trim_m']} m, sides {j['side_trim_m']} m) or the minimum share inside one plot "
                f"({j['min_share']}) are too strict")
        applied = {}
        for name in self.inputs:
            p = (f.get("asapplied") or {}).get(name)
            if p:
                a = eio.read_vector(p).to_crs(self.epsg)
                rc = (c.get("applied") or {}).get(name) or eio.find_col(a, "applied_rate")
                c.setdefault("applied", {})[name] = rc
                a = self._convert_applied(a, rc, name)
                applied[name] = ej.applied_points(a, rc, self.inner, self.zone_r, self.grid)
        surf = self.stack.mean if self.stack is not None else None
        self.units, self.pieces, uinfo = ej.build_units(pieces, self.plots, self.inputs, self.grid, self.zone_r, surf,
                                                        applied, j["min_obs"], j["min_area_m2"], c["machine"], j["unit"])
        rate_checks = self._check_applied_vs_design()
        if int(uinfo.get("n_units", 0)) == 0:
            n_raw = int(uinfo.get("n_units_raw", 0))
            if n_raw == 0:
                raise ValueError(
                    f"{int(att['kept']):,} yield observation(s) lie in the trimmed plots, but none of them is on a cell "
                    f"of the zone map ({int(uinfo.get('dropped_no_zone_obs') or 0):,} dropped with no zone under them): "
                    "check that the zones (step 3) cover the trial plots")
            raise ValueError(
                f"All {n_raw} plot x zone unit(s) were dropped: {int(uinfo.get('dropped_few_obs', 0))} with fewer than "
                f"{int(uinfo.get('min_obs', j['min_obs']))} observations (min_obs) and "
                f"{int(uinfo.get('dropped_small', 0))} smaller than {float(uinfo.get('min_area_m2', j['min_area_m2'])):g} m2 "
                f"(min_area_m2), from {int(att['kept']):,} observation(s) in the trimmed plots: lower these thresholds "
                "in step 4, or check the trims")
        self.plot_tbl = ej.plot_table(self.units, self.inputs)
        self.applied = applied
        # Without a swath width (point mode) every yield point weighs 1.0: the weights are counts, not m2, so no
        # area in acres is reported (join.yield_polygons marks the frame with attrs['area_known'] = False).
        area_known = bool(ypoly.attrs.get("area_known", True)) and bool(uinfo.get("area_known", True))
        tot_ac = float(ypoly.w_area.sum() / eio.M2_PER_AC) if area_known else None
        # which length is missing: point mode has no area when the swath OR the distance is missing, and blaming the
        # swath width for a file that only lacks its distance sent the user to a column that was fine
        area_missing = None
        if not area_known:
            miss = [lab for col, lab in (("swath_m", "swath"), ("dist_m", "distance"))
                    if col not in ypoly.columns or not pd.to_numeric(ypoly[col], errors="coerce").notna().any()]
            area_missing = " and ".join(miss) if miss else None
        yq = ez.zone_quality(self.units.zone.values, self.units["yield"].values) if self.units.zone.nunique() > 1 else {}
        self.res["join"] = dict(mode=mode, attrition=att, units=uinfo, harvested_ac=tot_ac, area_known=area_known,
                                area_missing=area_missing,
                                analysed_ac=float(self.units.area_ac.sum()) if area_known else None, yield_col=c["yield"],
                                yield_r2_by_zone=yq.get("r2"), yield_means_by_zone=yq.get("means"),
                                rate_source={n: self.units[f"rate_src_{n}"].value_counts().to_dict() for n in self.inputs},
                                columns={k: c.get(k) for k in ("yield", "heading", "distance", "swath", "machine")},
                                applied_cols=c.get("applied"), rate_checks=rate_checks, swath_check=swath_check,
                                length_units={"swath": c.get("swath_unit") or eio.length_unit_from_name(c.get("swath")),
                                              "distance": c.get("distance_unit") or eio.length_unit_from_name(c.get("distance"))},
                                # an as-applied file whose rate column held no usable number comes back empty with
                                # a warning (join.applied_points): the report and the app must say why the design
                                # rate was used instead, not only the python warnings log
                                applied_warnings={n: str(a.attrs.get("warning")) for n, a in applied.items()
                                                  if a.attrs.get("warning")})
        self.log(f"join: {att['kept']:,} of {att['n_obs']:,} observations in plots -> {uinfo['n_units']} units in {uinfo['n_plots']} plots")
        return self.res["join"]

    def _swath_check(self, ypoly, mode):
        """The swath width against the spacing of the yield passes. The swath unit comes from the column name or
        the user's choice (ft / m), and a metric export read as feet gives rectangles 3.3 x too narrow: every area,
        the containment test and the unit sizes are then wrong while the weighted means look fine. The passes
        themselves give the truth: the median distance from a point to the nearest point of a NEIGHBOURING pass
        (across the heading, within a few metres along it) is the pass spacing, which equals the cut width less the
        overlap. A ratio far from 1 is reported (results['join']['swath_check']); the run goes on."""
        if mode != "polygon" or "swath_m" not in ypoly or len(ypoly) < 200:
            return None
        try:
            from scipy.spatial import cKDTree
            cs = shapely.get_coordinates(ypoly.geometry.values).reshape(len(ypoly), -1, 2)
            u = cs[:, 1] - cs[:, 0]
            u /= np.maximum(np.linalg.norm(u, axis=1), 1e-9)[:, None]
            v = np.c_[u[:, 1], -u[:, 0]]
            P = np.c_[ypoly.px.values, ypoly.py.values]
            sw = float(np.nanmedian(ypoly.swath_m))
            ds = float(np.nanmedian(ypoly.dist_m)) if "dist_m" in ypoly else 2.0
            rng = np.random.default_rng(0)
            idx = rng.choice(len(P), min(3000, len(P)), replace=False)
            tree = cKDTree(P)
            r = max(3.0 * sw, 10.0)
            spacing = []
            for i in idx:
                nb = tree.query_ball_point(P[i], r)
                if len(nb) < 2:
                    continue
                d = P[nb] - P[i]
                perp = np.abs(d @ v[i])
                along = np.abs(d @ u[i])
                # same-pass points are told by an absolute offset (GPS noise ~1 m), not by a fraction of the swath:
                # a swath read 3.3 x too wide would otherwise hide the true neighbouring pass
                ok = (perp > min(0.3 * sw, 2.0)) & (along < max(2.0 * ds, 3.0))
                if ok.any():
                    spacing.append(float(perp[ok].min()))
            if len(spacing) < 50:
                return None
            sp = float(np.median(spacing))
            ratio = sp / sw if sw > 0 else np.nan
            out = dict(swath_m=sw, pass_spacing_m=sp, ratio=float(ratio), n=len(spacing), warning=None)
            if np.isfinite(ratio) and abs(ratio - 1.0) > 0.30:
                # the ft / m factor is 3.28 less the overlap (ratio 2.3-3.3 one way, 0.3-0.45 the other); a genuine
                # 2 x error in the cut width is far rarer than the unit mix-up
                hint = ("the swath looks like metres read as feet" if ratio > 2.0 else
                        "the swath looks like feet read as metres" if ratio < 0.5 else
                        "the swath column may not be the cut width, or the passes overlap a lot")
                out["warning"] = (f"Swath width {sw:.2f} m but the yield passes are {sp:.2f} m apart (median of "
                                  f"{len(spacing)} points): {hint}. Check the unit (ft / m) of the swath column in "
                                  "step 1; every area and the containment test depend on it.")
                self.log("WARNING: " + out["warning"])
            return out
        except Exception as exc:                                        # a check, never a stop
            self.log(f"swath check skipped: {exc}")
            return None

    def _check_applied_vs_design(self):
        """Per input with both an applied and a design rate on the units: the median ratio applied / design. Near
        the nutrient fraction (or its inverse) the map and the design are in different bases (lb of product against
        lb of N); far from 1 otherwise, the unit or the column is wrong. The warnings are logged and kept in
        results['join']['rate_checks'] for the app and the PDF; the run goes on, since the ratio is only a hint."""
        out = {}
        for name in self.inputs:
            rc, tc = f"rate_{name}", f"target_{name}"
            if rc not in self.units or tc not in self.units:
                continue
            src = self.units.get(f"rate_src_{name}")
            m = self.units[tc].notna() & self.units[rc].notna() & (self.units[tc] > 0)     # a 0-rate check plot has no ratio
            if src is not None:
                m &= src.astype(str).str.startswith("applied")
            if m.sum() < 5:
                continue
            ratio = float((self.units.loc[m, rc] / self.units.loc[m, tc]).median())
            ip = self.prices.input(name)
            fr = float(ip.fraction)
            chk = dict(median_ratio=ratio, n_units=int(m.sum()), warning=None)
            # a basis mix-up moves the ratio to the fraction or its inverse; for products with a fraction near 1
            # (anhydrous 0.82) that band overlaps ordinary applicator error, so it is only read for fractions <= 0.6.
            # A kg/ha vs lb/ac mix-up (x 0.892 or 1.121) is inside ordinary applicator error and cannot be told here.
            if 0.0 < fr <= 0.6 and (abs(ratio - fr) <= 0.10 * fr or abs(ratio - 1 / fr) <= 0.10 / fr):
                chk["warning"] = (f"the as-applied rates of {name} are {ratio:.2f} x the design rates, about "
                                  f"{'the nutrient fraction' if ratio < 1 else '1 / the nutrient fraction'} of "
                                  f"{ip.product} ({fr:g}): the map and the design seem to be in different bases (lb of "
                                  "product against lb of nutrient); set what the as-applied map logs in step 1")
            elif abs(ratio - 1.0) > 0.25:
                chk["warning"] = (f"the as-applied rates of {name} are {ratio:.2f} x the design rates on the units "
                                  "that have both: check the basis and the rate column of the map, and whether the "
                                  "drill applied what the design asked")
            if chk["warning"]:
                self.log("WARNING: " + chk["warning"])
            out[name] = chk
        return out

    # ------------------------------------------------------------------------------------------ 5 analyse
    def _n_design(self, name):
        """Number of distinct design rates of input `name` (target_<input>) on the analysis units."""
        tcol = f"target_{name}"
        return (int(pd.to_numeric(self.units[tcol], errors="coerce").dropna().nunique())
                if tcol in self.units.columns else 0)

    def _usable_control(self, name):
        """Whether input `name` can be a control (other input) in another input's model: it is not skipped (2 or more
        design rates), its design varies inside at least one original zone, and its rate column (rate_ or target_,
        per the rate source) holds at least one finite value.
        Why: a skipped input with no design and no as-applied map has an all-NaN rate; response.prep then scaled it
        with a NaN mean and sd, every predicted curve became NaN, decide() took argmax of NaN (the lowest trial rate)
        and every net return, gain and the verdict of the analysed input came out NaN. An input whose design rate is
        set by zone (zone 1 = 80, zone 2 = 120: no design degree of freedom inside the original zones) cannot be
        estimated itself (analyse skips it), and as a control it is collinear with the zone levels of the model."""
        if self._n_design(name) < 2:
            return False
        tcol = f"target_{name}"
        if er.within_design_df(self.units[tcol].values, self.units["zone"].values) < 1:
            return False
        c = f"rate_{name}" if self.cfg["join"]["rate_source"] == "applied" else f"target_{name}"
        if c not in self.units.columns:
            return False
        v = pd.to_numeric(self.units[c], errors="coerce").astype(float)
        return bool(np.isfinite(v).any())

    def _control_why(self, name):
        """Why _usable_control rejected input `name`, in words for the model notes."""
        if self._n_design(name) < 2:
            return "not analysed itself (fewer than 2 design rates)"
        if er.within_design_df(self.units[f"target_{name}"].values, self.units["zone"].values) < 1:
            return "its design rate does not vary inside any zone (confounded with the zone levels)"
        return "no usable rate on the units"

    def analyse(self, progress=None, inputs=None):
        P, m = self.prices, self.cfg["model"]
        out = {}
        realised_tables = {}
        for name in (inputs or self.inputs):
            ip = P.input(name)
            # A response curve needs at least 2 distinct DESIGN rates of this input (target_<input>). A treatment-only
            # design (strips labelled VR / Flat / None with no rate column), or a second input with no design column,
            # has none: fitting it used to crash (rate_zone_table KeyError) before the treatment comparison below
            # was reached. Such an input is skipped with the reason, and the comparison still runs.
            tcol = f"target_{name}"
            n_design = self._n_design(name)
            if n_design < 2:
                why = ("no design rate" if n_design == 0 else "only 1 distinct design rate")
                out[name] = dict(input=name, stamp=self.analysis_stamp(name),
                                 skipped=f"{name}: the analysed plots have {why} ({tcol}); a response curve needs at "
                                         "least 2, so no response model or economics was run for this input"
                                         + (" (the treatment comparison is reported instead)"
                                            if self.plots.treatment.notna().any() else ""))
                self.log(f"analysis of {name} skipped: {n_design} distinct design rate(s)")
                continue
            # only inputs that were analysed themselves and have finite rates are controls (see _usable_control)
            others = [o for o in self.inputs if o != name and self._usable_control(o)]
            not_controls = [o for o in self.inputs if o != name and o not in others]
            # The response form is decided BEFORE the zones are merged. A field with fewer distinct design rates than
            # the configured form needs (a 2-rate strip trial under the default quadratic) can only fit a straight
            # line. That fallback used to happen inside _fit_models, AFTER merge_small_zones had applied the curve
            # rule (3 design rates per zone): every zone of a 2-rate design was merged into one, so the zone model,
            # the zone x rate test and the VR cross-fit never ran although each zone supported its own line, and
            # the verdict said the zones lacked the data. The merge, the zone fit, the bootstrap and the cross-fits
            # all use the form chosen here.
            form, form_note = m["form"], None
            if form != "linear" and n_design < er.min_design_rates(form):
                need = er.min_design_rates(form)
                form_lab = {"qplateau": "quadratic-plateau"}.get(form, form)
                form_note = (f"{name}: the field has {n_design} distinct design rates and a {form_lab} curve needs "
                             f"{need} -> straight-line response used, for the field and per zone (its optimum always "
                             "lies at an edge of the trial range)")
                form = "linear"
                self.log(f"analysis of {name}: {n_design} design rates -> straight-line response")
            spec = er.Spec(input=name, others=others,
                           others_sq={o: self.status_quo(o) for o in others},
                           form=form, machine=m["machine"], spatial=m["spatial"],
                           rate_source=self.cfg["join"]["rate_source"])
            pc, pi, sq = P.crop_price_bu, ip.price_per_rate_unit, self.status_quo(name)
            # The merge rule follows the response form, like check_support and the bootstrap / cross-fit support rule:
            # a straight line needs 2 design rates (1 design degree of freedom inside the original zones), a curve 3.
            min_units, min_rates = int(m.get("min_units_zone", 20)), int(er.min_design_rates(form))
            units, W, zmap, znotes = er.merge_small_zones(self.units, self.zone_ac, tcol, min_units=min_units,
                                                          min_rates=min_rates,
                                                          ndvi_means=(self.res.get("zones") or {}).get("ndvi_means"),
                                                          form=form)
            # Zones are merged when they lack the support for a curve of their OWN; a zone LEVEL (a yield shift) needs
            # no such support. Every model fitted on these units (the single curve, the curve-per-zone model, the
            # NDVI model, and their bootstrap and cross-fit, which resample these same rows) therefore reads the
            # ORIGINAL zones as level covariates from 'zone_level' (response.fit): the zone model takes its
            # intercepts from C(zone_level), nested in the analysis zone, and keeps its rate terms per analysis zone.
            # With the merged zones only, the between-zone yield difference became residual noise and could hide a
            # clear rate response.
            units = units.copy()
            units["zone_level"] = (units["zone_orig"] if "zone_orig" in units.columns else units["zone"]).astype(int)
            # zone_notes are about zone SUPPORT only (merge_small_zones); what happened to the model itself (linear
            # fallback, NDVI model not fitted) goes to model_notes, so the report does not print a field-wide switch
            # to a straight line under a zone-merge heading with the zone threshold appended
            R = dict(input=name, price_crop=pc, price_input=pi, status_quo=sq, vr_cost=P.vr_cost_ac,
                     zone_map=zmap, zone_notes=znotes, model_notes=[], zone_acres=W,
                     zone_rule=dict(min_units=min_units, min_rates=min_rates))
            if form_note:
                R["model_notes"].append(form_note)
            if not_controls:
                # the reason per input: fewer than 2 design rates, a design that does not vary inside any original
                # zone (confounded with the zone levels), or no usable rate on the units
                R["model_notes"].append(
                    "; ".join(f"{o} not used as a control in this model: {self._control_why(o)}" for o in not_controls))
            # One input whose response cannot be estimated (its design rate set by zone, or any other ValueError
            # of its fits) used to abort step 5 for EVERY input: the inputs already fitted and the treatment
            # comparison were lost and no PDF was written. It is skipped with the reason instead, like an input with
            # fewer than 2 design rates, and the other inputs continue.
            try:
                fits, spec = _fit_models(name, units, spec, R, self.stack is not None)
            except ValueError as exc:
                msg = str(exc).strip().rstrip(".")
                if not msg.lower().startswith(f"{name.lower()}: "):
                    msg = f"{name}: {msg}"
                out[name] = dict(input=name, stamp=self.analysis_stamp(name),
                                 skipped=msg + ("; no response model or economics was run for this input"
                                                + (" (the treatment comparison is reported instead)"
                                                   if self.plots.treatment.notna().any() else "")))
                self.log(f"analysis of {name} skipped: {exc}")
                continue
            R["models"] = {k: dict(formula=v.formula, tests=v.tests, form=getattr(v.spec, "form", spec.form),
                                   form_note=getattr(v, "form_note", None),
                                   # the formulas name the other inputs (o_0, ...) and the combines (m_0, ...) by
                                   # alias: keep the keys, so the params can be read
                                   aliases={**{a: n for n, a in (v.scale.get("other_cols") or {}).items()},
                                            **{a: f"combine share {n}" for a, n in (v.scale.get("machine_names") or {}).items()}},
                                   # the controls the fit really used (the report's method sentence reads them)
                                   controls=_controls_used(v),
                                   params={a: float(b) for a, b in (v.params or {}).items() if np.isscalar(b)})
                           for k, v in fits.items()}
            R["table"] = er.rate_zone_table(units, spec, pc, pi).to_dict("records")
            # The main model is the curve-per-zone model whenever there are 2 or more analysis zones. Which model it
            # is comes from the models that were fitted, never from Fit.kind: a model's kind and its functional
            # form are separate things (a zoned quadratic-plateau fit is still the zone model), and choosing the
            # weights from fit.kind once gave a zoned fit the weights {None: 1}, which crashed decide().
            kind = "zone" if "zone" in fits else "pooled"
            main = fits[kind]
            wts = W if kind == "zone" else {None: 1.0}
            # the PDF used to print a fixed list of controls (combine shares, trend surface, the other input) whatever
            # the model held: with one combine, 30 units or fewer, spatial off, or the other input not usable as a
            # control, it named controls that were not in the model
            R["controls"] = R["models"][kind].get("controls") or []
            # With one analysis zone there is no VR strategy: the "VR" rate is the flat optimum itself. Charging it
            # the VR cost printed the same flat rate with two net returns (and a VR gain of exactly -VR cost) in the
            # summary, realised and cross-fit rows, so the VR cost is only charged when there are zones to vary by.
            single = kind == "pooled"
            vc = 0.0 if single else float(P.vr_cost_ac or 0.0)
            R["model_kind"] = kind
            R["single_zone"] = single
            R["vr_cost"] = vc
            R["decision"] = ee.analyse(main, wts, pc, pi, sq, vc)
            R["pooled"] = ee.analyse(fits["pooled"], {None: 1.0}, pc, pi, sq, 0.0)
            # functional-form sensitivity: the quadratic overestimates the optimum when the true response has a
            # plateau (Cerrato & Blackmer 1990; Bullock & Bullock 1994), so the other form is always reported too
            R["forms"] = _form_sensitivity(units, spec, W, pc, pi, sq, vc)
            if "ndvi" in fits:
                # a pixel-wise NDVI prescription IS a variable rate, so it is charged the VR cost as configured
                vals = self.stack.mean[self.grid.mask]
                R["ndvi_rx"] = ee.ndvi_prescription(fits["ndvi"], vals, np.full(vals.shape, self.grid.cell_area_m2()),
                                                    pc, pi, sq, P.vr_cost_ac)
            if m.get("B", 0):
                bt = ee.bootstrap(units, spec, kind, wts, pc, pi, sq, vc, B=int(m["B"]), progress=progress)
                # replicates without the design rates for their curves (n_dropped) and replicates whose fit failed
                # (n_failed, in bt.attrs and so in summarise_boot) are both left out of bt; the report prints the
                # two counts and the number actually used, B - n_dropped - n_failed
                nd = (getattr(bt, "attrs", None) or {}).get("n_dropped")
                if nd is None:
                    nd = int(m["B"]) - len(bt)
                R["bootstrap"] = ee.summarise_boot(bt, n_dropped=int(nd))
                R["bootstrap"]["B"] = int(m["B"])
                R["bootstrap"]["n_used"] = int(len(bt))
                bt.to_csv(os.path.join(self.out, f"bootstrap_{name}.csv"), index=False)
            if m.get("R", 0):
                cf, cs = ee.crossfit(units, spec, kind, wts, pc, pi, sq, vc, R=int(m["R"]), progress=progress)
                R["crossfit"] = cs
                # why half-splits were skipped (no design support, no rate overlap, a failed fit): the verdict and
                # the report name the real cause when no VR gain was cross-fitted, instead of always the design rates
                R["crossfit_skips"] = _cf_skips(cf)
                cf.to_csv(os.path.join(self.out, f"crossfit_{name}.csv"), index=False)
                if kind == "zone" and sq is not None:
                    # The best flat rate vs the status quo, cross-fitted on the SAME basis as the in-sample row of the
                    # summary table (summary_basis): the flat rate is chosen on the training half by the single curve
                    # (as the verdict chooses it), and it and the status quo are valued on the held-out half with the
                    # zone model over the field's zone acres. A cross-fit of the single curve judged on a held-out
                    # single curve at the plots' zone mix was the out-of-sample version of a different number (it
                    # ignored the zone heterogeneity the in-sample row is penalised for), so the two figures printed
                    # side by side, and gated together in the verdict, could differ several-fold. With one analysis
                    # zone the main cross-fit above is already on that basis (the single curve, {None: 1}).
                    cf_flat, cs_flat = ee.crossfit(units, spec, "pooled", W, pc, pi, sq, 0.0, R=int(m["R"]),
                                                   progress=progress, eval_fit_kind="zone")
                    R["crossfit_flat"] = cs_flat
                    # marks results made on the summary basis (the report labels them; older results lack it)
                    R["crossfit_flat_basis"] = "zone"
                    R["crossfit_flat_skips"] = _cf_skips(cf_flat)
            a, b = P.input_price_range_pct
            pc_rng = (pc * (1 + a / 100), pc * (1 + b / 100))
            pi_rng = (pi * (1 + a / 100), pi * (1 + b / 100))
            # A zero-width range ([0, 0] in a saved project.json or the CLI) or an input price of 0 gives one price
            # repeated n times on that axis: the report's pivot then failed on duplicate entries after the whole
            # bootstrap and cross-fit had run. One value is one row / column of the grid.
            n_pg = int(m.get("n_price", 7))
            if np.isclose(*pc_rng) and np.isclose(*pi_rng):
                n_pg = 1
            pg = ee.price_grid(main, wts, pc_rng, pi_rng, sq, vc, n=n_pg)
            if len(pg):
                pg = pg.drop_duplicates(["input_price", "crop_price"]).reset_index(drop=True)
            R["price_grid"] = pg.to_dict("records")
            # every strategy on ONE basis (the main model on the field's zone acres), so the summary table's net
            # returns can be subtracted from each other; the verdict's in-sample flat gain is read from it too
            R["summary_basis"] = summary_basis(R, pc, pi)
            R["verdict"] = verdict(R, float(m.get("min_gain_ac", 3.0)))
            R["summary_basis"]["recommended_flat"] = _flat_row(R["decision"], R["verdict"]["flat_rate"], pc, pi, sq)
            # the realised trial needs the verdict's recommended flat rate, so it is computed after the verdict
            rl, rs, rz = ee.realised(units, main, R["decision"], pc, pi, sq, vr_cost=vc,
                                     flat_rate=R["verdict"]["flat_rate"])
            R["realised"] = dict(summary=rs, by_zone=rz.to_dict("records"))
            rl.to_csv(os.path.join(self.out, f"realised_units_{name}.csv"), index=False)
            realised_tables[name] = rl
            R["stamp"] = self.analysis_stamp(name)
            out[name] = R
            self._fits = getattr(self, "_fits", {})
            self._fits[name] = fits
        if len(realised_tables) > 1:
            # the cost of the trial to the grower with EVERY tested input at its status quo at once: each input's
            # realised table moves only that input, and printing its own compensation under each input gave two
            # contradictory sentences for one trial (Trial 22: seed 'earned +4.58', urea 'cost 8.02')
            jt = realised_joint(realised_tables, out, P.crop_price_bu)
            if jt:
                out["_realised_joint"] = jt
        if self.plots.treatment.notna().any():
            tp = {n: P.input(n).price_per_rate_unit for n in self.inputs}
            t = et.compare(self.plot_tbl, P.crop_price_bu, tp, vr_cost=P.vr_cost_ac)
            if t:
                out["_treatments"] = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                                      for k, v in t.items() if k != "plots"}
        self.res["analysis"] = out
        return out

    # ------------------------------------------------------------------------------------------ 6 outputs
    def write_outputs(self):
        o = self.out
        self.units.to_csv(os.path.join(o, "units.csv"), index=False)
        self.plot_tbl.to_csv(os.path.join(o, "plots.csv"), index=False)
        up = ej.unit_polygons(self.pieces, self.units)
        up.to_file(os.path.join(o, "units.gpkg"), driver="GPKG")
        z = self.zones.copy()
        meaning = {c: ("", _ZONE_FIELD_MEANING.get(c, "zone attribute")) for c in z.columns if c != "geometry"}
        for name, R in self.res.get("analysis", {}).items():
            # an input skipped by analyse (fewer than 2 design rates) has no decision and so no prescription
            if name.startswith("_") or R.get("skipped"):
                continue
            dec = R["decision"]
            rz = {(None if k in (None, "None") else int(k)): v for k, v in dec["r_zone"].items()}
            # The zone polygons carry the ORIGINAL zone codes, while the optima are keyed by the ANALYSIS zones left
            # after merge_small_zones: a zone merged into a neighbour must get its analysis zone's rate (as in the
            # PDF prescription map), not the flat fallback.
            zmap = {int(a): (None if b is None else int(b)) for a, b in (R.get("zone_map") or {}).items()}
            fallback = rz.get(None, dec["r_uniform"])
            an = [zmap.get(int(k), int(k)) for k in z.zone]
            z[f"Zan_{name}"] = an
            z[f"Rx_{name}"] = [rz.get(a, fallback) for a in an]
            # the flat rate the report recommends (the single-curve optimum, or the status quo when the trial
            # cannot tell them apart), not the uniform optimum of the zone model
            z[f"Flat_{name}"] = (R.get("verdict") or {}).get("flat_rate", dec["r_uniform"])
            meaning[f"Zan_{name}"] = (name, "analysis zone (original zones merged when they lacked the support for a curve)")
            meaning[f"Rx_{name}"] = (name, "zone prescription rate")
            meaning[f"Flat_{name}"] = (name, "recommended flat rate (verdict)")
        z.to_file(os.path.join(o, "zones_rx.gpkg"), driver="GPKG")
        zz = z.to_crs(4326)
        # dBASE field names hold 10 characters: cutting them can make two inputs share a name (Flat_input2 and
        # Flat_input3 both become Flat_input), which geopandas refuses after minutes of bootstrap. Unique names are
        # built instead, and zones_rx_fields.csv says which field belongs to which input.
        names = _shp_field_names([c for c in zz.columns if c != "geometry"])
        zz = zz.rename(columns=names)
        zz.to_file(os.path.join(o, "zones_rx.shp"))
        pd.DataFrame([dict(field=names[c], input=meaning.get(c, ("", ""))[0], meaning=meaning.get(c, ("", ""))[1],
                           gpkg_column=c) for c in names],
                     columns=["field", "input", "meaning", "gpkg_column"]).to_csv(
            os.path.join(o, "zones_rx_fields.csv"), index=False)
        with open(os.path.join(o, "results.json"), "w", encoding="utf-8") as f:
            json.dump(self.res, f, indent=1, default=_json_default)
        save(self.cfg, self.workdir, summary=self.summary())
        return o

    def analysis_stamp(self, name=None):
        """See analysis_stamp(cfg): the same for every input (the prices hold all the input definitions)."""
        return analysis_stamp(self.cfg)

    def summary(self):
        s = dict(n_plots=self.res.get("design", {}).get("n_plots"), n_units=self.res.get("join", {}).get("units", {}).get("n_units"))
        for name, R in self.res.get("analysis", {}).items():
            if name.startswith("_"):
                continue
            if R.get("skipped"):
                s[name] = dict(skipped=R["skipped"])
                continue
            d = R["decision"]
            s[name] = dict(r_uniform=d["r_uniform"], r_zone=d["r_zone"], gain_vr_vs_uniform=d["gain_vr_vs_uniform"],
                           gain_vr_vs_sq=d.get("gain_vr_vs_sq"), verdict=R.get("verdict", {}).get("text"))
        return s


# load_ndvi's refusal of a source-'fetch' project fetched before the image list and its settings were recorded; the
# app matches the start of this message to show it as a warning with the Fetch button, so keep the prefix stable
OLD_FETCH_MSG = "This project was fetched with an older version: fetch again"


def _boundary_sha1(boundary):
    """sha1 of the field boundary (its CRS and the WKB of its union, coordinates snapped to 1 cm so that a re-read of
    the same file gives the same hash): a new boundary needs a new NDVI fetch."""
    geom = boundary.geometry.union_all()
    try:
        import shapely
        geom = shapely.set_precision(geom, 0.01)
    except Exception:                                                 # pragma: no cover - shapely < 2
        pass
    return hashlib.sha1(str(boundary.crs).encode("utf-8") + geom.wkb).hexdigest()


_ZONE_FIELD_MEANING = {"zone": "engine zone code 1..k (original zones before merging; renumbered from the zone "
                               "map, see 'label')",
                       "area_ac": "zone area (ac)",
                       "area_pct": "share of the field area (%)", "ndvi_rel": "mean relative NDVI of the zone",
                       "label": "the zone's own label: the grower's value in the zone map (the code as text for NDVI "
                                "zones)"}


def fetch_status(log, years):
    """{'fetched': [...], 'failed': [...]} from the log rows of ndvi_fetch.fetch_years for the selected `years`.

    A season is fetched when every one of its log rows is 'ok' or 'no clear date' (a new fetch would find the same
    nothing), and failed when any of its rows failed on the network ('statistics failed' / 'download failed'): a
    season with one of its dates lost is fetched again, and the dates already on disk are reused. A failed season with
    at least one 'ok' row is also listed in 'partial' (a subset of 'failed'): its downloaded dates are stored with the
    others and load_ndvi builds the season from them, so it is in the zones with fewer dates, not missing from them.
    A selected season with no log row at all was left out by fetch_years because Sentinel-2 L2A has nothing for it
    (before 2017 or in the future): fetching it again would change nothing, so it counts as fetched and does not
    trigger a fetch on every run."""
    rows = {}
    for r in log or []:
        try:
            rows.setdefault(int(r.get("year")), []).append(str(r.get("status") or ""))
        except (TypeError, ValueError):
            continue
    fetched, failed, partial = [], [], []
    for y in sorted({int(v) for v in years}):
        sts = rows.get(y)
        if sts is None or all(t in ("ok", "no clear date") for t in sts):
            fetched.append(y)
        else:
            failed.append(y)
            if "ok" in sts:
                partial.append(y)
    return dict(fetched=fetched, failed=failed, partial=partial)


def _fit_models(name, units, spec, R, has_stack):
    """The models of one input: 'pooled' (one curve), 'zone' (a curve per analysis zone, when there are 2 or more)
    and 'ndvi' (a curve along the NDVI, when there is an NDVI). Returns (fits, spec), spec after any straight-line
    fallback; model notes go to R['model_notes']. Raises ValueError when the input's response cannot be estimated:
    analyse() then skips that input with the message and goes on with the others."""
    fits = {}
    for kind in ("pooled", "zone", "ndvi"):
        if kind == "zone" and units.zone.nunique() < 2:
            continue
        if kind == "ndvi" and (not has_stack or units.ndvi.isna().all()):
            continue
        try:
            fits[kind] = er.fit(units, spec, kind)
        except ValueError as exc:
            if kind == "zone":
                # merge_small_zones guarantees the support of the zone curves, so this is not expected; it still
                # skips this input only
                raise ValueError(f"{name}: the curve-per-zone model could not be fitted: {exc}") from exc
            if kind == "pooled":
                if spec.form != "linear":
                    # two design rates cannot identify a curvature: fall back to a straight line for the field
                    spec = er.Spec(**{**spec.__dict__, "form": "linear"})
                    try:
                        fits[kind] = er.fit(units, spec, kind)
                    except ValueError as exc2:
                        raise ValueError(_confounded_msg(name, exc2)) from exc2
                    R["model_notes"].append(f"{exc} -> straight-line response used for the whole field (its "
                                            "optimum always lies at an edge of the trial range)")
                else:
                    raise ValueError(_confounded_msg(name, exc)) from exc
            else:
                R["model_notes"].append(f"NDVI model not fitted: {exc}")
    return fits, spec


def _confounded_msg(name, exc):
    """Not even a line: every original zone holds a single design rate, so the rate cannot be told apart from the
    zone's own yield level (a design problem, not a data problem)."""
    return (f"{name}: the response to the rate cannot be estimated: {str(exc).rstrip('.')}. The design rates are "
            "confounded with the zones (each zone saw one rate); a randomised design with every rate in every zone "
            "is needed")


def _controls_used(fit):
    """The controls a fitted model really holds, in words: the other inputs by their real name, 'combine shares' and
    'trend surface'. Read from response._controls on the fit's own data and spec (the terms the formula was built
    from), so they follow spec.machine / spec.spatial, the more-than-30-units rule of the trend surface, a single
    combine (no varying share) and the other inputs left out of spec.others. None when it cannot be told."""
    try:
        terms = er._controls(fit.data, fit.spec, fit.scale)
    except Exception:
        return None
    alias = {a: n for n, a in (fit.scale.get("other_cols") or {}).items()}
    out = []
    machines = set(fit.scale.get("machines") or [])
    for t in terms:
        lab = (alias.get(t) or ("combine shares" if (t in machines or str(t).startswith("sh_")) else None)
               or ("trend surface" if t == getattr(er, "SPATIAL", None) else None))
        if lab and lab not in out:
            out.append(lab)
    return out


def _cf_skips(cf):
    """The skip counts of one economics.crossfit run (cf.attrs): half-splits tried, skipped without the design support
    of a curve, without a common applied-rate range, or because a fit failed (with the first error), and used."""
    a = getattr(cf, "attrs", None) or {}
    out = {k: int(a[k]) for k in ("n_tried", "n_no_support", "n_no_overlap", "n_failed", "n_used")
           if a.get(k) is not None}
    if a.get("first_error"):
        out["first_error"] = str(a["first_error"])
    return out


def cf_missing_txt(sk):
    """Why the cross-fit gave no VR gain, from its skip counts (R['crossfit_skips']). Why: the verdict always blamed
    the design rates, while a fit that fails in every fold is a model problem and a missing rate overlap points at the
    as-applied rates; the user was sent to fix the wrong thing."""
    sk = sk or {}
    n = int(sk.get("n_tried") or 0)
    if not n:
        # results computed before the counts were kept
        return "no cross-fit split had the design support for the zone curves in both halves"
    ns, no, nf = (int(sk.get(k) or 0) for k in ("n_no_support", "n_no_overlap", "n_failed"))
    err = f" (first error: {str(sk['first_error'])[:120]})" if nf and sk.get("first_error") else ""
    if ns == n:
        return (f"none of the {n} cross-fit half-splits kept the design support for the zone curves in both halves "
                "(design rates in every zone, and design degrees of freedom inside the original zones)")
    if no == n:
        return f"none of the {n} cross-fit half-splits had a common applied-rate range in its two halves"
    if nf == n:
        return f"the zone model failed to fit in every one of the {n} cross-fit half-splits{err}"
    parts = [t for c, t in ((ns, f"{ns} without the design support for the zone curves"),
                            (no, f"{no} without a common applied-rate range"),
                            (nf, f"{nf} whose fit failed{err}")) if c]
    return f"no cross-fit split gave a VR gain ({n} half-splits: {', '.join(parts)})"


def main_form(R):
    """Functional form of the main model of one input's result (after any straight-line fallback)."""
    M = R.get("models") or {}
    k = R.get("model_kind") or ("zone" if "zone" in M else "pooled")
    return (M.get(k) or M.get("pooled") or {}).get("form") or "quadratic"


def sq_extrapolation_txt(R):
    """How the yield of a status quo outside the trial range is valued (economics.bounded_curve), in words that
    follow the fitted form and the side of the range. Why: one sentence 'extrapolated linearly ... (held at the edge
    value on a plateau)' was wrong for a plateau curve above the range, which is extended with its slope at the upper
    edge (0 once the plateau is reached inside the range, the rising slope otherwise)."""
    d = R.get("decision") or {}
    sq, g = R.get("status_quo"), d.get("grid") or []
    if sq is None or not len(g):
        return ""
    lo = float(g[0])
    if main_form(R) == "qplateau":
        if float(sq) < lo:
            return ("its yield is extended from the lower edge with the slope of the fitted plateau curve's rising "
                    "segment there, never below 0")
        return ("its yield is extended from the upper edge with the one-sided slope of the fitted plateau curve there "
                "(0 when the plateau is reached inside the range, the rising slope otherwise), never below 0")
    return "its yield is extrapolated linearly with the slope of the fitted curve at the nearest edge, never below 0"


def realised_joint(tables, results, pc):
    """The realised trial with every input at its status quo AT ONCE, on the units common to the inputs' realised
    tables (economics.realised, one per input, each moving only its own input). The fitted yield is additive in the
    other inputs (they enter the model as controls), so the joint counterfactual yield is the observed yield plus the
    sum of each input's own shift: y_cf = y_obs + sum_i (y_sq_i - y_obs); the net returns charge every input at its
    trial rate (nr_trial) or at its status quo (nr_sq). Inputs without a priced status quo are left out and named;
    None when fewer than two inputs have one."""
    use = {}
    for n, t in tables.items():
        R = results.get(n) or {}
        if R.get("status_quo") is None or R.get("skipped") or "y_sq" not in t.columns:
            continue
        if (R.get("realised") or {}).get("summary", {}).get("sq_too_far"):
            continue
        use[n] = (t, float(R["price_input"]), float(R["status_quo"]))
    if len(use) < 2:
        return None
    names = list(use)
    base = use[names[0]][0][["unit_id", "plot_id", "zone", "area_ac", "yield_obs"]].copy()
    for n in names:
        t = use[n][0][["unit_id", "rate", "y_sq"]].rename(columns={"rate": f"rate_{n}", "y_sq": f"ysq_{n}"})
        base = base.merge(t, on="unit_id", how="inner")
    if base.empty:
        return None
    y_cf = base.yield_obs + sum(base[f"ysq_{n}"] - base.yield_obs for n in names)
    base["y_sq"] = y_cf
    base["nr_trial"] = pc * base.yield_obs - sum(use[n][1] * base[f"rate_{n}"] for n in names)
    base["nr_sq"] = pc * y_cf - sum(use[n][1] * use[n][2] for n in names)
    w = base.area_ac
    summ = {c: float(np.average(base[c], weights=w)) for c in ("nr_trial", "nr_sq", "yield_obs", "y_sq")}
    summ["trial_minus_sq"] = summ["nr_trial"] - summ["nr_sq"]
    summ["area_ac"] = float(w.sum())
    summ["n_units"] = int(len(base))
    byz = (base.groupby("zone")
           .apply(lambda q: pd.Series({c: np.average(q[c], weights=q.area_ac) for c in ("nr_trial", "nr_sq", "yield_obs", "y_sq")}
                                      | {"area_ac": q.area_ac.sum()}), include_groups=False)
           .reset_index())
    left_out = [n for n in tables if n not in use]
    # by_zone follows the analysis zones of the first input (each input merges small zones on its own)
    return dict(inputs=names, left_out=left_out, summary=summ, by_zone=byz.to_dict("records"), zones_of=names[0])


def _shp_field_names(columns, limit=10):
    """{column: shapefile field name}: unique names of at most `limit` characters (dBASE). A name whose cut form is
    already taken gets an index suffix (Flat_input, Flat_inpu1, ...); dBASE compares names without case."""
    used, out = set(), {}
    for c in columns:
        c = str(c)
        name, i = c[:limit], 1
        while name.lower() in used:
            s = str(i)
            name, i = c[:limit - len(s)] + s, i + 1
        used.add(name.lower())
        out[c] = name
    return out


def _form_sensitivity(units, spec, W, pc, pi, sq, vr_cost):
    """The main functional form and the alternatives, each fitted as one curve and as a curve per zone: the other of
    quadratic / quadratic-plateau, and both of them when the main form is a straight line (chosen by the user, or
    the fallback of a 2-rate trial, where both usually fail and are reported as errors). The report builds its
    heading from the forms that were fitted. {form: dict(r_flat, gain_flat_vs_sq, basis, r_zone, gain_vr_vs_flat,
    curve_flat, grid)} or {form: dict(error=...)}.

    r_flat is the form's single-curve optimum (the rate the verdict would judge under that form). Its gain over the
    status quo is valued like the summary table's 'Best flat' row (summary_basis): on the curve-per-zone model of the
    SAME form on the field's zone acres (basis 'zone'), and on the single curve only when there is no zone model
    (basis 'pooled'). Valuing it on the single curve's own plot zone mix printed a second, different gain for the
    same rate and form a few rows below the summary table. gain_vr_vs_flat is the zone model's VR gain over its own
    flat optimum (decide), net of `vr_cost`."""
    others = {"linear": ("quadratic", "qplateau"), "qplateau": ("quadratic",)}.get(spec.form, ("qplateau",))
    forms = {}
    for form in (spec.form,) + others:
        try:
            sp2 = er.Spec(**{**spec.__dict__, "form": form})
            fp = er.fit(units, sp2, "pooled")
            dp = ee.analyse(fp, {None: 1.0}, pc, pi, sq, 0.0)
            dz, zerr = None, None
            if units.zone.nunique() > 1:
                try:
                    dz = ee.analyse(er.fit(units, sp2, "zone"), W, pc, pi, sq, vr_cost)
                except Exception as exc:                 # the form's zone fit failed: fall back to the single curve
                    zerr = str(exc)
            row = _flat_row(dz, dp["r_uniform"], pc, pi, sq) if dz is not None else None
            forms[form] = dict(r_flat=dp["r_uniform"],
                               gain_flat_vs_sq=(row or {}).get("gain_vs_sq") if dz is not None
                               else dp.get("gain_uniform_vs_sq"),
                               basis="zone" if dz is not None else "pooled",
                               r_zone=dz["r_zone"] if dz else None,
                               gain_vr_vs_flat=dz["gain_vr_vs_uniform"] if dz else None,
                               curve_flat=dp["curves"], grid=dp["grid"])
            if zerr:
                forms[form]["zone_error"] = zerr
        except Exception as exc:
            forms[form] = dict(error=str(exc))
    return forms


def _flat_row(dec, rate, pc, pi, sq=None):
    """Yield and partial net return of ONE flat rate on the basis of the decision `dec`: its own curves (one per
    analysis zone, or the single curve) weighted by its field weights. A rate equal to the status quo takes the
    decision's own status-quo figures, which economics.decide extrapolates along the curve when the status quo lies
    outside the trial range."""
    if rate is None:
        return None
    rate = float(rate)
    too_far = bool(dec.get("sq_too_far"))
    if sq is not None and abs(rate - float(sq)) < 1e-9 and too_far:
        # a status quo further than 25 % of the trial range outside it is not valued (economics.decide): no net
        # return and no gain over it, rather than figures read off a curve extrapolated that far
        return dict(rate=rate, nr=None, y=None, gain_vs_sq=None, too_far=True)
    if sq is not None and abs(rate - float(sq)) < 1e-9 and dec.get("nr_sq") is not None:
        row = dict(rate=rate, nr=float(dec["nr_sq"]), y=dec.get("y_sq"))
    else:
        g = np.asarray(dec["grid"], float)
        y = sum(float(w) * float(np.interp(rate, g, np.asarray(dec["curves"][str(z)], float)))
                for z, w in dec["weights"].items())
        row = dict(rate=rate, nr=float(pc * y - pi * rate), y=float(y))
    if too_far:
        row["gain_vs_sq"] = None
    elif dec.get("nr_sq") is not None:
        row["gain_vs_sq"] = row["nr"] - float(dec["nr_sq"])
    return row


def summary_basis(R, pc, pi):
    """Every strategy of the summary table on ONE basis: the main model of the decision (the curve per zone when the
    field has 2 or more analysis zones, else the single curve) on the field's zone acres.

    Why: the single-curve model's own net returns are averaged over the trial's analysis units, i.e. the zone mix of
    the PLOTS, while the zone model weights the zones by the FIELD acres. Printing levels from both side by side made
    the rows impossible to subtract (VR could look $100/ac worse than the status quo while its gain over the status
    quo was positive). Here the best flat rate (the single-curve optimum the verdict uses) and the recommended flat
    rate are only CHOSEN by their own model and then valued on the main model, like the status quo and VR, so every
    difference between two rows is the gain between them."""
    dec, sq = R["decision"], R.get("status_quo")
    vr = dict(rates=dec["r_zone"], nr=float(dec["nr_vr"]), y=dec.get("y_vr"), gain_vs_sq=dec.get("gain_vr_vs_sq"))
    return dict(model=R.get("model_kind") or ("zone" if len(dec["r_zone"]) > 1 else "pooled"),
                status_quo=_flat_row(dec, sq, pc, pi, sq) if sq is not None else None,
                best_flat=_flat_row(dec, (R.get("pooled") or dec)["r_uniform"], pc, pi, sq),
                model_flat=dict(rate=float(dec["r_uniform"]), nr=float(dec["nr_uniform"]), y=dec.get("y_uniform"),
                                gain_vs_sq=dec.get("gain_uniform_vs_sq")),
                vr=vr, recommended_flat=None)


def verdict(R, min_gain=3.0):
    """A plain-language recommendation, in two parts.

    1. Variable rate vs the flat optimum of the zone model (the zone model's own best flat rate: the VR gains of the
       decision and the cross-fit are measured against it). In-sample, zone rates can never lose to one flat rate
       BEFORE the VR cost (they are chosen on the same data), so the in-sample gain and its bootstrap say how big the
       gain might be, not whether it exists.
       Without a cross-fit (0 repeats, or every split skipped: the reason comes from the skip counts in
       R['crossfit_skips'], cf_missing_txt) variable rate is NOT tested, and the verdict says so instead of claiming
       that the zones do not pay. With one analysis zone the headline tells a field that HAS one zone (a one-zone map,
       no NDVI: nothing to compare) from one whose zones were merged for lack of data (VR could not be evaluated).
       A cross-fitted gain within 1 $/ac of the minimum worthwhile gain is borderline (Pannell 2006: payoff functions
       are flat near the optimum) only where that margin could flip the advice: VR vs flat when the in-sample and the
       cross-fitted VR gains agree in sign; best flat vs status quo when the switch's evidence holds (the field
       responds to the rate, or the best flat rate is lower than the status quo and was cross-fitted) and the
       in-sample and cross-fitted gains of the best flat rate are both positive (otherwise the response test or the
       in-sample gate decides, whatever the margin). The reason then says the two choices are economically
       equivalent, verdict['borderline'] is True and verdict['borderline_on'] lists the comparisons.
       The evidence is the zone x rate test and the cross-fitted gain (rates chosen on half of the plots, judged on the
       other half: sample splitting as in Chernozhukov et al. 2018 and Athey & Imbens 2016, because the in-sample
       value of VR is biased upward by construction, Mieno, Li & Bullock 2024), which must also reach the minimum
       worthwhile gain: payoff functions are
       flat near the optimum (Pannell 2006) and small consistent gains appear by chance (synthetic control in
       tests/test_engine.py).
    2. Which flat rate. The best flat rate replaces the status quo when the cross-fitted gain of the best flat rate
       over the status quo is worthwhile (>= the minimum, positive in >= 70 % of the repeats) AND its in-sample gain
       printed in the advice (the summary basis: the main model on the field's zone acres) is positive. Both gains
       are on that basis (the flat rate chosen by the single curve, valued with the main model over the field's zone
       acres; the cross-fit does so on held-out plots, R['crossfit_flat']), but an in-sample and a cross-fitted gain
       can still have opposite signs; advising a switch next to a negative gain would contradict itself.
       The direction matters. A HIGHER rate must buy its extra yield, so it also needs the field to respond to the
       rate (pooled test p < 0.10): a gain made of a chance slope is not advised. A LOWER rate saves input with
       certainty and loses only the yield the response gives up, so the response test is NOT required for it: when
       the yield does not respond, the H0 the test cannot reject is exactly the case in which the cut pays (Trial
       22 seed: 101 instead of 134, p = 0.55, +17.5 $/ac cross-fitted in 93 % of the splits; the old rule kept 134
       and printed the opposite evidence beside it). The reason then says how much of the gain is the input saving.
       Otherwise the advice is to keep the status quo (the trial could not tell them apart)."""
    d = R["decision"]
    cfs = R.get("crossfit") or {}
    cf = cfs.get("gain_vr_vs_uniform") if len(d["r_zone"]) > 1 else None
    cfu = (R.get("crossfit_flat") or cfs).get("gain_uniform_vs_sq")
    zt = (R["models"].get("zone") or {}).get("tests", {}).get("zone_x_rate")
    rt = (R["models"].get("pooled") or {}).get("tests", {}).get("rate")
    pooled = R.get("pooled") or d
    g_in = d["gain_vr_vs_uniform"]
    sq = R.get("status_quo")
    reasons = []
    # one analysis zone: merged for lack of data (merge notes, or 2 or more original zones mapped together), or the
    # field had a single zone to begin with (a one-zone map, no NDVI and no zone map)
    zmap_ = R.get("zone_map") or {}
    merged = bool(R.get("zone_notes")) or len({str(k) for k in zmap_}) > 1
    # ------------------------------------------------ 1 VR
    if len(d["r_zone"]) < 2:
        reasons.append("a single analysis zone: no zone had the data for a curve of its own (the zones were merged)"
                       if merged else "the field has one zone: there is no zone strategy to compare")
    if zt is not None:
        reasons.append(f"zone x rate test p = {zt['p']:.3f} (H0: the same response in every zone)")
    if cf is not None:
        reasons.append(f"cross-fitted VR gain over the flat optimum of the zone model {cf['mean']:+.2f} $/ac "
                       f"({100 * cf['share_positive']:.0f} % of {cf['n']} repeated 2-fold splits positive)")
    if len(d["r_zone"]) > 1:
        reasons.append(f"in-sample VR gain over the flat optimum of the zone model {g_in:+.2f} $/ac "
                       "(optimistic by construction)")
    ok_cf = cf is not None and cf["mean"] >= min_gain and cf["share_positive"] >= 0.70
    ok_lean = cf is not None and cf["mean"] >= min(1.0, min_gain / 3) and cf["share_positive"] >= 0.60
    ok_test = zt is not None and zt["p"] < 0.10
    if cf is not None and 0 < cf["mean"] < min_gain:
        reasons.append(f"the cross-fitted VR gain is below the minimum worthwhile gain of {min_gain:.2f} $/ac")
    # a zone model whose VR gain was never cross-fitted: 'crossfit' missing = not run (0 repeats); present but without
    # the VR gain = every split skipped (a half without the design rates of a zone, or no rate overlap)
    cf_missing_why = None
    if len(d["r_zone"]) > 1 and cf is None:
        cf_missing_why = ("the cross-fit was not run (0 cross-fit repeats)" if "crossfit" not in R
                          else cf_missing_txt(R.get("crossfit_skips")))
        reasons.append(f"{cf_missing_why}: the value of variable rate was not tested")
    border = []
    # borderline only when the in-sample and cross-fitted VR gains agree in sign: when they disagree the evidence is
    # contradictory, not two economically equivalent choices
    if cf is not None and abs(float(cf["mean"]) - min_gain) <= 1.0 and (g_in > 0) == (float(cf["mean"]) > 0):
        border.append("VR vs flat")
    # ------------------------------------------------ 2 flat rate
    flat = pooled["r_uniform"]
    corner = any(pooled.get("corner", {}).values())
    responds = rt is not None and rt["p"] < 0.10
    if rt is not None:
        reasons.append(f"field response to the rate p = {rt['p']:.3f}")
    too_far = sq is not None and bool(d.get("sq_too_far"))
    if too_far:
        reasons.append(f"the status quo {sq:g} lies more than 25 % of the trial range outside it ({d['grid'][0]:.0f} to "
                       f"{d['grid'][-1]:.0f}): the trial cannot value it, so no gain over the status quo is computed")
    elif sq is not None and d.get("sq_extrapolated"):
        reasons.append(f"the status quo {sq:g} lies outside the trial range ({d['grid'][0]:.0f} to {d['grid'][-1]:.0f}): "
                       f"{sq_extrapolation_txt(R)}, so the gains over it are less certain")
    if sq is None:
        flat_rec = flat
        flat_txt = f"Best flat rate {flat:.0f}"
        change = True
    elif too_far:
        # there is no gain over the status quo to test: the best flat rate is advised only when the field responds
        # to the rate inside the trial range; otherwise the trial has nothing to say against the status quo
        if responds:
            flat_rec, change = flat, abs(flat - sq) > 1e-9
            flat_txt = (f"Best flat rate {flat:.0f} inside the trial range (the status quo {sq:.0f} lies too far outside "
                        "it for the gain to be computed)")
        else:
            flat_rec, change = sq, False
            p_txt = f" (p = {rt['p']:.2f})" if rt is not None else ""
            if flat > sq:
                flat_txt = (f"Keep the status quo {sq:.0f}: the yield did not respond clearly to the rate{p_txt}, and "
                            "the status quo lies too far outside the trial range to be compared with the trial rates")
            else:
                # a lower rate is judged on its gain over the status quo, which cannot be computed this far out
                flat_txt = (f"Keep the status quo {sq:.0f}: it lies too far outside the trial range for the gain of the "
                            f"lower rate {flat:.0f} to be computed, so the trial cannot value the cut{p_txt}")
    else:
        # in-sample gain of the best flat rate over the status quo on the same basis as the summary table (the main
        # model on the field's zone acres; summary_basis), not on the trial plots' zone mix
        bf = (R.get("summary_basis") or {}).get("best_flat") or {}
        g_flat = bf.get("gain_vs_sq")
        if g_flat is None:
            g_flat = pooled.get("gain_uniform_vs_sq", 0.0) or 0.0
        if cfu is not None:
            reasons.append(f"cross-fitted gain of the best flat rate over the status quo {cfu['mean']:+.2f} $/ac "
                           f"({100 * cfu['share_positive']:.0f} % of {cfu.get('n', '?')} repeated splits positive)")
            worth_cf = cfu["mean"] >= min_gain and cfu["share_positive"] >= 0.70
        else:
            worth_cf = g_flat >= min_gain
        # a higher rate has to buy its extra yield (the response test); a lower one saves input with certainty and
        # is judged on its gains alone (see the docstring)
        needs_response = flat > sq + 1e-9
        # a lower rate without a response test still needs the cross-fit to vouch for it: with 0 cross-fit repeats
        # the in-sample gain alone would advise the cut, and the reason could not claim the out-of-sample check
        evidence_ok = responds or (not needs_response and cfu is not None)
        if cfu is not None and (abs(float(cfu["mean"]) - min_gain) <= 1.0 and evidence_ok and g_flat > 0
                                and float(cfu["mean"]) > 0):
            # only when every other condition of the switch holds (and the cross-fitted gain is positive too, so
            # the two gains agree in sign) can a margin within 1 $/ac flip the advice; otherwise 'economically
            # equivalent; either is defensible' sat next to a negative in-sample gain or a field that did not
            # respond to the rate
            border.append("best flat vs status quo")
        # the gain printed next to the advice must agree with it (see the docstring)
        worth = worth_cf and g_flat > 0
        change = evidence_ok and worth and abs(flat - sq) > 1e-9
        if change:
            flat_rec = flat
            flat_txt = f"Best flat rate {flat:.0f} instead of the status quo {sq:.0f} ({g_flat:+.2f} $/ac)"
            if not needs_response and not responds and rt is not None:
                # the gain of a lower rate = the certain input saving + what the fitted curve credits (or charges)
                # for the yield at the lower rate; the text follows the numbers, which the curve can put either way
                pi_ = R.get("price_input")
                txt_ = f"the yield did not respond clearly to the rate (p = {rt['p']:.2f})"
                if pi_ is not None:
                    sav = pi_ * (sq - flat)
                    yld = g_flat - sav
                    txt_ += (f"; the input saving is {sav:+.2f} of the {g_flat:+.2f} $/ac gain of the lower rate, and "
                             + (f"the fitted curve credits {yld:+.2f} $/ac of yield to it" if yld > 0 else
                                f"the fitted curve charges {-yld:.2f} $/ac of yield against it, which the saving more "
                                "than covers"))
                txt_ += (": a lower rate needs no response test, and the cross-fitted gain vouches for it. Under-application "
                         "costs more than over-application when the true response is steeper than estimated (Babcock 1992; "
                         "Rajsic & Weersink 2008), so repeat the trial before cutting further")
                reasons.append(txt_)
        else:
            flat_rec = sq
            if needs_response and not responds:
                why = ((f"the yield did not respond clearly to the rate (p = {rt['p']:.2f})" if rt is not None
                        else "the response test is not available") + ", and a higher rate has to buy its extra yield")
                if cfu is not None:
                    why += (f" (the best flat rate {flat:.0f} gains {g_flat:+.2f} $/ac in-sample and {cfu['mean']:+.2f} "
                            f"cross-fitted, {100 * cfu['share_positive']:.0f} % of the splits positive, but a gain made "
                            "of an unproven slope is not advised)")
            elif not needs_response and cfu is None and worth_cf and g_flat > 0:
                why = (f"the best flat rate {flat:.0f} gains {g_flat:+.2f} $/ac in-sample, but no cross-fit of the flat "
                       "rate was run (0 repeats) to vouch for the lower rate")
            elif evidence_ok and worth_cf and g_flat <= 0 and cfu is not None:
                why = (f"the best flat rate {flat:.0f} gains {cfu['mean']:+.2f} $/ac cross-fitted but "
                       f"{g_flat:+.2f} $/ac in-sample (both valued on the main model over the field's zone acres): "
                       "the two do not agree")
            else:
                why = (f"the best flat rate {flat:.0f} gains only {g_flat:+.2f} $/ac in-sample"
                       + (f" and {cfu['mean']:+.2f} cross-fitted" if cfu is not None else ""))
            flat_txt = f"Keep the status quo {sq:.0f}: {why}"
    if corner and change:
        flat_txt += " (at the edge of the trial range: the true optimum may lie outside it)"
    if border:
        reasons.append(f"borderline: within 1 $/ac of the minimum worthwhile gain of {min_gain:.2f} $/ac; the two "
                       "choices are economically equivalent")
    if ok_cf and ok_test:
        rates = ", ".join(f"zone {z}: {r:.0f}" for z, r in d["r_zone"].items())
        level, text = "vr", f"Variable rate by zone is supported ({rates})."
        if "VR vs flat" in border:
            text = text[:-1] + ", but only just: the cross-fitted gain is within 1 $/ac of the minimum worthwhile gain."
        # The flat advice (and why a best flat rate with a positive cross-fitted gain is not taken) was dropped here,
        # while the summary table still prints 'Recommended flat (verdict)' next to that cross-fitted gain: the
        # reader saw a contradiction with no explanation. It goes to the reasons, as the advice for staying flat.
        adv = flat_txt[:1].lower() + flat_txt[1:]
        if adv.startswith("best flat rate"):
            adv = "use the " + adv
        reasons.append(f"if the field stays at one flat rate, {adv}")
    elif ok_lean and (ok_test or ok_cf):
        level, text = "lean_vr", ("Variable rate may pay a little, but the evidence is weak or the gain small: stay flat "
                                  "and repeat the trial before relying on the zone rates. " + flat_txt + ".")
    elif len(d["r_zone"]) < 2:
        # one analysis zone: there is no zone model, no zone x rate test and no VR cross-fit, so the zones were never
        # evaluated and the headline must not claim a tested negative result. A field with ONE zone to begin with
        # lacks nothing; blaming the data there misattributed the cause.
        if merged:
            level, text = "flat", ("Flat rate: no zone had the data for a curve of its own (the zones were merged "
                                   "into one), so variable rate could not be evaluated. " + flat_txt + ".")
        else:
            level, text = "flat", ("Flat rate: the field has one zone, so there is no variable-rate strategy to "
                                   "compare. " + flat_txt + ".")
    elif cf_missing_why:
        # no cross-fitted VR gain: a flat advice by default, never the tested negative claim of the branch below
        level, text = "flat", (f"Flat rate: {cf_missing_why}, so variable rate was not tested. " + flat_txt + ".")
    else:
        level, text = "flat", "Flat rate: the zones do not change the best rate enough to pay. " + flat_txt + "."
    zc = [str(z) for z, c in d.get("corner", {}).items() if c]
    if zc and level != "flat":
        reasons.append("zone optimum at the edge of the trial range in zone(s) " + ", ".join(zc))
    return dict(level=level, text=text, reasons=reasons, flat_rate=flat_rec, change_flat=bool(change),
                borderline=bool(border), borderline_on=border)


def run(cfg, workdir, log=print, progress=None, fetch=True):
    st = State(cfg, workdir, log)
    st.load_design()
    # a fetch made with another window, dates per season or boundary is fetched again (load_ndvi would refuse it)
    fetched = set((st.cfg["files"].get("ndvi_fetch_settings") or {}).get("years") or [])
    wanted = {int(y) for y in (st.cfg["ndvi"].get("years") or [])}
    if st.cfg["ndvi"].get("exclude_trial_year", True):
        wanted.discard(int(st.cfg["year"]))          # never fetched, so it must not trigger a fetch on every run
    wanted = {y for y in wanted if y >= 2017}         # Sentinel-2 L2A starts in 2017
    # a project fetched with the old code (no fetch settings, no image list) is fetched again too: load_ndvi refuses it
    if fetch and st.cfg["ndvi"]["source"] == "fetch" and (not st.cfg["files"].get("ndvi_dir") or st.ndvi_fetch_changes()
                                                         or not st.cfg["files"].get("ndvi_fetch_settings")
                                                         or st.cfg["files"].get("ndvi_paths") is None
                                                         or not wanted <= {int(y) for y in fetched}):
        st.fetch_ndvi(progress)
    st.load_ndvi()
    st.make_zones()
    st.join()
    st.analyse(progress)
    st.write_outputs()
    from . import report
    report.figures(st)
    report.make_pdf(st)
    return st
