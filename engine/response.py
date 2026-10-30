# -*- coding: utf-8 -*-
"""Yield response to the tested rate, by NDVI zone or along the continuous NDVI.

Models (weighted by unit area; standard errors clustered by plot, the unit of randomisation):
    pooled       y = level_zone + f(r) + controls             -> one response for the field (the flat-rate optimum)
    zone         y = f_zone(r) + controls                     -> one curve per zone (the zone VR optimum)
    ndvi         y = f(r) + g(r) x NDVI + controls            -> the response changes continuously with NDVI
f is quadratic (default), linear, or quadratic-plateau (fitted by non-linear least squares; inference by bootstrap).
Fit.kind is always the STRUCTURE of the model ('pooled' | 'zone' | 'ndvi') and Fit.spec.form its functional form
('quadratic' | 'linear' | 'qplateau'): downstream code (weights per zone, bootstrap, cross-fit) branches on the
structure, so a plateau fit must keep it. The NDVI model has no plateau version: asked for one, fit() fits the
quadratic and says so in Fit.form_note.
A curve needs support: every zone of a zone model (the field, for the other models) must hold at least 3 distinct
DESIGN rates for a quadratic or a plateau and 2 for a line. With fewer, the intercept and the rate terms are collinear
inside the zone and least squares returns an arbitrary (pseudo-inverse) curvature, or one identified only by the
as-applied noise; fit() raises ValueError instead, so the bootstrap and the cross-fit skip such a resample.
When the model has an intercept per ORIGINAL zone (zone_level, see below), the rate terms are identified only by the
rate variation INSIDE each original zone: an original zone with a single design rate has it absorbed by its own
intercept. The design degrees of freedom of a curve are then the sum over its original zones of (distinct design
rates - 1), and that sum must also reach 2 for a quadratic or a plateau (1 for a line). Counting the rates of the
union only let a merged zone {90} + {150, 170} pass with 3 rates while one design degree of freedom informed slope
and curvature (its curvature swung from -9 to +3 with the seed and its optimum ran to the grid edge).
merge_small_zones applies the same rule for the same form (it takes the form), so the main fit passes it.
Controls: the other tested input (linear + quadratic, so that a confounded second factor does not leak into the
first; Trial 19), the combine shares (Machine is a fixed effect in every OFPE model), and a quadratic trend surface in
the coordinates (spatial field trend; Bullock et al. 2019 recommend accounting for spatial structure).
The other inputs enter the formulas under safe aliases (o_0, o_1, ... in the order of spec.others), never under their
own names: patsy parses a name such as 'urea N' or 'N-rate' as code and the fit crashed. scale['others'] and
spec.others_sq keep the real names; scale['other_cols'] maps each real name to its alias.
The yield levels come from the ORIGINAL zones ('zone_level', set by the project before the zones are merged) when the
units carry them, in BOTH models: merging zones for the curves must not also merge their yield levels, or a level
difference between two merged zones leaks into the slope again. merge_small_zones merges a zone exactly when it lacks
design rates, so inside a merged analysis zone the rates are unbalanced across the original zones by construction and
a level gap between them is confounded with the curvature (synthetic trial with one true response everywhere: the
merged zone's optimum moved from 130 to 90 or 152 with a 15 bu/ac gap). The pooled model therefore has C(zone_level)
intercepts and one curve; the zone model has C(zone_level) intercepts (zone_level is nested in the analysis zone) and
its rate terms per ANALYSIS zone (C(zone):rc ...), so each analysis zone still has one curve, read marginally over
that zone's units (each at its own original-zone level). The plateau fit mirrors this when it partials out the levels.
The quadratic-plateau join point is constrained to the supported design range: at or above the second-lowest design
rate. Below it the rising segment would be fitted to the as-applied scatter around ONE design rate, so its curvature
(and the optimum next to it) would be set by noise; above the highest rate it is allowed (no plateau seen).

The rate is centred and scaled inside the models (rc = (r - mid) / half-range) for numerical stability; every
prediction function takes and returns real units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import least_squares
from scipy.stats import f as f_dist

SPATIAL = "xs + ys + I(xs**2) + I(xs*ys) + I(ys**2)"


def is_rate_term(name):
    """True for a model term that involves the tested rate (rc, I(rc ** 2), and their interactions)."""
    return any(p == "rc" or p.startswith("I(rc") for p in name.split(":"))


def within_design_df(rates, groups):
    """Design degrees of freedom left for the rate terms when every group (original zone) has its own intercept:
    the sum over the groups of (distinct design rates in the group - 1). A group with one design rate (or none)
    adds nothing, because its intercept absorbs that rate."""
    r = pd.to_numeric(pd.Series(np.asarray(rates)), errors="coerce")
    k = r.groupby(np.asarray(groups)).nunique()
    return int(np.maximum(k.values - 1, 0).sum()) if len(k) else 0


def merge_small_zones(units, zone_ac, rate_col, min_units=20, min_rates=None, ndvi_means=None, form="quadratic"):
    """Zones without the support for a curve of their own are merged into the neighbouring zone (the adjacent NDVI
    class, the closer in mean NDVI when there are two) until every zone qualifies. A zone lacks support with fewer
    than `min_units` units, fewer than `need` distinct design rates, or fewer than `need - 1` design degrees of freedom
    INSIDE its original zones (within_design_df): the zone model fitted afterwards gives every original zone its own
    intercept, so only the rate variation inside each original zone identifies the curve (check_support uses the same
    rule, so the main fit passes). Returns (units with the analysis zone in `zone` and the original in `zone_orig`,
    field acres per analysis zone, mapping original -> analysis zone, notes).

    `need` follows the response `form` the zones are merged for (min_design_rates, as check_support): 3 design rates
    and 2 within-zone degrees of freedom for a quadratic or a plateau, 2 and 1 for a line. A fixed 3 made the merge
    stricter than the fit it protects: with form 'linear' on a 2-rate strip design every zone was merged into one, so
    the zone model, the zone x rate test and the VR cross-fit never ran although each zone identified its own line.
    `min_rates` is kept for callers that predate `form` and is not used: the count always follows the form, because
    a looser count lets through a zone that check_support then rejects (the main fit fails) and a stricter one (the
    old fixed 3 passed with form 'linear') merges zones the fit can use. A caller that stores the rule for the report
    should store min_design_rates(form)."""
    min_rates = min_design_rates("quadratic" if form == "qplateau" else (form or "quadratic"))
    u = units.copy()
    u["zone_orig"] = u["zone"]
    mapping = {int(z): int(z) for z in set(zone_ac) | set(u.zone.unique())}
    nm = {int(k): v for k, v in (ndvi_means or {}).items()}
    notes = []
    while True:
        cur = sorted(set(mapping.values()))
        if len(cur) <= 1:
            break
        sup = {}
        for z in cur:
            m = u.zone == z
            sup[z] = (int(m.sum()), int(u.loc[m, rate_col].nunique()),
                      within_design_df(u.loc[m, rate_col].values, u.loc[m, "zone_orig"].values))
        bad = [z for z in cur if sup[z][0] < min_units or sup[z][1] < min_rates or sup[z][2] < min_rates - 1]
        if not bad:
            break
        z = min(bad, key=lambda k: sup[k][0])
        i = cur.index(z)
        nb = [cur[j] for j in (i - 1, i + 1) if 0 <= j < len(cur)]
        if len(nb) == 2 and nm:
            tgt = min(nb, key=lambda k: abs(nm.get(k, 0) - nm.get(z, 0)))
        elif len(nb) == 2:
            tgt = max(nb, key=lambda k: sup[k][0])
        else:
            tgt = nb[0]
        why = ""
        if sup[z][0] >= min_units and sup[z][1] >= min_rates:
            # merged for the within-original-zone rule only: say so, the rate count alone looks sufficient
            why = (f"; only {sup[z][2]} design degree(s) of freedom inside its original zones, "
                   f"{min_rates - 1} needed")
        notes.append(f"zone {z} ({sup[z][0]} units, {sup[z][1]} rates{why}) merged into zone {tgt}")
        for k, v in list(mapping.items()):
            if v == z:
                mapping[k] = tgt
        u["zone"] = u.zone_orig.map(mapping)
    W = {}
    for k, a in zone_ac.items():
        t = mapping.get(int(k), int(k))
        W[t] = W.get(t, 0.0) + a
    return u, W, mapping, notes


@dataclass
class Spec:
    input: str                       # primary input name (column rate_<input>)
    others: list = field(default_factory=list)
    others_sq: dict = field(default_factory=dict)     # status quo of each other input (prediction point)
    form: str = "quadratic"          # quadratic | linear | qplateau
    machine: bool = True
    spatial: bool = True
    rate_source: str = "applied"     # applied | target


def prep(units, spec):
    d = units.copy()
    rcol = f"rate_{spec.input}" if spec.rate_source == "applied" else f"target_{spec.input}"
    d["r"] = pd.to_numeric(d[rcol], errors="coerce")
    d = d[np.isfinite(d.r) & np.isfinite(d["yield"])].copy()
    lo, hi = float(d.r.min()), float(d.r.max())
    mid, half = (lo + hi) / 2, max((hi - lo) / 2, 1e-6)
    d["rc"] = (d.r - mid) / half
    x0, y0 = d.x_m.mean(), d.y_m.mean()
    d["xs"], d["ys"] = (d.x_m - x0) / 500.0, (d.y_m - y0) / 500.0
    nm, ns = d.ndvi.mean(), d.ndvi.std()
    d["nz"] = ((d.ndvi - nm) / (ns if ns and ns > 0 else 1.0)).fillna(0.0)
    oth, ocols = {}, {}
    for i, o in enumerate(spec.others):
        c = f"rate_{o}" if spec.rate_source == "applied" else f"target_{o}"
        v = pd.to_numeric(d[c], errors="coerce")
        om, oh = float(v.mean()), max(float(v.std()), 1e-6)
        # a positional alias, not o_<name>: the column is referred to inside patsy formulas, where a name with a
        # space, a hyphen or a leading digit is parsed as code ('o_urea N' -> SyntaxError, 'o_N-rate' -> o_N - rate)
        alias = f"o_{i}"
        d[alias] = ((v - om) / oh).fillna(0.0)
        oth[o] = (om, oh)
        ocols[o] = alias
    # the combine shares (join.build_units: sh_<machine name>) get positional aliases m_<i> for the same reason as
    # the other inputs: the machine name is whatever the yield file holds ('X9 1100', 'S680.1', 'Combine-2'), and
    # written verbatim into the formula a space, a dot or a hyphen is parsed as code and the fit crashes
    mcols, mnames = [], {}
    for i, c in enumerate(sorted(c for c in d.columns if c.startswith("sh_"))):
        alias = f"m_{i}"
        d[alias] = d[c]
        mcols.append(alias)
        mnames[alias] = c[3:]
    scale = dict(mid=mid, half=half, lo=lo, hi=hi, x0=x0, y0=y0, ndvi_mean=nm, ndvi_sd=ns, others=oth,
                 other_cols=ocols, machines=mcols, machine_names=mnames)
    return d, scale


def other_col(scale, name):
    """The formula column (alias o_<i>) of the other input `name`."""
    return scale["other_cols"][name]


def level_col(d):
    """The pooled model's zone-level covariate: 'zone_level' (the original zones, before merge_small_zones) when the
    units carry it with more than one value and no gap, else 'zone' (the analysis zones)."""
    if "zone_level" in d.columns:
        zl = d["zone_level"]
        if zl.notna().all() and zl.nunique() > 1:
            return "zone_level"
    return "zone"


def zone_intercept_col(d):
    """The zone model's intercept factor: 'zone_level' (the original zones) when level_col accepts it and it is nested
    in the analysis zone (each original zone inside one analysis zone, as merge_small_zones guarantees), else 'zone'.
    Nesting matters because the rate terms stay per analysis zone: an original zone spread over two analysis zones
    would tie their intercepts together, which is not the model the curves are read from."""
    if level_col(d) == "zone_level" and int(d.groupby("zone_level")["zone"].nunique().max()) == 1:
        return "zone_level"
    return "zone"


def _controls(d, spec, scale):
    terms = []
    for o in spec.others:
        c = other_col(scale, o)
        terms += [c, f"I({c}**2)"]
    mc = [c for c in scale["machines"] if d[c].std() > 0][1:] if spec.machine else []
    terms += mc
    if spec.spatial and len(d) > 30:
        terms.append(SPATIAL)
    return terms


def _rate_terms(form):
    return ["rc"] if form == "linear" else ["rc", "I(rc**2)"]


def min_design_rates(form):
    """Distinct design rates a curve of this form needs: 3 for a quadratic or a plateau, 2 for a line."""
    return 2 if form == "linear" else 3


def check_support(units, spec, kind):
    """Raise ValueError when the data cannot identify the requested curve: a zone (kind 'zone') or the field (other
    kinds) with fewer distinct DESIGN rates (target_<input>; the analysed rate when there is no design column) than
    the form needs. merge_small_zones guarantees this for the full data; a bootstrap resample or a cross-fit half
    can lose it, and a rank-deficient fit there would be kept as a valid draw with an invented optimum.

    When the model gives every ORIGINAL zone its own intercept (zone model: zone_intercept_col is 'zone_level';
    pooled model: level_col is 'zone_level'), the rates of an original zone with a single design rate are absorbed by
    its intercept, so the rule counts what the model can use: the sum over the original zones of (distinct design
    rates - 1) (within_design_df) must also reach need - 1, i.e. 2 for a quadratic or a plateau and 1 for a line, per
    analysis zone (zone model) or for the field (pooled). The NDVI model has no zone intercepts and keeps the plain
    count."""
    need = min_design_rates("quadratic" if spec.form == "qplateau" else spec.form)
    tcol = f"target_{spec.input}"
    if tcol in units and pd.to_numeric(units[tcol], errors="coerce").notna().any():
        col = pd.to_numeric(units[tcol], errors="coerce")
    else:
        rcol = f"rate_{spec.input}" if spec.rate_source == "applied" else tcol
        col = pd.to_numeric(units[rcol], errors="coerce")
    # the intercept factor the fit will use: only original-zone intercepts change what identifies the curve
    nested = False
    if "zone_level" in units.columns:
        if kind == "zone":
            nested = zone_intercept_col(units) == "zone_level"
        elif kind == "pooled":
            nested = level_col(units) == "zone_level"
    if kind == "zone":
        n = col.groupby(units["zone"]).nunique()
        bad = {int(z): int(k) for z, k in n.items() if k < need}
        if bad:
            raise ValueError(f"{spec.form} response of {spec.input}: zone(s) "
                             + ", ".join(f"{z} ({k} design rate{'s' if k != 1 else ''})" for z, k in bad.items())
                             + f" need at least {need} distinct design rates for a curve of their own")
        if nested:
            # positional frame: a resample can carry a repeated index
            t = pd.DataFrame({"r": col.values, "zl": units["zone_level"].values, "z": units["zone"].values})
            dfw = {int(z): within_design_df(g["r"].values, g["zl"].values) for z, g in t.groupby("z")}
            badw = {z: k for z, k in dfw.items() if k < need - 1}
            if badw:
                raise ValueError(
                    f"{spec.form} response of {spec.input}: zone(s) "
                    + ", ".join(f"{z} ({k} design degree{'s' if k != 1 else ''} of freedom)" for z, k in badw.items())
                    + " inside their original zones; each original zone has its own yield level, so a curve needs "
                    + f"at least {need - 1} (the sum over the original zones of distinct design rates - 1)")
    elif col.nunique() < need:
        raise ValueError(f"{spec.form} response of {spec.input}: the field has {col.nunique()} distinct design rates, "
                         f"at least {need} are needed")
    elif nested:
        k = within_design_df(col.values, units["zone_level"].values)
        if k < need - 1:
            raise ValueError(f"{spec.form} response of {spec.input}: only {k} design degree(s) of freedom inside the "
                             f"zones (each zone has its own yield level), at least {need - 1} are needed")


@dataclass
class Fit:
    kind: str                        # model structure: pooled | zone | ndvi (the functional form is spec.form)
    spec: Spec
    scale: dict
    data: pd.DataFrame
    model: object = None
    params: dict = None
    zones: list = None               # the zones for kind 'zone', [None] otherwise
    tests: dict = None
    formula: str = ""
    form_note: str = ""              # why the fitted form differs from the requested one (NDVI model + plateau)

    # ----------------------------------------------------------- prediction (real units)
    def _rc(self, r):
        return (np.asarray(r, float) - self.scale["mid"]) / self.scale["half"]

    def _base_row(self, zone=None):
        d = self.data if zone is None else self.data[self.data.zone == zone]
        w = d.area_m2
        row = {"xs": float(np.average(d["xs"], weights=w)), "ys": float(np.average(d["ys"], weights=w)), "nz": 0.0}
        for c in self.scale["machines"]:
            row[c] = float(np.average(self.data[c], weights=self.data.area_m2))
        for o, (om, oh) in self.scale["others"].items():
            sq = self.spec.others_sq.get(o)
            row[other_col(self.scale, o)] = 0.0 if sq is None else (sq - om) / oh
        return row

    def curve(self, rgrid, zone=None, ndvi=None):
        """Predicted yield on `rgrid` (real rate units) for a zone (zone model), an NDVI value (ndvi model, the
        multi-year normalised NDVI) or the field (pooled). The prediction is averaged over the units (of the zone, or
        of the field) with their own controls - combine shares, position - and the rate set to each grid value
        (marginal standardisation), so the curve sits at the zone's real yield level; the other tested input is set
        to its status quo. A plateau fit has no linear model to predict with: its curve comes from _qp_curve, at the
        same level (see _fit_qplateau)."""
        rgrid = np.asarray(rgrid, float)
        if self.spec.form == "qplateau" and self.model is None:
            return self._qp_curve(rgrid, zone)
        d = self.data if (zone is None or self.kind != "zone") else self.data[self.data.zone == zone]
        w = d.area_m2.values / d.area_m2.values.sum()
        base = d.drop(columns=["geometry"], errors="ignore").copy()
        for o, (om, oh) in self.scale["others"].items():
            sq = self.spec.others_sq.get(o)
            if sq is not None:
                base[other_col(self.scale, o)] = (sq - om) / oh
        if self.kind == "ndvi" and ndvi is not None:
            sd = self.scale["ndvi_sd"] or 1.0
            base["nz"] = (float(ndvi) - self.scale["ndvi_mean"]) / sd
        rc = self._rc(rgrid)
        big = base.loc[base.index.repeat(len(rc))].copy()
        big["rc"] = np.tile(rc, len(base))
        p = np.asarray(self.model.predict(big), float).reshape(len(base), len(rc))
        return (w[:, None] * p).sum(axis=0)

    def slope_terms(self):
        return {k: v for k, v in self.params.items() if is_rate_term(k)}

    # ----------------------------------------------------------- quadratic plateau
    def _qp_curve(self, rgrid, zone):
        """Plateau curve in real units. Like curve(): a pooled fit gives the FIELD curve whatever zone is asked; a zone
        fit gives the zone's curve, or, with zone=None, the area-weighted average of the zone curves (the field)."""
        pars = self.params["zones"]
        x = self._rc(rgrid)

        def one(z):
            a, b, c, base = pars[z]
            xp = -b / (2 * c) if c < 0 else np.inf
            xx = np.minimum(x, xp)
            return a + b * xx + c * xx ** 2 + base
        if self.kind != "zone":
            return one(None)
        if zone is not None:
            return one(zone)
        aw = self.data.groupby("zone").area_m2.sum()
        return sum(aw[z] / aw.sum() * one(z) for z in self.zones)


def fit(units, spec, kind="zone"):
    """Fit the response model of structure `kind` ('pooled' | 'zone' | 'ndvi') and form spec.form. The returned Fit
    always has kind == `kind`. Raises ValueError when a zone (or the field) lacks the design rates for the curve."""
    if kind not in ("pooled", "zone", "ndvi"):
        raise ValueError(f"unknown model kind {kind!r} (pooled | zone | ndvi)")
    note = ""
    if spec.form == "qplateau" and kind == "ndvi":
        # the plateau is fitted per zone / for the field; along the continuous NDVI the plateau would need a
        # non-linear NDVI interaction the trial cannot support, so the NDVI model stays quadratic
        spec = Spec(**{**spec.__dict__, "form": "quadratic"})
        note = ("quadratic-plateau requested: the continuous NDVI model has no plateau version, "
                "so the quadratic was fitted")
    d, scale = prep(units, spec)
    check_support(d, spec, kind)
    ctrl = _controls(d, spec, scale)
    rt = _rate_terms(spec.form if spec.form != "qplateau" else "quadratic")
    if spec.form == "qplateau":
        return _fit_qplateau(d, spec, scale, kind, ctrl)
    if kind == "pooled":
        # one response for the whole field, but each zone keeps its own yield level: the level difference between
        # zones is a pre-treatment covariate, and leaving it out lets a chance imbalance of rates between zones bias
        # the slope (synthetic trial, seed 1: flat optimum 80 instead of 128). The ORIGINAL zones carry the levels
        # (level_col): a zone merged into its neighbour for lack of design rates still has its own yield level.
        lc = level_col(d)
        rhs = ([f"C({lc})"] if d[lc].nunique() > 1 else []) + rt
    elif kind == "zone":
        # intercepts per ORIGINAL zone (nested in the analysis zone; see the module docstring), curves per analysis
        # zone: without the original levels a merged zone's level gap is fitted as curvature and sets its optimum.
        # No global intercept: the intercept factor carries one level per zone. patsy codes C(zone):rc in full (one
        # slope per analysis zone) because no bare rc term is in the model.
        ic = zone_intercept_col(d)
        rhs = [f"0 + C({ic})"] + [f"C(zone):{t}" for t in rt]
    else:                                                           # ndvi
        rhs = rt + ["nz"] + [f"{t}:nz" for t in rt]
    formula = "Q('yield') ~ " + " + ".join(rhs + ctrl)
    m = smf.wls(formula, data=d, weights=d.area_m2).fit(cov_type="cluster", cov_kwds={"groups": d.plot_id})
    f = Fit(kind=kind, spec=spec, scale=scale, data=d, model=m, params=dict(m.params), formula=formula,
            zones=sorted(d.zone.unique().tolist()) if kind == "zone" else [None], form_note=note)
    f.tests = _tests(f)
    return f


def _wald(m, cons, G):
    """Wald test of the constraints `cons` with the cluster-robust covariance, judged against F(h, G - 1) instead of
    chi2(h): the clusters are the plots (G of them), and with few clusters the chi-square reference over-rejects
    (Cameron & Miller 2015, sect. VI: G = 6 at a nominal 0.05 rejects 0.12-0.21); F(h, G - 1) is their recommended
    small-sample reference, and it tends to chi2(h) / h as G grows, so nothing changes on large trials."""
    w = m.wald_test(", ".join(cons), scalar=True)
    h = len(cons)
    stat = float(w.statistic)
    df2 = max(int(G) - 1, 1)
    return dict(stat=stat, df=h, df2=df2, p=float(f_dist.sf(stat / h, h, df2)), p_chi2=float(w.pvalue),
                reference=f"F({h}, {df2}); clusters = plots")


def _tests(f):
    m, names = f.model, list(f.model.params.index)
    G = int(f.data.plot_id.nunique())
    out = dict(n_units=int(m.nobs), n_plots=G, r2=float(m.rsquared), aic=float(m.aic), bic=float(m.bic))
    try:
        if f.kind == "zone" and len(f.zones) > 1:
            # H0: the rate terms are equal in every zone (same response everywhere)
            rate_terms = [n for n in names if n.startswith("C(zone)") and is_rate_term(n)]
            by_term = {}
            for n in rate_terms:
                t = n.split(":", 1)[1]
                by_term.setdefault(t, []).append(n)
            cons = [f"{v[0]} = {w}" for v in by_term.values() for w in v[1:]]
            if cons:
                out["zone_x_rate"] = dict(**_wald(m, cons, G), h0="the response to the rate is the same in every zone")
        if f.kind == "ndvi":
            cons = [n for n in names if ":nz" in n and is_rate_term(n)]
            if cons:
                out["ndvi_x_rate"] = dict(**_wald(m, [f"{c} = 0" for c in cons], G),
                                          h0="the response to the rate does not change with NDVI")
        if f.kind == "pooled":
            cons = [n for n in names if is_rate_term(n)]
            out["rate"] = dict(**_wald(m, [f"{c} = 0" for c in cons], G), h0="no response to the rate")
    except Exception as exc:                                         # pragma: no cover
        out["test_error"] = str(exc)
    return out


def _fit_qplateau(d, spec, scale, kind, ctrl):
    """Quadratic-plateau per zone (kind 'zone') or for the field (kind 'pooled'), returned as a Fit of that kind.

    The controls (and, for the pooled fit, the zone levels) are partialled out first with the quadratic model of the
    same kind, so the plateau is fitted on yields moved to one reference point (field-mean position and combine
    shares, the other input at its status quo, the first zone's level for the pooled fit). For the zone fit whose
    quadratic has original-zone intercepts (zone_intercept_col), every unit is moved to the first original zone of its
    ANALYSIS zone, so a level gap between the original zones of a merged zone is removed before the plateau is fitted
    (it would otherwise be fitted as curvature, as in the quadratic zone model). That reference is not a
    real yield level: the pooled curve would sit at the first zone's level (about 20 bu/ac low on a 3-zone synthetic
    field) and understate every absolute net return. Because the controls and the zone levels enter the quadratic
    model additively (they do not interact with the rate), the gap between the marginally standardised prediction
    used by Fit.curve (each unit at its own controls and zone, the other input at its status quo) and the reference
    prediction does not depend on the rate; its area-weighted mean over the units (of the zone, or of the field) is
    stored as the curve's `base` and added back, so plateau and quadratic curves sit at the same level.

    The plateau is parameterised by its level P, its join xp and its curvature c < 0 (y = P + c (min(x, xp) - xp)^2,
    stored as a + b x + c x^2 with b = -2 c xp and a = P + c xp^2), and xp is bounded below by the second-lowest
    design rate of the zone (the field): with only the lowest design rate on the rising segment, least squares put the
    join inside the gap between the two lowest rates with a curvature of -20 to -700 in rc units, fitted to the
    as-applied scatter of one rate, and corner optima followed. For a given xp the model is linear in (P, c), so xp
    is profiled on a grid and the best point polished with least_squares: the residual is not smooth in xp and a
    single local start often stopped in that gap. Zones whose join sits at the bound are listed in Fit.form_note."""
    qspec = Spec(**{**spec.__dict__, "form": "quadratic"})
    base = fit(d.drop(columns=["rc", "xs", "ys", "nz"], errors="ignore"), qspec, kind)
    # remove the control contribution (everything that is not zone / rate) with the quadratic fit
    full = np.asarray(base.model.predict(d), float)
    nocontrol = d.copy()
    for k, v in base._base_row().items():
        if k in nocontrol and k not in ("nz",):
            nocontrol[k] = v
    lc = level_col(d)
    if kind == "pooled" and d[lc].nunique() > 1:
        nocontrol[lc] = sorted(d[lc].unique())[0]             # the zone levels are removed with the controls
    elif kind == "zone" and zone_intercept_col(d) == "zone_level":
        # the original-zone levels inside each analysis zone are removed too (the reference is the zone's first
        # original zone); the analysis zone's own level comes back through `shift` below, averaged over its units
        first = d.groupby("zone")["zone_level"].transform(lambda v: sorted(v.unique())[0])
        nocontrol["zone_level"] = first.values
    ref = np.asarray(base.model.predict(nocontrol), float)
    adj = d["yield"].values - (full - ref)
    # the level Fit.curve would give: each unit at its own controls and zone, the other input at its status quo
    atsq = d.copy()
    for o, (om, oh) in scale["others"].items():
        sq = spec.others_sq.get(o)
        if sq is not None:
            atsq[other_col(scale, o)] = (sq - om) / oh
    shift = np.asarray(base.model.predict(atsq), float) - ref
    zones = sorted(d.zone.unique().tolist()) if kind == "zone" else [None]
    pars = {}
    at_bound = []
    w = np.sqrt(d.area_m2.values / d.area_m2.mean())
    tcol = f"target_{spec.input}"
    design = pd.to_numeric(d[tcol], errors="coerce") if tcol in d.columns else None
    if design is None or design.isna().all():
        design = d.r                                  # no design column: the analysed rate (as check_support)
    for z in zones:
        m = (d.zone == z).values if z is not None else np.ones(len(d), bool)
        x, y, ww = d.rc.values[m], adj[m], w[m]
        level = float(np.average(shift[m], weights=d.area_m2.values[m]))
        sol, x2 = _plateau_one(x, y, ww, np.sort(design[m].dropna().unique()), scale)
        P_, xp, c = sol
        pars[z] = (P_ + c * xp ** 2, -2.0 * c * xp, c, level)
        if xp <= x2 + 1e-6:
            at_bound.append((z, x2 * scale["half"] + scale["mid"]))
    note = ""
    if at_bound:
        where = ", ".join(("the field" if z is None else f"zone {z}") + f" ({r:.0f})" for z, r in at_bound)
        note = ("quadratic-plateau: the join point was held at the second-lowest design rate in " + where
                + ", the lowest join the design supports (below it the rising segment would rest on one design rate)")
    f = Fit(kind=kind, spec=spec, scale=scale, data=d, model=None, params={"zones": pars}, zones=zones,
            formula=("quadratic-plateau per zone" if kind == "zone" else "quadratic-plateau for the field")
            + " on control-adjusted yield", form_note=note)
    # the plateau has no closed-form tests: the rate and zone x rate tests of the quadratic model of the same
    # structure (the one that partialled out the controls) are reported, and labelled as such
    f.tests = {**(base.tests or {}), "n_units": int(len(d)), "n_plots": int(d.plot_id.nunique()),
               "tests_from": "quadratic model of the same structure (used to partial out the controls)"}
    return f


# upper bound of the plateau join in rc units (the trial range is [-1, 1]): far enough to mean "no plateau seen"
# (the curve is the rising parabola over the whole range), finite so that least_squares has a bounded box
QP_XP_MAX = 50.0


def _plateau_one(x, y, ww, design_rates, scale):
    """Least-squares plateau y = P + c (min(x, xp) - xp)^2 with c <= -1e-6 and xp in [x2, QP_XP_MAX], x2 = the rc of
    the second-lowest design rate (the lowest data point when there is only one). Returns ((P, xp, c), x2).

    For fixed xp the model is linear in (P, c): every xp of a grid over [x2, max(x)] and a few points beyond is
    solved in closed form (weighted least squares, c capped at -1e-6), and the best is polished with least_squares
    inside the same bounds."""
    if len(design_rates) >= 2:
        x2 = float((design_rates[1] - scale["mid"]) / scale["half"])
    else:
        x2 = float(np.min(x))
    x2 = min(x2, QP_XP_MAX - 1.0)
    W = ww ** 2
    top = max(float(np.max(x)), x2)
    cand = np.unique(np.concatenate([np.linspace(x2, top, 41), top + np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])]))
    cand = cand[(cand >= x2) & (cand <= QP_XP_MAX)]

    def lin(xp):
        t = (np.minimum(x, xp) - xp) ** 2
        tm, ym = np.average(t, weights=W), np.average(y, weights=W)
        vt = np.average((t - tm) ** 2, weights=W)
        c = np.average((t - tm) * (y - ym), weights=W) / vt if vt > 0 else -1e-6
        c = min(c, -1e-6)
        P_ = ym - c * tm
        return float(np.sum(W * (P_ + c * t - y) ** 2)), P_, c

    best = min((lin(xp) + (xp,) for xp in cand), key=lambda r: r[0])
    _, P0, c0, xp0 = best

    def res(p):
        P_, xp, c = p
        return (P_ + c * (np.minimum(x, xp) - xp) ** 2 - y) * ww
    lo_b, hi_b = [-np.inf, x2, -np.inf], [np.inf, QP_XP_MAX, -1e-6]
    p0 = [P0, float(np.clip(xp0, x2 + 1e-9, QP_XP_MAX - 1e-9)), min(c0, -2e-6)]
    try:
        sol = least_squares(res, p0, bounds=(lo_b, hi_b))
        out = tuple(float(v) for v in sol.x)
        if float(np.sum(res(out) ** 2)) > float(np.sum(res(p0) ** 2)):
            out = tuple(float(v) for v in p0)
    except ValueError:                                 # pragma: no cover - p0 is inside the box by construction
        out = tuple(float(v) for v in p0)
    return out, x2


# columns of rate_zone_table, also those of its empty result (no unit with a design rate)
RATE_ZONE_COLUMNS = ["zone", "design_rate", "applied", "yield_mean", "yield_se", "n_units", "n_plots", "area_ac",
                     "net_return"]


def rate_zone_table(units, spec, price_crop, price_input):
    """Design-based summary: mean yield and partial net return by design rate x zone (units), with the plot count
    and a plot-clustered standard error of the mean yield.

    Units without a design rate (NaN target_<input>: a treatment-only design, or a second input with an as-applied
    map but no design column) have no row here, because the table is about the DESIGN rates. When no unit has one
    the table is empty but keeps its columns, so callers (project, report, app) can test .empty instead of failing
    on a KeyError from sorting a column-less frame."""
    d, _ = prep(units, spec)
    tcol = f"target_{spec.input}"
    if tcol not in d.columns:
        return pd.DataFrame(columns=RATE_ZONE_COLUMNS)
    rows = []
    for (z, r), g in d.groupby(["zone", tcol]):
        pm = g.groupby("plot_id").apply(lambda q: np.average(q["yield"], weights=q.area_m2), include_groups=False)
        se = float(pm.std(ddof=1) / np.sqrt(len(pm))) if len(pm) > 1 else np.nan
        y = float(np.average(g["yield"], weights=g.area_m2))
        ra = float(np.average(g.r, weights=g.area_m2))
        rows.append(dict(zone=int(z), design_rate=float(r), applied=ra, yield_mean=y, yield_se=se, n_units=len(g),
                         n_plots=int(g.plot_id.nunique()), area_ac=float(g.area_ac.sum()),
                         net_return=price_crop * y - price_input * ra))
    if not rows:
        return pd.DataFrame(columns=RATE_ZONE_COLUMNS)
    return pd.DataFrame(rows, columns=RATE_ZONE_COLUMNS).sort_values(["zone", "design_rate"]).reset_index(drop=True)
