# -*- coding: utf-8 -*-
"""Variable rate vs flat rate.

For the tested input with price p_i per rate unit and the crop at p_c per bushel, on a grid of rates INSIDE the trial
range (no extrapolation; Mieno et al. 2024 clip corner solutions to the trial range):
    NR_z(r)   = p_c * yhat_z(r) - p_i * r                      partial net return of zone z
    r*_z      = argmax NR_z                                    the zone (VR) optimum
    r_u       = argmax sum_z w_z NR_z(r)                       the best FLAT rate for the whole field (uniform optimum)
    r_sq      = the grower's flat rate (status quo)
with w_z = the share of the FIELD (boundary) in zone z, not of the plots, because the prescription is for the field.
Gains, $/ac of field:
    VR vs status quo      sum_z w_z [NR_z(r*_z) - NR_z(r_sq)] - VR cost
    VR vs best flat       sum_z w_z [NR_z(r*_z) - NR_z(r_u)]  - VR cost      <- the value of the zone information
    best flat vs SQ       sum_z w_z [NR_z(r_u)  - NR_z(r_sq)]
The value of VR estimated in-sample is optimistic (the same data choose and judge the zone rates; Mieno et al. 2024
show GWR-based VR value is overstated), so the decision rests on:
    * a plot-cluster bootstrap (resample plots, refit, recompute everything): median and 90 % interval of each gain
      and of the optima, P(gain > 0);
    * a repeated two-fold plot cross-fit: the zone rates and the flat rate are chosen on one half of the plots and
      their net returns judged by the model fitted on the other half (out-of-sample value of VR).
The continuous NDVI model gives r*(NDVI) per pixel: a smooth prescription, judged the same way.

The status quo is the grower's rate and may lie outside the trial range. The optima never leave the range, but the
status quo is priced where it is: its input cost is always p_i * r_sq (clamping the whole net return to the edge of
the grid would charge the input at the edge rate and drop p_i * (r_sq - edge) from every gain over the status quo),
and its yield is extrapolated LINEARLY from the nearest edge with the curve's slope there, floored at 0 (flagged
sq_in_range False, sq_extrapolated True). A plateau fit is extended the same way, with its one-sided slope at the
nearest edge, never a negative one: above the range that slope is 0 when the plateau is reached inside the range and
the supported rising slope when the join lies beyond the highest rate ("no plateau seen"); below the range it is the
slope of the rising segment at the lowest rate, which the two lowest design rates identify (the join is bounded at
or above the second-lowest design rate, response._plateau_one). Holding the edge value on one side credited the
status quo with the yield of the edge rate for less (or more) input, so it biased every gain over it, and against
the quadratic of the same data it reported an extrapolation rule as a difference between the two forms. The fitted
curve itself is never followed
outside the data: a quadratic's curvature, or a plateau's rising segment, continued 30 lb/ac below the lowest rate gave
yields of -250 bu/ac and "gains" over the status quo of thousands of $/ac. Even the linear extension is a guess, so
when the status quo lies more than SQ_MAX_OUT (25 %) of the trial range outside it, nothing is priced at it: its
yield, net return and every gain over it are None and sq_too_far is True (sq_extrapolated False).

The cross-fit judges both halves on their COMMON support (the overlap of the two halves' applied-rate ranges): the
training optima are clipped into it before the held-out curve prices them, so a held-out plateau or quadratic is
never extrapolated beyond its own data, where one steep fold dominated the per-repeat mean and share_positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import response as er


def rate_grid(fit, n=161, lo=None, hi=None):
    lo = fit.scale["lo"] if lo is None else lo
    hi = fit.scale["hi"] if hi is None else hi
    return np.linspace(lo, hi, n)


def zone_curves(fit, grid):
    """{zone: yhat(grid)} for a zone model, {None: yhat(grid)} for the field (pooled or NDVI model at its mean)."""
    if fit.kind == "zone" and fit.zones and fit.zones[0] is not None:
        return {z: fit.curve(grid, zone=z) for z in fit.zones}
    return {None: fit.curve(grid)}


# a status quo further than this share of the trial range outside it is not priced (see the module docstring)
SQ_MAX_OUT = 0.25


def sq_position(sq, lo, hi):
    """(inside, too_far) of the rate `sq` against the trial range [lo, hi]: too_far when it lies more than
    SQ_MAX_OUT of the range outside it."""
    sq, lo, hi = float(sq), float(lo), float(hi)
    out = max(lo - sq, sq - hi, 0.0)
    return out == 0.0, out > SQ_MAX_OUT * max(hi - lo, 1e-9)


def is_plateau(fit):
    """True for a quadratic-plateau fit (its curve has no linear model)."""
    return getattr(fit.spec, "form", None) == "qplateau" and fit.model is None


def bounded_curve(curve_fn, rates, lo, hi, plateau=False):
    """Yield of `curve_fn` (real rates -> yields) at `rates`, followed only inside [lo, hi]. Outside, the yield is
    extended linearly from the nearest edge with the one-sided slope of the curve at that edge and floored at 0.
    A plateau curve (plateau True) is extended the same way but never with a negative slope (a plateau does not
    fall): above the range its slope at hi is 0 when the plateau is reached inside the range (so nothing changes)
    and the supported rising slope when the join lies beyond the highest rate; below the range the slope of its
    rising segment at lo, identified by the two lowest design rates. Holding the edge value below the range instead
    gave a status quo under the trial the yield of the lowest rate for less input, so it could never lose to that
    rate, and the quadratic of the same data (which does slope) differed from it by the rule, not by the form.
    Rates inside the range are the curve itself."""
    r = np.atleast_1d(np.asarray(rates, float))
    lo, hi = float(lo), float(hi)
    y = np.asarray(curve_fn(np.clip(r, lo, hi)), float).copy()
    out = (r < lo) | (r > hi)
    if out.any() and hi > lo:
        h = (hi - lo) * 1e-3
        e = np.asarray(curve_fn(np.array([lo, lo + h, hi - h, hi])), float)
        s_lo, s_hi = (e[1] - e[0]) / h, (e[3] - e[2]) / h
        if plateau:
            # the one-sided slopes of the plateau curve, never negative (a plateau does not fall; a tiny negative
            # difference would only be numerical)
            s_lo, s_hi = max(s_lo, 0.0), max(s_hi, 0.0)
        b, a = r < lo, r > hi
        y[b] = e[0] + s_lo * (r[b] - lo)
        y[a] = e[3] + s_hi * (r[a] - hi)
    if out.any():
        y[out] = np.maximum(y[out], 0.0)
    return y


def _curve_fns(fit):
    """{zone: rates -> yields} like zone_curves, as functions (for bounded_curve)."""
    if fit.kind == "zone" and fit.zones and fit.zones[0] is not None:
        return {z: (lambda g, z=z: fit.curve(g, zone=z)) for z in fit.zones}
    return {None: lambda g: fit.curve(g)}


def sq_yields(fit, sq, lo=None, hi=None):
    """{zone: predicted yield at the status quo}: the model curve inside [lo, hi] (default the trial range), the
    bounded linear extension of bounded_curve outside it (a plateau: its one-sided slope at the nearest edge, never
    negative)."""
    lo = fit.scale["lo"] if lo is None else lo
    hi = fit.scale["hi"] if hi is None else hi
    return {z: float(bounded_curve(fn, [float(sq)], lo, hi, is_plateau(fit))[0]) for z, fn in _curve_fns(fit).items()}


def decide(curves, grid, weights, pc, pi, sq=None, vr_cost=0.0, sq_curves=None):
    """Optima and gains from yield curves {zone: yhat(grid)} and field weights {zone: acres}.

    The optima are searched on `grid` (the trial range) only. The status quo is priced at its own rate: its yield is
    interpolated on the grid when it lies inside it, else taken from `sq_curves` ({zone: yhat(sq)}, the bounded
    extension of sq_yields; sq_extrapolated True) or, without them, held at the edge of the grid (sq_extrapolated
    False); its input cost is always pi * sq. A status quo more than SQ_MAX_OUT of the range outside the grid is not
    priced: y_sq, nr_sq and the gains over it are None (also per zone) and sq_too_far is True."""
    zs = [z for z in curves if (weights.get(z, 0) if z is not None else 1) > 0]
    if not zs:
        if any(z is not None for z in curves):
            # zone curves with no positive field weight: the caller passed weights of another model (e.g. {None: 1}
            # for a zone fit). Dividing by their zero total used to raise ZeroDivisionError deep in here.
            raise ValueError(f"no positive field weight for the curves {sorted(map(str, curves))}: "
                             f"weights given for {sorted(map(str, weights))}")
        zs = list(curves)
    tot = sum(weights.get(z, 0) for z in zs) if zs[0] is not None else 1.0
    w = {z: (weights.get(z, 0) / tot if z is not None else 1.0) for z in zs}
    NR = {z: pc * curves[z] - pi * grid for z in zs}
    avg = sum(w[z] * NR[z] for z in zs)
    iu = int(np.argmax(avg))
    r_u = float(grid[iu])
    opt = {z: float(grid[int(np.argmax(NR[z]))]) for z in zs}
    nr_at = lambda z, r: float(np.interp(r, grid, NR[z]))           # rates on the grid (optima, best flat)
    y_at = lambda z, r: float(np.interp(r, grid, curves[z]))
    out = dict(r_uniform=r_u, r_zone=opt, weights=w,
               nr_vr=sum(w[z] * nr_at(z, opt[z]) for z in zs) - vr_cost,
               nr_uniform=sum(w[z] * nr_at(z, r_u) for z in zs),
               y_vr=sum(w[z] * y_at(z, opt[z]) for z in zs), y_uniform=sum(w[z] * y_at(z, r_u) for z in zs),
               rate_vr_mean=sum(w[z] * opt[z] for z in zs), corner={z: opt[z] in (float(grid[0]), float(grid[-1])) for z in zs})
    out["gain_vr_vs_uniform"] = out["nr_vr"] - out["nr_uniform"]
    ysq = nrsq = {}
    if sq is not None:
        sq = float(sq)
        inside, too_far = sq_position(sq, grid[0], grid[-1])
        extrap = (not inside) and (not too_far) and sq_curves is not None and all(z in sq_curves for z in zs)
        out["sq_in_range"] = inside
        out["sq_extrapolated"] = extrap
        out["sq_too_far"] = too_far
        span = float(grid[-1] - grid[0])
        out["sq_limits"] = [float(grid[0]) - SQ_MAX_OUT * span, float(grid[-1]) + SQ_MAX_OUT * span]
        if too_far:
            # nothing is priced at a status quo this far from the data (module docstring)
            ysq = nrsq = {z: None for z in zs}
            out.update(nr_sq=None, y_sq=None, gain_vr_vs_sq=None, gain_uniform_vs_sq=None)
        else:
            ysq = {z: (float(sq_curves[z]) if extrap else y_at(z, sq)) for z in zs}
            nrsq = {z: pc * ysq[z] - pi * sq for z in zs}           # the input is always charged at the status quo
            out["nr_sq"] = sum(w[z] * nrsq[z] for z in zs)
            out["y_sq"] = sum(w[z] * ysq[z] for z in zs)
            out["gain_vr_vs_sq"] = out["nr_vr"] - out["nr_sq"]
            out["gain_uniform_vs_sq"] = out["nr_uniform"] - out["nr_sq"]
    priced = sq is not None and not out.get("sq_too_far")
    out["per_zone"] = [dict(zone=z, weight=w[z], r_opt=opt[z], y_opt=y_at(z, opt[z]), nr_opt=nr_at(z, opt[z]),
                            y_uniform=y_at(z, r_u), nr_uniform=nr_at(z, r_u),
                            y_sq=ysq[z] if priced else None,
                            nr_sq=nrsq[z] if priced else None,
                            gain_vs_uniform=nr_at(z, opt[z]) - nr_at(z, r_u),
                            gain_vs_sq=(nr_at(z, opt[z]) - nrsq[z]) if priced else None) for z in zs]
    return out


def analyse(fit, weights, pc, pi, sq=None, vr_cost=0.0, n=161):
    """decide() on the fit's curves over the trial range; a status quo outside the range gets the bounded linear
    extension of sq_yields (see decide; not priced when it is too far outside)."""
    g = rate_grid(fit, n)
    cur = zone_curves(fit, g)
    sqc = sq_yields(fit, sq) if sq is not None else None
    res = decide(cur, g, weights, pc, pi, sq, vr_cost, sq_curves=sqc)
    res["grid"] = g.tolist()
    res["curves"] = {str(k): v.tolist() for k, v in cur.items()}
    return res


# ================================================================================================ continuous NDVI
def ndvi_prescription(fit_ndvi, ndvi_values, areas, pc, pi, sq=None, vr_cost=0.0, n=121, bins=None):
    """r*(NDVI) along the NDVI values of the field's pixels (or bins of them). Returns the per-bin optimum, the
    field-weighted gains and the function to map a raster.

    Each bin is evaluated at the area-weighted mean NDVI of ITS pixels, not at the midpoint of its quantile edges:
    the end bins run to the field's extreme pixels (a slough, a yard), so their midpoint can sit far outside the NDVI
    the trial units cover and the rate x NDVI interaction would be extrapolated there for 5 % of the field. Bins
    whose mean lies outside the units' NDVI range are flagged (in_range False), and their share of the field's pixel
    area is res['share_out_of_range']."""
    g = rate_grid(fit_ndvi, n)
    v = np.asarray(ndvi_values, float)
    a = np.asarray(areas, float)
    ok = np.isfinite(v)
    v, a = v[ok], a[ok]
    nb = bins or 20
    edges = np.quantile(v, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, nb - 1)
    wb = np.bincount(idx, weights=a, minlength=nb)
    sv = np.bincount(idx, weights=a * v, minlength=nb)
    used = [i for i in range(nb) if wb[i] > 0]                     # repeated quantile edges leave empty bins
    centres = {i: float(sv[i] / wb[i]) for i in used}
    un = pd.to_numeric(fit_ndvi.data.get("ndvi"), errors="coerce") if "ndvi" in fit_ndvi.data else pd.Series(dtype=float)
    nlo, nhi = (float(un.min()), float(un.max())) if un.notna().any() else (-np.inf, np.inf)
    curves = {i: fit_ndvi.curve(g, ndvi=centres[i]) for i in used}
    # the status quo of each bin, with the same bounded extension as sq_yields (the NDVI model is never a plateau)
    sqc = ({i: float(bounded_curve(lambda gg, c=centres[i]: fit_ndvi.curve(gg, ndvi=c), [float(sq)], g[0], g[-1],
                                   is_plateau(fit_ndvi))[0]) for i in used}
           if sq is not None else None)
    res = decide(curves, g, {i: wb[i] for i in used}, pc, pi, sq, vr_cost, sq_curves=sqc)
    res["bins"] = [dict(ndvi=centres[i], r_opt=res["r_zone"][i], weight=float(res["weights"].get(i, 0)),
                        in_range=bool(nlo <= centres[i] <= nhi), lo=float(edges[i]), hi=float(edges[i + 1]),
                        corner=bool((res.get("corner") or {}).get(i, False)))
                   for i in used if i in res["r_zone"]]
    res["ndvi_unit_range"] = [nlo, nhi] if np.isfinite([nlo, nhi]).all() else None
    # share of the field (pixel area) in bins whose mean NDVI lies outside the units' NDVI range: their optimum, and
    # their part of the gains, rest on the rate x NDVI interaction extrapolated beyond the trial; the report states it
    tot = float(sum(wb[i] for i in used)) or 1.0
    res["share_out_of_range"] = float(sum(wb[i] for i in used if not (nlo <= centres[i] <= nhi)) / tot)
    # share of the field whose bin optimum sits at an end of the trial range: with the rate x NDVI interaction the
    # curvature changes sign along the NDVI, and a convex (or still rising) bin curve has no optimum inside the range,
    # so its "optimum" is whichever end is cheaper: a corner, not a measured rate. The report says so and treats a
    # prescription made of corners as a diagnostic only.
    res["share_corner"] = float(sum(wb[i] for i in used if (res.get("corner") or {}).get(i)) / tot)
    res["edges"] = edges.tolist()
    res.pop("per_zone", None)
    return res


# ================================================================================================ uncertainty
def _resample_plots(units, rng):
    plots = units.plot_id.unique()
    pick = rng.choice(plots, len(plots), replace=True)
    parts = []
    for k, p in enumerate(pick):
        q = units[units.plot_id == p].copy()
        q["plot_id"] = f"{p}#{k}"
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def _supported(d, spec, kind, zones=None):
    """True when the resample still holds the design rates for every curve: at least 3 distinct design rates (2
    for a line) in every zone of a zone model, in the field otherwise, and, when every original zone has its own
    intercept (zone_level), at least 2 design degrees of freedom (1 for a line) inside the original zones of each
    curve (response.check_support). Same rule as merge_small_zones (given the same form) and fit().
    For a zone model every zone of `zones` (those of the full data) must be present: a zone missing from the
    resample has no design rate at all."""
    if kind == "zone" and zones is not None and not set(zones) <= set(d.zone.unique()):
        return False
    try:
        er.check_support(d, spec, kind)
        return True
    except ValueError:
        return False


def bootstrap(units, spec, kind, weights, pc, pi, sq=None, vr_cost=0.0, B=200, seed=1, progress=None):
    """Plot-cluster bootstrap: resample the plots with replacement, refit, recompute the decision.

    A resample in which a zone of the full data is missing, or a zone (zone model) or the field (pooled / NDVI model)
    has fewer than 3 distinct design rates (2 for a line), or, with an intercept per original zone, fewer than 2 design
    degrees of freedom (1 for a line) inside those original zones (see
    response.check_support) cannot identify its curve: least squares would return a pseudo-inverse curvature (or one fitted to the as-applied
    noise) and a corner or invented optimum, which widens or shifts the 90 % interval and P(gain > 0). Such replicates
    are dropped, like the cross-fit drops such splits; their number is in bt.attrs['n_dropped'] (and replicates whose
    fit failed in bt.attrs['n_failed']), and summarise_boot reports it."""
    rng = np.random.default_rng(seed)
    rows = []
    n_dropped = n_failed = 0
    by_plot = {p: g for p, g in units.groupby("plot_id")}
    plots = list(by_plot)
    zones = units.zone.unique().tolist()
    for b in range(B):
        pick = rng.choice(len(plots), len(plots), replace=True)
        d = pd.concat([by_plot[plots[i]].assign(plot_id=f"{plots[i]}#{k}") for k, i in enumerate(pick)],
                      ignore_index=True)
        if not _supported(d, spec, kind, zones):
            n_dropped += 1
            continue
        try:
            f = er.fit(d, spec, kind)
            r = analyse(f, weights, pc, pi, sq, vr_cost, n=81)
            row = dict(b=b, r_uniform=r["r_uniform"], gain_vr_vs_uniform=r["gain_vr_vs_uniform"],
                       gain_vr_vs_sq=r.get("gain_vr_vs_sq"), gain_uniform_vs_sq=r.get("gain_uniform_vs_sq"))
            for z, ro in r["r_zone"].items():
                row[f"r_zone_{z}"] = ro
            rows.append(row)
        except Exception:
            n_failed += 1
            continue
        if progress and b % 10 == 0:
            progress(b / B, f"bootstrap {b}/{B}")
    bt = pd.DataFrame(rows)
    bt.attrs["n_dropped"] = n_dropped
    bt.attrs["n_failed"] = n_failed
    bt.attrs["B"] = B
    return bt


def summarise_boot(bt, n_dropped=None):
    """Median, 5-95 % interval and P(> 0) of each bootstrap quantity, plus (top level) 'n_dropped': the replicates
    left out because a zone (or the field) lost the design rates for its curve (taken from bt.attrs when not given)
    and 'n_failed': the replicates whose fit failed."""
    out = {}
    for c in bt.columns:
        if c == "b":
            continue
        v = pd.to_numeric(bt[c], errors="coerce").dropna()
        if len(v) < 5:
            continue
        out[c] = dict(median=float(v.median()), lo=float(v.quantile(0.05)), hi=float(v.quantile(0.95)),
                      p_pos=float((v > 0).mean()) if c.startswith("gain") else None, n=int(len(v)))
    out["n_dropped"] = int(bt.attrs.get("n_dropped", 0) if n_dropped is None else n_dropped)
    out["n_failed"] = int(bt.attrs.get("n_failed", 0))
    return out


def crossfit(units, spec, kind, weights, pc, pi, sq=None, vr_cost=0.0, R=50, seed=2, progress=None,
             eval_fit_kind=None):
    """Repeated 2-fold split of the PLOTS, stratified by design rate: decisions from one half, judged on the other.

    Uncertainty: the two folds of a repeat are complementary halves and every repeat reuses the same plots, so the
    2R fold results are strongly dependent and std/sqrt(2R) is only the Monte-Carlo error of the split average (it
    shrinks as R grows with no new data). The summary therefore works on the per-repeat average of the two folds:
    mean, sd (their spread), lo / hi (5th / 95th percentiles), share_positive and n (repeats). There is no 'se'.
    Common support: the two halves have different applied-rate ranges (as-applied noise), and the training optima are
    often at the training grid's edge. Each fold is therefore judged on the overlap [max(train lo, test lo),
    min(train hi, test hi)]: the training optima (zone rates and flat rate, chosen on the training grid) are clipped
    into it and priced on the held-out curve evaluated on it, so the held-out curve is never extrapolated beyond its own
    data. A fold without an overlap is skipped.
    Every skipped fold is counted by its reason, so the verdict and the report can say WHY the value of VR was not
    cross-fitted instead of always blaming the design rates (a fit failing in every fold points at the model, a
    missing overlap at the as-applied rates, a lack of support at the design): cf.attrs holds 'n_tried' (half-splits
    tried, 2 R), 'n_no_support' (the training or the held-out half lost the design support for a curve:
    _supported, i.e. design rates, a zone, or design degrees of freedom inside the original zones), 'n_no_overlap'
    (no common applied-rate support), 'n_failed' (a fit or the decision raised; 'first_error' keeps the first
    message) and 'n_used' (folds that gave a row). n_tried = n_no_support + n_no_overlap + n_failed + n_used.
    The status quo is judged at its own rate on the held-out curve: inside the overlap it is read on it, outside it
    gets the bounded linear extension from the overlap's edge (sq_yields), and its input is charged at pi * sq. When
    the status quo is too far outside the FULL trial range (sq_position on all the units; the same rule as decide) no
    gain over it is computed (cf.attrs['sq_too_far'] True).

    eval_fit_kind (optional): the model fitted on the HELD-OUT half to value the decisions, when it differs from the
    model `kind` that chooses them on the training half. `weights` are then the weights of the held-out model's
    curves (the field's zone acres for 'zone'); the training model's decision uses them only for the curves it has
    (a single curve gets {None: 1}). Why: the summary table values the single-curve optimum (the verdict's best flat
    rate) and the status quo on the main zone model over the field's zone acres (project.summary_basis). A cross-fit
    of the single curve judged on a held-out single curve at the plots' zone mix is the out-of-sample version of a
    DIFFERENT number, so it was no check of the in-sample gain printed beside it. With kind 'pooled' and
    eval_fit_kind 'zone' the flat rate is chosen on the training half by the single curve and it and the status quo
    are valued on the held-out zone model over the field acres: the same basis as the in-sample row. Only
    gain_uniform_vs_sq is computed then (a single curve has no VR strategy to judge)."""
    ek = kind if eval_fit_kind is None else eval_fit_kind
    mixed = ek != kind
    rng = np.random.default_rng(seed)
    tcol = f"target_{spec.input}"
    # plots without a design rate (NaN target, e.g. check strips of a partly rated design) form their own stratum:
    # groupby drops NaN keys by default, which left every such plot out of `half` and put it in the second fold of
    # every repeat, so the folds were neither balanced nor random for them
    pr = (units.groupby("plot_id")[tcol].first() if tcol in units.columns
          else pd.Series(np.nan, index=pd.Index(units.plot_id.unique(), name="plot_id")))
    zones = units.zone.unique().tolist()
    sq_far = False
    if sq is not None:
        full_scale = er.prep(units, spec)[1]
        sq_far = sq_position(sq, full_scale["lo"], full_scale["hi"])[1]
    n_tried = n_no_support = n_no_overlap = n_failed = 0
    first_error = None
    rows = []
    for rep in range(R):
        half = set()
        for r, ids in pr.groupby(pr, dropna=False):
            ids = list(ids.index)
            rng.shuffle(ids)
            half |= set(ids[: len(ids) // 2])
        A = units[units.plot_id.isin(half)]
        Bh = units[~units.plot_id.isin(half)]
        for tr, te in ((A, Bh), (Bh, A)):
            n_tried += 1
            # each half must still support a curve per zone (3 design rates, and 2 design degrees of freedom inside
            # the original zones when they carry the intercepts; 2 and 1 for a line), else the split is skipped
            # (with eval_fit_kind, the held-out half must support the evaluation model instead)
            if not (_supported(tr, spec, kind, zones) and _supported(te, spec, ek, zones)):
                n_no_support += 1
                continue
            try:
                ftr, fte = er.fit(tr, spec, kind), er.fit(te, spec, ek)
                lo_c = max(ftr.scale["lo"], fte.scale["lo"])
                hi_c = min(ftr.scale["hi"], fte.scale["hi"])
                if not hi_c > lo_c:
                    n_no_overlap += 1
                    continue
                g_tr = rate_grid(ftr, 81)
                ctr = zone_curves(ftr, g_tr)
                # the training decision weights its own curves: the evaluation weights when they name them, {None: 1}
                # for a single curve judged on a zone model (mixed mode)
                wtr = weights if not mixed or all(z in weights for z in ctr) else {z: 1.0 for z in ctr}
                dtr = decide(ctr, g_tr, wtr, pc, pi, None, vr_cost)
                g = rate_grid(ftr, 81, lo_c, hi_c)                  # the common support
                cte = zone_curves(fte, g)
                if mixed:
                    # the held-out model's curves at the evaluation weights (the field's zone acres)
                    zs = [z for z in cte if (weights.get(z, 0) if z is not None else 1) > 0]
                    w = {z: float(weights.get(z, 0)) if z is not None else 1.0 for z in zs}
                else:
                    zs = [z for z in dtr["r_zone"] if z in cte]
                    w = {z: dtr["weights"][z] for z in zs}
                if mixed and not zs:
                    raise ValueError("no held-out curve with a positive evaluation weight")
                tot = sum(w.values()) or 1.0
                clip = lambda r: float(np.clip(r, lo_c, hi_c))
                nr = lambda z, r: float(np.interp(clip(r), g, pc * cte[z] - pi * g))
                un = sum(w[z] / tot * nr(z, dtr["r_uniform"]) for z in zs)
                row = dict(rep=rep)
                has_vr = not mixed or all(z in dtr["r_zone"] for z in zs)
                if has_vr:
                    vr = sum(w[z] / tot * nr(z, dtr["r_zone"][z]) for z in zs) - vr_cost
                    row["gain_vr_vs_uniform"] = vr - un
                if sq is not None and not sq_far:
                    if lo_c <= sq <= hi_c:
                        ysq = {z: float(np.interp(sq, g, cte[z])) for z in zs}
                    else:
                        ysq = sq_yields(fte, sq, lo_c, hi_c)
                    s = sum(w[z] / tot * (pc * ysq[z] - pi * sq) for z in zs)
                    if has_vr:
                        row["gain_vr_vs_sq"] = vr - s
                    row["gain_uniform_vs_sq"] = un - s
                rows.append(row)
            except Exception as e:
                # counted, not hidden: a fold that fails every time is a model problem the report must name
                n_failed += 1
                if first_error is None:
                    first_error = f"{type(e).__name__}: {e}"[:300]
                continue
        if progress and rep % 5 == 0:
            progress(rep / R, f"cross-fit {rep}/{R}")
    cf = pd.DataFrame(rows)
    cf.attrs["n_tried"] = n_tried
    cf.attrs["n_no_support"] = n_no_support
    cf.attrs["n_no_overlap"] = n_no_overlap
    cf.attrs["n_failed"] = n_failed
    cf.attrs["n_used"] = len(rows)
    cf.attrs["first_error"] = first_error
    cf.attrs["sq_too_far"] = bool(sq_far)
    cf.attrs["eval_fit_kind"] = ek
    summ = {}
    for c in [c for c in cf.columns if c.startswith("gain")]:
        v = cf[["rep", c]].dropna().groupby("rep")[c].mean()        # per-repeat average of the two folds
        if len(v):
            summ[c] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                           lo=float(v.quantile(0.05)), hi=float(v.quantile(0.95)),
                           share_positive=float((v > 0).mean()), n=int(len(v)))
    return cf, summ


def price_grid(fit, weights, pc_range, pi_range, sq=None, vr_cost=0.0, n=7):
    rows = []
    for pc in np.linspace(*pc_range, n):
        for pi in np.linspace(*pi_range, n):
            r = analyse(fit, weights, pc, pi, sq, vr_cost, n=81)
            rows.append(dict(crop_price=float(pc), input_price=float(pi), r_uniform=r["r_uniform"],
                             gain_vr_vs_uniform=r["gain_vr_vs_uniform"], gain_vr_vs_sq=r.get("gain_vr_vs_sq"),
                             **{f"r_zone_{z}": v for z, v in r["r_zone"].items()}))
    return pd.DataFrame(rows)


# ================================================================================================ realised trial
def realised(units, fit, decision, pc, pi, sq=None, vr_cost=0.0, flat_rate=None):
    """Per unit: the net return actually obtained at the trial rate, and what the same unit would have returned at
    the flat status quo, at the best flat rate, at the recommended flat rate (`flat_rate`, strategy 'flat': the
    verdict's advice, which may be the status quo) and at the zone VR rate. The counterfactual keeps the unit's own
    residual: y_cf = y_obs + yhat(r_cf) - yhat(r_applied) (the model only moves the unit along its zone curve).

    The strategies are priced as in decide(): the VR net return is net of `vr_cost` ($/ac), so the realised table
    and the decision table agree on the same strategy, and a status quo (or flat rate) outside the trial range gets
    the same bounded linear extension (bounded_curve), with the input charged at its own rate. A status quo (or flat
    rate) too far outside the range (sq_position) is not priced: its columns are left out and the differences with
    it are None (summary 'sq_too_far'). The summary carries the area-weighted observed yield (yield_obs)."""
    d, _ = er.prep(units, fit.spec)
    lo, hi = float(fit.scale["lo"]), float(fit.scale["hi"])
    far = lambda r: sq_position(r, lo, hi)[1]
    sq_far = sq is not None and far(sq)
    flat_far = flat_rate is not None and far(flat_rate)
    extra = [float(r) for r in (sq, flat_rate) if r is not None]
    g = np.unique(np.concatenate([rate_grid(fit, 161), extra, [float(d.r.min()), float(d.r.max())]]))
    plat = is_plateau(fit)
    cur = {z: bounded_curve(fn, g, lo, hi, plat) for z, fn in _curve_fns(fit).items()}
    key = lambda z: z if z in cur else (None if None in cur else list(cur)[0])
    yhat = lambda z, r: np.interp(r, g, cur[key(z)])
    strategies = [("uniform", decision["r_uniform"])]
    if flat_rate is not None and not flat_far:
        strategies.append(("flat", float(flat_rate)))
    rows = []
    for _, u in d.iterrows():
        z = int(u.zone)
        base = yhat(z, u.r)
        rz = decision["r_zone"].get(key(z), decision["r_uniform"])
        r_ = dict(unit_id=u.unit_id, plot_id=u.plot_id, zone=z, area_ac=u.area_ac, rate=u.r, yield_obs=u["yield"],
                  nr_trial=pc * u["yield"] - pi * u.r)
        for lab, r in strategies + [("vr", rz)] + ([("sq", sq)] if sq is not None and not sq_far else []):
            y = u["yield"] + yhat(z, r) - base
            r_[f"y_{lab}"] = y
            r_[f"nr_{lab}"] = pc * y - pi * r - (vr_cost if lab == "vr" else 0.0)
            r_[f"rate_{lab}"] = r
        rows.append(r_)
    R = pd.DataFrame(rows)
    w = R.area_ac
    keep = ("nr_", "y_", "rate", "yield_obs")
    summ = {c: float(np.average(R[c], weights=w)) for c in R.columns if c.startswith(keep)}
    has_sq = sq is not None and not sq_far
    summ["trial_minus_sq"] = summ["nr_trial"] - summ["nr_sq"] if has_sq else None
    summ["vr_minus_sq"] = summ["nr_vr"] - summ["nr_sq"] if has_sq else None
    summ["vr_minus_uniform"] = summ["nr_vr"] - summ["nr_uniform"]
    if flat_rate is not None:
        summ["vr_minus_flat"] = summ["nr_vr"] - summ["nr_flat"] if not flat_far else None
        summ["flat_minus_sq"] = summ["nr_flat"] - summ["nr_sq"] if (has_sq and not flat_far) else None
    summ["sq_too_far"] = bool(sq_far)
    summ["vr_cost"] = float(vr_cost)
    summ["area_ac"] = float(w.sum())
    byz = R.groupby("zone").apply(lambda q: pd.Series({c: np.average(q[c], weights=q.area_ac) for c in R.columns
                                                       if c.startswith(("nr_", "y_", "yield_obs"))} | {"area_ac": q.area_ac.sum()}),
                                  include_groups=False).reset_index()
    return R, summ, byz
