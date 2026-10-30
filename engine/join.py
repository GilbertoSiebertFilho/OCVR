# -*- coding: utf-8 -*-
"""Putting the trial together: design plots, transition trims, yield polygons, as-applied rates, zones -> analysis
units.

The rules come from the OFPE / DIFM practice of this group and the literature behind it:
  * the border / headland polygon is not a plot (it received the grower's rate, not a trial rate). It is found by
    the type column first (a value containing one of BORDER_WORDS); without that, by geometry: a polygon far larger
    than the plots (> 5 x the median, 1 to 3 of them), one that surrounds or contains most plots, or one whose
    oriented envelope covers > 60 % of the design extent (a ring, an L-shaped headland). Narrow headland strips
    (up to 5 x a plot, or more than 3 large strips) are NOT found from the geometry (a rule for them also flagged real double plots and was
    removed): a design that has them needs a type column. Each border polygon keeps a `border_rule` column naming
    the rule that found it;
  * each plot is trimmed at both ends (default 10 m: the applicator's rate change and the residual yield-monitor delay
    after the cleaning app corrected it; DIFM's ~10 m "transitional buffer zones", Bullock et al. 2019 p. 2743;
    Trevisan et al. 2021) and optionally on the sides;
  * every yield observation becomes a rectangle (swath width x distance travelled, centred on the logged point,
    along the heading corrected to grid north), and is kept only when at least `min_share` of it lies inside ONE
    trimmed plot (default 0.95, the DIFM "at least 95 % of the yield polygon in one treatment"); `min_share = 0`
    switches to clip-and-weight (each piece counts with its area inside the plot);
  * the applied rate of a unit is the median of the as-applied points inside it (the target rate when there is no
    as-applied file);
  * the analysis unit is the piece of a plot inside one zone (plot x zone). A plot that crosses two NDVI zones gives
    two units with the same treatment, which is what lets the response differ by zone. Units with fewer than
    `min_obs` observations or less than `min_area_m2` are dropped (the area test only when the yield file gives a
    real area per observation, see `area_known` in yield_polygons). The plot (unit of randomisation) is the cluster
    of every standard error downstream.
Nothing is filtered here on the yield value: the map is assumed cleaned (OCCD / yield-map-cleaning) before it arrives.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Polygon
from shapely.affinity import translate
from pyproj import Transformer

from . import io as eio

BORDER_WORDS = ("head", "border", "buffer", "bord", "edge", "outside", "check_strip_none")

# Column names the join creates itself. A cleaned yield map often already carries some of them (a trial-clipped map
# from the OFPE / OCCD pipeline has a plot_id; a re-exported one may have zone or share). Kept, they collide in the
# overlay / sjoin / merge (pandas renames them plot_id_x / plot_id_y and groupby("plot_id") fails), and a stale user
# "zone" or "plot_id" is never the one this run computes. So they are dropped from the yield frame on the way in;
# the yield value column chosen by the caller is read before the drop and is never lost.
ENGINE_COLS = ("plot_id", "piece_m2", "share", "obs_id", "index_right", "index_left", "zone", "zone_unit", "x_m", "y_m")


def _drop_engine_cols(df, keep=()):
    """`df` without the columns that clash with the engine's own names (see ENGINE_COLS), except those in `keep`."""
    drop = [c for c in ENGINE_COLS if c in df.columns and c not in keep and c != df.geometry.name]
    return df.drop(columns=drop) if drop else df


# ================================================================================================ design
def _border_by_geometry(g, area):
    """Border polygons found from the geometry alone (no usable type column). Returns (flag Series, reason Series).

    A design without a type column still has to lose its headland / buffer: if it is kept, it becomes an extra plot
    at the grower's rate and its headland yield enters the response fit as status-quo observations. The rules:
      * area > 5 x the median polygon, when only 1 to 3 polygons qualify (a single headland polygon or a few large
        end strips);
      * surrounds or contains most plots: the polygon's outline, holes filled, holds > 50 % of the other polygons'
        interior points (a buffer ring, or a field-wide polygon drawn under the plots);
      * its envelope covers > 60 % of the design extent, measured in the design's own oriented frame (the minimum
        rotated rectangle of the whole design), so a long plot of a design drawn at 45 deg does not look as if it
        covered the extent (the axis-aligned box of a diagonal strip would).
    A rule based on "large and touching many plots" was tried and removed: in a grid of touching plots it also flags
    a real double or quadruple plot. Headland strips up to 5 x a plot (or more than 3 large strips) are therefore NOT found from the
    geometry: give the design a type column (headland / border) when it has them.
    A rule that flags half of the polygons or more describes the plots, not a border, and is ignored."""
    n = len(g)
    none = pd.Series(False, index=g.index), pd.Series("", index=g.index, dtype=object)
    if n < 4:
        return none
    med = float(area.median())
    reason = pd.Series("", index=g.index, dtype=object)
    big = area > 5 * med
    if 1 <= big.sum() <= 3:
        reason[big] = "area > 5 x median"
    # oriented frame of the whole design
    c, u, v, hl, hw = _oriented(g.geometry.union_all())
    ext = max(4 * hl * hw, 1e-9)
    env_share, surr_share = [], []
    rep_pts = g.geometry.representative_point().values
    for i, geom in enumerate(g.geometry.values):
        xy = shapely.get_coordinates(geom) - c
        s_, t_ = xy @ u, xy @ v
        env_share.append(float((s_.max() - s_.min()) * (t_.max() - t_.min()) / ext) if len(xy) else 0.0)
        parts = getattr(geom, "geoms", [geom])
        filled = shapely.union_all([Polygon(pp.exterior) for pp in parts if pp.geom_type == "Polygon"])
        inside = shapely.contains(filled, rep_pts)
        inside[i] = False
        surr_share.append(inside.sum() / (n - 1))
    env_share, surr_share = np.array(env_share), np.array(surr_share)
    surrounds = pd.Series(surr_share > 0.5, index=g.index)
    covers = pd.Series(env_share > 0.6, index=g.index)
    for flag, why in ((surrounds, "surrounds / contains most plots"),
                      (covers, "envelope covers > 60 % of the design extent")):
        if 0 < flag.sum() < n / 2:
            reason[flag & (reason == "")] = why      # the first rule that fits names the polygon
    border = reason != ""
    if border.sum() >= n / 2:
        return none
    return border, reason


def prepare_design(design, rate_col=None, id_col=None, type_col=None, treatment_col=None, block_col=None, epsg=None,
                   extra_rate_cols=None):
    """Plots (UTM) with plot_id, design_rate, treatment, block; the border polygon(s) apart (their column
    `border_rule` says which rule found each one: the type column or a geometric rule, see _border_by_geometry).

    Design rates are always float: an integer rate field of a shapefile / GeoJSON (100, 120, ...) would otherwise stay
    int64, and build_units later writes the float as-applied medians into a copy of it, which pandas >= 3 refuses
    (TypeError: invalid value for dtype int64) instead of upcasting.

    `extra_rate_cols` = {input_name: column}: the design rate of a second (third, ...) input when it lives in another
    column of THIS design file (e.g. seed rate in `Rate` and urea rate in `N_rate` on the same plots). Each becomes
    the numeric column design_<input_name> of the plots, the same column attach_rate makes from a separate file."""
    g, epsg = eio.to_utm(design, epsg)
    g = g.reset_index(drop=True)
    area = g.geometry.area
    border = pd.Series(False, index=g.index)
    reason = pd.Series("", index=g.index, dtype=object)
    if type_col and type_col in g:
        # the type column is the designer's own statement: it wins over any geometric guess
        t = g[type_col].astype(str).str.lower()
        border = t.apply(lambda s: any(w in s for w in BORDER_WORDS)).astype(bool)
        reason[border] = f"type column '{type_col}'"
    if not border.any():
        border, reason = _border_by_geometry(g, area)
    plots = g[~border].copy()
    bord = g[border].copy()
    bord["border_rule"] = reason[border].values
    if id_col and id_col in plots:
        plots["plot_id"] = plots[id_col].astype(str)
    else:
        plots["plot_id"] = [str(i + 1) for i in range(len(plots))]
    if plots.plot_id.duplicated().any():
        plots["plot_id"] = plots.plot_id + "_" + plots.groupby("plot_id").cumcount().astype(str)
    plots["design_rate"] = (pd.to_numeric(plots[rate_col], errors="coerce").astype(float)
                            if rate_col and rate_col in plots else np.nan)
    plots["treatment"] = plots[treatment_col].astype(str) if treatment_col and treatment_col in plots else None
    plots["block"] = plots[block_col].astype(str) if block_col and block_col in plots else None
    plots["area_m2"] = plots.geometry.area
    extra = []
    for name, col in (extra_rate_cols or {}).items():
        if not col:
            continue
        if col not in plots:
            raise ValueError(f"Design rate column '{col}' for input '{name}' is not in the design file")
        plots[f"design_{name}"] = pd.to_numeric(plots[col], errors="coerce").astype(float)
        extra.append(f"design_{name}")
    keep = ["plot_id", "design_rate"] + extra + ["treatment", "block", "area_m2", "geometry"]
    return plots[keep].reset_index(drop=True), bord, epsg


def attach_rate(plots, other_design, rate_col, name):
    """The rate of a second input (another design file on the same plots): taken at each plot's interior point.
    Cast to float for the same reason as in prepare_design (as-applied medians are written into a copy of it)."""
    o = other_design.to_crs(plots.crs).reset_index(drop=True)
    pts = gpd.GeoDataFrame(plots[["plot_id"]], geometry=plots.geometry.representative_point(), crs=plots.crs).reset_index(drop=True)
    j = gpd.sjoin(pts, o[[rate_col, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    plots[f"design_{name}"] = pd.to_numeric(j[rate_col], errors="coerce").astype(float).values
    return plots


def _partition_1d(s, k):
    """Optimal split of the SORTED values s into k contiguous groups minimising the within-group sum of squares (the
    one-dimensional k-means / Jenks solution, by dynamic programming, so it does not depend on a starting point).
    Returns the group index (0 .. k-1, in value order) of each value."""
    s = np.asarray(s, dtype=float)
    n = len(s)
    cs, cs2 = np.concatenate([[0.0], np.cumsum(s)]), np.concatenate([[0.0], np.cumsum(s * s)])
    D = np.full((k + 1, n + 1), np.inf)
    B = np.zeros((k + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for kk in range(1, k + 1):
        for j in range(kk, n + 1):
            i = np.arange(kk - 1, j)
            m = j - i
            ssq = cs2[j] - cs2[i] - (cs[j] - cs[i]) ** 2 / m
            v = D[kk - 1, i] + ssq
            b = int(np.argmin(v))
            D[kk, j], B[kk, j] = v[b], i[b]
    grp = np.zeros(n, dtype=int)
    j = n
    for kk in range(k, 0, -1):
        i = B[kk, j]
        grp[i:j] = kk - 1
        j = i
    return grp


def _merged_levels(s, grp, floor, raise_jump=6.0, raise_width=2.5, hint_jump=3.5, hint_width=2.0):
    """Signs that a group of the chosen split holds more than one design level (the separation score alone prefers
    a coarser split when the outer steps of the design are far larger than the inner ones, or when two close
    levels drift into each other while the others stay apart). Two readings of the SORTED values s and their group
    index grp:
      * a jump: in a group of 8 or more plots the largest spacing between consecutive values reaches the noise
        floor AND exceeds raise_jump (6) x the 90th percentile of that group's spacings (chance spacings inside
        one level are of one size; a step between two merged levels stands out of them: 8 to 11 x on
        100/120/200/300 at sd 2). At 4 x, the review's figure, the chance spacings of a level of 33 plots with
        +-6 % uniform drift (Trial 22) fired in 3 of 100 clean maps; at 6 x in 1 of 300, and every merge of the
        review's designs is still caught (the width rule alone catches them);
      * a wide group: a group's width (p90 - p10 of its plots) reaches 2 x the floor AND exceeds raise_width (2.5)
        x the smallest width among the groups of 4 or more plots, in absolute terms and ALSO relative to the
        group's own value (a drill's noise may scale with the rate: Trial 22's 262 level is three times wider than
        its 88 and must not read as merged; a level of 0 has no scale, which is why the absolute ratio is required
        too). Two levels one step apart merged at a plot-to-plot sd of a fifth of the step are 2.8 to 3 x as wide
        as a single level, which is why the ratio is 2.5 and not the review's 3 (at 3, one seed in 100 of
        90/100/110/130 at sd 2 x 35 plots still came back as 3 levels); 2.5 costs 1 to 2 raises in 100 on clean
        maps of 14 plots per level, where the p10-p90 of a level is itself noisy, and nothing on Trial 22.
    Returns (reason, hint): `reason` names the first group that fails one of these rules (empty when none), `hint`
    the first group that fails only the softer version (hint_jump 3.5, hint_width 2), for a warning."""
    k = int(grp.max()) + 1
    groups = [s[grp == g] for g in range(k)]
    widths = np.array([np.percentile(x, 90) - np.percentile(x, 10) for x in groups])
    scale = np.array([max(abs(float(np.median(x))), 1e-9) for x in groups])
    big = np.array([len(x) >= 4 for x in groups])
    # the reference widths are never taken below the noise floor: a level the controller logged exactly (a 0-rate
    # check strip, a target held to the decimal) has width 0, and every level with ordinary drift would otherwise
    # be "wider than the tightest" by any ratio
    wf = np.maximum(widths, floor)
    w_min = float(wf[big].min()) if big.any() else np.inf
    r_min = float((wf / scale)[big].min()) if big.any() else np.inf
    reason = hint = ""
    for g, x in enumerate(groups):
        ratio = min(wf[g] / w_min, (wf[g] / scale[g]) / r_min) if big.any() else 0.0
        # a group narrower than 2 x the floor is within the noise whatever the ratio says (the text keeps the ratio)
        wide = widths[g] >= 2.0 * floor and ratio > hint_width
        jump = None
        if len(x) >= 8:
            d = np.diff(x)
            jump = float(d.max()) / max(float(np.percentile(d, 90)), 1e-9)
            if float(d.max()) < floor:
                jump = None          # a jump under the floor is noise, whatever its ratio: not evaluated
        msg = (f"the level near {np.median(x):.1f} ({len(x)} plots) looks like two design levels: its plots spread "
               f"{widths[g]:.1f} (p10-p90), {ratio:.1f} x the tightest level"
               + (f", and the largest step inside it is {jump:.1f} x its other spacings" if jump is not None else ""))
        if ((jump is not None and jump > raise_jump) or (wide and ratio > raise_width)) and not reason:
            reason = msg
        elif ((jump is not None and jump > hint_jump) or wide) and not hint:
            hint = msg
    return reason, hint


def rate_levels(values, precision=0.0, max_levels=15, min_plots=2, rel_tol=0.25, min_rel_step=0.02, min_score=2.0,
                max_ambiguity=0.85, max_single_share=0.10, min_plots_free=24, small_min_score=(4.0, 2.5),
                max_warn_score=5.0, name="input"):
    """The design levels hidden in one applied rate per plot (the plot medians of an as-applied map).

    For k = 2 .. max_levels the sorted plot values are split into k groups (_partition_1d, the optimal 1-D
    partition) and each split gets a SEPARATION SCORE: the smallest gap between two adjacent groups (the empty
    interval between the highest value of the lower group and the lowest of the upper one) divided by the largest
    spacing between consecutive values INSIDE any group (never taken below 3 x precision or 0.5 % of the median,
    so exact values do not divide by zero). The k with the best score wins:
      * at the true number of levels the score is (smallest real step gap) / (largest chance spacing inside a
        level), well above 1;
      * one level too many cuts a real level at its largest chance spacing, so the smallest gap becomes a chance
        spacing and the score falls to about 1;
      * one level too few merges two real levels, so a real gap becomes a spacing inside a group and the score
        falls below 1 WHEN the steps of the design are roughly equal. When they are not (100/120/200/300: the
        merged 100 + 120 step is small next to the 200 and 300 gaps), or when two close levels drift into each
        other while the others stay apart (90/100/110/130 with a plot-to-plot sd of 1.5 or more), the coarser
        split can score HIGHER than the true one, so the score is not trusted alone: see the shape rules below.
    The noise floor is max(6 x precision, min_rel_step x the median value): `precision` is the precision of ONE
    plot value (the standard error of a plot median, about 0.5 x (p90 - p10 of the plot's points) / sqrt(points));
    no trial steps its rates by less than 2 %. A split is only eligible when its smallest gap reaches that floor and
    when the groups of fewer than `min_plots` plots (misfires: one plot that got a rate of its own) hold at most
    max_single_share (10 %) of the plots, so a drifting design cannot be read as one level per plot. A split whose
    gap is under the floor is still scored (scores_below_floor) and still counts as a rival in the ambiguity test.
    The best split must pass four tests, else it is not returned:
      * score at least min_score (2.0), and no other k (eligible or not) within max_ambiguity (0.85) of it;
      * shape (_merged_levels): no group of 8 or more plots with a jump inside it of at least the floor and over
        6 x its other spacings, and no group wider (p10-p90) than 2.5 x the tightest level (both in absolute terms
        and relative to the level's value), with a width of at least 2 x the floor. These are the marks of two
        design levels read as one; on the sweeps of the review they stopped every silent merge of 90/100/110/130
        and 100/120/200/300 at sd 1.5-2 (which the score alone passed in up to 16 of 20 seeds, scoring 2.7-9)
        at the cost of about 1 raise in 300 clean Trial-22-like maps (+-6 % drift, 168 plots);
      * size: with fewer than min_plots_free (24) plots the chance that a gradient or continuous prescription
        splits into 2-3 "levels" that pass the score test is real (uniform(60, 200) on 8-20 plots passed in 6 to
        24 % of seeds), so a small map is only read when its best score reaches small_min_score: (4.0, 2.5) = 4 at
        8 plots or fewer, falling in a straight line to 2.5 at 23. Designs with realistic drift score 2 to 5
        (+-6 % uniform drift on Trial 22's steps: 2-4; 100/120/140/160 x 5 plots at sd 2: 3-6; exactly logged
        rates: 11-48), so the rule is a trade-off, measured on 100-200 seeds per case: it reads 90 % of
        100/120/140/160 x 5 (sd 2) and 93 % of 100/110 x 6 (sd 1), raising the rest (the backup read 97-98 %),
        and a gradient still passes it in 1.5-4.5 % of seeds (7, 9, 6, 3 of 200 at 8, 12, 16, 20 plots; the
        backup 22, 24, 16, 6 %), each time with the small-design warning of rates_from_asapplied attached.
        A fixed 6 would cut the gradient to about 1 % but refuse 34-39 % of those clean designs. Otherwise the
        error says the design is too small for its levels to be read from the map.
    When the softer shape rules fire (a jump over 3.5 x, a width over 2 x) and the score is under max_warn_score
    (5), the levels are returned with a `warning` saying that a finer step may be hidden, to be checked against
    the design (about 6 % of the clean Trial-22-like maps, which score 2 to 4, carry it).
    When the best split fails, ONE level is returned only when NO split separated the values (none eligible, or
    the best under min_score) and the values are one tight cluster (their p10 - p90 within max(10 % of the
    median, 6 x precision): a map that applied one rate). A split that scored but was refused as ambiguous,
    merged or on too small a map never falls back to one level (100/110 on 6 plots must raise, not read as 105),
    and in every other case a ValueError says that no discrete levels were found (a gradient or continuous
    prescription, more than max_levels levels, plot-to-plot drift that runs the levels into each other, or too
    small a design) instead of returning a level structure that is not there. The reading is reliable when the
    design steps are roughly equal, or when the plot-to-plot drift is well under half the smallest step; outside
    that it raises rather than guesses, and a design file or column is the answer.

    A group of fewer than `min_plots` plots is not a design level: its plots get no design rate unless they lie
    within the tolerance of a kept level, and the number of such groups is reported. Every plot is given the level
    nearest to its value when it lies within that level's tolerance: rel_tol (0.25) of the local step, widened to
    0.8 x the level's own width (p90 - p10 of its plots) but never past 0.45 of the step, so a plot halfway between
    two levels stays off both.
    Returns dict(level=array aligned with values (NaN = no level), levels=sorted list (one decimal), tol=tolerance
    per level, widths=p90 - p10 of the plots of each level, plots_per_level={level: n}, k=..., unassigned=number
    of finite values given no level, step=the smallest step between the kept levels, single_plot_levels=groups of
    fewer than min_plots plots, score=the separation score of the chosen split (None for one level), scores={k:
    score} of the eligible splits, scores_below_floor={k: score} of the splits whose gap is under the floor,
    warning=text or None, precision)."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    res = dict(level=out, levels=[], tol=[], widths=[], plots_per_level={}, k=0, unassigned=0, step=0.0,
               single_plot_levels=0, score=None, scores={}, scores_below_floor={}, warning=None,
               precision=float(precision or 0.0))
    if not ok.any():
        return res
    s = np.sort(v[ok])
    n = len(s)
    n_u = len(np.unique(s))
    med_all = float(np.median(s))
    floor = max(6.0 * float(precision or 0.0), min_rel_step * abs(med_all), 1e-9)
    # the spacing inside a level is never taken below 3 x the precision of a plot value (or 0.5 % of the median):
    # exact values (spacing 0) would otherwise give an infinite score
    denom_floor = max(3.0 * float(precision or 0.0), 0.005 * abs(med_all), 1e-9)
    best = None
    scores, below = {}, {}
    for k in range(2, min(max_levels, n_u) + 1):
        grp = _partition_1d(s, k)
        sizes = np.bincount(grp, minlength=k)
        if (sizes < min_plots).sum() and sizes[sizes < min_plots].sum() > max_single_share * n:
            continue
        lo_v = np.array([s[grp == g].min() for g in range(k)])
        hi_v = np.array([s[grp == g].max() for g in range(k)])
        gaps = lo_v[1:] - hi_v[:-1]
        within = max([float(np.diff(s[grp == g]).max()) for g in range(k) if sizes[g] > 1] + [0.0])
        sc = float(gaps.min()) / max(within, denom_floor)
        if float(gaps.min()) < floor:
            # its levels run into each other, so it cannot be chosen; it still rivals the best in the ambiguity test
            below[k] = sc
            continue
        scores[k] = sc
        if best is None or sc > best[0]:
            best = (sc, k, grp)
    res.update(scores=scores, scores_below_floor=below)
    # ambiguous: another number of levels separates almost as well (second best score > max_ambiguity x the best),
    # which on the sweeps of noisy maps is where a wrong number of levels came from; it is an error, not a guess
    rivals = {**below, **scores}
    runner = max([v_ for k_, v_ in rivals.items() if best is not None and k_ != best[1]] + [0.0])
    ambiguous = best is not None and runner > max_ambiguity * best[0]
    # the shape of the chosen groups: a merged pair of design levels shows as a jump inside a group or as a group
    # far wider than the tightest level (_merged_levels); a small map needs a score no gradient reaches by chance
    merged, hint = _merged_levels(s, best[2], floor) if best is not None else ("", "")
    # the score a small map must reach: small_min_score[0] at 8 plots or fewer, falling in a straight line to
    # small_min_score[1] at min_plots_free - 1 plots
    lo_n, hi_n = 8, min_plots_free - 1
    need = (small_min_score[0] + (small_min_score[1] - small_min_score[0]) * (min(max(n, lo_n), hi_n) - lo_n)
            / max(hi_n - lo_n, 1))
    too_small = best is not None and n < min_plots_free and best[0] < need
    if best is not None and best[0] >= min_score and not ambiguous and not merged and not too_small:
        sc, k, grp = best
        meds = np.array([np.median(s[grp == g]) for g in range(k)])
        widths = np.array([np.percentile(s[grp == g], 90) - np.percentile(s[grp == g], 10) for g in range(k)])
        counts = np.bincount(grp, minlength=k)
        if hint and sc < max_warn_score:
            res["warning"] = (f"the {k} levels of {name} may hide a finer step: {hint} (separation score {sc:.1f}); "
                              "check them against the trial design")
    else:
        spread = float(np.percentile(s, 90) - np.percentile(s, 10))
        # one level only when NO split separated the values (none eligible, or the best under min_score): a split
        # that scored but was refused as ambiguous, merged or on too small a map is a level structure in doubt,
        # and a design whose levels span under 10 % (100/110) must not come back silently as one level
        rejected = best is not None and best[0] >= min_score
        if not rejected and spread <= max(0.10 * abs(med_all), 6.0 * float(precision or 0.0), 1e-9):
            sc, k = None, 1
            meds, widths, counts = np.array([med_all]), np.array([spread]), np.array([n])
        else:
            if best is None:
                why = "no split leaves gaps of at least the noise floor between its levels"
            elif best[0] < min_score:
                why = (f"the best split ({best[1]} levels) separates its levels by only {best[0]:.2f} x the largest "
                       f"spacing inside a level (at least {min_score} is needed)")
            elif ambiguous:
                k2 = max((k_ for k_ in rivals if k_ != best[1]), key=lambda k_: rivals[k_])
                why = (f"{best[1]} and {k2} levels separate them about equally well (scores {best[0]:.2f} and "
                       f"{rivals[k2]:.2f}), so the number of levels is ambiguous")
            elif merged:
                why = (f"the best split ({best[1]} levels, score {best[0]:.2f}) is not trusted because {merged}, "
                       "so two levels may have run into each other")
            else:
                why = (f"only {n} plots carry a usable rate and the best split ({best[1]} levels) scores "
                       f"{best[0]:.2f}, not far enough above what a gradient reaches by chance on so few plots "
                       f"(at least {need:.1f} is needed at {n} plots, {small_min_score[0]:g} at 8 and "
                       f"{small_min_score[1]:g} at {min_plots_free - 1}): the design is too small for its levels "
                       "to be read from the map")
            raise ValueError(
                f"The as-applied map of {name} shows no discrete rate levels: the plot medians spread from "
                f"{s[0]:.1f} to {s[-1]:.1f} (p10-p90 {spread:.1f}) and {why}. A gradient or continuous "
                f"prescription, more than {max_levels} levels, plot-to-plot drift that runs the levels into each "
                f"other, or an unreplicated design (a level needs at least {min_plots} plots) cannot be read this "
                "way: give a design file for this input, or choose its design column.")
    keep = counts >= min_plots
    # one decimal: the level is a design rate (88.3, not 88.3078), and it names the strata and the Rx columns
    levels = [float(round(x, 1)) for x in meds[keep]]
    res.update(score=sc, single_plot_levels=int((~keep).sum()), widths=[float(w) for w in widths[keep]])
    if not levels:
        return res
    L = np.array(levels)
    lsteps = level_steps(L)
    tol = np.minimum(0.45 * lsteps, np.maximum(rel_tol * lsteps, 0.8 * widths[keep]))
    lev, hit = assign_levels(v[ok], L, tol=tol)
    out[ok] = lev
    res.update(level=out, levels=levels, tol=[float(t) for t in tol], k=int(len(levels)),
               step=float(np.diff(L).min()) if len(L) > 1 else 0.0,
               plots_per_level={float(x): int(np.sum(lev == x)) for x in levels},
               unassigned=int((~hit).sum()))
    return res


def level_steps(L):
    """The local step of each level: the gap to its nearest neighbouring level (one level: its own value, so that a
    relative tolerance still has a scale)."""
    L = np.asarray(L, dtype=float)
    if len(L) == 1:
        return np.array([abs(L[0]) if L[0] else 1.0])
    gaps = np.diff(L)
    return np.array([min(gaps[i - 1] if i > 0 else np.inf, gaps[i] if i < len(gaps) else np.inf) for i in range(len(L))])


def assign_levels(x, L, rel_tol=0.25, tol=None):
    """Nearest level of each value when it lies within that level's tolerance (`tol` per level, else rel_tol x the
    level's local step); (level or NaN, hit)."""
    x = np.asarray(x, dtype=float)
    L = np.asarray(L, dtype=float)
    tol = rel_tol * level_steps(L) if tol is None else np.asarray(tol, dtype=float)
    d = np.abs(x[:, None] - L[None, :])
    j = d.argmin(axis=1)
    hit = d[np.arange(len(j)), j] <= tol[j]
    return np.where(hit, L[j], np.nan), hit


def _mixed_plots(tbl, L, use, mix_frac=0.5, typical_spread=0.0):
    """Plots whose points carry two rates, judged against the level structure: p10 and p90 lie nearest to two
    different levels, or their spread (p90 - p10) exceeds mix_frac x the local step of the plot's nearest level.
    Either sign is only believed when the spread also exceeds 3 x the typical spread of a plot (the controller's
    own noise): on 10 % steps the p10 and p90 of a PURE plot with 4-5 % point noise already sit nearest to two
    levels, and without this floor 8-10 of 20 such plots were called mixed. Only plots in `use` are judged; the
    others are False."""
    L = np.asarray(L, dtype=float)
    steps = level_steps(L)
    lo, hi, med = tbl.p10.values, tbl.p90.values, tbl["median"].values
    j_lo = np.abs(lo[:, None] - L[None, :]).argmin(axis=1)
    j_hi = np.abs(hi[:, None] - L[None, :]).argmin(axis=1)
    j_md = np.abs(med[:, None] - L[None, :]).argmin(axis=1)
    floor = 3.0 * float(typical_spread or 0.0)
    mixed = ((j_lo != j_hi) | ((hi - lo) > mix_frac * steps[j_md])) & ((hi - lo) > floor)
    return mixed & use.values & np.isfinite(med)


def rates_from_asapplied(plots, inner, asapplied_utm, rate_col, name="input", min_pts=5, max_levels=15, min_plots=2,
                         mix_frac=0.5):
    """Design rates of an input read from its as-applied map, for a trial whose design file has no column for that
    input (Trial 22: the seed rates are in the design, the urea rates only on the drill's as-applied map, one rate
    per plot).

    Each plot takes the median rate of the as-applied points inside its trimmed interior (`inner`, so the ramps of
    the controller at the plot ends are left out); the plot medians are grouped into levels (rate_levels) and each
    plot gets its level as its design rate. The analysis units still take the applied medians of their own points
    (build_units), as for any input; the design rate is what counts the trial's distinct rates (the response form,
    the zone support, the cross-fit strata).

    Mixed plots (two rates inside one plot: a controller change, a pass overlap) are judged against the levels, in
    two passes: the levels are read from every plot with enough points, the plots whose p10 and p90 sit nearest to
    two different levels or whose spread exceeds mix_frac x the local step (and, either way, 3 x the typical
    spread of a plot: _mixed_plots) are set aside, and the levels are read again without them (two half-and-half
    plots could otherwise invent a level between two real ones).
    A plot gets NO design rate (NaN, counted in info) when it holds fewer than `min_pts` points, when it is mixed,
    or when its median lies off every level (rate_levels). Such a plot still carries its applied rate, like a check
    strip of a partly rated design: with rate_source 'applied' it enters the response with that rate, with
    rate_source 'target' it is left out.
    Returns (pandas Series of design rates aligned with `plots`, info dict)."""
    if not rate_col or rate_col not in asapplied_utm.columns:
        raise ValueError(f"The rates of {name} are to be read from its as-applied map, but its rate column "
                         f"{rate_col!r} is not in that file: choose the applied-rate column of {name} in step 1")
    a = asapplied_utm[[rate_col, "geometry"]].copy().reset_index(drop=True)
    a["rate"] = pd.to_numeric(a[rate_col], errors="coerce")
    a = a[np.isfinite(a.rate) & a.geometry.notna()]
    a = a[~a.geometry.is_empty]
    if not len(a):
        raise ValueError(f"The as-applied map of {name} has no numeric value in its rate column {rate_col!r}")
    notpt = (a.geom_type != "Point").values
    if notpt.any():
        a.loc[notpt, "geometry"] = a.geometry[notpt].representative_point()
    j = gpd.sjoin(a[["rate", "geometry"]], inner[["plot_id", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated()]
    if not len(j):
        raise ValueError(f"None of the {len(a):,} as-applied points of {name} falls inside a trimmed plot (another "
                         "field, or a coordinate-system problem?), so no design rate can be read from that map")
    g = j.groupby("plot_id").rate
    tbl = pd.DataFrame({"n": g.size(), "median": g.median(), "p10": g.quantile(0.10), "p90": g.quantile(0.90)})
    tbl["spread"] = tbl.p90 - tbl.p10
    tbl = tbl.reindex(plots.plot_id.values)
    few = tbl.n.isna() | (tbl.n < min_pts)
    use = ~few
    if not use.any():
        raise ValueError(f"No plot holds {min_pts} or more as-applied points of {name} inside its trimmed interior "
                         f"({int(len(j)):,} points in the plots): the map may be sparse, or the trims too wide")
    # precision of one plot median: about 0.5 x (p90 - p10 of its points) / sqrt(points), the typical one
    prec = float(np.median(0.5 * tbl.loc[use, "spread"] / np.sqrt(tbl.loc[use, "n"])))
    kw = dict(precision=prec, max_levels=max_levels, min_plots=min_plots, name=name)
    # pass 1 leaves out the plots whose points spread far more than the typical plot (4 x the median spread,
    # floored by the noise floor of a level): a plot split between two levels has a median anywhere between them
    # and would widen or bridge the groups. Pass 2 reads the levels again from every plot that pass 1's levels do
    # not call mixed, so a level made only of such wide but pure plots (a check level logging residual readings
    # next to exactly logged rates) is not lost; when pass 2 finds fewer levels than pass 1 (pure plots flagged
    # on a noisy map), pass 1's levels are kept.
    typical = float(tbl.loc[use, "spread"].median())
    wide = use & (tbl.spread > 4.0 * max(typical, 0.02 * abs(float(tbl.loc[use, "median"].median())), 1e-9))
    first = use & ~wide if (use & ~wide).sum() >= 2 else use
    lv1 = rate_levels(tbl.loc[first, "median"].values, **kw)
    lv, mixed = lv1, pd.Series(False, index=tbl.index)
    if lv1["levels"]:
        mixed[:] = _mixed_plots(tbl, lv1["levels"], use, mix_frac, typical)
        try:
            lv2 = rate_levels(tbl.loc[use & ~mixed, "median"].values, **kw)
        except ValueError:
            lv2 = None
        if lv2 is not None and len(lv2["levels"]) >= len(lv1["levels"]):
            lv = lv2
            mixed[:] = _mixed_plots(tbl, lv["levels"], use, mix_frac, typical)
    level = pd.Series(np.nan, index=plots.index, dtype=float)
    if lv["levels"]:
        m = (use & ~mixed).values
        level[m], _ = assign_levels(tbl.loc[m, "median"].values, lv["levels"], tol=lv["tol"])
    off = (use & ~mixed).values & ~np.isfinite(level.values)
    ppl = {float(x): int(np.sum(level.values == x)) for x in lv["levels"]}
    # warnings, joined into one text: a separation score under 3 passed the rules but is in the range where noisy
    # synthetic maps were sometimes read with a wrong number of levels; rate_levels' own shape warning (a finer
    # step may be hidden); and a map of fewer than 24 usable plots, where the level structure rests on too few
    # plots to be sure of (rate_levels only reads it at a score of 6 or more): each says to check the levels
    # against the trial design, with the plots per level and the width (p10-p90) of each level for that check
    warn = []
    if lv.get("score") is not None and lv["score"] < 3.0:
        warn.append(f"the levels of {name} are only moderately separated (score {lv['score']:.1f}; clean drill "
                    "maps score above 3): check them against the trial design")
    if lv.get("warning"):
        warn.append(lv["warning"])
    n_usable = int((use & ~mixed).sum())
    if n_usable < 24 and lv["levels"]:
        per = ", ".join(f"{x:g} ({ppl[x]} plots, width {w:.1f})" for x, w in zip(lv["levels"], lv.get("widths") or []))
        warn.append(f"only {n_usable} plots of {name} carry a usable rate, too few to be sure of the levels read from "
                    f"the map: {per}; check them against the trial design")
    info = dict(input=name, column=rate_col, levels=lv["levels"], plots_per_level=ppl, score=lv.get("score"),
                n_plots=int(len(plots)), n_with_level=int(np.isfinite(level.values).sum()),
                n_few_points=int(few.sum()), n_mixed=int(mixed.sum()), n_off_level=int(off.sum()),
                min_pts=int(min_pts), mix_frac=float(mix_frac), precision=prec, step=lv.get("step"),
                widths=lv.get("widths") or [], warning="; ".join(warn) if warn else None,
                single_plot_levels=int(lv.get("single_plot_levels") or 0),
                plot_medians={str(p): (None if not np.isfinite(m_) else float(m_))
                              for p, m_ in zip(plots.plot_id.values, tbl["median"].values)},
                points_in_plots=int(len(j)))
    if not lv["levels"]:
        spl = int(lv.get("single_plot_levels") or 0)
        raise ValueError(
            f"No design level could be read from the as-applied map of {name}: {int((use & ~mixed).sum())} plot(s) "
            f"with a usable rate ({int(few.sum())} with fewer than {min_pts} points, {int(mixed.sum())} with mixed "
            f"rates) fall into {spl} level(s) of a single plot each, and a design level needs at least {min_plots} "
            "plots (an unreplicated design, one plot per rate, cannot be read this way: give a design file)")
    return level, info


def _oriented(poly):
    """Centre, long-axis unit vector, short-axis unit vector, half length, half width of the minimum rotated
    rectangle of a polygon."""
    r = poly.minimum_rotated_rectangle
    xy = np.asarray(r.exterior.coords)[:4]
    e1, e2 = xy[1] - xy[0], xy[2] - xy[1]
    l1, l2 = np.hypot(*e1), np.hypot(*e2)
    if l1 >= l2:
        u, L, v, W = e1 / l1, l1, e2 / l2, l2
    else:
        u, L, v, W = e2 / l2, l2, e1 / l1, l1
    c = xy.mean(axis=0)
    return c, u, v, L / 2, W / 2


def _shrink_along(geom, axis, d):
    """`geom` with `d` metres taken off both of its ends along the unit vector `axis`: the part of the plot that is
    still inside it when it is slid d forward and d back. A point is kept only when the plot also holds the
    points d ahead and d behind it, so the kept part is exactly d from every end edge whatever the edge's shape;
    on a rectangle this is the rectangle shortened by d at both ends."""
    if d <= 0:
        return geom
    dx, dy = d * float(axis[0]), d * float(axis[1])
    return geom.intersection(translate(geom, dx, dy)).intersection(translate(geom, -dx, -dy))


def trim_plots(plots, end_m=10.0, side_m=0.0):
    """The analysed part of each plot: the plot shortened by `end_m` at both ends of its long axis and `side_m` on
    both sides (_shrink_along, with the axes of the plot's minimum rotated rectangle). Adds the long-axis azimuth
    (deg from grid north).

    The ends are taken off the plot itself, not off its rotated rectangle: a trapezoid or wedge (a plot clipped to
    a field boundary) has an end edge that runs inside its rectangle, and the rectangle shortened by end_m kept
    0 m of trim along part of that edge (9 m at one side of the plot, none at the other). A plot too short for the
    trim (under 2 x end_m + 1 m along its axis, or 2 x side_m + 1 m across) keeps the old reading, a 1 m strip
    across its middle, so its records are still counted (and dropped downstream by the unit's min_obs rule)."""
    out, az = [], []
    for geom in plots.geometry:
        c, u, v, hl, hw = _oriented(geom)
        inner_g = None
        if hl - end_m >= 0.5 and (side_m <= 0 or hw - side_m >= 0.5):
            inner_g = _shrink_along(_shrink_along(geom, u, end_m), v, side_m)
        if inner_g is None or inner_g.is_empty:
            hl2, hw2 = max(0.5, hl - end_m), max(0.5, hw - side_m)
            pts = [c + s1 * hl2 * u + s2 * hw2 * v for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            inner_g = Polygon(pts).intersection(geom)
        out.append(inner_g)
        az.append(float(np.degrees(np.arctan2(u[0], u[1])) % 180))
    inner = plots.copy()
    inner["geometry"] = out
    inner["axis_deg"] = az
    inner["inner_m2"] = inner.geometry.area
    return inner


# ================================================================================================ yield
def _grid_rotation(gdf_utm):
    """Angle (deg) to ADD to a true azimuth to get the grid azimuth at the field (meridian convergence)."""
    c = gdf_utm.geometry.union_all().centroid
    to_ll = Transformer.from_crs(gdf_utm.crs, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, gdf_utm.crs, always_xy=True)
    lon, lat = to_ll.transform(c.x, c.y)
    x1, y1 = to_utm.transform(lon, lat + 0.001)
    return float(np.degrees(np.arctan2(x1 - c.x, y1 - c.y)))


def yield_polygons(pts_utm, ycol, heading_col=None, dist_col=None, swath_col=None, heading_is="true",
                   default_swath_m=None, swath_unit=None, dist_unit=None):
    """Rectangles swath x distance centred on each point along the heading. Without heading, distance or swath the
    points are kept as points (mode 'point'). `swath_unit` / `dist_unit` ('ft' | 'm', cfg columns.swath_unit /
    distance_unit) name the unit of those columns when the user chose one; without them the column name decides
    (io.length_to_m).

    The frame carries attrs['area_known']: True when every observation has a real area in m2 (polygon mode, or point
    mode with both swath and distance), False otherwise. In point mode without a swath (no swath column and no
    `default_swath_m`) or without a distance, an observation has no area: every point then weighs 1.0, a plain
    count. That weight is fine for an area-weighted mean inside one unit (the points are evenly spaced along the
    passes), but it is NOT square metres: the unit area test (min_area_m2) and any acreage built from it are
    meaningless, which is what area_known = False tells assign_yield, build_units and the report."""
    g = pts_utm.copy()
    g["y"] = pd.to_numeric(g[ycol], errors="coerce")
    g = _drop_engine_cols(g, keep=(ycol,))
    g = g[np.isfinite(g["y"])].copy()
    if swath_col and swath_col in g:
        sw = eio.length_to_m(g[swath_col], swath_col, swath_unit)
    else:
        sw = pd.Series(np.nan if default_swath_m is None else float(default_swath_m), index=g.index, dtype=float)
    ds = (eio.length_to_m(g[dist_col], dist_col, dist_unit) if dist_col and dist_col in g
          else pd.Series(np.nan, index=g.index))
    sw = pd.to_numeric(sw, errors="coerce").astype(float)
    ds = pd.to_numeric(ds, errors="coerce").astype(float)
    sw = sw.where(sw > 0)
    ds = ds.where(ds > 0)
    g["swath_m"], g["dist_m"] = sw, ds
    # rectangles need a heading, a positive distance and a positive swath for most records (a distance column of
    # zeros would give degenerate rings)
    have = bool(heading_col and heading_col in g and dist_col and dist_col in g and (sw.notna().mean() > 0.5)
                and (ds.notna().mean() > 0.5))
    if not have:
        if sw.notna().any() and ds.notna().any():
            g["w_area"] = sw.fillna(sw.median()) * ds.fillna(ds.median())
            g.attrs["area_known"] = True
        else:
            # no real area: a uniform weight (a count), flagged so nobody reads it as m2
            g["w_area"] = 1.0
            g.attrs["area_known"] = False
        return g, "point"
    sw = sw.fillna(sw.median())
    ds = ds.fillna(ds.median())
    hd = pd.to_numeric(g[heading_col], errors="coerce").fillna(0).values
    if heading_is == "true":
        hd = hd + _grid_rotation(g)
    a = np.radians(hd)
    ux, uy = np.sin(a), np.cos(a)          # travel direction
    vx, vy = uy, -ux                       # across
    x, y = g.geometry.x.values, g.geometry.y.values
    hl, hw = ds.values / 2, sw.values / 2
    corners = np.stack([
        np.c_[x - hl * ux - hw * vx, y - hl * uy - hw * vy],
        np.c_[x + hl * ux - hw * vx, y + hl * uy - hw * vy],
        np.c_[x + hl * ux + hw * vx, y + hl * uy + hw * vy],
        np.c_[x - hl * ux + hw * vx, y - hl * uy + hw * vy],
        np.c_[x - hl * ux - hw * vx, y - hl * uy - hw * vy]], axis=1)
    g["px"], g["py"] = x, y
    g = g.set_geometry(shapely.polygons(corners), crs=pts_utm.crs)
    g["w_area"] = g.geometry.area
    g.attrs["area_known"] = True
    return g, "polygon"


def assign_yield(ypoly, inner, mode="polygon", min_share=0.95):
    """Observations inside the trimmed plots. Returns (pieces, attrition dict). A piece = the part of an observation
    inside one plot, with its area as the weight (a count of 1 per point when the yield frame has
    attrs['area_known'] False; the flag is passed on in pieces.attrs and att['area_known'])."""
    area_known = bool(ypoly.attrs.get("area_known", True))
    ypoly = _drop_engine_cols(ypoly.reset_index(drop=True))
    ypoly["obs_id"] = np.arange(len(ypoly))
    n0 = len(ypoly)
    if mode == "point":
        j = gpd.sjoin(ypoly, inner[["plot_id", "geometry"]], how="inner", predicate="within")
        j = j[~j.obs_id.duplicated()]
        pieces = j.drop(columns=["index_right"]).copy()
        pieces["piece_m2"] = pieces.w_area
        pieces["share"] = 1.0
        att = dict(n_obs=n0, kept=int(len(pieces)), outside_trimmed_plots=int(n0 - len(pieces)), straddling=0,
                   area_known=area_known)
        pieces.attrs["area_known"] = area_known
        return pieces, att
    ov = gpd.overlay(ypoly[["obs_id", "geometry"]], inner[["plot_id", "geometry"]], how="intersection",
                     keep_geom_type=True)
    ov["piece_m2"] = ov.geometry.area
    ov = ov.merge(ypoly.drop(columns="geometry"), on="obs_id", how="left")
    ov["share"] = ov.piece_m2 / ov.w_area
    best = ov.groupby("obs_id").share.max()
    tot = ov.groupby("obs_id").share.sum()
    if min_share > 0:
        good = best[best >= min_share].index
        pieces = ov[ov.obs_id.isin(good)].sort_values("share", ascending=False).drop_duplicates("obs_id")
    else:
        pieces = ov[ov.share >= 0.02].copy()
    touched = set(ov.obs_id)
    straddle = int(((best < min_share) & (tot >= min_share)).sum()) if min_share > 0 else 0
    att = dict(n_obs=n0, kept=int(pieces.obs_id.nunique()),
               outside_trimmed_plots=int(n0 - len(touched)),
               partly_in_transition_or_edge=int(len(touched) - pieces.obs_id.nunique() - straddle),
               straddling=straddle, min_share=float(min_share), area_known=area_known)
    pieces = gpd.GeoDataFrame(pieces, geometry="geometry", crs=inner.crs)
    pieces.attrs["area_known"] = area_known
    return pieces, att


# ================================================================================================ as-applied
def applied_points(asapplied_utm, rate_col, inner, zone_raster=None, grid=None):
    """As-applied points inside the trimmed plots with their plot and zone.

    The zone follows the yield-piece rule of build_units (_nearest_zone): a point on an unzoned cell takes the zone
    of the nearest valid cell among its 8 neighbours, and a point further out keeps zone NaN (it then counts in
    neither a unit median nor the plot-median fallback).

    When the rate column is missing (none recognised and none chosen), not in the file, or holds no numeric value
    (a text column chosen by mistake), the result is an EMPTY frame with attrs['warning'] saying why, and a
    UserWarning is raised: build_units then uses the target (design) rate for that input, as it does when there is
    no as-applied file, instead of the whole join stopping on a KeyError / IndexError that names nothing."""
    def _empty(msg):
        warnings.warn(msg, UserWarning, stacklevel=3)
        e = gpd.GeoDataFrame({"rate": pd.Series(dtype=float), "plot_id": pd.Series(dtype=object)},
                             geometry=gpd.GeoSeries([], crs=inner.crs), crs=inner.crs)
        if zone_raster is not None and grid is not None:
            e["zone"] = pd.Series(dtype=float)
        e.attrs["warning"] = msg
        return e

    if not rate_col or rate_col not in asapplied_utm.columns:
        return _empty(f"As-applied file: rate column {rate_col!r} not found; the target rate is used instead")
    a = asapplied_utm[[rate_col, "geometry"]].copy().reset_index(drop=True)
    a["rate"] = pd.to_numeric(a[rate_col], errors="coerce")
    a = a[np.isfinite(a.rate) & a.geometry.notna()]
    a = a[~a.geometry.is_empty]
    if not len(a):
        return _empty(f"As-applied file: column {rate_col!r} has no numeric rate; the target rate is used instead")
    notpt = (a.geom_type != "Point").values
    if notpt.any():
        a.loc[notpt, "geometry"] = a.geometry[notpt].representative_point()
    j = gpd.sjoin(a[["rate", "geometry"]], inner[["plot_id", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated()]
    if not len(j):
        # valid rates, but no point inside a trimmed plot: an as-applied map of another field, a wrong coordinate
        # system, or the wrong file; say so instead of silently using the target rate
        return _empty(f"As-applied file: none of its {len(a):,} points with a rate falls inside a trimmed plot "
                      "(another field, or a coordinate-system problem?); the target rate is used instead")
    if zone_raster is not None and grid is not None:
        z = grid.sample(zone_raster, j.geometry.x.values, j.geometry.y.values)
        z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
        # points on a cell without a zone take the zone of the nearest valid cell among its 8 neighbours, with
        # _nearest_zone, exactly the rule build_units applies to the yield pieces: an edge cell whose centre is
        # just outside the boundary mask keeps its points (so edge units do not lose their medians), while a
        # point more than one cell outside the zone map keeps NaN, as the yield piece at the same place is
        # dropped. An unlimited nearest-cell fill (distance transform) would put into the edge unit's median the
        # rates of ground whose yield is never analysed, typically the applicator ramp where a plot runs past a
        # boundary drawn tighter than the design.
        bad = ~np.isfinite(z)
        if bad.any():
            z[bad] = _nearest_zone(grid, zone_raster, j.geometry.x.values[bad], j.geometry.y.values[bad])
        j["zone"] = z
    return j


# ================================================================================================ units
def _nearest_zone(grid, zone_raster, x, y):
    """Zone of the nearest valid (> 0) cell among the 8 neighbours of the cell holding (x, y); NaN when none.

    The boundary mask is rasterised with all_touched=False, so a 10 m cell on the field edge whose CENTRE is outside
    the boundary has zone 0 even though part of it (and a yield piece in it) is inside the trimmed plot. Taking the
    neighbouring cell's zone keeps those edge pieces; a piece more than one cell outside the zone map stays unzoned."""
    t = grid.transform
    r, c = grid.xy_to_rc(x, y)
    best_d = np.full(len(r), np.inf)
    best_z = np.full(len(r), np.nan)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            ok = (rr >= 0) & (rr < grid.height) & (cc >= 0) & (cc < grid.width)
            zz = np.full(len(r), np.nan)
            zz[ok] = zone_raster[rr[ok], cc[ok]]
            ok &= np.isfinite(zz) & (zz > 0)
            cx, cy = t.c + (cc + 0.5) * t.a, t.f + (rr + 0.5) * t.e
            d = np.hypot(np.asarray(x) - cx, np.asarray(y) - cy)
            better = ok & (d < best_d)
            best_d[better], best_z[better] = d[better], zz[better]
    return best_z


def build_units(pieces, plots, inputs, grid=None, zone_raster=None, ndvi_surface=None, applied=None,
                min_obs=10, min_area_m2=150.0, machine_col=None, unit="plot_zone", area_known=None):
    """Analysis units. `inputs` = list of input names; plots carry design_rate (first input) and design_<name> for
    the others; `applied` = {name: applied points DataFrame}. unit = plot_zone (default) or plot.

    `area_known` (default: pieces.attrs['area_known'], else True): when False the piece weights are point counts,
    not m2 (point mode without a swath), so the min_area_m2 test is skipped (a count compared with m2 would drop
    every unit with fewer than 150 points) and info['area_known'] = False tells the caller that area_m2 / area_ac
    are relative weights, not areas.

    Pieces whose centroid falls on no zone (zone 0 / NaN: the cell centre is outside the boundary mask or the zone
    map) first take the zone of the nearest valid neighbouring cell; those still without a zone are dropped and
    counted in info (dropped_no_zone_pieces / _obs / _m2, filled_nearest_zone_pieces), so the attrition reported to
    the user adds up to what actually reaches the units."""
    if area_known is None:
        area_known = bool(pieces.attrs.get("area_known", True))
    p = pieces.copy()
    c = p.geometry.centroid
    p["x_m"], p["y_m"] = c.x.values, c.y.values
    n_filled = 0
    if grid is not None and zone_raster is not None:
        z = grid.sample(zone_raster, p.x_m.values, p.y_m.values)
        z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
        miss = ~np.isfinite(z)
        if miss.any():
            zn = _nearest_zone(grid, zone_raster, p.x_m.values[miss], p.y_m.values[miss])
            z[miss] = zn
            n_filled = int(np.isfinite(zn).sum())
        p["zone"] = z
    else:
        p["zone"] = 1
    if grid is not None and ndvi_surface is not None:
        p["ndvi"] = grid.sample(ndvi_surface, p.x_m.values, p.y_m.values)
    else:
        p["ndvi"] = np.nan
    nz = ~np.isfinite(p.zone.astype(float))
    obs_kept = set(p.loc[~nz, "obs_id"])
    no_zone = dict(dropped_no_zone_pieces=int(nz.sum()),
                   dropped_no_zone_obs=int(len(set(p.loc[nz, "obs_id"]) - obs_kept)),
                   dropped_no_zone_m2=float(p.loc[nz, "piece_m2"].sum()) if area_known else None,
                   filled_nearest_zone_pieces=n_filled)
    p = p[~nz].copy()
    p["zone"] = p.zone.astype(int)
    if unit == "plot":
        p["zone_unit"] = 0
    else:
        p["zone_unit"] = p.zone
    w = p.piece_m2
    p["_wy"], p["_wx"], p["_wyy"] = p["y"] * w, p.x_m * w, p.y_m * w
    p["_wn"] = p.ndvi.fillna(0) * w
    p["_wnv"] = np.isfinite(p.ndvi) * w
    key = ["plot_id", "zone_unit"]
    agg = p.groupby(key).agg(area_m2=("piece_m2", "sum"), n_obs=("obs_id", "nunique"), _wy=("_wy", "sum"),
                             _wx=("_wx", "sum"), _wyy=("_wyy", "sum"), _wn=("_wn", "sum"), _wnv=("_wnv", "sum"),
                             yield_sd=("y", "std"))
    agg["yield"] = agg._wy / agg.area_m2
    agg["x_m"] = agg._wx / agg.area_m2
    agg["y_m"] = agg._wyy / agg.area_m2
    agg["ndvi"] = np.where(agg._wnv > 0, agg._wn / agg._wnv.replace(0, np.nan), np.nan)
    zmaj = p.groupby(key + ["zone"]).piece_m2.sum().reset_index().sort_values("piece_m2").drop_duplicates(key, keep="last")
    agg = agg.join(zmaj.set_index(key)["zone"])
    if machine_col and machine_col in p:
        p["_m"] = p[machine_col].astype(str)
        sh = p.pivot_table(index=key, columns="_m", values="piece_m2", aggfunc="sum", fill_value=0.0)
        sh = sh.div(sh.sum(axis=1), axis=0)
        sh.columns = [f"sh_{c}" for c in sh.columns]
        agg = agg.join(sh)
        agg["machine"] = sh.idxmax(axis=1).str[3:]
    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("_")]).reset_index()
    pl = plots.drop(columns="geometry").set_index("plot_id")
    agg = agg.join(pl[[c for c in pl.columns if c.startswith("design") or c in ("treatment", "block")]], on="plot_id")
    # applied rate per input: unit median, else plot median, else design
    for k, name in enumerate(inputs):
        dcol = "design_rate" if k == 0 else f"design_{name}"
        # float on purpose: plots built outside prepare_design may still carry an int64 design rate, and the
        # as-applied medians written into rate_ below would then raise under pandas >= 3 (no silent upcast)
        agg[f"target_{name}"] = agg[dcol].astype(float) if dcol in agg else np.nan
        agg[f"rate_{name}"] = agg[f"target_{name}"].astype(float)
        agg[f"rate_src_{name}"] = "target"
        A = (applied or {}).get(name)
        if A is not None and len(A):
            if unit != "plot" and "zone" in A:
                A = A.copy()
                A["zone_unit"] = A.zone
                u = A.groupby(["plot_id", "zone_unit"]).rate.agg(["median", "count", "std"])
                agg = agg.join(u.rename(columns={"median": f"_au_{name}", "count": f"_an_{name}",
                                                 "std": f"rate_sd_{name}"}), on=key)
            # the plot-median fallback uses the same points as the yield: with a zone map, points left without a
            # zone (more than one cell outside it, see applied_points) are on ground whose yield pieces were
            # dropped, so they are left out here too
            Ap = A[np.isfinite(A.zone.astype(float))] if "zone" in A else A
            plm = Ap.groupby("plot_id").rate.median().rename(f"_ap_{name}")
            agg = agg.join(plm, on="plot_id")
            use_u = f"_au_{name}" in agg
            r = agg[f"rate_{name}"].copy()
            src = agg[f"rate_src_{name}"].copy()
            if f"_ap_{name}" in agg:
                m = agg[f"_ap_{name}"].notna()
                r[m], src[m] = agg.loc[m, f"_ap_{name}"], "applied (plot median)"
            if use_u:
                m = agg[f"_au_{name}"].notna() & (agg[f"_an_{name}"] >= 5)
                r[m], src[m] = agg.loc[m, f"_au_{name}"], "applied (unit median)"
            agg[f"rate_{name}"], agg[f"rate_src_{name}"] = r, src
            agg = agg.drop(columns=[c for c in agg.columns if c.startswith(("_au_", "_an_", "_ap_"))])
    n_before = len(agg)
    small = (agg.area_m2 < min_area_m2) if area_known else pd.Series(False, index=agg.index)
    agg["dropped"] = np.where(agg.n_obs < min_obs, "few observations", np.where(small, "small area", ""))
    kept = agg[agg.dropped == ""].drop(columns="dropped").reset_index(drop=True)
    # with area_known False this is a relative weight (count / M2_PER_AC), kept so downstream weighting still works
    kept["area_ac"] = kept.area_m2 / eio.M2_PER_AC
    kept.insert(0, "unit_id", [f"{a}_{b}" for a, b in zip(kept.plot_id, kept.zone_unit)])
    info = dict(n_units_raw=int(n_before), n_units=int(len(kept)),
                dropped_few_obs=int((agg.dropped == "few observations").sum()),
                dropped_small=int((agg.dropped == "small area").sum()),
                n_plots=int(kept.plot_id.nunique()), min_obs=int(min_obs),
                min_area_m2=float(min_area_m2), unit=unit, area_known=bool(area_known),   # area test applied only if known
                **no_zone)
    return kept, p, info


def unit_polygons(pieces_with_units, units):
    """One polygon per unit (union of its observation pieces), for maps and the shapefile export."""
    p = pieces_with_units.copy()
    p["unit_id"] = [f"{a}_{b}" for a, b in zip(p.plot_id, p.zone_unit)]
    p = p[p.unit_id.isin(set(units.unit_id))]
    d = p[["unit_id", "geometry"]].dissolve("unit_id")
    d["geometry"] = d.geometry.buffer(0.3).buffer(-0.3)
    return gpd.GeoDataFrame(units.set_index("unit_id").join(d), geometry="geometry", crs=p.crs).reset_index()


def plot_table(units, inputs):
    """Plot-level summary (the unit of randomisation): area-weighted yield, NDVI, zone shares, rates."""
    rows = []
    for pid, g in units.groupby("plot_id"):
        w = g.area_m2
        r = dict(plot_id=pid, area_ac=float(g.area_ac.sum()), n_units=len(g),
                 yield_=float(np.average(g["yield"], weights=w)),
                 ndvi=float(np.average(g.ndvi.fillna(g.ndvi.mean()), weights=w)) if g.ndvi.notna().any() else np.nan,
                 zone_major=int(g.loc[w.idxmax(), "zone"]), x_m=float(np.average(g.x_m, weights=w)),
                 y_m=float(np.average(g.y_m, weights=w)))
        for z, gz in g.groupby("zone"):
            r[f"zone{z}_share"] = float(gz.area_m2.sum() / w.sum())
        for n in inputs:
            r[f"rate_{n}"] = float(np.average(g[f"rate_{n}"], weights=w))
            r[f"target_{n}"] = float(g[f"target_{n}"].iloc[0]) if f"target_{n}" in g else np.nan
        for c in ("treatment", "block", "machine"):
            if c in g:
                r[c] = g[c].iloc[0]
        for c in [c for c in g.columns if c.startswith("sh_")]:
            r[c] = float(np.average(g[c], weights=w))
        rows.append(r)
    return pd.DataFrame(rows).rename(columns={"yield_": "yield"})


def zone_field_areas(zone_raster, grid):
    m = grid.mask & (zone_raster > 0)
    ks, n = np.unique(zone_raster[m], return_counts=True)
    ac = n * grid.cell_area_m2() / eio.M2_PER_AC
    return {int(k): float(a) for k, a in zip(ks, ac)}
