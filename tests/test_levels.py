# -*- coding: utf-8 -*-
"""Level reading from as-applied maps (join.rate_levels / rates_from_asapplied / _mixed_plots) and the plot trims
(join.trim_plots) on the cases of the 2026-10 audit: merged levels on 10 %-step and unequal-spacing designs, a
gradient on a small design, pure plots with noisy points, a trapezoid plot."""
import os
import sys

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import box, Point, Polygon, LineString

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import join as ej  # noqa: E402
import synth  # noqa: E402


def sweep(gen, prec, want, n=20):
    """(right, wrong, raised) counts of the number of levels read over n seeds: a ValueError is acceptable, a wrong
    number of levels returned silently is not."""
    right = wrong = err = 0
    for seed in range(n):
        try:
            k = len(ej.rate_levels(gen(np.random.default_rng(seed)), prec)["levels"])
            right += k == want
            wrong += k != want
        except ValueError:
            err += 1
    return right, wrong, err


def test_rate_levels_merged_never_returned_silently():
    """Defect A: 90/100/110/130 with plot-to-plot sd 1.5-2 and 100/120/200/300 with sd 2 used to come back as 2 or 3
    levels, every plot at a level and no warning (9 to 16 of 20 seeds). They may raise; they may not be wrong."""
    for per_level in (20, 35):
        for sd in (1.5, 2.0):
            for design in ((90, 100, 110, 130), (100, 120, 200, 300)):
                gen = lambda r, d=design, s=sd, m=per_level: np.concatenate([l + r.normal(0, s, m) for l in d])
                right, wrong, err = sweep(gen, 0.2, 4)
                assert wrong == 0, (design, sd, per_level, right, wrong, err)
    # the same designs with the drift well under half the smallest step are still read
    gen = lambda r: np.concatenate([l + r.normal(0, 1.0, 20) for l in (90, 100, 110, 130)])
    right, wrong, err = sweep(gen, 0.2, 4)
    assert wrong == 0 and right >= 18


def test_rate_levels_clean_t22_map_still_five():
    """A clean Trial-22-like map (88/131/175/219/262, 168 plots, +-6 % plot-to-plot drift) keeps its 5 levels in at
    least 95 of 100 seeds: the shape rules of the fix must not cost more than a few clean maps."""
    T22 = (88, 131, 175, 219, 262)
    gen = lambda r: np.concatenate([l * (1 + r.uniform(-.06, .06, nn)) for l, nn in zip(T22, (34, 34, 34, 33, 33))])
    right, wrong, err = sweep(gen, 0.17, 5, n=100)
    assert wrong == 0 and right >= 95, (right, wrong, err)


def test_rate_levels_gradient_on_small_design_raises():
    """Defect B: a gradient prescription (uniform 60-200) on 8 to 20 plots was read as 2-3 levels in 6 to 24 % of
    seeds (backup: 9, 10, 6, 1 of 50). Below 24 plots the score must reach 4 (at 8 plots) to 2.5 (at 23); the
    message says the design is too small. The rule is a trade-off with the clean small designs of the next test,
    so a few gradients still pass (measured 1, 1, 0, 0 of 50): the limit here is 3 of 50 per size."""
    for n in (8, 12, 16, 20):
        right, wrong, err = sweep(lambda r, n=n: r.uniform(60, 200, n), 0.2, 0, n=50)
        assert wrong <= 3 and err >= 47, (n, right, wrong, err)
    with pytest.raises(ValueError, match="too small for its levels to be read from the map|no discrete rate levels"):
        ej.rate_levels(np.random.default_rng(1).uniform(60, 200, 12), 0.2, name="urea")
    # a small but clean map (three levels of six plots, 20 % steps) still reads, with a warning through the as-applied
    # path (below 24 usable plots: levels, plots per level and widths are listed)
    r = ej.rate_levels(np.concatenate([l + np.random.default_rng(0).normal(0, .5, 6) for l in (80, 100, 120)]), 0.2)
    assert len(r["levels"]) == 3 and len(r["widths"]) == 3


def test_rate_levels_small_clean_designs_mostly_read():
    """The other side of the small-design rule: plausible small OFPE designs with realistic drift must mostly be
    read, never wrong. Measured on 100 seeds (backup / now): 100/120/140/160 x 5 at sd 2: 97 / 90 right;
    100/110 x 6 at sd 1: 98 / 93; 100/105/110 x 8 at sd 0.6: 96 / 94; the rest raise. A T22-like 5-level design
    on 4 plots per level (+-6 % drift) is not readable by either (backup 34 right and 6 WRONG, now 8 right and 0
    wrong): it must at least never be wrong."""
    cases = [
        ((100, 120, 140, 160), lambda r: np.concatenate([l + r.normal(0, 2, 5) for l in (100, 120, 140, 160)]), 0.2, 85),
        ((100, 110), lambda r: np.concatenate([l + r.normal(0, 1, 6) for l in (100, 110)]), 0.2, 88),
        ((100, 105, 110), lambda r: np.concatenate([l + r.normal(0, .6, 8) for l in (100, 105, 110)]), 0.2, 88),
        ((88, 131, 175, 219, 262),
         lambda r: np.concatenate([l * (1 + r.uniform(-.06, .06, 4)) for l in (88, 131, 175, 219, 262)]), 0.17, 0),
    ]
    for design, gen, prec, min_right in cases:
        right, wrong, err = sweep(gen, prec, len(design), n=100)
        assert wrong == 0 and right >= min_right, (design, right, wrong, err)


def test_rate_levels_two_close_levels_never_one():
    """Defect 1 of the second review: a design whose levels span under 10 % of the median (100/110) used to fall
    through to the one-level fallback when its split was refused (too small, merged, ambiguous) and come back
    silently as ONE level of 105. Now: two levels or a ValueError, never one."""
    for seed in range(100):
        r = np.random.default_rng(seed)
        try:
            k = len(ej.rate_levels(np.concatenate([l + r.normal(0, 1, 6) for l in (100, 110)]), 0.2)["levels"])
            assert k == 2, (seed, k)
        except ValueError:
            pass
    # 100/105/110 x 8 (seed 23 was read as [105.2] after a 'merged' refusal) too
    for seed in range(50):
        r = np.random.default_rng(seed)
        try:
            k = len(ej.rate_levels(np.concatenate([l + r.normal(0, .6, 8) for l in (100, 105, 110)]), 0.2)["levels"])
            assert k == 3, (seed, k)
        except ValueError:
            pass
    # a map that applied one rate is still one level
    one = ej.rate_levels(np.full(20, 100.0) + np.random.default_rng(0).normal(0, 0.3, 20), 0.2)
    assert len(one["levels"]) == 1 and one["levels"][0] == pytest.approx(100.0, abs=0.5) and one["score"] is None


def test_rate_levels_exactly_logged_check_level():
    """Defect 2 of the second review: a level the controller logged exactly (a 0-rate check strip: width 0) made
    every other level 'wider than the tightest' by any ratio and the map raised as merged. 0/100/150/200 with
    +-6 % drift on 20 plots per level reads 4 levels in every seed (backup: the same)."""
    gen = lambda r: np.concatenate([np.zeros(20)] + [l * (1 + r.uniform(-.06, .06, 20)) for l in (100, 150, 200)])
    assert sweep(gen, 0.2, 4, n=50) == (50, 0, 0)
    r = ej.rate_levels(gen(np.random.default_rng(0)), 0.2)
    assert r["levels"][0] == 0.0 and r["plots_per_level"][0.0] == 20
    # the warning / error text carries real ratios and omits the jump clause when it was not evaluated
    reason, hint = ej._merged_levels(np.sort(gen(np.random.default_rng(0))), np.repeat([0, 1, 2, 3], 20), 2.0)
    assert reason == "" and "e+" not in hint and "0.0 x its other" not in hint


def _plots_and_points(levels, per_level, n_x=8, n_y=8, cv=0.04, seed=0, crs=synth.EPSG):
    """Plots of 40 x 80 m in a row, `per_level` of each level, with n_x x n_y pure points per plot whose rate carries
    a relative noise of `cv` (the controller's own noise)."""
    rng = np.random.default_rng(seed)
    plots, pts = [], []
    L = [l for l in levels for _ in range(per_level)]
    for i, l in enumerate(L):
        plots.append(dict(plot_id=str(i), geometry=box(i * 50, 0, i * 50 + 40, 80)))
        for x in np.linspace(i * 50 + 2, i * 50 + 38, n_x):
            for y in np.linspace(12, 68, n_y):
                pts.append(dict(r=l * (1 + rng.normal(0, cv)), geometry=Point(x, y)))
    return gpd.GeoDataFrame(plots, crs=crs), gpd.GeoDataFrame(pts, crs=crs)


def test_mixed_plots_pure_with_noisy_points_are_not_mixed():
    """Defect C: on 10 % steps the p10 and p90 of a pure plot with 4-5 % point noise sit nearest to two different
    levels, and 8-10 of 20 such plots were called mixed. With the noise floor on the spread none is."""
    for cv in (0.04, 0.05):
        P_, A_ = _plots_and_points((90, 100, 110, 130), 8, cv=cv, seed=3)
        inner = ej.trim_plots(P_, 5.0, 0.0)
        lev, info = ej.rates_from_asapplied(P_, inner, A_, "r", name="urea")
        assert info["n_mixed"] == 0, (cv, info["n_mixed"])
        assert np.allclose(info["levels"], [90, 100, 110, 130], atol=1.5), (cv, info["levels"])
        assert info["n_with_level"] == 32 and info["n_off_level"] == 0
    # a plot that really carries two rates is still mixed when its spread stands out of the controller noise: half
    # of plot 0 one level up on a 1 % map (on a 4 % map a mix of two adjacent 10 % levels is inside the noise floor
    # of the rule, 3 x the typical spread, and cannot be told from a pure plot: that is the price of the floor)
    P_, A_ = _plots_and_points((90, 100, 110, 130), 8, cv=0.01, seed=3)
    inside = A_.within(P_.geometry.iloc[0]) & (A_.geometry.y < 40)
    A_.loc[inside, "r"] = A_.loc[inside, "r"] / 90 * 100
    lev, info = ej.rates_from_asapplied(P_, ej.trim_plots(P_, 5.0, 0.0), A_, "r", name="urea")
    assert info["n_mixed"] == 1 and np.isnan(lev.iloc[0]) and info["n_with_level"] == 31


def test_asapplied_small_design_warns():
    """Below 24 usable plots rates_from_asapplied always sets info['warning'] with the levels, plots per level and
    widths (even when the levels are clear)."""
    P_, A_ = _plots_and_points((80, 100, 120), 4, cv=0.01, seed=5)
    lev, info = ej.rates_from_asapplied(P_, ej.trim_plots(P_, 5.0, 0.0), A_, "r", name="urea")
    assert np.allclose(info["levels"], [80, 100, 120], atol=1)
    assert info["warning"] and "only 12 plots" in info["warning"] and "width" in info["warning"]
    assert all(f"({n} plots" in info["warning"] for n in info["plots_per_level"].values())


def _x_range_at(geom, y):
    """Extent along x of `geom` on the row y (None when the row misses it)."""
    cut = geom.intersection(LineString([(-1e3, y), (1e3, y)]))
    return None if cut.is_empty else (cut.bounds[0], cut.bounds[2])


def test_trim_plots_trapezoid_keeps_end_trim_everywhere():
    """Defect D: the rectangle-based trim kept 0 m of end trim along part of a non-rectangular plot's width. The
    trimmed interior must stay end_m away from both end edges on every row, and a rectangle must trim as before."""
    trap = Polygon([(0, 0), (80, 0), (50, 30), (0, 30)])      # slanted end edge x = 80 - y
    P_ = gpd.GeoDataFrame([dict(plot_id="t", geometry=trap)], crs=synth.EPSG)
    inner = ej.trim_plots(P_, 10.0, 0.0)
    g = inner.geometry.iloc[0]
    assert not g.is_empty and g.within(trap.buffer(1e-6))
    for y in np.arange(1.0, 30.0, 2.0):
        xr = _x_range_at(g, y)
        assert xr is not None, y
        assert xr[0] >= 10 - 1e-6 and xr[1] <= (80 - y) - 10 + 1e-6, (y, xr)
    # nothing is lost beyond the trim: the row at y = 15 keeps exactly 10 .. 55
    assert _x_range_at(g, 15.0) == pytest.approx((10.0, 55.0), abs=1e-6)
    assert "axis_deg" in inner and inner.inner_m2.iloc[0] == pytest.approx(g.area)
    # the old behaviour on the same trapezoid, for the record: the shortened rectangle intersected with the plot
    c, u, v, hl, hw = ej._oriented(trap)
    old = Polygon([c + s1 * (hl - 10) * u + s2 * hw * v for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]).intersection(trap)
    assert _x_range_at(old, 29.0)[1] > (80 - 29) - 1     # 0 m of trim at the top row
    # a rectangle gives the same result as before (symmetric difference 0), with and without a side trim
    rect = box(0, 0, 80, 30)
    R = gpd.GeoDataFrame([dict(plot_id="r", geometry=rect)], crs=synth.EPSG)
    assert ej.trim_plots(R, 10.0, 0.0).geometry.iloc[0].symmetric_difference(box(10, 0, 70, 30)).area < 1e-6
    assert ej.trim_plots(R, 10.0, 5.0).geometry.iloc[0].symmetric_difference(box(10, 5, 70, 25)).area < 1e-6
    # a rotated rectangle too
    from shapely import affinity
    rot = affinity.rotate(rect, 37, origin=(0, 0))
    want = affinity.rotate(box(10, 0, 70, 30), 37, origin=(0, 0))
    Rr = gpd.GeoDataFrame([dict(plot_id="r", geometry=rot)], crs=synth.EPSG)
    assert ej.trim_plots(Rr, 10.0, 0.0).geometry.iloc[0].symmetric_difference(want).area < 1e-6
    # a plot too short for the trim (18 m along its long axis, under 2 x 10 + 1) keeps a 1 m strip across its
    # middle (the old reading), not an empty polygon
    S = gpd.GeoDataFrame([dict(plot_id="s", geometry=box(0, 0, 18, 12))], crs=synth.EPSG)
    short = ej.trim_plots(S, 10.0, 0.0).geometry.iloc[0]
    assert not short.is_empty and short.area == pytest.approx(12.0, rel=1e-6)
