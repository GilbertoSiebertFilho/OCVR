# -*- coding: utf-8 -*-
"""Engine tests on synthetic trials with a known answer (tests/synth.py) and on small pieces."""
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import prices as EP, response as er, economics as ee, project as P, io as eio, join as ej  # noqa: E402
import synth  # noqa: E402


# ------------------------------------------------------------------------------------------------ small pieces
def test_input_price_basis():
    urea = EP.InputPrice(name="N", product="urea 46-0-0", price_per_lb_product=0.46, rate_basis="nutrient")
    assert urea.price_per_rate_unit == pytest.approx(1.0)
    urea_p = EP.InputPrice(name="urea", product="urea 46-0-0", price_per_lb_product=0.49, rate_basis="product")
    assert urea_p.price_per_rate_unit == pytest.approx(0.49)


def test_tonne_bushel():
    assert EP.per_tonne_to_per_bu(327.0, "wheat") == pytest.approx(8.90, abs=0.01)
    assert EP.per_bu_to_per_tonne(EP.per_tonne_to_per_bu(250, "barley"), "barley") == pytest.approx(250)


def test_parse_date():
    assert eio.parse_date("Field_NDVI_2017-08-12_10m_EPSG32612.tif") == "2017-08-12"
    assert eio.parse_date("ndvi_20210725.parquet") == "2021-07-25"
    assert eio.parse_date("nothing.tif") is None


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return str(path)


def test_unpack_reuses_one_folder_per_content(tmp_path, monkeypatch):
    """Re-reading a zip adds nothing to the temp folder: this leak filled the disk on 2026-10-28."""
    monkeypatch.setattr(eio, "ZIP_CACHE", str(tmp_path / "cache"))
    a = _zip(tmp_path / "a.zip", {"plots.shp": "shp", "plots.dbf": "dbf"})
    copy = _zip(tmp_path / "copy_of_a.zip", {"plots.shp": "shp", "plots.dbf": "dbf"})
    other = _zip(tmp_path / "b.zip", {"plots.shp": "other"})
    d = eio.unpack(a)
    assert eio.unpack(a) == d and eio.unpack(copy) == d
    assert eio.unpack(other) != d
    assert len(os.listdir(tmp_path / "cache")) == 2
    assert eio.list_layers(a) == [os.path.join(d, "plots.shp")]


def test_unpack_nested_zip_before_the_folder_is_shared(tmp_path, monkeypatch):
    monkeypatch.setattr(eio, "ZIP_CACHE", str(tmp_path / "cache"))
    inner = _zip(tmp_path / "inner.zip", {"plots.shp": "shp"})
    with open(inner, "rb") as f:
        outer = _zip(tmp_path / "outer.zip", {"inner.zip": f.read()})
    d = eio.unpack(outer)
    assert os.path.isfile(os.path.join(d, "inner", "plots.shp"))
    assert [p for p in os.listdir(tmp_path / "cache") if p.startswith("new_")] == []


def test_unpack_restores_a_folder_that_lost_files(tmp_path, monkeypatch):
    monkeypatch.setattr(eio, "ZIP_CACHE", str(tmp_path / "cache"))
    a = _zip(tmp_path / "a.zip", {"plots.shp": "shp", "plots.dbf": "dbf"})
    d = eio.unpack(a)
    os.remove(os.path.join(d, "plots.dbf"))
    assert eio.unpack(a) == d
    assert os.path.isfile(os.path.join(d, "plots.dbf"))


def test_unpack_removes_folders_not_read_for_a_week(tmp_path, monkeypatch):
    monkeypatch.setattr(eio, "ZIP_CACHE", str(tmp_path / "cache"))
    old = eio.unpack(_zip(tmp_path / "old.zip", {"plots.shp": "old"}))
    recent = eio.unpack(_zip(tmp_path / "recent.zip", {"plots.shp": "recent"}))
    eight_days_ago = time.time() - 8 * 86400
    os.utime(old, (eight_days_ago, eight_days_ago))
    eio.unpack(_zip(tmp_path / "new.zip", {"plots.shp": "new"}))
    assert not os.path.exists(old) and os.path.exists(recent)


def test_rate_term():
    assert er.is_rate_term("rc") and er.is_rate_term("I(rc ** 2)") and er.is_rate_term("C(zone)[2]:rc")
    assert not er.is_rate_term("Intercept") and not er.is_rate_term("xs")


def test_decide_known_curves():
    g = np.linspace(80, 200, 121)
    cur = {1: synth.truth(g, False), 2: synth.truth(g, True)}
    d = ee.decide(cur, g, {1: 50.0, 2: 50.0}, synth.PC, synth.PI, sq=140.0)
    assert d["r_zone"][1] == pytest.approx(87.8, abs=1.0)
    assert d["r_zone"][2] == pytest.approx(167.8, abs=1.0)
    assert 120 < d["r_uniform"] < 135
    # analytic gain: 0.001 * pc * ((ru - r1)^2 + (ru - r2)^2) / 2
    ru = d["r_uniform"]
    expect = 0.001 * synth.PC * ((ru - 87.8) ** 2 + (ru - 167.8) ** 2) / 2
    assert d["gain_vr_vs_uniform"] == pytest.approx(expect, rel=0.05)


def test_rate_levels():
    # five equal steps with controller noise, one plot off every level (300): 5 levels, 1 unassigned
    r = ej.rate_levels(np.array([88, 89, 87, 131, 130, 132, 175, 176, 174, 219, 220, 218, 262, 261, 263, 300.0]),
                       precision=0.5)
    assert np.allclose(r["levels"], [88, 131, 175, 219, 262]) and r["unassigned"] == 1
    assert r["plots_per_level"] == {88.0: 3, 131.0: 3, 175.0: 3, 219.0: 3, 262.0: 3}
    # unequal steps (90, 110, 130, 170) are still four levels; a flat map is one level; two levels of three plots
    assert np.allclose(ej.rate_levels(np.array([90, 91, 89, 110, 111, 109, 130, 131, 129, 170, 171, 169.0]), 0.2)["levels"],
                       [90, 110, 130, 170])
    assert ej.rate_levels(np.full(20, 100.0), 0.2)["levels"] == [100.0]
    assert np.allclose(ej.rate_levels(np.array([80, 80, 80, 120, 120, 120, 80.5, 119.5]), 0.0)["levels"], [80, 120])
    # noisy levels (sd 1.5 on steps of 20) and a NaN: every plot assigned
    rng = np.random.default_rng(1)
    v = np.concatenate([np.repeat([40, 60, 80, 100, 120], 30) + rng.normal(0, 1.5, 150), [np.nan]])
    r = ej.rate_levels(v, precision=0.6)
    assert len(r["levels"]) == 5 and r["unassigned"] == 0 and np.isnan(r["level"][-1])
    assert np.allclose(r["levels"], [40, 60, 80, 100, 120], atol=1.5)
    # Seed sweeps of the review's cases (20 seeds each). Clean maps must give the right number of levels in every
    # seed: Trial-22 geometry with +-6 % plot-to-plot drift, 20 % steps with +-5 % drift, 90/100/110 with sd 1 on 8
    # plots (the controller noise of the POINTS enters only through the precision of a plot median), 10 and 13
    # levels, a 0-rate check level, 8 levels plus one misfire. Maps whose levels run into each other (+-8 % drift on
    # 20 % steps) may raise, but must almost never return a wrong number of levels. A gradient always raises.
    def sweep(gen, prec, want, n=20):
        right = wrong = err = 0
        for seed in range(n):
            try:
                k = len(ej.rate_levels(gen(np.random.default_rng(seed)), prec)["levels"])
                right += k == want
                wrong += k != want
            except ValueError:
                err += 1
        return right, wrong, err
    T22 = (88, 131, 175, 219, 262)
    clean = [
        (lambda r: np.concatenate([l * (1 + r.uniform(-.06, .06, 33)) for l in T22]), 0.17, 5),
        (lambda r: np.concatenate([l * (1 + r.uniform(-.05, .05, 14)) for l in (40, 60, 80, 100, 120)]), 0.3, 5),
        (lambda r: np.concatenate([l * (1 + r.normal(0, .01, 33)) for l in (101, 118, 134, 151, 168)]), 0.1, 5),
        (lambda r: np.concatenate([l + r.normal(0, .5, 8) for l in range(40, 221, 20)]), 0.2, 10),
        (lambda r: np.concatenate([l + r.normal(0, .5, 6) for l in range(40, 281, 20)]), 0.2, 13),
        (lambda r: np.concatenate([np.abs(r.normal(0, .3, 6)), 40 + r.normal(0, .3, 6), 80 + r.normal(0, .3, 6)]), 0.1, 3),
        (lambda r: np.concatenate([np.concatenate([l + r.normal(0, .5, 8) for l in range(40, 181, 20)]), [110.0]]), 0.2, 8),
        (lambda r: np.concatenate([l + r.normal(0, .8, 6) for l in (90, 110, 130, 170)]), 0.2, 4),
    ]
    for gen, prec, want in clean:
        assert sweep(gen, prec, want) == (20, 0, 0), (want, sweep(gen, prec, want))
    right, wrong, err = sweep(lambda r: np.concatenate([l + r.normal(0, 1, 8) for l in (90, 100, 110)]), 0.63, 3)
    assert wrong == 0 and right >= 18
    right, wrong, err = sweep(lambda r: np.concatenate([l * (1 + r.uniform(-.08, .08, 14)) for l in (40, 60, 80, 100, 120)]), 0.3, 5)
    assert wrong <= 1
    assert sweep(lambda r: r.uniform(60, 200, 72), 0.2, 0) == (0, 0, 20)
    # the misfire: its group is counted, its plot gets no level
    r = ej.rate_levels(np.concatenate([np.concatenate([l + np.random.default_rng(4).normal(0, .5, 8) for l in range(40, 181, 20)]), [110.0]]), 0.2)
    assert r["unassigned"] == 1 and r["single_plot_levels"] == 1
    # no discrete levels (a gradient prescription) is an error naming the input; an unreplicated design too
    with pytest.raises(ValueError, match="urea shows no discrete rate levels"):
        ej.rate_levels(np.random.default_rng(3).uniform(60, 200, 72), 0.2, name="urea")
    with pytest.raises(ValueError, match="a level needs at least 2 plots"):
        ej.rate_levels(np.array([40, 60, 80, 100, 120, 140.0]), 0.3)
    # levels carry one decimal
    r = ej.rate_levels(np.array([88.3078, 88.31, 88.29, 130.6178, 130.62, 130.6]), 0.05)
    assert r["levels"] == [88.3, 130.6]


def test_rates_from_asapplied_mixed_plots():
    """Mixed plots are judged against the levels: a 70/30 mix of two ADJACENT levels 20 % apart, two half-and-half
    plots (which could invent a level between two real ones) and a 0-rate check level with residual readings."""
    import geopandas as gpd
    from shapely.geometry import box, Point
    rng = np.random.default_rng(2)
    levels = [100, 120, 144] * 4
    plots, pts = [], []
    for i in range(12):
        plots.append(dict(plot_id=str(i), geometry=box(i * 50, 0, i * 50 + 40, 80)))
        for x in np.linspace(i * 50 + 2, i * 50 + 38, 10):
            for y in np.linspace(12, 68, 8):
                L = levels[i]
                if i == 9 and y < 12 + 56 * 0.3:
                    L = 120                      # 30 % of plot 9 (level 100) at the next level
                if i in (10, 11) and y < 40:
                    L = 100                      # half of plots 10 (level 120) and 11 (level 144) at 100
                if i in (4, 8):
                    L = 0                        # two check plots: residual readings only
                pts.append(dict(r=abs(L + rng.normal(0, 0.5)) if L == 0 else L + rng.normal(0, 0.5), geometry=Point(x, y)))
    P_ = gpd.GeoDataFrame(plots, crs=32612)
    inner = ej.trim_plots(P_, 5.0, 0.0)
    A_ = gpd.GeoDataFrame(pts, crs=32612)
    lev, info = ej.rates_from_asapplied(P_, inner, A_, "r", name="urea")
    assert info["n_mixed"] == 3 and info["n_off_level"] == 0 and info["n_few_points"] == 0
    assert np.allclose(info["levels"], [0, 100, 120, 144], atol=1.0)
    assert list(info["plots_per_level"].values()) == [2, 3, 2, 2]
    assert np.isnan(lev.iloc[9]) and np.isnan(lev.iloc[10]) and np.isnan(lev.iloc[11])
    assert lev.iloc[8] == pytest.approx(0, abs=1) and lev.iloc[0] == pytest.approx(100, abs=1)
    # a repeated index on the as-applied frame drops no point
    A2 = A_.copy()
    A2.index = np.arange(len(A2)) % 300
    lev2, _ = ej.rates_from_asapplied(P_, inner, A2, "r", name="urea")
    assert lev2.equals(lev)
    # an unreplicated design names the rule in its message
    with pytest.raises(ValueError, match="a level needs at least 2 plots"):
        one = gpd.GeoDataFrame([dict(plot_id=str(i), geometry=box(i * 50, 0, i * 50 + 40, 80)) for i in range(6)], crs=32612)
        pts1 = [dict(r=40 + 20 * i + rng.normal(0, 0.3), geometry=Point(x, y)) for i in range(6)
                for x in np.linspace(i * 50 + 2, i * 50 + 38, 6) for y in np.linspace(12, 68, 6)]
        ej.rates_from_asapplied(one, ej.trim_plots(one, 5.0, 0.0), gpd.GeoDataFrame(pts1, crs=32612), "r", name="urea")


def test_merge_small_zones():
    u = pd.DataFrame(dict(zone=[1] * 5 + [2] * 30 + [3] * 30, target_seed=[90, 115, 140, 170, 200] * 13))
    u2, W, m, notes = er.merge_small_zones(u, {1: 5.0, 2: 50.0, 3: 50.0}, "target_seed", min_units=20)
    assert m[1] == 2 and W[2] == pytest.approx(55.0) and notes


# ------------------------------------------------------------------------------------------------ synthetic trials
@pytest.fixture(scope="module")
def vr_trial(tmp_path_factory):
    d = tmp_path_factory.mktemp("vr")
    files = synth.make(str(d / "data"))
    st = P.run(synth.config(files, "synth_vr"), str(d / "proj"), log=lambda m: None, fetch=False)
    return st


@pytest.fixture(scope="module")
def flat_trial(tmp_path_factory):
    d = tmp_path_factory.mktemp("flat")
    files = synth.make(str(d / "data"), same_response=True, seed=11)
    st = P.run(synth.config(files, "synth_flat"), str(d / "proj"), log=lambda m: None, fetch=False)
    return st


def test_design_and_border(vr_trial):
    D = vr_trial.res["design"]
    assert D["n_plots"] == 72 and D["n_border"] == 1
    assert D["rates"] == [80.0, 110.0, 140.0, 170.0, 200.0]


def test_zones_recover_halves(vr_trial):
    st = vr_trial
    g = st.grid
    cols = np.arange(g.width)
    xs = g.transform.c + g.res * (cols + 0.5)
    east = np.tile(xs > synth.X0 + synth.NC * synth.PW / 2, (g.height, 1))
    z = st.zone_r
    m = g.mask & (z > 0)
    agree = ((z == 2) == east)[m].mean()
    assert agree > 0.9, agree


def test_join_containment(vr_trial):
    J = vr_trial.res["join"]
    a = J["attrition"]
    assert J["mode"] == "polygon"
    assert a["kept"] > 0.4 * a["n_obs"]
    u = vr_trial.units
    assert (u.rate_src_seed.str.startswith("applied")).all()
    assert np.allclose(u.rate_seed, u.target_seed, rtol=0.02)


def test_vr_detected(vr_trial):
    R = vr_trial.res["analysis"]["seed"]
    d = R["decision"]
    assert d["r_zone"][1] == pytest.approx(87.8, abs=20)
    assert d["r_zone"][2] == pytest.approx(167.8, abs=20)
    assert R["models"]["zone"]["tests"]["zone_x_rate"]["p"] < 0.01
    assert R["crossfit"]["gain_vr_vs_uniform"]["mean"] > 5
    assert R["verdict"]["level"] == "vr"
    # the single curve keeps a yield level per original zone (README: without it the best flat rate came out at 80
    # instead of 128); the best flat rate is the single-curve optimum, and the status quo 140 is kept because 125
    # gains too little over it
    assert "C(zone_level)" in R["models"]["pooled"]["formula"]
    assert R["pooled"]["r_uniform"] == pytest.approx(128, abs=15)
    assert R["verdict"]["flat_rate"] == pytest.approx(140.0) and R["verdict"]["change_flat"] is False
    assert os.path.exists(vr_trial.report_path)
    assert os.path.exists(os.path.join(vr_trial.out, "zones_rx.shp"))


def test_flat_when_same_response(flat_trial):
    R = flat_trial.res["analysis"]["seed"]
    assert R["verdict"]["level"] == "flat"
    assert R["crossfit"]["gain_vr_vs_uniform"]["mean"] < 3     # a chance gain, below the worthwhile threshold
    assert R["pooled"]["r_uniform"] == pytest.approx(87.8, abs=20)


def _verdict_input(cf_mean, cf_share, p_zone, p_rate=0.5, flat=120.0, sq=120.0, g_flat=0.0, cfu=None, g_in=5.0):
    """A minimal result dict for project.verdict: two zones, the cross-fit summaries and the tests given."""
    grid = list(np.linspace(80, 200, 161))
    R = dict(decision=dict(r_zone={1: 100.0, 2: 160.0}, r_uniform=flat, gain_vr_vs_uniform=g_in, grid=grid,
                           corner={1: False, 2: False}, nr_sq=100.0, sq_too_far=False, sq_extrapolated=False),
             pooled=dict(r_uniform=flat, corner={None: False}, gain_uniform_vs_sq=g_flat),
             crossfit=dict(gain_vr_vs_uniform=dict(mean=cf_mean, share_positive=cf_share, n=30)),
             models=dict(zone=dict(tests=dict(zone_x_rate=dict(p=p_zone))), pooled=dict(tests=dict(rate=dict(p=p_rate)))),
             status_quo=sq, price_input=0.26, zone_map={1: 1, 2: 2}, zone_notes=[],
             summary_basis=dict(best_flat=dict(gain_vs_sq=g_flat)))
    if cfu is not None:
        R["crossfit_flat"] = dict(gain_uniform_vs_sq=dict(mean=cfu[0], share_positive=cfu[1], n=30))
    return R


def test_verdict_rules():
    """The decision rules of README 'How it decides' 6, on a minimal result: the minimum worthwhile gain, the 70 %
    share, the zone x rate test and the direction-aware flat-rate gate."""
    V = P.verdict
    assert V(_verdict_input(2.0, 0.83, 0.015))["level"] == "lean_vr"      # the README's own chance case: not 'vr'
    assert V(_verdict_input(3.5, 0.83, 0.015))["level"] == "vr"
    assert V(_verdict_input(3.5, 0.83, 0.30))["level"] == "lean_vr"       # gain without the zone x rate test
    assert V(_verdict_input(0.5, 0.65, 0.015))["level"] == "flat"
    assert V(_verdict_input(3.5, 0.60, 0.015))["level"] == "lean_vr"      # mean reached, share not
    assert V(_verdict_input(2.0, 0.83, 0.015), min_gain=1.0)["level"] == "vr"
    # a higher flat rate needs the field to respond to the rate (p < 0.10) and a worthwhile gain in and out of sample
    v = V(_verdict_input(0.0, 0.5, 0.5, p_rate=0.04, flat=150.0, sq=120.0, g_flat=6.0, cfu=(5.0, 0.9)))
    assert v["change_flat"] and v["flat_rate"] == 150.0
    v = V(_verdict_input(0.0, 0.5, 0.5, p_rate=0.30, flat=150.0, sq=120.0, g_flat=6.0, cfu=(5.0, 0.9)))
    assert not v["change_flat"] and v["flat_rate"] == 120.0
    # a LOWER flat rate: the saving is certain, so no response test is required; the cross-fitted gain, its share
    # and the in-sample gain must agree
    v = V(_verdict_input(0.0, 0.5, 0.5, p_rate=0.55, flat=100.0, sq=134.0, g_flat=18.0, cfu=(17.5, 0.93)))
    assert v["change_flat"] and v["flat_rate"] == 100.0
    assert any("input saving" in r for r in v["reasons"])
    v = V(_verdict_input(0.0, 0.5, 0.5, p_rate=0.55, flat=100.0, sq=134.0, g_flat=18.0, cfu=(2.0, 0.6)))
    assert not v["change_flat"] and v["flat_rate"] == 134.0
    v = V(_verdict_input(0.0, 0.5, 0.5, p_rate=0.55, flat=100.0, sq=134.0, g_flat=-1.0, cfu=(17.5, 0.93)))
    assert not v["change_flat"]
    # without a cross-fit of the flat rate (0 repeats) the lower rate is not advised on the in-sample gain alone
    v = V(_verdict_input(0.0, 0.5, 0.5, p_rate=0.55, flat=100.0, sq=134.0, g_flat=18.0))
    assert not v["change_flat"] and not any("cross-fitted gain vouches" in r for r in v["reasons"])


def test_machine_names_in_formula(vr_trial):
    """Combine names with spaces, dots or hyphens (Ops Center / SMS 'Machine' values) must not reach the patsy
    formula: the shares get positional aliases (response.prep)."""
    from engine import response as er
    u = vr_trial.units.copy()
    sh = [c for c in u.columns if c.startswith("sh_")]
    assert len(sh) >= 2
    u = u.rename(columns={sh[0]: "sh_X9 1100", sh[1]: "sh_S780 - 1"})
    spec = er.Spec(input="seed", others=[], others_sq={}, form="quadratic", machine=True, spatial=True,
                   rate_source="applied")
    for kind in ("pooled", "zone", "ndvi"):
        f_ = er.fit(u, spec, kind)
        assert f_.scale["machines"] == ["m_0", "m_1"]
        assert set(f_.scale["machine_names"].values()) == {"X9 1100", "S780 - 1"}
        assert "sh_" not in f_.formula
    assert "combine shares" in (P._controls_used(er.fit(u, spec, "pooled")) or [])
    with pytest.raises(ValueError):
        EP.Prices.from_dict(dict(inputs=[dict(name="seed"), dict(name="seed")]))


def test_input_names_and_rate_units():
    from engine import prices as pr
    assert pr.Prices.from_dict(dict(inputs=[dict(name="seed"), dict(name="urea")])).inputs[1].name == "urea"
    for bad in (["seed", "Seed"], ["rate"], ["urea N"], ["2nd"]):
        with pytest.raises(ValueError):
            pr.Prices.from_dict(dict(inputs=[dict(name=n) for n in bad]))
    assert pr.rate_unit_factor("lb/ac") == 1.0
    assert pr.rate_unit_factor("kg/ha") == pytest.approx(0.892179, abs=1e-6)
    with pytest.raises(ValueError):
        pr.rate_unit_factor("t/ha")


def test_plateau_extension_below_range():
    """economics.bounded_curve: a plateau curve is extended BELOW the trial range with its slope at the lowest rate
    (never held flat there), above it with its slope at hi (0 once the plateau is reached inside the range)."""
    from engine import economics as ee
    xp = 150.0
    curve = lambda r: 50 + 0.4 * (np.minimum(np.asarray(r, float), xp) - 80) - 0.001 * (np.minimum(np.asarray(r, float), xp) - 80) ** 2
    y = ee.bounded_curve(curve, [60.0, 80.0, 220.0], 80.0, 200.0, plateau=True)
    assert y[0] < y[1]                                     # sloped below the range
    assert y[1] - y[0] == pytest.approx(20 * 0.4, rel=0.05)
    assert y[2] == pytest.approx(float(curve(200.0)), abs=1e-6)      # flat above once the plateau is reached


@pytest.fixture(scope="module")
def urea_trial(tmp_path_factory):
    """Seed x urea on the same plots; the urea rates exist only on the urea as-applied map (two plots mixed)."""
    d = tmp_path_factory.mktemp("urea")
    files = synth.make(str(d / "data"), seed=5, urea=True, mixed_plots=2)
    st = P.run(synth.config(files, "synth_urea", urea=True), str(d / "proj"), log=lambda m: None, fetch=False)
    return st, files


def test_urea_rates_from_asapplied(urea_trial):
    st, files = urea_trial
    D = st.res["design"]
    rb = D["rates_by_input"]
    assert D["rates"] == [80.0, 110.0, 140.0, 170.0, 200.0]                       # the seed design is untouched
    assert rb["seed"]["source"].startswith("design column")
    assert rb["urea"]["source"].startswith("as-applied map")
    assert np.allclose(rb["urea"]["rates"], synth.N_RATES, atol=1.0)              # the five levels the drill delivered
    fa = rb["urea"]["from_applied"]
    assert fa["n_mixed"] == 2 and fa["n_with_level"] == 70 and fa["n_few_points"] == 0 and fa["n_off_level"] == 0
    # every unmixed plot got its true level
    p = st.plots.set_index("plot_id")
    for pid, n in files["n_by_plot"].items():
        got = p.loc[str(pid), "design_urea"]
        if pid in files["mixed_ids"]:
            assert np.isnan(got)
        else:
            assert got == pytest.approx(n, abs=1.0)
    # the units carry both designs and both applied rates; the mixed plots have an applied urea rate but no target
    u = st.units
    assert {"target_seed", "rate_seed", "target_urea", "rate_urea"} <= set(u.columns)
    mixed = u[u.plot_id.isin([str(i) for i in files["mixed_ids"]])]
    assert len(mixed) and mixed.target_urea.isna().all() and np.isfinite(mixed.rate_urea).all()


def test_urea_analysed_with_seed_control(urea_trial):
    st, _ = urea_trial
    A = st.res["analysis"]
    assert not A["urea"].get("skipped") and not A["seed"].get("skipped")
    # each input's model used the other as a control
    assert "seed" in A["urea"]["controls"] and "urea" in A["seed"]["controls"]
    # the urea response is the same in both zones: flat (tested, not merged or untested), with the optimum near the
    # true 91 lb/ac
    assert A["urea"]["verdict"]["level"] == "flat"
    assert len(A["urea"]["decision"]["r_zone"]) > 1                       # zones kept, VR really tested
    cf = A["urea"]["crossfit"]["gain_vr_vs_uniform"]
    assert cf["n"] > 0 and cf["mean"] < 3
    sk = A["urea"]["crossfit_skips"]
    assert sk["n_used"] == sk["n_tried"] > 0                               # no cross-fit split skipped
    assert A["urea"]["pooled"]["r_uniform"] == pytest.approx(91.0, abs=20)
    # the seed answer is still VR with the zone optima
    d = A["seed"]["decision"]
    assert A["seed"]["verdict"]["level"] == "vr"
    assert d["r_zone"][1] == pytest.approx(87.8, abs=20) and d["r_zone"][2] == pytest.approx(167.8, abs=20)
    assert os.path.exists(st.report_path)
    # the prescription carries both inputs
    import geopandas as gpd
    rx = gpd.read_file(os.path.join(st.out, "zones_rx.gpkg"))
    assert any(c.startswith("Rx_urea") for c in rx.columns) and any(c.startswith("Rx_seed") for c in rx.columns)


def test_from_applied_needs_the_map(urea_trial, tmp_path):
    """FROM_APPLIED without an as-applied map for that input, or with a rate column that is not in the map, stops
    'Load the trial' with a message naming the input."""
    st, files = urea_trial
    cfg = synth.config(files, "synth_urea_bad", urea=True)
    cfg["files"]["asapplied"].pop("urea")
    s = P.State(cfg, str(tmp_path / "p1"), log=lambda m: None)
    with pytest.raises(ValueError, match="no as-applied map is given for urea"):
        s.load_design()
    cfg = synth.config(files, "synth_urea_bad2", urea=True)
    cfg["columns"]["applied"]["urea"] = "no_such_column"
    s = P.State(cfg, str(tmp_path / "p2"), log=lambda m: None)
    with pytest.raises(ValueError, match="not in that file"):
        s.load_design()


def test_project_roundtrip(vr_trial):
    rec = P.load(vr_trial.workdir)
    assert rec["config"]["name"] == "synth_vr"
    assert rec["summary"]["seed"]["r_zone"]
