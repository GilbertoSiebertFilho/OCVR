# -*- coding: utf-8 -*-
"""A synthetic trial with a known answer, for the engine tests.

Field 540 m (E-W) x 560 m (N-S) in UTM 12N near Olds. 12 x 6 plots of 45 m x 80 m (3 swaths of 15 m) inside a border.
The west half is the low-NDVI zone, the east half the high one (four seasons of NDVI GeoTIFFs, wheat). Rates 80, 110,
140, 170, 200 lb/ac randomised over the plots. True yield (bu/ac):
    west  50 + 0.20 r - 0.001 r^2         (optimum at 9.00 $/bu and 0.22 $/lb: r = 87.8)
    east  70 + 0.36 r - 0.001 r^2         (optimum r = 167.8)
with plot effects (sd 1.5) and record noise (sd 4). With `same_response=True` both zones get the west slope (only the
level differs), so variable rate should not pay.
"""
from __future__ import annotations

import os

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Point

EPSG = 32612
X0, Y0 = 700000.0, 5740000.0
PW, PL, NC, NR = 45.0, 80.0, 12, 6
MARGIN_X, MARGIN_Y = 0.0, 40.0
RATES = [80, 110, 140, 170, 200]
PC, PI = 9.0, 0.22


def truth(r, east, same_response=False):
    if east and not same_response:
        return 70 + 0.36 * r - 0.001 * r ** 2
    return (70 if east else 50) + 0.20 * r - 0.001 * r ** 2


# A second input (urea, lb product/ac) whose rates are NOT in the design file: the drill logged them (as-applied
# map), one level per plot, crossed with the seed rates. Its true effect is the same in both zones:
#     + 0.20 n - 0.0008 n^2   (optimum at 9.00 $/bu and 0.49 $/lb: n = (0.20 - 0.49 / 9) / 0.0016 = 91.0; a clear
#     hump: 11.6 bu/ac at 91 against 6.7 at 40 and 8.0 at 200, so 72 plots identify it)
N_RATES = [40, 80, 120, 160, 200]
PN = 0.49


def truth_n(n):
    return 0.20 * n - 0.0008 * n ** 2


def make(folder, same_response=False, seed=7, urea=False, mixed_plots=0):
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    W, H = NC * PW, NR * PL + 2 * MARGIN_Y
    field = box(X0, Y0, X0 + W, Y0 + H)
    plots = []
    rates = np.array(RATES * (NC * NR // len(RATES) + 1))[: NC * NR]
    rng.shuffle(rates)
    k = 0
    for i in range(NC):
        for j in range(NR):
            g = box(X0 + i * PW, Y0 + MARGIN_Y + j * PL, X0 + (i + 1) * PW, Y0 + MARGIN_Y + (j + 1) * PL)
            plots.append(dict(Obj__Id=k + 1, Tgt_Rate_l=float(rates[k]), geometry=g))
            k += 1
    P = gpd.GeoDataFrame(plots, crs=EPSG)
    # urea level per plot: a Latin-square-like crossing with the seed rates (every seed rate meets every N rate)
    nr = np.array(N_RATES * (NC * NR // len(N_RATES) + 1))[: NC * NR]
    rng.shuffle(nr)
    P["n_rate"] = nr.astype(float)
    # `mixed_plots` plots get two N rates (the first half of the plot one level up): no design level can be read
    mixed_ids = set(P.Obj__Id.iloc[:mixed_plots].tolist()) if mixed_plots else set()
    border = field.difference(P.geometry.union_all())
    D = pd.concat([P, gpd.GeoDataFrame([dict(Obj__Id=k + 1, Tgt_Rate_l=140.0, geometry=border)], crs=EPSG)], ignore_index=True)
    D.to_crs(4326).to_file(os.path.join(folder, "design.shp"))
    peff = {pid: rng.normal(0, 1.5) for pid in P.Obj__Id}
    # harvest: N-S passes, one every 15 m, points every 2 m
    recs = []
    xc = X0 + 7.5
    t = 0
    for p_ in range(int(W // 15)):
        x = xc + 15 * p_
        ys = np.arange(Y0 + 1, Y0 + H - 1, 2.0)
        if p_ % 2:
            ys = ys[::-1]
        for y in ys:
            east = x > X0 + W / 2
            hit = P[P.contains(Point(x, y))]
            r = float(hit.Tgt_Rate_l.iloc[0]) if len(hit) else 140.0
            pe = peff[int(hit.Obj__Id.iloc[0])] if len(hit) else 0.0
            yv = truth(r, east, same_response) + pe + rng.normal(0, 4)
            if urea:
                nv = _n_at(hit, x, y, mixed_ids) if len(hit) else 120.0
                yv += truth_n(nv) - truth_n(120.0)
            recs.append(dict(Yld_Vol_Dr=yv, Track_deg_=0.0 if p_ % 2 == 0 else 180.0, Distance_f=2.0 / 0.3048,
                             Swth_Wdth_=15.0 / 0.3048, Machine="A" if p_ % 2 else "B", geometry=Point(x, y)))
            t += 1
    Y = gpd.GeoDataFrame(recs, crs=EPSG)
    Y.to_crs(4326).to_file(os.path.join(folder, "yield.shp"))
    # as-applied: points every 5 m along drill passes, rate + 1 % noise
    aa = []
    for x in np.arange(X0 + 2.5, X0 + W, 5.0):
        for y in np.arange(Y0 + 2.5, Y0 + H, 5.0):
            hit = P[P.contains(Point(x, y))]
            r = float(hit.Tgt_Rate_l.iloc[0]) if len(hit) else 140.0
            aa.append(dict(Rate_lbac=r * (1 + rng.normal(0, 0.01)), geometry=Point(x, y)))
    gpd.GeoDataFrame(aa, crs=EPSG).to_crs(4326).to_file(os.path.join(folder, "asapplied.shp"))
    if urea:
        au = []
        for x in np.arange(X0 + 2.5, X0 + W, 5.0):
            for y in np.arange(Y0 + 2.5, Y0 + H, 5.0):
                hit = P[P.contains(Point(x, y))]
                nv = _n_at(hit, x, y, mixed_ids) if len(hit) else 120.0
                au.append(dict(Rt_Apd_Ms_=nv * (1 + rng.normal(0, 0.01)), geometry=Point(x, y)))
        gpd.GeoDataFrame(au, crs=EPSG).to_crs(4326).to_file(os.path.join(folder, "asapplied_urea.shp"))
    # NDVI: four seasons, east brighter
    nd = os.path.join(folder, "ndvi")
    os.makedirs(nd, exist_ok=True)
    res = 10.0
    nx, ny = int(W / res) + 6, int(H / res) + 6
    tr = from_origin(X0 - 30, Y0 + H + 30, res, res)
    xs = X0 - 30 + res * (np.arange(nx) + 0.5)
    for year in (2021, 2022, 2023, 2024):
        east = (xs > X0 + W / 2).astype(float)
        a = 0.70 + 0.10 * np.tile(east, (ny, 1)) + rng.normal(0, 0.015, (ny, nx)) + rng.normal(0, 0.02)
        with rasterio.open(os.path.join(nd, f"synth_NDVI_{year}-07-15_10m_EPSG{EPSG}.tif"), "w", driver="GTiff", height=ny,
                           width=nx, count=2, dtype="float32", crs=f"EPSG:{EPSG}", transform=tr, nodata=None) as dst:
            dst.write(a.astype("float32"), 1)
            dst.write(np.ones_like(a, dtype="float32"), 2)
    out = dict(design=os.path.join(folder, "design.shp"), yield_=os.path.join(folder, "yield.shp"),
               asapplied=os.path.join(folder, "asapplied.shp"), ndvi_dir=nd,
               n_by_plot={int(i): float(n) for i, n in zip(P.Obj__Id, P.n_rate)}, mixed_ids=sorted(mixed_ids))
    if urea:
        out["asapplied_urea"] = os.path.join(folder, "asapplied_urea.shp")
    return out


def _n_at(hit, x, y, mixed_ids):
    """The N rate applied at (x, y) inside the plot `hit`: its level, one level up in the southern half of a mixed
    plot."""
    n = float(hit.n_rate.iloc[0])
    if int(hit.Obj__Id.iloc[0]) in mixed_ids:
        miny = hit.geometry.iloc[0].bounds[1]
        if y < miny + PL / 2:
            n = min(n + 40.0, 200.0) if n < 200 else n - 40.0
    return n


def config(files, name="synth", urea=False):
    from engine import project as P
    cfg = P.default_config()
    cfg.update(name=name, crop="wheat", year=2026, field=name)
    cfg["files"].update(design=files["design"], ndvi_dir=files["ndvi_dir"])
    cfg["files"]["yield"] = files["yield_"]
    cfg["files"]["asapplied"] = {"seed": files["asapplied"]}
    cfg["ndvi"].update(source="files", crop_history={str(y): "wheat" for y in (2021, 2022, 2023, 2024)})
    cfg["zones"].update(k=2, method="quantile", min_patch_ha=0.2)
    cfg["prices"]["crop_price_bu"] = PC
    cfg["prices"]["inputs"] = [dict(name="seed", product="seed", price_per_lb_product=PI, rate_basis="product",
                                    rate_unit="lb/ac", status_quo=140.0)]
    cfg["model"].update(B=20, R=10, spatial=False)
    if urea:
        from engine.project import FROM_APPLIED
        cfg["files"]["asapplied"]["urea"] = files["asapplied_urea"]
        cfg["columns"]["extra_design_rate"] = {"urea": FROM_APPLIED}
        cfg["columns"]["applied"] = {"seed": "Rate_lbac", "urea": "Rt_Apd_Ms_"}
        cfg["prices"]["inputs"].append(dict(name="urea", product="urea 46-0-0", price_per_lb_product=PN,
                                            rate_basis="product", rate_unit="lb/ac", status_quo=120.0))
    return cfg
