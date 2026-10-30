# -*- coding: utf-8 -*-
"""Peak-season NDVI per year from Copernicus (Sentinel-2 L2A, 2017 on), through the NDVI Map Viewer's client.

The Copernicus code is not duplicated here: GIT/NDVIMAP/server (cdse, render, geo, evalscripts, store) is imported, so
the credentials saved in that app (data/config.json, or CDSE_CLIENT_ID / CDSE_CLIENT_SECRET) are the ones used.

For each year:
  1. Statistics API over the season window (default 15 Jun - 15 Aug: heading to grain fill of spring cereals and
     canola flowering in central Alberta), one row per acquisition with the field mean and the share of usable
     pixels after the SCL cloud / shadow / snow mask;
  2. the dates with at least `min_valid_pct` usable pixels and the highest field-mean NDVI (default the three best:
     the season is their per-pixel median, built in ndvi.build_stack);
  3. Process API: band 1 = NDVI float32 (NaN masked), band 2 = validity, 10 m, the field's UTM zone, clipped to the
     boundary; written as <field>_NDVI_<date>_<res>m_EPSG<epsg>_b<hash>.tif: the viewer's download name plus a short
     hash of the boundary geometry.
Images already on disk are not fetched again, but only when they were clipped to the same boundary: the image of a
date depends on the boundary (clip and grid), not on the window or the number of dates, so the boundary hash is the
whole cache key. A new boundary therefore fetches again instead of reusing an image clipped to the old one (which would
then see only part of the new field and fail min_valid). fetch_years returns only the images of this run; the folder
may hold images of earlier runs, so the caller must use the returned paths, not every file of the folder.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys

def _find_ndvimap():
    """The NDVI Map Viewer folder: OCVR_NDVIMAP_DIR, else a copy shipped inside the OCVR folder (NDVIMAP\\, the
    portable package) or beside it (..\\NDVIMAP, the Olds College apps bundle), else the viewer's usual place under
    the user's GIT folder."""
    env = os.environ.get("OCVR_NDVIMAP_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(here, "NDVIMAP"), os.path.join(os.path.dirname(here), "NDVIMAP"),
                 os.path.join(os.path.expanduser("~"), "GIT", "NDVIMAP")):
        if os.path.isdir(os.path.join(cand, "server")):
            return cand
    return os.path.join(here, "NDVIMAP")


NDVIMAP_DIR = _find_ndvimap()


def _server():
    if NDVIMAP_DIR not in sys.path:
        sys.path.insert(0, NDVIMAP_DIR)
    from server import cdse, render, geo, evalscripts, store   # noqa: E402
    return cdse, render, geo, evalscripts, store


def available():
    try:
        _server()
        return os.path.isdir(NDVIMAP_DIR)
    except Exception:
        return False


def client():
    cdse, _, _, _, store = _server()
    cfg = store.load_config()
    if not cfg.get("client_id") or not cfg.get("client_secret"):
        raise RuntimeError("No Copernicus credentials: open the NDVI Map Viewer (GIT/NDVIMAP/run.bat), panel 1, "
                           "and save the client id and secret once.")
    return cdse.CDSEClient(cfg["client_id"], cfg["client_secret"])


def boundary_geojson(boundary_gdf):
    """The field boundary as a clean WGS 84 GeoJSON geometry (the viewer's normaliser: ring closing, winding,
    simplification to at most 1,200 vertices)."""
    from shapely.geometry import MultiPolygon, Polygon
    from . import io as eio
    _, _, geo, _, _ = _server()
    b, _ = eio.to_utm(boundary_gdf)
    u = b.geometry.union_all().buffer(5).buffer(-5).simplify(2)
    parts = [p for p in getattr(u, "geoms", [u]) if isinstance(p, Polygon) and p.area > 10000]
    u = Polygon(parts[0].exterior) if len(parts) == 1 else MultiPolygon([Polygon(p.exterior) for p in parts])
    g = b.iloc[:1].set_geometry([u]).to_crs(4326).geometry.iloc[0]
    return geo.normalize_geometry(json.loads(json.dumps(g.__geo_interface__)))


def boundary_key(geom4326):
    """Short, stable hash of the (normalised WGS 84) boundary geometry: the cache key of the downloaded images."""
    txt = json.dumps(geom4326, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:10]


def season_stats(cli, geom4326, year, window=("06-15", "08-15"), res=10.0, max_cloud=80):
    cdse, _, geo, evalscripts, _ = _server()
    utm = geo.utm_epsg(*geo.centroid(geom4326))
    grid = geo.compute_grid(geom4326, utm, res)
    geom_utm = geo.reproject(geom4326, 4326, utm)
    bounds = cdse.build_bounds(geom_utm, grid.bbox, utm, clip=True)
    script = evalscripts.stats_evalscript("ndvi", "sentinel-2-l2a", True, True, False, False)
    payload = cdse.build_stats_payload(bounds, "sentinel-2-l2a", f"{year}-{window[0]}T00:00:00Z",
                                       f"{year}-{window[1]}T23:59:59Z", script, res, res, max_cloud=max_cloud)
    resp = cli.statistics(payload)
    area_m2 = geo.planar_area(geom_utm)
    expected = max(1.0, area_m2 / (res * res))
    rows = []
    for e in resp.get("data") or []:
        day = str((e.get("interval") or {}).get("from") or "")[:10]
        out = (e.get("outputs") or {}).get("index") or {}
        band = (out.get("bands") or {}).get("B0") or {}
        st = band.get("stats") or {}
        n = float(st.get("sampleCount") or 0) - float(st.get("noDataCount") or 0)
        mean = st.get("mean")
        try:
            mean = float(mean)
            if mean != mean:
                mean = None
        except (TypeError, ValueError):
            mean = None
        rows.append(dict(date=day, mean=mean, valid_pct=round(min(100.0, 100.0 * max(0.0, n) / expected), 1)))
    rows.sort(key=lambda r: r["date"])
    return rows


def pick_dates(rows, min_valid_pct=95.0, n=1):
    ok = [r for r in rows if r["mean"] is not None and r["valid_pct"] >= min_valid_pct]
    ok.sort(key=lambda r: -r["mean"])
    return ok[:n]


def download(cli, geom4326, date, out_dir, field_name="field", res=10.0):
    """The NDVI image of `date` clipped to `geom4326`, from the cache when the same boundary was fetched before.
    The file name carries the boundary hash (see boundary_key), so an image clipped to another boundary is never
    reused."""
    _, render, geo, _, _ = _server()
    spec = render.RenderSpec(kind="index", index="ndvi", raw=True, fmt="tiff", crs_mode="utm", res_m=res,
                             clip=True, max_dim=geo.MAX_DIM)
    grid = render.plan(geom4326, spec)
    stem, ext = os.path.splitext(render.filename_for("index_tif", field_name, spec, date, grid))
    path = os.path.join(out_dir, f"{stem}_b{boundary_key(geom4326)}{ext}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    data, grid, meta = render.render(cli, geom4326, date, spec, grid=grid, timeout=240)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def fetch_years(boundary_gdf, years, out_dir, field_name="field", window=("06-15", "08-15"), min_valid_pct=95.0,
                n_per_year=3, progress=None):
    """Download the peak-season image(s) of every year into `out_dir`. Returns (paths, log rows): `paths` are ONLY
    the images fetched or reused for this boundary, years, window and number of dates; `out_dir` may also hold images
    of earlier runs (another boundary, window or season list), which must not enter the stack."""
    cli = client()
    geom = boundary_geojson(boundary_gdf)
    paths, log = [], []
    years = sorted(int(y) for y in years if int(y) >= 2017 and int(y) <= dt.date.today().year)
    for k, y in enumerate(years):
        if progress:
            progress(k / max(1, len(years)), f"{y}: reading the season statistics")
        try:
            rows = season_stats(cli, geom, y, window=window)
        except Exception as exc:                                     # pragma: no cover - network
            log.append(dict(year=y, status=f"statistics failed: {exc}"))
            continue
        best = pick_dates(rows, min_valid_pct, n_per_year)
        if not best:
            near = max((r for r in rows if r["mean"] is not None), key=lambda r: r["valid_pct"], default=None)
            log.append(dict(year=y, status="no clear date", n_dates=len(rows),
                            best_valid_pct=near["valid_pct"] if near else None))
            continue
        for b in best:
            if progress:
                progress((k + 0.5) / max(1, len(years)), f"{y}: downloading {b['date']}")
            try:
                p = download(cli, geom, b["date"], out_dir, field_name=field_name)
                if p not in paths:
                    paths.append(p)
                log.append(dict(year=y, status="ok", date=b["date"], mean=b["mean"], valid_pct=b["valid_pct"],
                                n_dates=len(rows), file=os.path.basename(p)))
            except Exception as exc:                                 # pragma: no cover - network
                log.append(dict(year=y, status=f"download failed: {exc}", date=b["date"]))
    if progress:
        progress(1.0, f"{len(paths)} images")
    return paths, log
