# -*- coding: utf-8 -*-
"""Reading the field's files and recognising their columns.

Vector inputs: .shp (+ parts), a .zip holding a shapefile (any depth), .geojson/.json, .gpkg, .kml/.kmz, .parquet
(GeoParquet). Every layer is returned in the field's UTM zone (metres), chosen from the centroid.

Column recognition is a priority list per role (the names written by Ag Leader SMS, John Deere Operations Center, the
OCCD cleaning app, the DIFM trial design and the OFPE pipeline); the app shows what was picked and lets the user
change it.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

M_PER_FT = 0.3048
M2_PER_AC = 4046.8564224

# ------------------------------------------------------------------------------------------------ column roles
CANDIDATES = {
    # yield (dry, standard moisture) first, then the levelled / harmonised columns of OCCD, then wet
    "yield": ["Yld_Lvl_Sy", "Yld_Level", "Yld_Dry_Sy", "Yld_Vol_Dr", "Yld_Vol13", "VRYIELDVOL", "Yld_Adj", "DryYield",
              "yield13", "yield", "Yield", "YIELD", "Yld_Vol_We", "Yld_Wet"],
    "heading": ["Track_deg_", "Heading", "heading", "Track", "TRACK", "Direction"],
    "distance": ["Distance_f", "Dist_ft", "DISTANCE", "Distance", "dist_ft", "Dist_m", "distance_m"],
    "swath": ["Swth_Wdth_", "Swath_ft", "SWATHWIDTH", "Swath_Wdth", "SwathWidth", "width_ft", "Swath_m", "width_m"],
    "machine": ["Machine", "machine", "Serial", "VEHICLE", "Vehicle", "Combine"],
    "moisture": ["Moisture__", "Moisture", "moisture"],
    "speed": ["Speed_mph_", "Speed_mph", "speed_mph", "Speed"],
    "plot_id": ["plot_id", "Plot_Id", "plot", "Plot", "PLOT", "id", "ID", "Obj__Id", "Obj_Id", "FID"],
    "design_rate": ["tgt_seed", "tgt_urea", "tgt_n", "Tgt_Rate_l", "Tgt_Rate", "TgtRate", "TargetRate", "rate", "Rate",
                    "Rate_l", "RATE", "design_rate"],
    "type": ["type", "Type", "TYPE", "kind"],
    "treatment": ["treat", "trt", "Treatment", "treatment", "TRT", "Trt"],
    "block": ["block", "Block", "BLOCK", "rep", "Rep", "REP"],
    "applied_rate": ["Rate_lbac", "AppliedRate", "Rt_Apd_Ms_", "Rt_Apd_Vl_", "Rate_Mass", "Rate_Appli", "Applied",
                     "applied", "WMRate", "Rate", "RATE", "AppldRate"],
    "zone": ["zone", "Zone", "ZONE", "MZ", "mz", "Zone_ID", "ZoneID", "Class", "class", "GRIDCODE", "gridcode", "DN"],
    "ndvi": ["NDVI", "ndvi", "mean_ndvi", "NDVI_mean"],
}


def find_col(df, role, exclude=()):
    """First column of `df` matching the role's priority list (exact name, then case-insensitive, then a prefix
    match for the tgt*/Tgt* design-rate columns). None when nothing matches."""
    cols = [c for c in df.columns if c != "geometry" and c not in exclude]
    for c in CANDIDATES.get(role, []):
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in CANDIDATES.get(role, []):
        if c.lower() in low:
            return low[c.lower()]
    if role == "design_rate":
        for c in cols:
            if re.match(r"(?i)^tgt", c) and pd.api.types.is_numeric_dtype(df[c]):
                return c
    return None


def numeric_cols(df):
    return [c for c in df.columns if c != "geometry" and pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------------------------------------------ files
VECTOR_EXT = (".shp", ".geojson", ".json", ".gpkg", ".kml", ".kmz", ".parquet")
RASTER_EXT = (".tif", ".tiff")
ZIP_CACHE = os.path.join(tempfile.gettempdir(), "ocvr_zip_cache")
ZIP_CACHE_DAYS = 7      # an extracted folder not read for this long is removed


def _zip_key(z):
    """The same content gives the same key wherever the zip sits: names, sizes and checksums from the zip's index."""
    h = hashlib.sha1()
    for i in z.infolist():
        h.update(f"{i.filename}|{i.file_size}|{i.CRC}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def _zip_intact(folder, z):
    """Every member still in `folder` at its size: Windows' temp clean-up can take single files out of it."""
    for i in z.infolist():
        p = os.path.join(folder, i.filename)
        if not i.is_dir() and not (os.path.isfile(p) and os.path.getsize(p) == i.file_size):
            return False
    return True


def _prune_zip_cache():
    cutoff = time.time() - ZIP_CACHE_DAYS * 86400
    for e in os.scandir(ZIP_CACHE):
        if e.is_dir() and e.stat().st_mtime < cutoff:
            shutil.rmtree(e.path, ignore_errors=True)


def _unpack_nested(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".zip"):
                unpack(os.path.join(root, f), os.path.join(root, os.path.splitext(f)[0]))


def _extract_cached(path):
    """Folder holding a zip's content, extracted once: a project reads the same zips on every run and the app on every
    step, and one new temp folder per read filled the disk (257 folders, 4.7 GB in a day). It is built under a temp
    name and renamed when complete, so another thread or process never sees it half written."""
    with zipfile.ZipFile(path) as z:
        dest = os.path.join(ZIP_CACHE, _zip_key(z))
        if os.path.isdir(dest):
            if _zip_intact(dest, z):
                os.utime(dest)
                return dest
            shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(ZIP_CACHE, exist_ok=True)
        _prune_zip_cache()
        tmp = tempfile.mkdtemp(prefix="new_", dir=ZIP_CACHE)
        z.extractall(tmp)
    _unpack_nested(tmp)
    try:
        os.rename(tmp, dest)
    except OSError:     # another thread or process got there first
        if not os.path.isdir(dest):
            return tmp
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def unpack(path, dest=None):
    """A zip is extracted into its own folder (nested zips too) and that folder is returned; anything else is
    returned unchanged. Without `dest` the folder is shared by every read of the same content: read it, never write
    to it."""
    if not str(path).lower().endswith(".zip"):
        return path
    if dest is None:
        return _extract_cached(path)
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)
    _unpack_nested(dest)
    return dest


def list_layers(path):
    """Vector layer files under `path` (a file is returned as itself)."""
    path = unpack(path)
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(VECTOR_EXT) and not f.lower().endswith(".shp.xml"):
                out.append(os.path.join(root, f))
    return out


def list_rasters(path):
    path = unpack(path)
    if os.path.isfile(path):
        return [path] if path.lower().endswith(RASTER_EXT) else []
    out = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(RASTER_EXT):
                out.append(os.path.join(root, f))
    return out


def read_vector(path, layer=None):
    """GeoDataFrame from any supported vector file (a zip / folder: its largest layer unless `layer` names one)."""
    p = str(path)
    if not os.path.exists(p):
        # pyogrio raises a RuntimeError subclass with a traceback for a missing file; a plain message instead
        raise FileNotFoundError(f"Vector file not found: {p}")
    if p.lower().endswith(".zip") or os.path.isdir(p):
        layers = list_layers(p)
        if not layers:
            raise ValueError(f"No vector layer found in {os.path.basename(p)}")
        if layer:
            match = [q for q in layers if os.path.splitext(os.path.basename(q))[0] == layer]
            p = match[0] if match else layers[0]
        else:
            p = max(layers, key=lambda q: os.path.getsize(q))
    if p.lower().endswith(".parquet"):
        g = gpd.read_parquet(p)
    elif p.lower().endswith(".kmz"):
        d = unpack_kmz(p)
        g = gpd.read_file(d)
    else:
        g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    return g


def unpack_kmz(path):
    d = _extract_cached(path)
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".kml"):
                return os.path.join(root, f)
    raise ValueError("The .kmz holds no .kml")


def utm_epsg_for(gdf):
    """EPSG of the UTM zone (WGS 84) at the layer's centroid."""
    c = gdf.to_crs(4326).geometry.union_all().centroid
    zone = int((c.x + 180) // 6) + 1
    return (32600 if c.y >= 0 else 32700) + zone


def to_utm(gdf, epsg=None):
    epsg = epsg or utm_epsg_for(gdf)
    return gdf.to_crs(epsg), epsg


def length_unit_from_name(colname):
    """'m' for a column named ..._m / metres / meters, else 'ft' (SMS and Ops Center write feet). A default only:
    a metric SMS export truncates 'Swth Wdth(m)' to the same shapefile name as the imperial one."""
    n = (colname or "").lower()
    return "m" if (n.endswith("_m") or "metre" in n or "meter" in n) else "ft"


def length_to_m(series, colname, unit=None):
    """Distance / width column to metres, in `unit` ('ft' | 'm'; cfg columns.<role>_unit) or, without one, in the
    unit its name suggests (length_unit_from_name)."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    u = (unit or length_unit_from_name(colname)).strip().lower()
    if u in ("m", "metre", "metres", "meter", "meters"):
        return s
    if u in ("ft", "feet", "foot"):
        return s * M_PER_FT
    raise ValueError(f"Unknown length unit {unit!r} for column {colname!r}: use 'ft' or 'm'")


def parse_date(text):
    """First date in a file name or string: 2026-07-21, 2026_07_21, 20260721 -> '2026-07-21'; None if none."""
    s = os.path.basename(str(text))
    m = re.search(r"(19|20)(\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?([0-2]\d|3[01])", s)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}-{m.group(3)}-{m.group(4)}"
