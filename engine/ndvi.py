# -*- coding: utf-8 -*-
"""Multi-year NDVI: one layer per season, put on one grid, normalised, and summarised per pixel.

A layer is one GeoTIFF (band 1 = the index as float, NaN = masked; band 2, when present, = 1 where valid: the
"GeoTIFF - index values" product of GIT/NDVIMAP and of ndvi_fetch). The date comes from the file name, the crop from
the crop-history table the user fills in (the yield history of the same crop is what predicts the next one; the app
defaults to the same-crop years when there are enough of them).

Each season is normalised before averaging, because absolute NDVI changes with date, crop, sensor and season while the
spatial pattern is what a zone map needs (Blackmore 2000, normalised yield; the same treatment for imagery in
Maestrini & Basso 2018):
    relative   v / mean(v in field)            (default; 1.0 = field average)
    zscore     (v - mean) / sd
Per pixel: the temporal mean of the normalised values, their standard deviation, the coefficient of variation
(sd / mean for "relative") and the number of seasons that saw the pixel.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from . import io as eio


@dataclass
class Layer:
    path: str
    date: str | None = None
    year: int | None = None
    crop: str | None = None
    source: str = "file"
    field_mean: float | None = None
    valid_frac: float | None = None
    use: bool = True

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _bare_year(path):
    """A season year written alone in a file name (field_NDVI_2019.tif): the only 4-digit 19xx / 20xx run not part
    of a longer number. None when there is none or when several different years appear."""
    import re
    ys = {int(m) for m in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", os.path.basename(str(path)))}
    return ys.pop() if len(ys) == 1 else None


def layers_from_paths(paths, crop_by_year=None):
    """One Layer per file. The date comes from the file name; when the name holds only a year (own composites named
    field_NDVI_2019.tif) that year is used, so the crop history, the trial-year exclusion and the one-season-per-year
    rule still apply to it."""
    out = []
    for p in paths:
        d = eio.parse_date(p)
        y = int(d[:4]) if d else _bare_year(p)
        out.append(Layer(path=str(p), date=d, year=y, crop=(crop_by_year or {}).get(y)))
    out.sort(key=lambda L: (L.date or "", L.path))
    return out


@dataclass
class Grid:
    crs: str
    transform: object
    width: int
    height: int
    res: float
    mask: np.ndarray          # True inside the field boundary

    @property
    def bounds(self):
        t = self.transform
        return (t.c, t.f + self.height * t.e, t.c + self.width * t.a, t.f)

    def xy_to_rc(self, x, y):
        t = self.transform
        c = np.floor((np.asarray(x) - t.c) / t.a).astype(int)
        r = np.floor((np.asarray(y) - t.f) / t.e).astype(int)
        return r, c

    def sample(self, arr, x, y):
        """Nearest-pixel value of `arr` at the points (NaN outside the grid)."""
        r, c = self.xy_to_rc(x, y)
        ok = (r >= 0) & (r < self.height) & (c >= 0) & (c < self.width)
        out = np.full(len(r), np.nan)
        out[ok] = arr[r[ok], c[ok]]
        return out

    def cell_area_m2(self):
        return self.res * self.res


def make_grid(boundary_utm, res=10.0, pad=30.0):
    """A north-up grid in the boundary's UTM CRS, snapped to `res`, covering the boundary + pad."""
    x0, y0, x1, y1 = boundary_utm.total_bounds
    x0 = np.floor((x0 - pad) / res) * res
    y0 = np.floor((y0 - pad) / res) * res
    x1 = np.ceil((x1 + pad) / res) * res
    y1 = np.ceil((y1 + pad) / res) * res
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    tr = from_origin(x0, y1, res, res)
    mask = features.rasterize([(g, 1) for g in boundary_utm.geometry], out_shape=(h, w), transform=tr,
                              fill=0, all_touched=False, dtype="uint8").astype(bool)
    return Grid(crs=boundary_utm.crs.to_string(), transform=tr, width=w, height=h, res=res, mask=mask)


def read_on_grid(path, grid):
    """Band 1 of `path` resampled (bilinear) onto `grid`; band 2 (validity), when present, masks it."""
    with rasterio.open(path) as src:
        dst = np.full((grid.height, grid.width), np.nan, dtype="float32")
        b1 = src.read(1).astype("float32")
        nod = src.nodata
        if nod is not None and not np.isnan(nod):
            b1[b1 == nod] = np.nan
        if src.count >= 2:
            v = src.read(2)
            b1[v == 0] = np.nan
        reproject(b1, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=grid.transform,
                  dst_crs=grid.crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    dst[(dst < -1) | (dst > 1.5)] = np.nan
    return dst


@dataclass
class Stack:
    grid: Grid
    layers: list
    arr: np.ndarray                           # (n, H, W) raw index
    norm: str = "relative"
    normed: np.ndarray | None = None          # (n, H, W) normalised
    mean: np.ndarray | None = None            # temporal mean of the normalised values
    sd: np.ndarray | None = None
    cv: np.ndarray | None = None
    n_seen: np.ndarray | None = None
    used: list = field(default_factory=list)  # indices of the layers in the summary

    def summary_table(self):
        rows = []
        for i, L in enumerate(self.layers):
            rows.append(dict(i=i, date=L.date, year=L.year, crop=L.crop, field_mean=L.field_mean,
                             valid_pct=None if L.valid_frac is None else round(100 * L.valid_frac, 1),
                             used=i in self.used, file=os.path.basename(L.path)))
        return rows


def build_stack(layers, grid, per_year="median"):
    """Read every layer onto the grid. With per_year = "median" the images of one year (the clear dates near the
    season peak) become one season: their per-pixel median (more robust than any single date to residual haze,
    cloud-shadow edges and one-off canopy states). Layers without a date stay as they are."""
    arrs = []
    for L in layers:
        arrs.append(read_on_grid(L.path, grid))
    if per_year == "median" and layers:
        groups = {}
        for L, a in zip(layers, arrs):
            groups.setdefault(L.year if L.year is not None else L.path, []).append((L, a))
        layers2, arrs2 = [], []
        for key, items in groups.items():
            if len(items) == 1:
                layers2.append(items[0][0]); arrs2.append(items[0][1])
                continue
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = np.nanmedian(np.stack([x for _, x in items]), axis=0).astype("float32")
            L0 = items[0][0]
            L = Layer(path=";".join(os.path.basename(x.path) for x, _ in items),
                      date=" / ".join(x.date for x, _ in items if x.date), year=L0.year, crop=L0.crop,
                      source=f"median of {len(items)} dates")
            layers2.append(L); arrs2.append(a)
        layers, arrs = layers2, arrs2
    for L, a in zip(layers, arrs):
        inside = a[grid.mask]
        L.valid_frac = float(np.isfinite(inside).mean()) if inside.size else 0.0
        L.field_mean = float(np.nanmean(inside)) if np.isfinite(inside).any() else None
    arr = np.stack(arrs) if arrs else np.zeros((0, grid.height, grid.width), "float32")
    return Stack(grid=grid, layers=layers, arr=arr)


def select_layers(stack, crop=None, policy="same_crop", min_valid=0.90, min_years_same_crop=3, crop_group=None):
    """Which layers enter the summary. policy: same_crop (the trial's crop; when fewer than `min_years_same_crop` years
    qualify it falls back to the crop group, but only if the trial crop belongs to that group, and otherwise to all
    seasons, and says so), crop_group, all, or manual (the layers' own `use` flag). Crop names are compared in lower
    case, the group entries too (a user typing "Wheat, Barley" must match the crop history's "wheat"). Layers seeing
    less than `min_valid` of the field are always left out. One layer per year is kept (the first selected) so that a
    year with two images does not count twice; layers with no known year are all kept (nothing says they repeat a
    season). Returns (indices, note)."""
    ok = [i for i, L in enumerate(stack.layers) if (L.valid_frac or 0) >= min_valid and L.field_mean is not None]
    note = ""
    group_names = [str(g).strip().lower() for g in (crop_group or []) if str(g).strip()]

    def by_crop(pred):
        return [i for i in ok if stack.layers[i].crop and pred(stack.layers[i].crop.strip().lower())]

    def n_years(idx):
        return len({stack.layers[i].year for i in idx})

    sel = ok
    if policy == "manual":
        sel = [i for i in ok if stack.layers[i].use]
    elif policy in ("same_crop", "crop_group") and crop:
        c = crop.strip().lower().split()[0]
        same = by_crop(lambda k: c in k)
        group = by_crop(lambda k: any(g in k for g in group_names))
        # The crop group stands in for the trial crop's own crop type (wheat for a durum trial, cereals for barley).
        # For a canola, pea or flax trial the default cereal group would throw away the trial crop's own seasons and
        # zone it on a different crop family (NDVI scaling differs between cereals and canola), so the group is only
        # a fallback when the trial crop is one of its members; otherwise all seasons are used.
        crop_in_group = any(c in g or g in c for g in group_names)
        if policy == "same_crop" and n_years(same) >= min_years_same_crop:
            sel, note = same, f"{len(same)} {crop} seasons"
        elif group and n_years(group) >= min_years_same_crop and (crop_in_group or policy == "crop_group"):
            sel = group
            note = (f"only {n_years(same)} {crop} season(s); crop group {'/'.join(group_names)} used "
                    f"({len(group)} seasons)")
            if not crop_in_group:
                note += f" although {crop} is not in the group"
        else:
            sel = ok
            why = (f"the crop group {'/'.join(group_names)} does not contain {crop}"
                   if group_names and not crop_in_group and policy == "same_crop" else "")
            note = (f"fewer than {min_years_same_crop} seasons of {crop} with a known crop"
                    + (f" and {why}" if why else "")
                    + f": all {len(ok)} clear seasons used (fill in the crop history to restrict them)")
    seen, one = set(), []
    for i in sel:
        y = stack.layers[i].year
        if y is not None:
            if y in seen:
                continue
            seen.add(y)
        one.append(i)
    if len(one) < len(sel) and note:
        note += f"; {len(sel) - len(one)} extra layer(s) of an already used year left out"
    return one, note


def summarise(stack, used, norm="relative"):
    """Normalise each used layer within the field and compute the per-pixel temporal statistics."""
    m = stack.grid.mask
    normed = np.full(stack.arr.shape, np.nan, dtype="float32")
    for i in used:
        a = stack.arr[i]
        v = a[m]
        mu, sd = np.nanmean(v), np.nanstd(v)
        normed[i] = a / mu if norm == "relative" else (a - mu) / (sd if sd > 0 else 1.0)
    sub = normed[used] if used else normed[:0]
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(sub, axis=0) if len(used) else np.full(m.shape, np.nan)
        sd = np.nanstd(sub, axis=0) if len(used) else np.full(m.shape, np.nan)
        n = np.isfinite(sub).sum(axis=0) if len(used) else np.zeros(m.shape, int)
        cv = sd / np.abs(mean) if norm == "relative" else sd
    mean[~m] = np.nan
    sd[~m] = np.nan
    cv[~m] = np.nan
    stack.norm, stack.normed, stack.mean, stack.sd, stack.cv, stack.n_seen, stack.used = norm, normed, mean, sd, cv, n, list(used)
    return stack


def single_map_stack(path, grid, label="multi-year NDVI map"):
    """An already-composited multi-year map (raster) used as the temporal mean directly."""
    a = read_on_grid(path, grid)
    L = Layer(path=path, source="composite", use=True)
    st = Stack(grid=grid, layers=[L], arr=a[None])
    L.valid_frac = float(np.isfinite(a[grid.mask]).mean())
    L.field_mean = float(np.nanmean(a[grid.mask]))
    return summarise(st, [0], norm="relative")


def vector_map_stack(gdf_utm, col, grid):
    """A multi-year NDVI map delivered as polygons / points with a value column, rasterised onto the grid."""
    geoms = [(g, float(v)) for g, v in zip(gdf_utm.geometry, gdf_utm[col]) if g is not None and np.isfinite(v)]
    if gdf_utm.geom_type.isin(["Point", "MultiPoint"]).all():
        geoms = [(g.buffer(grid.res * 0.75), v) for g, v in geoms]
    a = features.rasterize(geoms, out_shape=(grid.height, grid.width), transform=grid.transform, fill=np.nan,
                           dtype="float32", merge_alg=features.MergeAlg.replace)
    L = Layer(path=f"vector:{col}", source="composite")
    st = Stack(grid=grid, layers=[L], arr=a[None])
    L.valid_frac = float(np.isfinite(a[grid.mask]).mean())
    L.field_mean = float(np.nanmean(a[grid.mask]))
    return summarise(st, [0], norm="relative")


def write_tif(path, arr, grid, nodata=np.nan, dtype="float32"):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=grid.height, width=grid.width, count=arr.shape[0],
                       dtype=dtype, crs=grid.crs, transform=grid.transform, nodata=nodata, compress="deflate") as dst:
        dst.write(arr.astype(dtype))
    return path


def save_stack(stack, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    write_tif(os.path.join(out_dir, "ndvi_mean_normalised.tif"), stack.mean, stack.grid)
    write_tif(os.path.join(out_dir, "ndvi_cv.tif"), stack.cv, stack.grid)
    if stack.used:
        write_tif(os.path.join(out_dir, "ndvi_seasons_normalised.tif"), stack.normed[stack.used], stack.grid)
    with open(os.path.join(out_dir, "ndvi_layers.json"), "w", encoding="utf-8") as f:
        json.dump(dict(norm=stack.norm, used=stack.used, layers=stack.summary_table()), f, indent=1)
    return out_dir
