# -*- coding: utf-8 -*-
"""Management zones.

From the multi-year NDVI summary (ndvi.Stack): the temporal mean of the normalised NDVI is lightly smoothed (a 3 x 3
NaN-aware mean, so a single 10 m pixel does not make its own zone), classified into k zones (k-means on the mean, the
1-D equivalent of the fuzzy k-means of MZA, Fridgen et al. 2004; or equal-area quantiles), and patches smaller than
`min_patch_ha` are merged into their neighbours (rasterio sieve), because a VR controller cannot follow a 0.1 ha
island. Zone 1 = the lowest NDVI. Optionally the temporal CV adds Blackmore's stability classes (stable-high,
stable-low, unstable).

From a zone map the grower already has (polygons with a zone column): rasterised on the same grid so that the join and
the economics treat both the same way. No-data codes (negative sentinels, an excluded 0 class, configured values) are
not zones; text labels are put in NDVI order (zone 1 = lowest) so that zone adjacency means the same as for NDVI
zones (see zones_from_vector).

zone_quality() reports the share of the NDVI (and later the yield) variance the zones explain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape

from . import io as eio


def _nan_smooth(a, mask, size=3):
    v = np.where(np.isfinite(a), a, 0.0)
    w = np.isfinite(a).astype(float)
    num = ndimage.uniform_filter(v, size=size, mode="nearest")
    den = ndimage.uniform_filter(w, size=size, mode="nearest")
    with np.errstate(all="ignore"):
        out = num / den
    out[~mask | (den == 0)] = np.nan
    return out


def kmeans_1d(values, k, seed=0):
    """Sorted cluster centres and the break points half-way between them (k-means, several starts)."""
    from sklearn.cluster import KMeans
    v = np.asarray(values, float).reshape(-1, 1)
    if len(v) > 60000:
        rng = np.random.default_rng(seed)
        v = v[rng.choice(len(v), 60000, replace=False)]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(v)
    c = np.sort(km.cluster_centers_.ravel())
    return c, (c[:-1] + c[1:]) / 2.0


def classify(surface, mask, k=3, method="kmeans", breaks=None):
    vals = surface[mask & np.isfinite(surface)]
    if vals.size == 0:
        # np.quantile / np.percentile on an empty array raise an opaque IndexError; say what is missing instead
        raise ValueError("no finite NDVI value inside the field: nothing to classify into zones")
    if method == "kmeans":
        # k-means follows outliers (a slough or a yard at 0.3 x the field mean would take a zone of its own):
        # the centres are found on the values winsorised at the 2nd / 98th percentiles
        lo, hi = np.percentile(vals, [2, 98])
        vals = np.clip(vals, lo, hi)
    if method == "quantile":
        br = np.quantile(vals, np.linspace(0, 1, k + 1)[1:-1])
    elif method == "breaks" and breaks is not None:
        br = np.sort(np.asarray(breaks, float))
    else:
        _, br = kmeans_1d(vals, k)
    z = np.zeros(surface.shape, dtype="int32")
    ok = mask & np.isfinite(surface)
    z[ok] = 1 + np.searchsorted(br, surface[ok], side="right")
    return z, [float(b) for b in br]


def fill_gaps(z, mask):
    """Pixels inside the field without a zone (NaN NDVI: a pixel no season saw) take the nearest zone."""
    miss = mask & (z == 0)
    if not miss.any() or not (z > 0).any():
        return z
    idx = ndimage.distance_transform_edt(z == 0, return_distances=False, return_indices=True)
    out = z[tuple(idx)]
    out[~mask] = 0
    return out.astype("int32")


def sieve(z, mask, min_pixels):
    if min_pixels <= 1:
        return z
    out = features.sieve(z.astype("int32"), size=int(min_pixels), mask=mask, connectivity=8)
    out = np.asarray(out, dtype="int32")
    out[~mask] = 0
    return fill_gaps(out, mask)


def relabel_contiguous(z):
    """Renumber the zone codes present in `z` (> 0) to 1..m, keeping their order. Returns (z, {old: new}).

    Sieving merges every patch smaller than min_patch_ha into its neighbours, so a class made only of small scattered
    patches (a low zone of isolated depressions) disappears; tied quantile breaks can also leave a class empty. The
    survivors are renumbered so that zone 1 is still the lowest NDVI and the codes run 1..k without holes, because the
    report, the zone table and the prescription all count zones as 1..k."""
    present = [int(c) for c in np.unique(z) if c > 0]
    cmap = {c: i + 1 for i, c in enumerate(present)}
    if all(o == n for o, n in cmap.items()):
        return z, cmap
    out = np.zeros_like(z)
    for o, n in cmap.items():
        out[z == o] = n
    return out, cmap


def zones_from_stack(stack, k=3, method="kmeans", smooth=3, min_patch_ha=0.5, breaks=None):
    """NDVI zones from the multi-year summary. info['k'] is the number of zones actually present after sieving (not
    the k requested), the codes are 1..k contiguous (1 = lowest NDVI) and info['labels'] = {code: label}; when a class
    vanished, info['k_requested'], info['class_map'] {class before sieving: final code} and info['note'] say so."""
    g = stack.grid
    if stack.mean is None or not np.isfinite(stack.mean[g.mask]).any():
        # select_layers kept no season (every image sees less than min_valid of the field, the folder is empty, or it
        # only holds trial-year images): stop with the cause rather than an IndexError deep inside classify
        n_lay = len(stack.layers)
        vf = ", ".join(f"{100 * (L.valid_frac or 0):.0f} %" for L in stack.layers[:12])
        raise ValueError(
            f"no NDVI season was usable for zones ({len(stack.used)} of {n_lay} layer(s) selected"
            + (f"; share of the field seen per layer: {vf}" if n_lay else "; the NDVI folder holds no image")
            + "). Each season must see at least min_valid of the field and must not be the trial year: lower "
              "min_valid, fetch / add images that cover the whole boundary, or check the crop history and seasons.")
    surf = _nan_smooth(stack.mean, g.mask, smooth) if smooth and smooth > 1 else stack.mean.copy()
    z, br = classify(surf, g.mask, k=k, method=method, breaks=breaks)
    z = fill_gaps(z, g.mask)
    z = sieve(z, g.mask, int(round(min_patch_ha * 10000 / g.cell_area_m2())))
    z, cmap = relabel_contiguous(z)
    m = len(cmap)
    info = dict(source="ndvi", method=method, k=int(m), labels={c: str(c) for c in range(1, m + 1)}, breaks=br,
                smooth_px=int(smooth or 1), min_patch_ha=float(min_patch_ha), norm=stack.norm,
                n_seasons=len(stack.used),
                seasons=[stack.layers[i].date or stack.layers[i].path for i in stack.used])
    if m != int(k):
        lost = [c for c in range(1, int(k) + 1) if c not in cmap]
        info.update(k_requested=int(k), class_map=cmap,
                    note=(f"{int(k)} zones were requested but only {m} remain: class(es) {lost} vanished because all "
                          f"their patches were under {min_patch_ha} ha and were merged into the neighbouring zones (or "
                          f"the class was empty), so the zones were renumbered 1..{m} (1 = lowest NDVI) and the "
                          f"stored NDVI breaks are still those of the {int(k)} requested classes."))
    return z, surf, info


def stability_classes(stack, cv_threshold=None):
    """Blackmore (2000): 1 = stable, below the field mean; 2 = stable, above; 3 = unstable (temporal CV above the
    threshold; default = the 80th percentile of the field's CV, reported)."""
    m = stack.grid.mask & np.isfinite(stack.mean) & np.isfinite(stack.cv)
    thr = float(np.nanpercentile(stack.cv[m], 80)) if cv_threshold is None else float(cv_threshold)
    ref = 1.0 if stack.norm == "relative" else 0.0
    s = np.zeros(stack.mean.shape, "int32")
    s[m & (stack.mean < ref)] = 1
    s[m & (stack.mean >= ref)] = 2
    s[m & (stack.cv > thr)] = 3
    return s, thr


# Known orders of text zone labels (case-blind, after collapsing spaces / '-' / '_'). A grower's "Low / Medium / High"
# map must keep low < medium < high: merge_small_zones merges a small zone into its code neighbours and the report
# lists zones in code order, and alphabetical order (High, Low, Medium) would make High the only neighbour of Low.
_KNOWN_LABEL_RANK = {"very low": 0, "low": 1, "med": 2, "medium": 2, "mid": 2, "moderate": 2, "high": 3,
                     "very high": 4}

# A map coded {0, 1, .., m} is read as 0-based (0 = a zone) by default. 0 is read as the excluded / unclassified
# class only when it is listed in the no-data codes, or when its polygons cover less than this share of the FIELD
# and reading it as no-data still leaves at least 2 zones (see zones_from_vector). The bound is absolute, not
# relative to the other codes: in a 0-based k-means / natural-breaks export the low-NDVI class is often the smallest
# one, and a relative rule threw away a real zone covering 30 % of the field (and turned a 2-zone map into 1 zone).
# An excluded remnant (yard, slough, sliver) is a few percent of the field at most.
ZERO_NODATA_MAX_FIELD_SHARE = 0.05
# a grower zone map that leaves more than this share of the field inside no zone polygon is refused
# (zones_from_vector): it is the map of another boundary, of the cropped part only, or of another field
ZONE_MAP_MAX_UNMAPPED = 0.30


def _norm_label(s):
    """Label key for the known low < medium < high order and for grouping spellings ('Low', ' low ', 'LOW')."""
    return " ".join(str(s).strip().lower().replace("_", " ").replace("-", " ").split())


def _nd_key(s):
    """Key for matching configured no-data values: case-blind and trimmed, but the minus sign is kept, so that the
    no-data value -1 never matches a zone labelled 1."""
    return " ".join(str(s).strip().lower().split())


def _burn(gdf, codes, grid):
    """Rasterise (geometry, code) pairs on the grid; cells outside the field mask are 0 (no zone)."""
    shapes = [(geom, int(c)) for geom, c in zip(gdf.geometry, codes) if geom is not None and not geom.is_empty]
    if shapes:
        z = features.rasterize(shapes, out_shape=(grid.height, grid.width), transform=grid.transform, fill=0,
                               dtype="int32")
    else:
        z = np.zeros((grid.height, grid.width), "int32")
    z[~grid.mask] = 0
    return z


def _nodata_sets(nodata_codes):
    """The configured no-data values as normalised text, as integers and as floats (those that parse as numbers).
    The integers serve integer-coded maps (a configured '2.0' matches the code 2); the floats serve maps whose values
    are fractional classes, where rounding would make a configured 0.5 match the class 0 (or 1)."""
    if nodata_codes is None:
        nodata_codes = []
    elif isinstance(nodata_codes, (str, bytes, int, float, np.number)):
        nodata_codes = [nodata_codes]
    nd_text = {_nd_key(x) for x in nodata_codes}
    nd_num, nd_float = set(), set()
    for x in nodata_codes:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            nd_num.add(int(round(f)))
            nd_float.add(f)
    return nd_text, nd_num, nd_float


def _fmt_value(v):
    """A fractional class value as the grower would write it: 0.8 -> '0.8', 1.0 -> '1' (no float noise)."""
    return np.format_float_positional(round(float(v), 9), trim="-")


# Values within this distance of an integer are read as that integer (float noise from a GIS export, 2.0000000001),
# so an integer-coded map stays on the integer path; anything further away is a fractional class value.
_INT_TOL = 1e-6
# More distinct fractional values than this is unusual for a zone map: the note asks to check that the column holds
# zone classes and not a continuous variable (a per-polygon mean NDVI or yield picked by mistake).
_MANY_CLASSES = 12


def zones_from_vector(gdf_utm, col, grid, surface=None, nodata_codes=None):
    """Rasterise an uploaded zone map and return (raster, info).

    The raster value 0 means "no zone" everywhere in the engine. Polygons that are NOT a management zone are not
    burnt, and fill_gaps gives their cells to the neighbouring zones (the area stays in the field, as part of the
    zone around it, instead of becoming a zone with its own curve and prescription rate):
      - rows with a null or blank value (an unclassified sliver, yard or slough);
      - numeric codes below 0 (-1, -9999: the usual no-data sentinels of MZA / SMS / GIS exports) and any value
        listed in `nodata_codes` (numbers compared as numbers, text case-blind);
      - the code 0 when it appears beside codes with gaps (0 beside 2, 3): the unclassified class.
    A map coded exactly {0, 1, .., m} (0-based k-means / GIS export) is read as 0-based by default: every code is
    shifted by +1 (info['code_shift'] = 1) so that its zone 0 is burnt. 0 is read as no zone instead only when it is
    listed in `nodata_codes`, or when its polygons cover less than ZERO_NODATA_MAX_FIELD_SHARE of the field AND
    zones 1..m still give at least 2 zones in the field; this rule never reduces the map to a single zone (a
    single-zone map has no variable-rate analysis at all, the costlier mistake). The note says which reading was
    taken and how to force the other.
    The no-data values found are listed in info['nodata_codes'] (their polygon area in info['nodata_ac']).

    Digit strings (a DBF character field) are parsed as numbers, so '10' does not sort between '1' and '2'. A
    character column is read as a numeric map when at least 2 distinct values (no-data values aside) parse as numbers
    and the distinct values that do not parse are at most 1 or at most a third of the distinct values; the decision is
    on distinct values, not rows, because a dissolved map has one row per zone. The unparsed values ('NA', 'Slough')
    are then no zone and listed in the note. Numeric codes keep their order. Numeric values that are not all
    integers (relative-productivity classes 0.8 / 1.0 / 1.2, half-integer codes 1.5 / 2.5) are ordered class labels:
    the distinct finite non-negative values are ranked 1..k in increasing order (info['label_order'] = 'ranked values')
    and are never merged by rounding (rounding turned 0.8 / 1.0 / 1.2 into a single zone, and round-half-to-even made
    1.5 and 2.5 the same zone, without a word); the 0-based / excluded-0 rules do not apply to them. Non-finite values
    (inf, -inf) are no zone. Text labels are ordered by the mean of `surface` (the multi-year NDVI mean on the same
    grid) over their cells, so zone 1 = the lowest NDVI as for NDVI zones; without a usable surface by the known order
    low < med/medium < high (case-blind); otherwise alphabetically. info['label_order'] = 'numeric' | 'ranked values' |
    'surface' | 'known' | 'alphabetical' says which. The order matters because merge_small_zones merges a small zone into its code
    neighbours. The codes present in the raster are then made contiguous 1..k (k = info['k']; a zone too small to
    cover one cell centre is lost and listed in info['labels_missing']), info['labels'] = {raster code: the grower's
    original value} keeps the link to the map, and info['note'] explains in one English sentence anything unusual
    (no-data codes, shifted codes, label order, lost zones) for the report and the app."""
    nd_text, nd_num, nd_float = _nodata_sets(nodata_codes)

    vals = gdf_utm[col]
    keep = vals.notna().to_numpy().copy()
    if not pd.api.types.is_numeric_dtype(vals):
        keep &= (vals.astype(str).str.strip() != "").to_numpy()
    gdf_utm, vals = gdf_utm[keep].reset_index(drop=True), vals[keep].reset_index(drop=True)

    parts = []           # pieces of info['note']
    shift = 0
    orig_txt = None      # the grower's spelling when the numbers came from text
    is_nd = np.zeros(len(vals), bool)
    if pd.api.types.is_numeric_dtype(vals) and not pd.api.types.is_bool_dtype(vals):
        num = vals.astype(float)
    else:
        txt = vals.astype(str).str.strip()
        txt_nd = txt.map(_nd_key).isin(nd_text).to_numpy()
        parsed = pd.to_numeric(txt, errors="coerce")
        # numeric when the values that are not listed no-data labels parse as numbers. A digit-coded character column
        # with a few stray text values ('NA', 'Slough') is still a numeric map: those values are no zone (and said so)
        # rather than turning the whole column into text labels numbered alphabetically ('10' between '1' and '2'),
        # or worse, the stray value becoming a management zone with its own curve and prescription rate.
        # The decision is taken on DISTINCT values, not rows: grower zone maps (MZA, SMS, GIS) are usually dissolved,
        # one row per zone, so with 3 or 4 rows a single 'NA' row is already 25-33 % of the rows and a row-share rule
        # would flip the whole map to text labels. The column is numeric when at least 2 distinct codes parse as
        # numbers and the distinct unparsed values are at most 1, or at most a third of all distinct values (a real
        # text-labelled map, 'A', 'B', '1', '2', stays text). Unparsed values are grouped case-blind, as text labels.
        ok = parsed[~txt_nd].notna()
        n_num = int(parsed[~txt_nd].dropna().nunique())
        bad_txt = txt[~txt_nd][~ok.to_numpy()]
        n_bad = int(bad_txt.map(_norm_label).nunique())
        n_distinct = n_num + n_bad
        if (~txt_nd).any() and ok.all():
            num, orig_txt, is_nd = parsed.where(~txt_nd), txt, txt_nd.copy()
        elif n_num >= 2 and (n_bad <= 1 or n_bad <= n_distinct / 3):
            stray = sorted(set(bad_txt))
            txt_nd = txt_nd | parsed.isna().to_numpy()
            num, orig_txt, is_nd = parsed.where(~txt_nd), txt, txt_nd.copy()
            parts.append("the text value(s) " + ", ".join(repr(x) for x in stray) + " in a numeric zone column were "
                         "read as no zone")
        else:
            num = None

    order = "numeric"
    fractional = False
    if num is not None:
        num = num.astype(float)
        # an infinite code ('inf' in a character column, or inf in a float column) is no zone, not a crash. It has no
        # integer to list among the no-data codes, so its name is kept apart (int(NaN) used to raise here).
        nonfinite = ~np.isfinite(num.to_numpy())
        nonfinite_found = sorted({str(float(v)) for v in num[nonfinite].dropna()}) if orig_txt is None else []
        num = num.where(~nonfinite)
        # the values that can be zones: finite, not negative (the -1 / -9999 sentinels) and not a configured no-data
        # value. Whether they are all integers decides how they are read.
        nd_f = np.zeros(len(num), bool)
        for f in nd_float:
            nd_f |= np.isclose(num.to_numpy(), f, rtol=0.0, atol=1e-9)
        cand = (num.notna() & (num >= 0)).to_numpy() & ~nd_f & ~is_nd
        fractional = bool((np.abs(num[cand] - num[cand].round()) > _INT_TOL).any())
        if fractional:
            # ordered class labels: rank the distinct values 1..k (zone 1 = the lowest value). Rounding them to
            # integers merged distinct classes in silence (0.8 / 1.0 / 1.2 -> one zone; 1.5 and 2.5 -> 2), and a
            # one-zone map has no variable-rate analysis at all. Values are keyed at 9 decimals so float noise
            # (1.2 vs 1.2000000000001) does not split a class.
            is_nd |= ~cand
            key = num.round(9)
            uniq = sorted(float(v) for v in key[cand].unique())
            rank = {v: i + 1 for i, v in enumerate(uniq)}
            orig = key.where(cand).map(rank)
            valid = list(range(1, len(uniq) + 1))
            order = "ranked values"
            shown = ", ".join(_fmt_value(v) for v in uniq[:8]) + (", ..." if len(uniq) > 8 else "")
            msg = (f"the zone values are not all integers ({shown}), so they were read as {len(uniq)} ordered "
                   f"classes ranked 1..{len(uniq)} in increasing order (zone 1 = the lowest value; distinct values "
                   "are never merged)")
            if 0.0 in rank:
                msg += (", and 0 was ranked as the lowest class (list 0 in the zones' no-data codes if it marks "
                        "excluded areas)")
            if len(uniq) > _MANY_CLASSES:
                msg += (f"; {len(uniq)} distinct values is many for a zone map: check that the column holds zone "
                        "classes and not a continuous variable such as a mean NDVI or yield per polygon")
            parts.append(msg)
        else:
            orig = num.round()
            is_nd |= (orig.isna() | (orig < 0) | orig.isin(list(nd_num))).to_numpy() | nd_f
            valid = sorted(int(c) for c in orig[~is_nd].unique())
        if 0 in valid:
            zero_is_zone = valid == list(range(len(valid)))
            gap_kept = False
            if not zero_is_zone and len(valid) == 2:
                # 0 beside a single other code ({0, 2}): reading 0 as no-data would leave one zone and no VR analysis
                # at all, so 0 stays a zone here too (list it in the no-data codes to exclude it)
                zero_is_zone, gap_kept = True, True
            small_zero = None
            if zero_is_zone and len(valid) > 2:
                # {0, 1, .., m} reads both as a 0-based map of m+1 zones and as zones 1..m plus a 0-coded excluded
                # class. The default is the 0-based reading; only a small remnant (in absolute terms: a share of the
                # field) is taken as excluded, and only when zones 1..m still make at least 2 zones. The shares are
                # measured on the grid inside the field mask, so polygons outside the boundary do not count. With
                # m = 1 ({0, 1}) the excluded reading would leave one zone, so it is never taken (len(valid) > 2).
                zc = _burn(gdf_utm[~is_nd], orig[~is_nd].astype(int) + 1, grid)
                n_field = int(grid.mask.sum())
                n_code = {c: int(((zc == c + 1) & grid.mask).sum()) for c in valid}
                others_in_field = sum(1 for c in valid if c and n_code[c] > 0)
                share0 = n_code[0] / n_field if n_field else 0.0
                if share0 < ZERO_NODATA_MAX_FIELD_SHARE and others_in_field >= 2:
                    zero_is_zone, small_zero = False, share0
            if zero_is_zone:
                shift = 1
                if gap_kept:
                    parts.append(f"the map has the codes {valid}; 0 was kept as a zone because excluding it would leave "
                                 "a single zone (list 0 in the zones' no-data codes if it marks excluded areas)")
                elif len(valid) > 1:
                    parts.append(f"the map is coded 0..{len(valid) - 1}, so 0 was read as a zone of a 0-based map and "
                                 "every code was shifted by +1 (if 0 marks excluded areas, list 0 in the zones' no-data "
                                 "codes)")
            else:
                # 0 beside codes with gaps, or a small 0-coded remnant beside 1..m: the unclassified / excluded class
                is_nd |= (orig == 0).to_numpy()
                valid.remove(0)
                if small_zero is not None:
                    parts.append(f"0 was taken as an excluded class rather than a zone of a 0-based map because its "
                                 f"polygons cover only {small_zero:.1%} of the field (less than "
                                 f"{ZERO_NODATA_MAX_FIELD_SHARE:.0%}) and {len(valid)} zones remain (recode the "
                                 "map 1..k if 0 is a zone)")
        if orig_txt is not None:
            nodata_found = sorted(set(orig_txt[is_nd]))
        else:
            # the finite no-data values (as integers when they are integers, else as numbers), then the non-finite
            # ones by name ('inf'); taken from `num`, not from the ranks, which are NaN for no-data rows
            fin_nd = num[is_nd].dropna().unique()
            nodata_found = sorted({int(round(c)) if abs(c - round(c)) <= _INT_TOL else float(c) for c in fin_nd})
            nodata_found += nonfinite_found
        codes = orig.where(~is_nd, 0).astype(int) + shift
        labels = {}
        for c in valid:
            if orig_txt is not None:
                # keep the grower's spelling of the label ('01' stays '01')
                labels[c + shift] = str(orig_txt[((orig == c).to_numpy()) & ~is_nd].iloc[0])
            elif fractional:
                labels[c + shift] = _fmt_value(uniq[c - 1])
            else:
                labels[c + shift] = str(c)
        z = _burn(gdf_utm[~is_nd], codes[~is_nd], grid)
    else:
        txt = vals.astype(str).str.strip()
        is_nd = txt.map(_nd_key).isin(nd_text).to_numpy()
        nodata_found = sorted(set(txt[is_nd]))
        # one zone per label whatever its spelling: 'Low', 'low' and ' LOW ' are the same zone (first spelling kept)
        canon = {}
        for t in txt:
            canon.setdefault(_norm_label(t), t)
        txt = txt.map(lambda t: canon[_norm_label(t)])
        uniq = sorted(txt[~is_nd].unique())
        alpha = {u: i + 1 for i, u in enumerate(uniq)}
        z = _burn(gdf_utm[~is_nd], txt[~is_nd].map(alpha), grid)
        order, rank, differs = "alphabetical", None, None
        if surface is not None and np.shape(surface) == z.shape:
            s = np.asarray(surface, float)
            means = {}
            for u, c in alpha.items():
                cells = z == c
                if cells.any():
                    fin = cells & np.isfinite(s)
                    means[u] = float(s[fin].mean()) if fin.any() else np.nan
            if means and all(np.isfinite(v) for v in means.values()):
                # labels absent from the raster are lost anyway: they go last, alphabetically
                rank = sorted(means, key=lambda u: (means[u], u)) + [u for u in uniq if u not in means]
                order = "surface"
                if all(_norm_label(u) in _KNOWN_LABEL_RANK for u in means):
                    named = sorted(means, key=lambda u: (_KNOWN_LABEL_RANK[_norm_label(u)], u))
                    if named != rank[:len(named)]:
                        # the grower's names and the NDVI disagree: say so, the zone numbers follow the NDVI
                        differs = ("their NDVI order (" + " < ".join(rank[:len(named)]) +
                                   ") differs from the order of their names")
        if rank is None and uniq and all(_norm_label(u) in _KNOWN_LABEL_RANK for u in uniq):
            rank = sorted(uniq, key=lambda u: (_KNOWN_LABEL_RANK[_norm_label(u)], u))
            order = "known"
        if rank is None:
            rank = uniq
        lut = {u: i + 1 for i, u in enumerate(rank)}
        remap = np.zeros(len(uniq) + 1, "int32")
        for u, c in alpha.items():
            remap[c] = lut[u]
        z = remap[z]
        labels = {c: u for u, c in lut.items()}
        parts.append({"surface": "the text labels were ordered by their mean NDVI (zone 1 = lowest NDVI)",
                      "known": "the text labels were ordered low < medium < high",
                      "alphabetical": "the text labels were numbered alphabetically (no NDVI to order them and not "
                                      "low / medium / high names), so zone 1 is not necessarily the lowest NDVI"
                      }[order])
        if differs:
            parts.append(differs)

    nd_ac = float(gdf_utm[is_nd].geometry.area.sum() / eio.M2_PER_AC) if nodata_found else 0.0
    if nodata_found:
        parts.insert(0, f"value(s) {nodata_found} were read as no zone ({nd_ac:.1f} ac of polygons, not burnt; "
                        "their cells went to the neighbouring zones)")
    if not (z > 0).any():
        raise ValueError(
            f"the zone map has no zone polygon inside the field (column '{col}'"
            + (f"; no-data value(s) {nodata_found} were left out" if nodata_found else "")
            + "): check the zone column, the no-data codes and that the map covers the boundary")
    # Field cells inside no zone polygon (no-data polygons apart): fill_gaps gives them the nearest zone, which is
    # right for slivers between polygons but not for a map clipped to an old boundary, of the cropped part only,
    # or of another field with a small overlap. The share is recorded and printed, and a map that leaves more than
    # ZONE_MAP_MAX_UNMAPPED of the field unclassified is refused rather than analysed as if it covered the field.
    nd_r = (_burn(gdf_utm[is_nd], pd.Series(1, index=gdf_utm.index[is_nd]), grid) > 0) if nodata_found else np.zeros_like(grid.mask)
    unmapped = grid.mask & (z == 0) & ~nd_r
    n_field = int(grid.mask.sum()) or 1
    share_unmapped = float(unmapped.sum() / n_field)
    ac_unmapped = float(unmapped.sum() * grid.cell_area_m2() / eio.M2_PER_AC)
    if share_unmapped > ZONE_MAP_MAX_UNMAPPED:
        raise ValueError(f"the zone map covers only {100 * (1 - share_unmapped):.0f} % of the field boundary "
                         f"({ac_unmapped:.1f} ac inside no zone polygon): check that it is the map of this field and "
                         "of the current boundary, or clip the boundary to the mapped area")
    if share_unmapped > 0.005:
        parts.append(f"{100 * share_unmapped:.1f} % of the field ({ac_unmapped:.1f} ac) lies outside every zone "
                     "polygon and was given the nearest zone")
    z = fill_gaps(z, grid.mask)
    z, cmap = relabel_contiguous(z)
    info = dict(source="zone map", column=col, labels={cmap[c]: labels[c] for c in cmap}, k=len(cmap),
                label_order=order, unmapped_share=share_unmapped, unmapped_ac=ac_unmapped)
    if shift:
        info["code_shift"] = int(shift)
    if nodata_found:
        info["nodata_codes"] = nodata_found
        info["nodata_ac"] = nd_ac
    missing = [labels[c] for c in labels if c not in cmap]
    if missing:
        info["labels_missing"] = missing
        parts.append(f"zone(s) {missing} cover no cell centre of the grid and were lost, so {len(cmap)} zone(s) "
                     "remain")
    if any(o != n for o, n in cmap.items()):
        info["code_map"] = {labels[o]: n for o, n in cmap.items()}
        if order == "numeric" and not missing:
            parts.append(f"the codes present were renumbered 1..{len(cmap)}")
    if parts:
        info["note"] = "Zone map: " + "; ".join(parts) + "."
    return z, info


def _extend_to_boundary(z, grid, boundary_geom):
    """The zone raster grown to every cell the boundary touches. grid.mask keeps only cells whose centre is inside
    the boundary, so polygons built from it stop up to half a cell (more in acute corners) short of the boundary and
    the prescription would leave an edge strip where the controller applies its out-of-map rate. Every point of the
    field lies in a cell the boundary touches, so after this the clipped zones cover the whole boundary; the added
    cells take the nearest zone."""
    touched = features.rasterize([(boundary_geom, 1)], out_shape=(grid.height, grid.width), transform=grid.transform,
                                 fill=0, all_touched=True, dtype="uint8").astype(bool)
    ext = touched | (z > 0)
    return fill_gaps(np.where(ext, z, 0).astype("int32"), ext)


def to_polygons(z, grid, boundary_utm=None, surface=None, labels=None):
    """Zone raster -> one MultiPolygon per zone (clipped to the boundary, which the zones cover entirely), with area,
    mean NDVI (the mean over the zone's cells inside the field mask) and 'label' (text): the grower's value from
    `labels` {raster code: label} (info['labels'] of zones_from_vector) when given, else the code as text. The label
    keeps the link from the prescription back to the grower's own zone map (code 2 may be his 'Medium' or 1.0)."""
    b = boundary_utm.geometry.union_all() if boundary_utm is not None else None
    zv = _extend_to_boundary(z, grid, b) if b is not None and not b.is_empty and (z > 0).any() else z
    recs = []
    for geom, val in features.shapes(zv.astype("int32"), mask=zv > 0, transform=grid.transform, connectivity=8):
        recs.append(dict(zone=int(val), geometry=shape(geom)))
    g = gpd.GeoDataFrame(recs, geometry="geometry", crs=grid.crs)
    if g.empty:
        return g
    g = g.dissolve("zone", as_index=False)
    if b is not None:
        g["geometry"] = g.geometry.intersection(b)
    g = g[~g.geometry.is_empty].copy()
    g["area_ac"] = g.geometry.area / eio.M2_PER_AC
    g["area_pct"] = 100 * g.area_ac / g.area_ac.sum()
    if surface is not None:
        g["ndvi_rel"] = [float(np.nanmean(surface[z == k])) if (z == k).any() else float("nan") for k in g.zone]
    lab = {int(k): str(v) for k, v in (labels or {}).items()}
    g["label"] = [lab.get(int(k), str(int(k))) for k in g.zone]
    return g.sort_values("zone").reset_index(drop=True)


def zone_quality(z, values, mask=None):
    """Share of the variance of `values` (same shape as z, or 1-D with z 1-D) explained by the zones, and the
    per-zone means. 0 = the zones separate nothing; the higher, the better the zones stratify the variable."""
    z = np.asarray(z).ravel()
    v = np.asarray(values, float).ravel()
    ok = (z > 0) & np.isfinite(v)
    if mask is not None:
        ok &= np.asarray(mask).ravel()
    z, v = z[ok], v[ok]
    if len(v) < 3:
        return dict(r2=None, means={})
    tot = ((v - v.mean()) ** 2).sum()
    within = sum(((v[z == k] - v[z == k].mean()) ** 2).sum() for k in np.unique(z))
    return dict(r2=float(1 - within / tot) if tot > 0 else None,
                means={int(k): float(v[z == k].mean()) for k in np.unique(z)},
                n={int(k): int((z == k).sum()) for k in np.unique(z)})
