# -*- coding: utf-8 -*-
"""Figures and the PDF report (English; Olds College palette, the same house style as OCCD).

    figures(state)  -> writes output/figures/*.png and returns {name: path}
    make_pdf(state) -> output/<name>_VR_economics_report.pdf
"""
from __future__ import annotations

import datetime as dt
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

NAVY, LIGHT_BLUE, GREEN, ORANGE, MUTED = "#00358E", "#009CDE", "#00B373", "#F5A623", "#4B5563"
ZONE_COLORS = ["#C0392B", "#F5A623", "#7FB247", "#1E8449", "#0B5345"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.edgecolor": "#9CA3AF", "axes.titleweight": "bold",
                     "axes.titlecolor": NAVY, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def _extent(grid):
    x0, y0, x1, y1 = grid.bounds
    return [x0, x1, y0, y1]


def _zone_cmap(k):
    cols = ZONE_COLORS[:k] if k <= 3 else ZONE_COLORS[:k]
    if k == 2:
        cols = [ZONE_COLORS[0], ZONE_COLORS[3]]
    if k == 3:
        cols = [ZONE_COLORS[0], ZONE_COLORS[1], ZONE_COLORS[3]]
    return ListedColormap(cols), BoundaryNorm(np.arange(0.5, k + 1.5), k)


def _outline(ax, st, lw=0.4):
    st.plots.boundary.plot(ax=ax, color="k", lw=lw)
    st.boundary.boundary.plot(ax=ax, color=NAVY, lw=1.0)


def _basis(R):
    """The summary basis of one input (project.summary_basis): every strategy valued on the main model on the field's
    zone acres. Rebuilt here for results computed before it existed."""
    b = R.get("summary_basis")
    if b:
        return b
    from .project import summary_basis, _flat_row
    b = summary_basis(R, R["price_crop"], R["price_input"])
    if R.get("verdict"):
        b["recommended_flat"] = _flat_row(R["decision"], R["verdict"]["flat_rate"], R["price_crop"], R["price_input"],
                                          R.get("status_quo"))
    return b


def _is_zone(R):
    return (R.get("model_kind") or ("zone" if len(R["decision"]["r_zone"]) > 1 else "pooled")) == "zone"


def _main_form(R):
    """Functional form of the main model (after any straight-line fallback): the one the bootstrap refits."""
    M = R.get("models") or {}
    k = R.get("model_kind") or ("zone" if "zone" in M else "pooled")
    return (M.get(k) or M.get("pooled") or {}).get("form") or "quadratic"


def _min_rates(form):
    """Distinct design rates a curve of this form needs (response.min_design_rates): 2 for a line, else 3."""
    return 2 if form == "linear" else 3


FORM_LABEL = {"quadratic": "quadratic", "qplateau": "quadratic-plateau", "linear": "straight line"}


def _cf_flat(R):
    """Cross-fitted gain of the best flat rate (the single-curve optimum, the verdict's flat rate) over the status quo.
    With a zone model it is R['crossfit_flat']: the rate chosen by the single curve on the training half and valued,
    like the status quo, with the zone model of the held-out half over the field's zone acres (the basis of the
    in-sample row, project.summary_basis). With one analysis zone the main cross-fit is already on that basis."""
    return (R.get("crossfit_flat") or R.get("crossfit") or {}).get("gain_uniform_vs_sq")


def _cf_flat_same_basis(R):
    """True when _cf_flat(R) is valued on the basis of the summary table's in-sample rows. Results made before
    project.analyse cross-fitted it on that basis (no 'crossfit_flat_basis') judged the single curve on held-out
    plots at their own zone mix: a different quantity, labelled as such so it is not read as a check of the in-sample
    gain beside it."""
    return bool(R.get("crossfit_flat_basis")) or not _is_zone(R)


def _cf_flat_label(R, short=False):
    """Row label of the best flat vs status quo cross-fit (summary table, uncertainty table, economics bars)."""
    if _cf_flat_same_basis(R):
        return "Best flat (cross-fitted)" if short else "best flat vs status quo (cross-fitted, same basis as the summary)"
    return ("Best flat (cross-fitted, one-curve model)" if short
            else "best flat (one curve on the plots) vs status quo (cross-fitted; not the summary's basis)")


def _cf_txt(c, short=False):
    """A cross-fit summary (economics.crossfit: mean of the per-repeat averages of the two folds, sd, 5th-95th
    percentile lo..hi, share of repeats > 0, n repeats) as text."""
    if not c:
        return "-"
    lo, hi = c.get("lo"), c.get("hi")
    rng = f"{lo:+.2f} to {hi:+.2f}" if lo is not None and hi is not None else "-"
    share = f"{100 * c['share_positive']:.0f} %" if c.get("share_positive") is not None else "-"
    if short:
        return f"{c['mean']:+.2f} (90 %: {rng}; {share} > 0)"
    return f"{c['mean']:+.2f}", f"{rng}; {share} of {c.get('n', '-')} repeats > 0"


def _cf_err(c):
    """Asymmetric error bar (below, above) from the cross-fit 90 % range lo..hi; zero when it is missing."""
    if not c or c.get("lo") is None or c.get("hi") is None:
        return 0.0, 0.0
    return max(0.0, c["mean"] - c["lo"]), max(0.0, c["hi"] - c["mean"])


def _zone_txt(z):
    """A zone key as printed: the single-curve result is keyed None ('None' after a json round trip) -> 'field'."""
    return "field" if z in (None, "None") else str(z)


def _share_clause(J):
    """'min share 0.95, ' for a polygon join; '' in point mode, where assign_yield keeps whole points inside a plot
    and stores no min_share (printing 'min share None' suggested a rule that did not apply)."""
    ms = (J.get("attrition") or {}).get("min_share")
    return "" if ms is None or J.get("mode") == "point" else f"min share {ms}, "


def _esc(s):
    """Text for a reportlab Paragraph (an error message may hold '<' or '&')."""
    from xml.sax.saxutils import escape
    return escape(str(s))


def _area_missing_txt(J):
    """Why the analysed area is unknown (point mode with no real area: project.join stores which length is missing in
    J['area_missing']). 'area unknown (no swath width ...)' only when the swath is missing; a file that only lacks the
    distance is sent to the distance column, which is the one to fix."""
    miss = J.get("area_missing")
    if miss == "distance":
        return ("area not computed (no distance column in the yield file: each yield point weighs 1; choose the "
                "distance column in step 1)")
    if miss == "swath and distance":
        return "area unknown (no swath width and no distance in the yield file: each yield point weighs 1)"
    if miss == "swath":
        return "area unknown (no swath width: each yield point weighs 1)"
    # results from before the missing length was recorded
    return "area unknown (no swath width or no distance: each yield point weighs 1)"


def _controls_txt(R, st):
    """'controls: ...' from the controls the main fit really used (R['controls'], project._controls_used): the other
    inputs by name, the combine shares and the trend surface only when they are in the model. Results from before
    R['controls'] existed keep the old fixed sentence."""
    ctl = R.get("controls")
    if ctl is None:
        return ("controls: the combine shares, a quadratic trend surface"
                + (", the other tested input" if len(st.inputs) > 1 else ""))
    words = []
    others = [c for c in ctl if c not in ("combine shares", "trend surface")]
    if "combine shares" in ctl:
        words.append("the combine shares")
    if "trend surface" in ctl:
        words.append("a quadratic trend surface")
    if others:
        words.append(("the other tested input " if len(others) == 1 else "the other tested inputs ")
                     + ", ".join(others))
    return "controls: " + ", ".join(words) if words else "no controls (no combine shares, trend surface or other input)"


def _cf_skips_txt(R):
    """One sentence on the cross-fit half-splits that were skipped, by reason (project._cf_skips); '' when none were
    skipped or the counts are not stored (older results)."""
    sk = R.get("crossfit_skips") or {}
    n = int(sk.get("n_tried") or 0)
    ns, no, nf = (int(sk.get(k) or 0) for k in ("n_no_support", "n_no_overlap", "n_failed"))
    if not n or not (ns or no or nf):
        return ""
    used = int(sk.get("n_used", n - ns - no - nf))
    parts = [t for c, t in ((ns, f"{ns} without the design support for a curve in one half"),
                            (no, f"{no} without a common applied-rate range"),
                            (nf, f"{nf} whose fit failed")) if c]
    err = f" (first error: {_esc(str(sk['first_error'])[:160])})" if nf and sk.get("first_error") else ""
    return f" Cross-fit: {used} of {n} half-splits were used; skipped: {', '.join(parts)}{err}."


def _rates_txt(rz):
    return ", ".join((f"z{z}: {r:.0f}" if z not in (None, "None") else f"field: {r:.0f}") for z, r in rz.items())


def _rate_src_txt(rs):
    """'seed: 260 units applied (unit median), 1 unit applied (plot median); urea: ...' from the join's rate_source
    ({input: {source: n units}}, project.join); the raw dict reprs were printed before."""
    parts = []
    for n, d in (rs or {}).items():
        if isinstance(d, dict):
            parts.append(f"{n}: " + ", ".join(f"{int(v)} unit{'' if int(v) == 1 else 's'} {k}" for k, v in d.items()))
        else:
            parts.append(f"{n}: {d}")
    return "; ".join(parts) or "-"


def _corner_bins(nx):
    """Per-bin corner flags of a continuous NDVI prescription (economics.ndvi_prescription), aligned with nx['bins'],
    and the share of the field at a corner. Newer results carry corner=True/False per bin and res['share_corner'];
    older ones only res['corner'] per bin index and the weights, from which both are derived here."""
    bins = nx.get("bins") or []
    cz = nx.get("corner") or {}
    if bins and "corner" in bins[0]:
        flags = [bool(b_.get("corner")) for b_ in bins]
    else:
        # res['corner'] and res['bins'] are both in the order of the used bins (decide keeps the zone order)
        vals = [bool(v) for v in cz.values()]
        flags = vals if len(vals) == len(bins) else [False] * len(bins)
    sh = nx.get("share_corner")
    if sh is None and cz:
        wt = nx.get("weights") or {}
        tot = sum(float(w) for w in wt.values()) or 1.0
        sh = sum(float(wt.get(k_, 0)) for k_, c_ in cz.items() if c_) / tot
    return flags, (float(sh) if sh is not None else 0.0)


def figures(st, parts=("ndvi", "units", "analysis")):
    """Write the figures of the steps that have run. parts: any of ndvi, units, analysis."""
    d = os.path.join(st.out, "figures")
    os.makedirs(d, exist_ok=True)
    F = dict(getattr(st, "figs", None) or {})
    g = st.grid
    ext = _extent(g)
    if "ndvi" in parts and getattr(st, "zone_r", None) is not None:
        _fig_ndvi(st, d, F, g, ext)
    if "units" in parts and getattr(st, "units", None) is not None:
        _fig_units(st, d, F)
    if "analysis" in parts and st.res.get("analysis"):
        _fig_analysis(st, d, F, g, ext)
    st.figs = F
    return F


def _fig_ndvi(st, d, F, g, ext):

    # ---------------------------------------------------------------- NDVI seasons
    if st.stack is not None and st.stack.used:
        used = st.stack.used
        n = len(used)
        cols = min(5, n)
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        vals = st.stack.normed[used][:, g.mask]
        lo, hi = np.nanpercentile(vals, [2, 98])
        for ax, i in zip(axs.ravel(), used):
            L = st.stack.layers[i]
            ax.imshow(np.where(g.mask, st.stack.normed[i], np.nan), extent=ext, cmap="RdYlGn", vmin=lo, vmax=hi)
            st.boundary.boundary.plot(ax=ax, color="k", lw=0.5)
            ax.set_title(f"{L.year}{' ' + L.crop if L.crop else ''}", fontsize=8)
        fig.suptitle("Peak-season NDVI per season, relative to the field mean", color=NAVY, fontweight="bold")
        F["ndvi_seasons"] = os.path.join(d, "fig_ndvi_seasons.png")
        fig.savefig(F["ndvi_seasons"])
        plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        im = axs[0].imshow(np.where(g.mask, st.stack.mean, np.nan), extent=ext, cmap="RdYlGn",
                           vmin=np.nanpercentile(st.stack.mean[g.mask], 2), vmax=np.nanpercentile(st.stack.mean[g.mask], 98))
        plt.colorbar(im, ax=axs[0], shrink=0.7, label="mean relative NDVI")
        axs[0].set_title(f"Temporal mean ({len(used)} seasons)")
        im = axs[1].imshow(np.where(g.mask, st.stack.cv, np.nan), extent=ext, cmap="magma_r",
                           vmax=np.nanpercentile(st.stack.cv[g.mask], 98))
        plt.colorbar(im, ax=axs[1], shrink=0.7, label="temporal CV")
        axs[1].set_title("Temporal variability")
        # zone codes are 1..k contiguous after sieving (zones.py), so the highest code sets the colour scale; the
        # title uses the number of zones actually present (info 'k')
        k = int(st.zone_r.max())
        cm, nm = _zone_cmap(k)
        axs[2].imshow(np.where(st.zone_r > 0, st.zone_r, np.nan), extent=ext, cmap=cm, norm=nm)
        axs[2].set_title(f"{(st.res.get('zones') or {}).get('k') or k} NDVI zones (1 = lowest)")
        for ax in axs:
            _outline(ax, st)
            ax.set_xticks([]); ax.set_yticks([])
        F["ndvi_summary"] = os.path.join(d, "fig_ndvi_zones.png")
        fig.savefig(F["ndvi_summary"])
        plt.close(fig)
    else:
        # zones from a zone map (or without NDVI): a season figure of an earlier build belongs to other zones and
        # must not be shown in step 2 or go into the PDF
        F.pop("ndvi_seasons", None)
        k = int(st.zone_r.max())
        cm, nm = _zone_cmap(max(k, 1))
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(np.where(st.zone_r > 0, st.zone_r, np.nan), extent=ext, cmap=cm, norm=nm)
        _outline(ax, st)
        ax.set_title("Zones")
        F["ndvi_summary"] = os.path.join(d, "fig_ndvi_zones.png")
        fig.savefig(F["ndvi_summary"])
        plt.close(fig)



def _fig_units(st, d, F):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    u = st.units
    sc = axs[0].scatter(u.x_m, u.y_m, c=u["yield"], s=12 + 30 * u.area_ac / u.area_ac.max(), cmap="RdYlGn",
                        vmin=u["yield"].quantile(0.02), vmax=u["yield"].quantile(0.98))
    plt.colorbar(sc, ax=axs[0], shrink=0.7, label="yield (bu/ac)")
    axs[0].set_title("Analysis units: yield")
    name0 = st.inputs[0]
    sc = axs[1].scatter(u.x_m, u.y_m, c=u[f"rate_{name0}"], s=14, cmap="viridis")
    plt.colorbar(sc, ax=axs[1], shrink=0.7, label=f"applied {name0} rate")
    axs[1].set_title("Analysis units: applied rate")
    for ax in axs:
        _outline(ax, st)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    F["units"] = os.path.join(d, "fig_units.png")
    fig.savefig(F["units"])
    plt.close(fig)


def _fig_analysis(st, d, F, g, ext):
    for name, R in st.res.get("analysis", {}).items():
        # an input analyse() skipped (fewer than 2 design rates) has no model, so no figure; the figures of an
        # earlier run of it (when it still had a model) are dropped so the app and the PDF do not show them
        if name.startswith("_"):
            continue
        if R.get("skipped"):
            for pre in ("response_", "economics_", "prices_", "ndvi_rx_", "rx_"):
                F.pop(pre + name, None)
            continue
        dec = R["decision"]
        grid_r = np.array(dec["grid"])
        fits = getattr(st, "_fits", {}).get(name, {})
        zf = fits.get("zone") or fits.get("pooled")
        dd = zf.data
        zs = sorted(dec["r_zone"])
        cm, _ = _zone_cmap(max(len(zs), 1))
        zcol = {z: cm(i) for i, z in enumerate(zs)}
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))
        for z in zs:
            m = dd.zone == z if z is not None else np.ones(len(dd), bool)
            axs[0].scatter(dd.r[m], dd["yield"][m], s=10, alpha=0.45, color=zcol[z])
            y = np.array(dec["curves"][str(z)])
            axs[0].plot(grid_r, y, color=zcol[z], lw=2, label=f"zone {z}" if z is not None else "field")
            nr = R["price_crop"] * y - R["price_input"] * grid_r
            axs[1].plot(grid_r, nr - nr.max(), color=zcol[z], lw=2)
            axs[1].axvline(dec["r_zone"][z], color=zcol[z], ls=":", lw=1)
        P = R["pooled"]
        yp = np.array(P["curves"]["None"])
        axs[0].plot(grid_r if len(grid_r) == len(yp) else np.array(P["grid"]), yp, color=NAVY, lw=2.2, ls="--", label="field (one curve)")
        if R.get("status_quo") is not None:
            for ax in axs:
                ax.axvline(R["status_quo"], color=MUTED, lw=1, ls="-.")
        axs[0].set_xlabel(f"{name} rate"); axs[0].set_ylabel("yield (bu/ac)")
        axs[0].set_title("Yield response by zone"); axs[0].legend(fontsize=8)
        axs[1].set_xlabel(f"{name} rate"); axs[1].set_ylabel("net return minus its maximum ($/ac)")
        axs[1].set_title("Partial net return by zone (dotted = zone optimum)")
        axs[1].set_ylim(bottom=max(-60, axs[1].get_ylim()[0]))
        F[f"response_{name}"] = os.path.join(d, f"fig_response_{name}.png")
        fig.savefig(F[f"response_{name}"])
        plt.close(fig)

        # economics bars
        # in-sample bars on the summary basis (main model, field zone acres); cross-fitted bars with the 90 % range
        # of the per-repeat averages (lo..hi) as asymmetric error bars
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        labels, vals, errs, cols = [], [], [], []
        bf = _basis(R).get("best_flat") or {}
        if bf.get("gain_vs_sq") is not None:
            labels.append("best flat\nvs status quo\n(in-sample)"); vals.append(bf["gain_vs_sq"]); errs.append((0.0, 0.0)); cols.append(LIGHT_BLUE)
        cff = _cf_flat(R)
        if cff:
            labels.append("best flat\nvs SQ\n(cross-fitted)" if _cf_flat_same_basis(R) else "best flat vs SQ\n(cross-fitted,\none curve)")
            vals.append(cff["mean"]); errs.append(_cf_err(cff)); cols.append(LIGHT_BLUE)
        # with one analysis zone there is no VR strategy (its "VR" rate is the flat optimum), so no VR bar is drawn
        if _is_zone(R):
            labels.append("VR vs flat\noptimum\n(in-sample)"); vals.append(dec["gain_vr_vs_uniform"]); errs.append((0.0, 0.0)); cols.append(ORANGE)
            cf = (R.get("crossfit") or {}).get("gain_vr_vs_uniform")
            if cf:
                labels.append("VR vs flat\noptimum\n(cross-fitted)"); vals.append(cf["mean"]); errs.append(_cf_err(cf)); cols.append(NAVY)
            cfs = (R.get("crossfit") or {}).get("gain_vr_vs_sq")
            if cfs:
                labels.append("VR vs\nstatus quo\n(cross-fitted)"); vals.append(cfs["mean"]); errs.append(_cf_err(cfs)); cols.append(GREEN)
        if vals:
            x = np.arange(len(vals))
            ax.bar(x, vals, color=cols, yerr=np.array(errs, float).T, capsize=4)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("$/ac of field")
            ax.set_title(f"Value of variable rate ({name})" if _is_zone(R) else f"Best flat rate vs status quo ({name})")
            # the value above a positive bar (below a negative one), past the end of its error bar and centred,
            # so it never sits on the bar edge or the error-bar cap
            e_ = np.array(errs, float)
            span = max(1e-9, float(np.nanmax(np.abs(np.array(vals, float)) + e_.max(axis=1))))
            pad = 0.04 * span
            for xi, v, (elo, ehi) in zip(x, vals, e_):
                yv = v + ehi + pad if v >= 0 else v - elo - pad
                ax.annotate(f"{v:+.2f}", (xi, yv), ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
                            color="k", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            # room for the labels: above the highest error bar and below the lowest one
            tops = [v + ehi + pad for v, (elo, ehi) in zip(vals, e_)]
            bots = [v - elo - pad for v, (elo, ehi) in zip(vals, e_)]
            lo_, hi_ = ax.get_ylim()
            neg = [b for b, v in zip(bots, vals) if v < 0]
            ax.set_ylim(min(lo_, (min(neg) - 0.12 * span) if neg else lo_), max(hi_, max(tops) + 0.12 * span))
            F[f"economics_{name}"] = os.path.join(d, f"fig_economics_{name}.png")
            fig.savefig(F[f"economics_{name}"])
        else:
            # one analysis zone and no status quo: there is no gain to draw (an old figure must not be shown)
            F.pop(f"economics_{name}", None)
        plt.close(fig)

        # price grid
        pg = pd.DataFrame(R["price_grid"])
        if len(pg):
            # a zero-width price range (or an input price of 0) repeats one price on that axis; pivot refused the
            # duplicates ('Index contains duplicate entries') after the whole analysis had run. One value is one row
            # (or column) of the grid; results from before project.analyse dropped the duplicates are cleaned here.
            pg = pg.drop_duplicates(["input_price", "crop_price"])
            # one analysis zone: there is no VR, so only the flat-rate panel is drawn
            if _is_zone(R):
                fig, axs = plt.subplots(1, 2, figsize=(11, 4))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
                axs = [None, ax1]
            pv = pg.pivot(index="input_price", columns="crop_price", values="gain_vr_vs_uniform")
            if axs[0] is not None:
                vmax = max(float(np.nanmax(np.abs(pv.values))), 1e-9)
                im = axs[0].imshow(pv.values, origin="lower", aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
                plt.colorbar(im, ax=axs[0], label="$/ac")
            # the price grid runs on the main model, so its flat rate is that model's uniform optimum (the zone
            # model's when there are zones), not the single-curve rate the verdict recommends
            flat_lab = "Flat optimum of the zone model" if _is_zone(R) else "Best flat rate (one curve)"
            if axs[0] is not None:
                axs[0].set_title("In-sample VR gain over the flat optimum")
            pv2 = pg.pivot(index="input_price", columns="crop_price", values="r_uniform")
            im = axs[1].imshow(pv2.values, origin="lower", aspect="auto", cmap="viridis")
            plt.colorbar(im, ax=axs[1], label=f"{name} rate")
            axs[1].set_title(flat_lab)
            for ax, p_ in [(a_, q_) for a_, q_ in zip(axs, (pv, pv2)) if a_ is not None]:
                ax.set_xticks(range(len(p_.columns))); ax.set_xticklabels([f"{v:.2f}" for v in p_.columns], rotation=45, fontsize=7)
                ax.set_yticks(range(len(p_.index))); ax.set_yticklabels([f"{v:.3f}" for v in p_.index], fontsize=7)
                ax.set_xlabel("crop price ($/bu)"); ax.set_ylabel(f"{name} price ($/unit)")
            F[f"prices_{name}"] = os.path.join(d, f"fig_prices_{name}.png")
            fig.savefig(F[f"prices_{name}"])
            plt.close(fig)

        # continuous NDVI prescription
        nx = R.get("ndvi_rx")
        if nx and nx.get("bins"):
            _fig_ndvi_rx(name, nx, d, F)
        else:
            # no NDVI model in this run (zones from a map, or the NDVI model not fitted): an NDVI prescription figure
            # of an earlier run describes a model that no longer exists
            F.pop(f"ndvi_rx_{name}", None)

        # prescription map
        rz = dec["r_zone"]
        zmap = R.get("zone_map") or {}
        rx = np.full(st.zone_r.shape, np.nan)
        for z0 in np.unique(st.zone_r[st.zone_r > 0]):
            za = zmap.get(int(z0), int(z0))
            rx[st.zone_r == z0] = rz.get(za, rz.get(None, dec["r_uniform"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.where(g.mask, rx, np.nan), extent=ext, cmap="viridis")
        plt.colorbar(im, ax=ax, shrink=0.7, label=f"{name} rate")
        _outline(ax, st, lw=0.2)
        flat_rec = (R.get("verdict") or {}).get("flat_rate", dec["r_uniform"])
        ax.set_title(f"Zone prescription ({name}); recommended flat = {flat_rec:.0f}")
        ax.set_xticks([]); ax.set_yticks([])
        F[f"rx_{name}"] = os.path.join(d, f"fig_rx_{name}.png")
        fig.savefig(F[f"rx_{name}"])
        plt.close(fig)


def _fig_ndvi_rx(name, nx, d, F):
    """The continuous NDVI prescription figure of one input (nx = R['ndvi_rx']): r*(NDVI) over the equal-area NDVI
    bins, the units' NDVI range shaded, out-of-range bins marked x and hatched, corner bins as hollow markers."""
    b = pd.DataFrame(nx["bins"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.step(b.ndvi, b.r_opt, where="mid", color=NAVY, lw=2, label="r*(NDVI)")
    ax.axhline(nx["r_uniform"], color=ORANGE, ls="--", label=f"flat optimum of this model {nx['r_uniform']:.0f}")
    # bins whose mean NDVI lies outside the units' NDVI range rest on the rate x NDVI interaction extrapolated
    # beyond the trial (economics.ndvi_prescription flags them in_range False): they are marked, and the
    # units' range is shaded, so the reader sees which part of r*(NDVI) the trial actually covered
    inr = b["in_range"].astype(bool) if "in_range" in b.columns else pd.Series(True, index=b.index)
    ur = nx.get("ndvi_unit_range")
    if ur:
        ax.axvspan(float(ur[0]), float(ur[1]), color=GREEN, alpha=0.08, label="NDVI range of the trial units")
    # bins whose optimum sits at an end of the trial range are corners, not measured optima (economics.ndvi_prescription:
    # res['corner']); nothing showed them, and a prescription made of corners looked like one
    cor = pd.Series(_corner_bins(nx)[0], index=b.index)
    if cor.any():
        ax.scatter(b.ndvi[cor], b.r_opt[cor], facecolors="none", edgecolors=NAVY, s=40, zorder=5,
                   label="optimum at an end of the trial range (corner)")
    if (~inr).any():
        ax.scatter(b.ndvi[~inr], b.r_opt[~inr], marker="x", color=ORANGE, s=40, zorder=6,
                   label="outside the units' NDVI range (extrapolated)")
    ax2 = ax.twinx()
    xlab = "multi-year relative NDVI"
    if {"lo", "hi"} <= set(b.columns):
        wid = (b.hi - b.lo).clip(lower=1e-4)
        ax2.bar(((b.lo + b.hi) / 2)[inr], b.weight[inr], width=wid[inr], alpha=0.18, color=LIGHT_BLUE)
        if (~inr).any():
            ax2.bar(((b.lo + b.hi) / 2)[~inr], b.weight[~inr], width=wid[~inr], alpha=0.30, color=ORANGE,
                    hatch="//", edgecolor=ORANGE, linewidth=0)
        # the quantile bins start at the field minimum and end at its maximum (a slough, a yard), so the two end
        # bars covered most of the axis: it is clipped to the inner bins, and the label says where the ends reach.
        # The end bins' own positions (b.ndvi, the area-weighted mean of their pixels) stay on the axis, so their
        # step and markers (the corners and x the text points to) are drawn; only the bars are cut
        if len(b) > 2:
            w_ = float((b.hi - b.lo).iloc[1:-1].median())
            if np.isfinite(w_) and w_ > 0:
                ax.set_xlim(min(float(b.lo.iloc[1]) - 2 * w_, float(b.ndvi.min()) - w_),
                            max(float(b.hi.iloc[-2]) + 2 * w_, float(b.ndvi.max()) + w_))
                xlab += (f" (axis clipped: the end bins reach the field extremes {float(b.lo.min()):.2f} and "
                         f"{float(b.hi.max()):.2f})")
    ax2.set_yticks([])
    ax.set_xlabel(xlab); ax.set_ylabel(f"optimum {name} rate")
    # the bins are equal-area quantiles, so the bar heights carry no information: the title says so
    ax.set_title(f"Optimum rate along the NDVI (continuous model); bars = the {len(b)} NDVI bins, each the same "
                 "share of the field")
    ax.legend(fontsize=8)
    F[f"ndvi_rx_{name}"] = os.path.join(d, f"fig_ndvi_rx_{name}.png")
    fig.savefig(F[f"ndvi_rx_{name}"])
    plt.close(fig)


def fig_design(st):
    """Plots coloured by design rate (or treatment), border and boundary: the check after step 1. One panel per
    input that has design rates on the plots (design_rate = the first input, design_<name> = the others), so a
    seed x urea trial shows both maps; plots without a design rate are hatched."""
    p = st.plots
    inputs = list(getattr(st, "inputs", []) or [])
    panels = []
    for k_, name in enumerate(inputs):
        col = "design_rate" if k_ == 0 else f"design_{name}"
        if col in p and pd.to_numeric(p[col], errors="coerce").notna().any():
            panels.append((name, col))
    if not panels:
        panels = [(None, None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, (name, col) in zip(axes[0], panels):
        if col is not None:
            v = pd.to_numeric(p[col], errors="coerce")
            p[v.notna()].plot(ax=ax, column=v[v.notna()], cmap="viridis", legend=True, edgecolor="k", lw=0.2,
                              legend_kwds={"shrink": 0.7, "label": f"{name} design rate"})
            if v.isna().any():
                p[v.isna()].plot(ax=ax, facecolor="none", edgecolor=ORANGE, hatch="///", lw=0.4)
        elif p.treatment.notna().any():
            p.plot(ax=ax, column="treatment", categorical=True, legend=True, edgecolor="k", lw=0.2)
        else:
            p.plot(ax=ax, facecolor="none", edgecolor="k", lw=0.2)
        if len(st.border):
            st.border.boundary.plot(ax=ax, color=ORANGE, lw=1)
        st.boundary.boundary.plot(ax=ax, color=NAVY, lw=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title((f"{name}: " if name else "") + f"{len(p)} plots (orange = border, navy = field"
                     + ("; hatched = no design rate" if col is not None and pd.to_numeric(p[col], errors="coerce").isna().any() else "")
                     + ")", fontsize=9)
    fig.tight_layout()
    return fig


# ==================================================================================================== PDF
def make_pdf(st, path=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

    if not getattr(st, "figs", None):
        figures(st)
    cfg = st.cfg
    # the same cleaning as the app's work folder: '/', '\\', ':' etc. in a project name gave a nested or invalid path,
    # or (':') an NTFS alternate data stream instead of a PDF
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', cfg.get("name") or "field")
    path = path or os.path.join(st.out, f"{name}_VR_economics_report.pdf")
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=colors.HexColor(NAVY), fontSize=16, spaceAfter=6)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor(NAVY), fontSize=12.5, spaceBefore=8)
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.2)
    S = ParagraphStyle("s", parent=B, fontSize=8, textColor=colors.HexColor(MUTED), leading=10)
    W = 7.0 * inch

    # table cell text: the table font (Helvetica 8) as a Paragraph, so a long cell wraps inside its column
    CELL = ParagraphStyle("cell", parent=ss["BodyText"], fontName="Helvetica", fontSize=8, leading=9.6)
    HEAD = ParagraphStyle("head", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)

    def _cell(v, style=CELL):
        """A text cell as a wrapping Paragraph (reportlab draws a plain string on one line, so a long one, e.g. the
        zone rates 'z1: 169, z2: 101, ...' of five zones, ran over the next column)."""
        if isinstance(v, str) and v:
            return Paragraph(_esc(v).replace(chr(10), "<br/>"), style)
        return v

    def tbl(rows, widths=None, header=True):
        # Cells wrap only when the column widths are set (every table of the report sets them): without widths
        # reportlab sizes each column to its longest plain string, which a Paragraph (no natural width) would break,
        # and a wide table then ran past the right margin.
        if widths:
            rows = [[_cell(v, HEAD if (header and i == 0) else CELL) for v in r] for i, r in enumerate(rows)]
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        sty = [("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D7E3F2")),
               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FC")])]
        if header:
            sty += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8)]
        t.setStyle(TableStyle(sty))
        return t

    def img(p, w=W, maxh=4.2 * inch):
        if not p or not os.path.exists(p):
            return Spacer(1, 1)
        from PIL import Image as PI
        with PI.open(p) as im:
            iw, ih = im.size
        h = w * ih / iw
        if h > maxh:
            w, h = maxh * iw / ih, maxh
        return Image(p, width=w, height=h)

    f = lambda v, d=1: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}"

    def comp_txt(tms, analysed_ac, how=""):
        """The grower-compensation sentence of the realised trial from trial_minus_sq (tms, $/ac on the analysed
        units); `how` names the counterfactual (which inputs are at their status quo). The trial can out-earn the
        status quo (e.g. a status quo above the optimum): then there is no cost and no compensation to quote, rather
        than a negative one. A straight line with design rates balanced around the status quo gives tms of about 0
        by construction: 'earned 0.00 $/ac more ... (the status quo is not the best rate here)' then contradicted a
        verdict that keeps the status quo. Within 0.5 $/ac the trial cost nothing; the claim about the best rate is
        gone (the verdict says which rate is best, with its evidence). The plot-acre total is an extrapolation from
        the analysed acres (the units inside the trimmed plots), and says so."""
        D_ = st.res.get("design", {})
        if abs(tms) < 0.5:
            return (f"The trial rates earned about the same as the status quo on these units{how} ({tms:+.2f} $/ac): "
                    "the trial cost nothing: no ex-post cost to compensate the grower for.")
        if tms < 0:
            return (f"The trial rates cost {-tms:.2f} $/ac against the status quo on these units{how} "
                    f"(x {f(D_.get('plot_area_ac'))} plot acres (extrapolated from the {f(analysed_ac)} analysed acres) "
                    f"= {-tms * (D_.get('plot_area_ac') or 0):,.0f} {P.currency}): the ex-post cost of the trial to the "
                    "grower, the quantity on-farm experimentation programmes compensate (DIFM pays a fixed amount per acre; "
                    "Puntel, Thompson & Mieno 2024), computed here from the realised units.")
        return (f"The trial rates earned {tms:.2f} $/ac more than the status quo on these units{how}, so the yield "
                "response left no ex-post cost to compensate the grower for.")
    story = []
    logo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "brand", "oc_logo_horizontal.png")
    if os.path.exists(logo):
        story += [Image(logo, width=2.0 * inch, height=0.5 * inch), Spacer(1, 6)]
    story.append(Paragraph(f"Variable rate vs flat rate: {cfg.get('name') or ''}", H1))
    P = st.prices
    story.append(Paragraph(f"{cfg.get('producer') or ''} {cfg.get('field') or ''} &middot; {P.crop} {cfg.get('year')} &middot; "
                           f"report {dt.date.today().isoformat()} &middot; OCVR (Olds College Smart Farm, OFPE)", S))
    D, J, Z, N = st.res.get("design", {}), st.res.get("join", {}), st.res.get("zones", {}), st.res.get("ndvi", {})
    A_all = {k: v for k, v in st.res.get("analysis", {}).items() if not k.startswith("_")}
    # inputs analyse() skipped (fewer than 2 design rates) get one line saying why, and no economics section
    A = {k: v for k, v in A_all.items() if not v.get("skipped")}
    skipped = {k: v["skipped"] for k, v in A_all.items() if v.get("skipped")}

    # ------------------------------------------------------------ 1 summary
    story.append(Paragraph("1. Summary", H2))
    for name, why in skipped.items():
        why_ = why.split(": ", 1)[1] if why.lower().startswith(f"{name.lower()}: ") else why
        # the reason may carry an exception message (project.analyse skips an input whose fits raised), which can
        # hold '<' or '&': escaped for the Paragraph markup
        story.append(Paragraph(f"<b>{name.capitalize()}: not analysed.</b> {_esc(why_[:1].upper() + why_[1:])}.", B))
    for name, R in A.items():
        v = R["verdict"]
        dec, pooled = R["decision"], R["pooled"]
        # escaped: a cross-fit reason can quote the first fit error (project.cf_missing_txt)
        story.append(Paragraph(f"<b>{name.capitalize()}: {_esc(v['text'])}</b>", B))
        story.append(Paragraph(_esc("; ".join(v["reasons"])) + ".", B))
        # ONE basis for every net return in the table (project.summary_basis): the main model on the field's zone
        # acres. The single-curve model only CHOOSES the best flat rate; its own levels (averaged over the trial
        # plots' zone mix) are not printed next to the zone model's, so the rows can be subtracted.
        Bs = _basis(R)
        sq = R.get("status_quo")
        gsq = lambda row: f((row or {}).get("gain_vs_sq"), 2) if sq is not None else "-"
        rows = [["", "rate", "net return $/ac", "vs status quo $/ac"]]
        if sq is not None and Bs.get("status_quo"):
            sqr = Bs["status_quo"]
            # a status quo too far outside the trial range is not valued (economics: sq_too_far)
            rows.append(["Status quo (flat)", f(sq, 0), f(sqr.get("nr"), 2),
                         "not valued" if sqr.get("too_far") or sqr.get("nr") is None else "0.00"])
        bf = Bs["best_flat"]
        rows.append(["Best flat (one-curve optimum)", f(bf["rate"], 0), f(bf["nr"], 2), gsq(bf)])
        cff = _cf_flat(R)
        if cff:
            rows.append([_cf_flat_label(R, short=True), "", "", _cf_txt(cff, short=True)])
        rf = Bs.get("recommended_flat")
        if rf:
            rows.append(["Recommended flat (verdict)", f(rf["rate"], 0), f(rf["nr"], 2), gsq(rf)])
        # with one analysis zone there is no VR strategy (its "VR" rate is the flat optimum, not charged the VR
        # cost: project.analyse), so the zone VR rows would only repeat the best flat row
        if _is_zone(R):
            # the comparator of the verdict's VR gain (the zone model's own best flat rate): without this row,
            # 'Zone VR' minus 'Best flat' did not match the in-sample VR gain of the reasons line
            mf = Bs.get("model_flat")
            if mf:
                rows.append(["Flat optimum of the zone model", f(mf.get("rate"), 0), f(mf.get("nr"), 2), gsq(mf)])
            vr = Bs["vr"]
            rows.append(["Zone VR (in-sample)", _rates_txt(dec["r_zone"]), f(vr["nr"], 2),
                         gsq(vr) if sq is not None else "-"])
            cf = R.get("crossfit") or {}
            if cf.get("gain_vr_vs_sq"):
                rows.append(["Zone VR (cross-fitted)", "", "", _cf_txt(cf["gain_vr_vs_sq"], short=True)])
        story += [Spacer(1, 4), tbl(rows, [1.9 * inch, 1.5 * inch, 1.1 * inch, 2.5 * inch]), Spacer(1, 4)]
        wtxt = ", ".join(f"z{z} {100 * w:.0f} %" for z, w in dec["weights"].items() if z not in (None, "None"))
        basis_txt = (f"the curve-per-zone model on the field's zone acres ({wtxt})" if _is_zone(R)
                     else "the single-curve model (one analysis zone)")
        # with one analysis zone the zone strategy is not charged the VR cost, but a continuous NDVI prescription
        # (section 4) is a variable rate and is: the sentence must not say that no VR cost is charged at all
        vr_txt = ("Zone VR is net of the VR cost; the VR gains of the verdict are measured against the flat optimum "
                  "of the zone model (its own best flat rate). " if _is_zone(R) else
                  "With one analysis zone there is no zone VR strategy to compare (and it is charged no VR cost"
                  + ("; the continuous NDVI prescription of the response section is" if R.get("ndvi_rx")
                     and P.vr_cost_ac else "") + "). ")
        if not cff:
            cf_flat_txt = ""
        elif _cf_flat_same_basis(R):
            cf_flat_txt = ("The cross-fitted best flat row is on the same basis as the in-sample rows: the flat rate is "
                           "chosen by the single curve on half of the plots, and it and the status quo are valued on "
                           "the other half with " + ("the curve-per-zone model over the field's zone acres. "
                                                     if _is_zone(R) else "the single curve. "))
        else:
            cf_flat_txt = ("The cross-fitted best flat row of these results was computed on another basis (the single "
                           "curve judged on held-out plots at their own zone mix), so it is not the out-of-sample "
                           "version of the in-sample row above it; run the analysis again to cross-fit it on the same "
                           "basis. ")
        story.append(Paragraph(
            f"All net returns and in-sample gains in this table come from one model, {basis_txt}, so the difference "
            "between two rows is the gain between them. The best flat rate is the single-curve optimum (the rate the "
            "verdict judges); it is valued on the same model as the other rows. " + vr_txt + cf_flat_txt +
            "Cross-fitted rows: the mean over repeated 2-fold plot splits of the average of the two folds, the 90 % "
            "range (5th to 95th percentile) of those averages, and the share of repeats above 0.", S))
        if sq is not None and dec.get("sq_too_far"):
            story.append(Paragraph(f"The status quo {sq:g} lies more than 25 % of the trial range outside it "
                                   f"({f(dec['grid'][0], 0)} to {f(dec['grid'][-1], 0)}): the trial cannot value it, so "
                                   "no gain over the status quo is computed ('-' in the tables).", S))
        elif sq is not None and dec.get("sq_extrapolated"):
            # the rule of economics.bounded_curve for every form: extended with the curve's one-sided slope at the
            # nearest edge (a plateau too, below the range); a plateau reached inside the range stays flat above it
            story.append(Paragraph(f"The status quo {sq:g} lies outside the trial range ({f(dec['grid'][0], 0)} to "
                                   f"{f(dec['grid'][-1], 0)}): its yield is extended with the one-sided slope of the "
                                   "fitted curve at the nearest edge (a plateau reached inside the range stays flat "
                                   "above it; never below 0), so the gains over it are less certain; the optima stay "
                                   "inside the trial range.", S))
        story.append(Paragraph(f"Prices: crop {P.crop_price_bu:.2f} {P.currency}/bu; {name} {R['price_input']:.3f} {P.currency} per rate unit"
                               # the VR cost is charged to the zone VR strategy AND to the continuous NDVI
                               # prescription (project.analyse: ndvi_rx), which exists with one analysis zone too;
                               # hiding it there left the NDVI gain 'net of the VR cost' unreconcilable
                               + (f"; VR cost {P.vr_cost_ac:.2f} $/ac" if P.vr_cost_ac and (_is_zone(R) or R.get("ndvi_rx"))
                                  else "")
                               + ". Net return = crop price x yield - "
                               "input price x rate (partial budget: costs that do not change with the rate are left out).", S))

    # ------------------------------------------------------------ 2 data
    story.append(Paragraph("2. Trial and data", H2))
    rows = [["Item", "Value"],
            ["Plots", f"{D.get('n_plots')} plots ({f(D.get('plot_area_ac'))} ac), {D.get('n_border')} border polygon(s)"
                      + ("" if D.get("rates_by_input") else f"; rates {', '.join(f(r, 0) for r in D.get('rates', []))}")],
            ["Field area used for the economics", f"{f(D.get('boundary_ac'))} ac ({D.get('boundary_source')})"]]
    # the rates of every input and where they came from (a design column, a design file of its own, or the levels
    # read from its as-applied map): a seed x urea trial has two designs, and the reader needs both
    applied_src = (cfg.get("join") or {}).get("rate_source", "applied") == "applied"
    for nm_, rb in (D.get("rates_by_input") or {}).items():
        fa_ = rb.get("from_applied") or {}
        if str(rb.get("source", "")).startswith("none"):
            rows.append([f"Rates of {nm_}", Paragraph(_esc("no design rates (" + str(rb.get("source")).split(": ", 1)[-1] + ")"), S)])
            continue
        rates_ = [float(r) for r in rb.get("rates", [])]
        nd_ = 1 if any(abs(r - round(r)) > 0.05 for r in rates_) else 0     # the same decimals for every number
        txt = ((", ".join(f(r, nd_) for r in rates_) or "none") + f" ({rb.get('source')}); plots per rate: "
               + (", ".join(f"{f(float(k_), nd_)}: {v}" for k_, v in (rb.get("plots_per_rate") or {}).items()) or "-"))
        if fa_:
            n_no = int(fa_.get("n_few_points") or 0) + int(fa_.get("n_mixed") or 0) + int(fa_.get("n_off_level") or 0)
            txt += (f". Levels read from column '{fa_.get('column')}' of the as-applied map: {fa_.get('n_with_level')} of "
                    f"{fa_.get('n_plots')} plots at a level")
            if n_no:
                txt += (f"; {fa_.get('n_few_points')} with fewer than {fa_.get('min_pts')} points, {fa_.get('n_mixed')} "
                        "with mixed rates (two levels inside the plot), "
                        f"{fa_.get('n_off_level')} off every level: those plots have no design rate and "
                        + ("enter the response with their applied rate only"
                           if applied_src else "are left out of the response (rate in the models = design rate)"))
        elif rb.get("n_without"):
            txt += (f"; {rb['n_without']} plot(s) without a design rate"
                    + ("" if applied_src else " (left out of the response: rate in the models = design rate)"))
        if fa_.get("warning"):
            txt += ". Note: " + fa_["warning"]
        cv_ = (D.get("rate_conversion") or {}).get(nm_) or {}
        if cv_.get("note"):
            # a design in kg/ha, or an as-applied map in lb of product under a nutrient price: converted on loading,
            # so every rate in this report and in the prescription is lb/ac in the input's basis
            txt += f". Unit: {cv_['note']}; every rate in this report and in the prescription is lb/ac"
        rows.append([f"Rates of {nm_}", Paragraph(_esc(txt), S)])
    rows += [
            ["Yield file / column", f"{os.path.basename(str(cfg['files'].get('yield')))} / {J.get('yield_col')}"],
            ["Observations", f"{J.get('attrition', {}).get('n_obs', 0):,} read; {J.get('attrition', {}).get('kept', 0):,} inside one trimmed plot "
                             f"({_share_clause(J)}ends trimmed {cfg['join']['end_trim_m']} m)"],
            ["Analysis units", f"{J.get('units', {}).get('n_units')} plot x zone units in {J.get('units', {}).get('n_plots')} plots, "
                               + (f"{f(J.get('analysed_ac'))} ac" if J.get("area_known", True) and J.get("analysed_ac") is not None
                                  else _area_missing_txt(J))],
            ["Rates used", _rate_src_txt(J.get("rate_source"))]]
    U_ = J.get("units") or {}
    if U_.get("dropped_no_zone_obs") or U_.get("filled_nearest_zone_pieces"):
        # observations on cells outside every zone (edges of the zone raster) are either given the nearest zone or
        # dropped (join.build_units); the reader needs both counts to reconcile 'kept' with the units
        rows.append(["Outside the zone map", Paragraph(
            f"{int(U_.get('dropped_no_zone_obs') or 0):,} observation(s) dropped (no zone under them); "
            f"{int(U_.get('filled_nearest_zone_pieces') or 0):,} piece(s) given the nearest zone", S)])
    for n_, w_ in (J.get("applied_warnings") or {}).items():
        rows.append([f"As-applied {n_}", Paragraph(f"{w_} (the design rate was used for this input)", S)])
    # checks that the join can only hint at (the run goes on): the as-applied rates against the design rates (a map
    # in another basis or unit) and the swath width against the spacing of the yield passes (a ft / m mix-up)
    for n_, ck_ in (J.get("rate_checks") or {}).items():
        if ck_.get("warning"):
            rows.append([f"Check: rates of {n_}", Paragraph(_esc(ck_["warning"][0].upper() + ck_["warning"][1:] + "."), S)])
    sc_ = J.get("swath_check") or {}
    if sc_.get("warning"):
        rows.append(["Check: swath width", Paragraph(_esc(sc_["warning"]), S)])
    story += [tbl(rows, [2.0 * inch, 5.0 * inch]), Spacer(1, 4)]
    story.append(img(st.figs.get("units"), maxh=3.2 * inch))

    # ------------------------------------------------------------ 3 NDVI and zones
    story.append(PageBreak())
    story.append(Paragraph("3. Multi-year NDVI and zones", H2))
    # The paragraph follows where the NDVI and the zones really came from. It used to describe fetched Sentinel-2
    # seasons whatever the source, so a grower's zone map with no NDVI read '0 season(s) ... classified into 3 zones
    # ... explain 0 % of the NDVI pattern', a method that was not used and an R2 nobody measured.
    files_ = cfg.get("files") or {}
    has_ndvi = bool(N) and N.get("note") != "no NDVI"
    nsrc = N.get("source") or ((cfg.get("ndvi") or {}).get("source") if has_ndvi else "none")
    zsrc = str(Z.get("source") or "")
    base_ = lambda p: os.path.basename(str(p)) if p else "-"
    if not has_ndvi:
        ndvi_txt = "No NDVI was used in this run (no NDVI covariate and no NDVI model)."
    elif nsrc == "map":
        ndvi_txt = (f"The NDVI is a multi-year NDVI map supplied by the user ({base_(files_.get('ndvi_map'))}), used as "
                    f"the temporal mean as is; {N.get('note', '')}.")
    elif nsrc == "files" and files_.get("ndvi_upload_names"):
        # files uploaded in the app live in a folder named by their hash: naming that folder as 'the user's folder'
        # pointed the reader to a path they never chose
        up_ = [str(x) for x in (files_.get("ndvi_upload_names") or [])]
        ndvi_txt = (f"{N.get('used', 0)} NDVI image(s) from {len(up_)} file(s) uploaded in the app "
                    f"({', '.join(up_[:6])}{', ...' if len(up_) > 6 else ''}) were normalised to the field mean and "
                    f"averaged; {N.get('note', '')}.")
    elif nsrc == "files":
        ndvi_txt = (f"{N.get('used', 0)} NDVI image(s) from the user's folder ({base_(files_.get('ndvi_dir'))}) were "
                    f"normalised to the field mean and averaged; {N.get('note', '')}.")
    else:
        ndvi_txt = (f"{N.get('used', 0)} season(s) of peak NDVI (Sentinel-2 L2A, 10 m; each season = the per-pixel "
                    "median of its clearest dates near the peak) were normalised to the field mean and averaged; "
                    f"{N.get('note', '')}.")
    if zsrc == "zone map":
        zone_txt = (f"The {Z.get('k')} zones come from the grower's zone map ({base_(files_.get('zones'))}, column "
                    f"{Z.get('column') or '-'}), not from a classification of the NDVI.")
    elif zsrc.startswith("none"):
        zone_txt = "There is no zone map and no NDVI to make zones from, so the whole field is one zone."
    else:
        what = "The NDVI map" if nsrc == "map" else "The temporal mean"
        zone_txt = (f"{what} was classified into {Z.get('k')} zones ({Z.get('method', zsrc)}; patches under "
                    f"{Z.get('min_patch_ha', '-')} ha merged).")
    fit_txt = []
    if Z.get("ndvi_r2") is not None:
        fit_txt.append(f"The zones explain {f(100 * Z['ndvi_r2'], 0)} % of the NDVI pattern.")
    if J.get("yield_r2_by_zone") is not None:
        fit_txt.append(f"They explain {f(100 * J['yield_r2_by_zone'], 0)} % of the yield variation between analysis "
                       "units (mean yield by zone: "
                       f"{', '.join(f'z{k} {v:.1f}' for k, v in (J.get('yield_means_by_zone') or {}).items())} bu/ac).")
    # seasons missing from the zones because the last fetch failed on the network (project.fetch_ndvi records them
    # apart) or because they were selected after it: without this the season table simply lacked them, with no reason
    miss_txt = []
    if N.get("years_fetch_failed"):
        miss_txt.append(f"Season(s) {', '.join(map(str, N['years_fetch_failed']))} failed to download (network) at the "
                        "last fetch and are not in the zones: fetch again to add them.")
    if N.get("years_fetch_partial"):
        # a season that lost only some of its dates is built from the others (project.load_ndvi): it is not missing,
        # and the season table shows whether it was used
        used_y = {str(L.get("year")) for L in (N.get("layers") or []) if L.get("used")}
        part = [str(y) for y in N["years_fetch_partial"]]
        p_in = [y for y in part if y in used_y]
        p_out = [y for y in part if y not in used_y]
        if p_in:
            miss_txt.append(f"Season(s) {', '.join(p_in)} lost some of their dates on the network at the last fetch: "
                            "they are in the zones with fewer dates, fetched again next time.")
        if p_out:
            miss_txt.append(f"Season(s) {', '.join(p_out)} lost some of their dates on the network at the last fetch "
                            "and are fetched again next time; with the dates they have, they were not selected for the "
                            "zones (see the season table).")
    if N.get("years_not_fetched"):
        miss_txt.append(f"Season(s) {', '.join(map(str, N['years_not_fetched']))} were selected after the last fetch "
                        "and are not in the zones.")
    story.append(Paragraph(" ".join([ndvi_txt] + miss_txt + [zone_txt] + fit_txt), B))
    if N.get("layers"):
        rows = [["Season", "Date(s)", "Crop", "Field mean NDVI", "Valid %", "Used"]]
        for L in N["layers"]:
            # the date cell wraps (a Paragraph), so the list of dates is not cut
            rows.append([str(L.get("year") or "-"), str(L.get("date") or "-"), L.get("crop") or "?", f(L.get("field_mean"), 3),
                         f(L.get("valid_pct"), 0), "yes" if L.get("used") else "no"])
        story += [tbl(rows, [0.7 * inch, 2.6 * inch, 0.9 * inch, 1.1 * inch, 0.7 * inch, 0.6 * inch]), Spacer(1, 4)]
    labels = Z.get("labels") or {}
    zrows = [["Zone", "Label", "Field acres", "Mean relative NDVI"]]
    for k, a in sorted((Z.get("field_acres") or {}).items()):
        lab = labels.get(k, labels.get(str(k), "")) if isinstance(labels, dict) else ""
        zrows.append([str(k), str(lab or "-"), f(a), f((Z.get("ndvi_means") or {}).get(k), 3)])
    story += [tbl(zrows, [0.8 * inch, 1.6 * inch, 1.1 * inch, 1.5 * inch]), Spacer(1, 4)]
    # zones.py explains anything unusual of the zoning in info['note'] (zones removed by sieving, shifted codes,
    # no-data codes, the order of text labels); the code-shift sentence is the fallback for older results
    if Z.get("note"):
        story.append(Paragraph(str(Z["note"]), S))
    elif Z.get("code_shift"):
        story.append(Paragraph(f"The zone map used 0 as a zone code, so every code was shifted by +{Z['code_shift']} "
                               "(0 is the no-data value of the zone raster).", S))
    story.append(img(st.figs.get("ndvi_summary"), maxh=2.8 * inch))
    story.append(img(st.figs.get("ndvi_seasons"), maxh=3.0 * inch))

    # ------------------------------------------------------------ 4 response and economics per input
    sec = 4
    for name, R in A.items():
        story.append(PageBreak())
        story.append(Paragraph(f"{sec}. Response and economics: {name}", H2))
        # notes about the MODEL (straight-line fallback for the whole field, NDVI model not fitted) are not zone
        # support notes: they get their own label and no zone threshold
        for n_ in R.get("model_notes") or []:
            story.append(Paragraph("Model: " + _esc(n_) + ".", S))
        # the threshold printed is the one the merge used (project.analyse stores it in zone_rule; older results:
        # cfg model.min_units_zone and 3 rates)
        zr = R.get("zone_rule") or {}
        mu = int(zr.get("min_units") or (cfg.get("model") or {}).get("min_units_zone", 20))
        mr = int(zr.get("min_rates") or 3)
        for n_ in R.get("zone_notes") or []:
            story.append(Paragraph("Zone support: " + _esc(n_) + f" (a zone needs {mu} units and {mr} design rates for its own curve).", S))
        M = R["models"]
        rows = [["Model", "Form", "Units / plots", "R2", "Test", "p"]]
        borrowed = None
        for k, lab, tk in (("pooled", "one curve", "rate"), ("zone", "curve per zone", "zone_x_rate"), ("ndvi", "curve x NDVI", "ndvi_x_rate")):
            if k in M:
                t = M[k]["tests"]
                # a quadratic-plateau fit has no closed-form tests: its R2 and p are those of the quadratic model of
                # the same structure (response._fit_qplateau, tests_from); they are marked * and explained below
                mark = ""
                if t.get("tests_from"):
                    mark, borrowed = " *", t["tests_from"]
                form_ = M[k].get("form")
                rows.append([lab, FORM_LABEL.get(form_, form_ or "-"), f"{t.get('n_units')} / {t.get('n_plots')}",
                             f(t.get("r2"), 2) + mark, (t.get(tk) or {}).get("h0", "-"),
                             f((t.get(tk) or {}).get("p"), 3) + mark])
        story += [tbl(rows, [1.1 * inch, 0.95 * inch, 0.85 * inch, 0.5 * inch, 3.0 * inch, 0.6 * inch]), Spacer(1, 3)]
        if borrowed:
            story.append(Paragraph(f"* R2 and test taken from the {borrowed}: the quadratic-plateau curve itself has no "
                                   "closed-form test.", S))
        for k in ("pooled", "zone", "ndvi"):
            if (M.get(k) or {}).get("form_note"):
                story.append(Paragraph(f"Functional form ({k} model): {M[k]['form_note']}", S))
        story.append(Paragraph("Weighted least squares on analysis units (weight = area), standard errors clustered by plot (the unit of "
                               "randomisation); " + _controls_txt(R, st) + ".", S))
        story.append(img(st.figs.get(f"response_{name}"), maxh=3.0 * inch))
        T = pd.DataFrame(R["table"])
        if len(T):
            rows = [["Zone", "Design rate", "Applied", "Yield bu/ac", "SE", "Units", "Plots", "Net return $/ac"]]
            # the decimals of section 2: one when a design rate is not whole (levels read from an as-applied map)
            dr_ = pd.to_numeric(T.design_rate, errors="coerce").dropna()
            nd_dr = 1 if len(dr_) and bool((abs(dr_ - dr_.round()) > 0.05).any()) else 0
            for _, r in T.iterrows():
                rows.append([str(int(r.zone)), f(r.design_rate, nd_dr), f(r.applied, 1), f(r.yield_mean, 1), f(r.yield_se, 2),
                             str(int(r.n_units)), str(int(r.n_plots)), f(r.net_return, 2)])
            story += [KeepTogether([Paragraph("Observed means by zone and design rate (model-free)", B),
                                    tbl(rows, [0.55 * inch, 0.9 * inch, 0.8 * inch, 0.95 * inch, 0.6 * inch, 0.6 * inch,
                                               0.6 * inch, 1.2 * inch])]),
                      Spacer(1, 4)]
        dec = R["decision"]
        rows = [["Zone", "Field share", "Zone optimum", "Yield at optimum", "Gain vs flat optimum $/ac\n(before the VR cost)",
                 "Gain vs status quo $/ac\n(before the VR cost)"]]
        for pz in dec["per_zone"]:
            rows.append([_zone_txt(pz["zone"]), f"{100 * pz['weight']:.0f} %", f(pz["r_opt"], 0), f(pz["y_opt"], 1), f(pz["gain_vs_uniform"], 2),
                         f(pz.get("gain_vs_sq"), 2)])
        zone_main = _is_zone(R)
        mflat = "flat optimum of the zone model" if zone_main else "best flat rate (one curve)"
        story += [Paragraph(f"Optima within the trial range ({f(dec['grid'][0], 0)} to {f(dec['grid'][-1], 0)}); {mflat} "
                            f"{f(dec['r_uniform'], 0)} (the flat rate recommended by the verdict: "
                            f"{f((R.get('verdict') or {}).get('flat_rate'), 0)})", B),
                  tbl(rows, [0.9 * inch, 0.8 * inch, 0.95 * inch, 1.05 * inch, 1.65 * inch, 1.65 * inch]), Spacer(1, 4)]
        if zone_main and float(R.get("vr_cost") or 0.0):
            # decide().per_zone does not charge the VR cost, the VR gains elsewhere do: say so, so that the weighted
            # sum of this table and the VR gain of the summary can be reconciled
            story.append(Paragraph(f"Per-zone gains are before the VR cost of {float(R['vr_cost']):.2f} $/ac: their "
                                   "field-share-weighted sum minus that cost is the in-sample VR gain of the summary.", S))
        bs, cf = R.get("bootstrap") or {}, R.get("crossfit") or {}
        rows = [["Quantity", "Estimate", "90 % interval / share > 0"]]
        vr_rows = zone_main      # one analysis zone: no VR strategy, so no VR rows
        # the estimate is the decision value (the figure of the summary table, the reasons line and the economics
        # bars); the plot-bootstrap median goes with the interval. It used to be printed as the estimate, and for
        # the in-sample VR gains it is biased upward (each resample's gain is floored at 0 before the VR cost):
        # Trial 22 seed printed 4.53 here next to +2.15 everywhere else
        for k, lab, nd_ in (("r_uniform", mflat, 0),) + ((("gain_vr_vs_uniform", f"VR gain vs the {mflat} (in-sample)", 2),
                                                        ("gain_vr_vs_sq", "VR gain vs status quo (in-sample)", 2)) if vr_rows else ()):
            if isinstance(bs.get(k), dict) and dec.get(k) is not None:
                b_ = bs[k]
                pp = f"; {100 * float(b_['p_pos']):.0f} % > 0" if b_.get("p_pos") is not None else ""
                rows.append([lab, f(dec[k], nd_), f"{f(b_['lo'], nd_)} to {f(b_['hi'], nd_)} (bootstrap median "
                                                  f"{f(b_['median'], nd_)}{pp})"])
        # the best flat vs status quo row is the same figure as the summary row and the verdict (_cf_flat: the
        # single-curve rate valued on the summary basis; the zone model's own cross-fit judges a different flat rate)
        for k, lab, src in ((("gain_vr_vs_uniform", f"VR gain vs the {mflat} (cross-fitted)", cf),
                             ("gain_vr_vs_sq", "VR gain vs status quo (cross-fitted)", cf)) if vr_rows else ()) + (
                            ("gain_uniform_vs_sq", _cf_flat_label(R), R.get("crossfit_flat") or cf),):
            c = src.get(k)
            if c:
                est, rng = _cf_txt(c)
                sd = f" (sd {f(c.get('sd'), 2)})" if c.get("sd") is not None else ""
                rows.append([lab, est + sd, rng])
        story += [tbl(rows, [2.8 * inch, 1.5 * inch, 2.7 * inch]), Spacer(1, 3)]
        # replicates without the design rates for their curves (n_dropped) and replicates whose fit failed (n_failed)
        # are both left out of the intervals, so the number used is B - n_dropped - n_failed
        B_ = int(bs.get("B") or (cfg.get("model") or {}).get("B", 0) or 0)
        nd, nf = bs.get("n_dropped"), int(bs.get("n_failed") or 0)
        boot_txt = ""
        if bs and nd is not None and B_:
            n_used = B_ - int(nd) - nf
            # economics._supported drops a resample for any lost design support (response.check_support): a zone
            # missing, too few design rates, or too few design degrees of freedom inside the original zones (a merged
            # zone can hold 3 rates and still fail that rule), so the sentence names all three
            mr_ = _min_rates(_main_form(R))
            boot_txt = (f" {n_used} of {B_} bootstrap resamples were used: {int(nd)} dropped because a zone, or the "
                        f"field, lost the design support for its curve (a zone missing from the resample, fewer than "
                        f"{mr_} design rates, or fewer than {mr_ - 1} design degree(s) of freedom inside its original "
                        f"zones), and {nf} whose fit failed.")
        boot_txt += _cf_skips_txt(R)
        story.append(Paragraph("The bootstrap resamples plots; the cross-fit chooses the rates on half of the plots (stratified by design rate) and "
                               "judges them with the response fitted on the other half, repeated; each repeat is the average of its two folds, "
                               "the range is the 5th to 95th percentile of those averages. In-sample VR gains before the VR cost are never "
                               "negative by construction (the rows above are net of it, so they can be), so "
                               "the cross-fitted gain is the evidence (Mieno, Li &amp; Bullock 2024); for the same reason the "
                               "bootstrap median of the in-sample VR gains is biased upward (each resample's in-sample gain is "
                               "floored at zero before the VR cost), which is why the estimate column holds the decision value."
                               + boot_txt, S))
        story.append(img(st.figs.get(f"economics_{name}"), maxh=2.8 * inch))
        fm = R.get("forms") or {}
        fitted = [FORM_LABEL.get(k, k) for k, v in fm.items() if not v.get("error")]
        failed = [FORM_LABEL.get(k, k) for k, v in fm.items() if v.get("error")]
        if fitted:
            rows = [["Functional form", "Best flat rate", "Flat vs SQ $/ac", "Valued on", "Zone optima",
                     "VR vs zone-model flat optimum $/ac"]]
            for k, v in fm.items():
                if "error" in v and v.get("error"):
                    continue
                # project._form_sensitivity values each form's flat rate like the summary table: on the zone model
                # of the same form on the field's zone acres ('zone'), else on the single curve ('pooled')
                bas = {"zone": "zone model, field acres", "pooled": "one curve"}.get(v.get("basis"), "one curve (plots)")
                rows.append([FORM_LABEL.get(k, k), f(v.get("r_flat"), 0), f(v.get("gain_flat_vs_sq"), 2) if R.get("status_quo") is not None else "-",
                             bas, ", ".join(f"z{z}: {r:.0f}" for z, r in (v.get("r_zone") or {}).items()) or "-",
                             f(v.get("gain_vr_vs_flat"), 2)])
            # the heading names the forms in the table (a 2-rate trial fits only the straight line)
            head = (" vs ".join(fitted) if len(fitted) > 1 else
                    f"{fitted[0]} only" + (f": the {' and '.join(failed)} could not be fitted" if failed else ""))
            # heading and table on the same page (the heading was left alone at the foot of a page)
            story += [KeepTogether([Paragraph(f"Functional form sensitivity ({head})", B),
                                    tbl(rows, [1.15 * inch, 0.8 * inch, 0.85 * inch, 1.3 * inch, 1.9 * inch, 1.0 * inch])]),
                      Paragraph("Best flat rate: the single-curve optimum of that form; its gain over the status quo is "
                                "valued as in the summary table (on the curve-per-zone model of the same form on the "
                                "field's zone acres when there is one). The VR column is measured against the zone "
                                "model's own flat optimum, net of the VR cost.", S), Spacer(1, 4)]
        story.append(img(st.figs.get(f"prices_{name}"), maxh=2.8 * inch))
        if R.get("ndvi_rx"):
            nx = R["ndvi_rx"]
            # the share of the field whose optimum rests on the rate x NDVI interaction extrapolated beyond the NDVI
            # of the trial units (economics.ndvi_prescription); older results have only the per-bin flag
            sh = nx.get("share_out_of_range")
            if sh is None and nx.get("bins"):
                wt = sum(float(b_.get("weight") or 0) for b_ in nx["bins"]) or 1.0
                sh = sum(float(b_.get("weight") or 0) for b_ in nx["bins"] if not b_.get("in_range", True)) / wt
            ur = nx.get("ndvi_unit_range")
            rng_txt = f" ({ur[0]:.3f} to {ur[1]:.3f})" if ur else ""
            if sh:
                oor = (f" {100 * sh:.0f} % of the field lies in NDVI bins outside the units' NDVI range{rng_txt}: "
                       "their optima, and their part of this gain, rest on the rate x NDVI interaction extrapolated "
                       "beyond the trial (marked x and hatched in the figure).")
            elif sh is not None:
                oor = f" Every NDVI bin of the field lies inside the units' NDVI range{rng_txt}."
            else:
                oor = ""
            # bins whose optimum sits at an end of the trial range (economics.ndvi_prescription: corner): there the
            # model's optimum lies outside the rates tested or the fitted curve is not concave, so the rate is a
            # corner, not a measured optimum. Nothing said so, and Trial 22 seed (20 of 20 bins at a corner) printed
            # its gain as a prescription.
            cflags, shc = _corner_bins(nx)
            cor_txt = ""
            if shc > 0 and nx.get("bins"):
                lo_, hi_ = float(dec["grid"][0]), float(dec["grid"][-1])
                mid_ = (lo_ + hi_) / 2
                ends = {("low" if float(b_.get("r_opt")) <= mid_ else "high")
                        for b_, c_ in zip(nx["bins"], cflags) if c_ and b_.get("r_opt") is not None}
                cor_txt = (f" {100 * shc:.0f} % of the field has its optimum at an end of the trial range "
                           f"({f(lo_, 0)} or {f(hi_, 0)}): there the model's optimum lies outside the rates tested or "
                           "the fitted curve is not concave, so that rate is a corner, not a measured optimum (open "
                           "markers in the figure).")
                if shc > 0.25 or len(ends) > 1:
                    cor_txt += (" The NDVI-model gain is not a usable prescription; it is reported as a diagnostic "
                                "only.")
            story.append(Paragraph(f"Continuous NDVI model: a prescription that follows the multi-year NDVI pixel by pixel would gain "
                                   f"{nx['gain_vr_vs_uniform']:+.2f} $/ac over this model's flat optimum "
                                   f"({f(nx.get('r_uniform'), 0)}; in-sample, net of the VR cost of "
                                   f"{float(P.vr_cost_ac or 0.0):.2f} $/ac) "
                                   f"(NDVI x rate test p = {f((M.get('ndvi', {}).get('tests', {}).get('ndvi_x_rate') or {}).get('p'), 3)})."
                                   + cor_txt + oor, B))
            story.append(img(st.figs.get(f"ndvi_rx_{name}"), maxh=2.6 * inch))
        rs = R["realised"]["summary"]
        # a two-input project: each input's realised table moves THIS input alone to its status quo (the other
        # stays at its trial rate), so the two compensation sentences contradicted each other (seed 'earned +4.58
        # $/ac ... no compensation', urea 'cost 8.02 $/ac'). The compensation is quoted once, from the joint
        # counterfactual with every tested input at its status quo at once (project.analyse: '_realised_joint'),
        # after the last input's realised section; older results without it keep each input's own sentence, worded
        # as the one-input counterfactual it is.
        multi = len(A) > 1
        RJ = st.res.get("analysis", {}).get("_realised_joint") if multi else None
        last = name == list(A)[-1]
        # the net returns of each input's table charge that input only (the joint paragraph charges every input), so
        # the 'trial as applied' rows of the two tables and of the joint paragraph differ in level, and say why
        story.append(Paragraph("Realised trial economics (ex post, on the analysed units"
                               + ("; this input alone, the other input(s) held at their trial rates; net returns charge "
                                  "this input only)" if multi else ")"), B))
        rows = [["Strategy on the same units", "Mean rate", "Yield bu/ac", "Net return $/ac"],
                ["Trial as applied", f(rs.get("rate"), 0), f(rs.get("yield_obs"), 1) if rs.get("yield_obs") else "-", f(rs.get("nr_trial"), 2)]]
        if rs.get("nr_sq") is not None:
            rows.append(["Status quo flat", f(rs.get("rate_sq"), 0), f(rs.get("y_sq"), 1), f(rs.get("nr_sq"), 2)])
        if rs.get("nr_flat") is not None:
            rows.append(["Recommended flat (verdict)", f(rs.get("rate_flat"), 0), f(rs.get("y_flat"), 1), f(rs.get("nr_flat"), 2)])
        rows.append(["Flat optimum of the zone model" if zone_main else "Best flat (one curve)", f(rs.get("rate_uniform"), 0),
                     f(rs.get("y_uniform"), 1), f(rs.get("nr_uniform"), 2)])
        if zone_main:
            # one analysis zone: the "VR" rate is the flat optimum of the row above (no VR strategy, no VR cost)
            rows.append(["Zone VR (net of the VR cost)", f(rs.get("rate_vr"), 0), f(rs.get("y_vr"), 1),
                         f(rs.get("nr_vr"), 2)])
        story += [tbl(rows, [2.6 * inch, 1.1 * inch, 1.2 * inch, 1.4 * inch]), Spacer(1, 3)]
        tms = rs.get("trial_minus_sq")
        if tms is not None and RJ is None:
            how = (" with this input alone at its status quo (the other input(s) at their trial rates)" if multi else "")
            story.append(Paragraph(comp_txt(tms, rs.get("area_ac"), how)
                                   + " Counterfactuals keep each unit's residual and move it along its zone curve.", S))
        if RJ is not None and last:
            js = RJ.get("summary") or {}
            names_ = " and ".join(RJ.get("inputs") or list(A))
            jtms = js.get("trial_minus_sq")
            # an input without a status quo cannot be moved to it: it stays at its trial rate (RJ['left_out'])
            lo_ = [str(x) for x in (RJ.get("left_out") or [])]
            left_ = f"; {', '.join(lo_)} held at its trial rate (no status quo given)" if lo_ else ""
            head_ = (f"With every tested input ({names_}) at its status quo at once{left_}, on the {js.get('n_units')} "
                     f"common units ({f(js.get('area_ac'))} ac), the trial as applied returned {f(js.get('nr_trial'), 2)} "
                     f"$/ac net of {'both inputs' if len(RJ.get('inputs') or list(A)) == 2 else 'every input'} against "
                     f"{f(js.get('nr_sq'), 2)} $/ac at the status quo (yield {f(js.get('yield_obs'), 1)} vs "
                     f"{f(js.get('y_sq'), 1)} bu/ac).")
            story.append(Paragraph(head_ + (" " + comp_txt(jtms, js.get("area_ac"), " with every tested input at its "
                                                                                     "status quo at once")
                                            if jtms is not None else "")
                                   + " Counterfactuals keep each unit's residual and move it along its zone curve.", S))
        story.append(img(st.figs.get(f"rx_{name}"), maxh=3.2 * inch))
        sec += 1

    T = st.res.get("analysis", {}).get("_treatments")
    if T:
        story.append(PageBreak())
        story.append(Paragraph(f"{sec}. Treatment comparison", H2))
        rows = [["Treatment", "Plots", "Yield bu/ac", "Input cost $/ac", "Net return $/ac"]]
        # an unknown rate gives an unknown cost (treatments.compare: NaN, printed '-'), never a cost of 0
        fnum = lambda v, d=2: f(float(v), d) if v is not None else "-"
        for r in T["table"]:
            rows.append([str(r["treatment"]), str(r["n_plots"]), fnum(r["yield_mean"], 1), fnum(r.get("input_cost")),
                         fnum(r.get("net_return"))])
        story += [tbl(rows, [2.4 * inch, 0.7 * inch, 1.1 * inch, 1.3 * inch, 1.5 * inch]), Spacer(1, 3)]
        if T.get("cost_note"):
            story.append(Paragraph(str(T["cost_note"]) + " ('-' = unknown).", S))
        if T.get("left_out_note"):
            story.append(Paragraph(str(T["left_out_note"]) + ".", S))
        for var in ("yield", "net_return"):
            if T.get(var) and "p_treatment" in T[var]:
                how = T[var].get("pairwise_method") or "Tukey HSD"
                # the model fitted for THIS variable (treatments.compare drops the blocks when the plots with a known
                # value sit in one block); older results: read it from the formula, else the trial-level flag
                blk = T[var].get("blocked")
                if blk is None:
                    blk = ("C(block)" in str(T[var]["model"])) if T[var].get("model") else bool(T.get("blocks"))
                story.append(Paragraph(f"{var}: treatment p = {T[var]['p_treatment']:.3f} ({'RCBD' if blk else 'one-way'} "
                                       f"ANOVA on plot means; pairwise: {how}, in results.json)", B))
            elif T.get(var) and T[var].get("error"):
                story.append(Paragraph(f"{var}: not compared ({T[var]['error']}).", B))
        sec += 1

    story.append(PageBreak())
    story.append(Paragraph(f"{sec}. Method and references", H2))
    for t in methods(cfg, st.res):
        story.append(Paragraph(t, B))
    story.append(Spacer(1, 4))
    for r in REFS:
        story.append(Paragraph(r, S))
    doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.6 * inch,
                            bottomMargin=0.6 * inch, title=f"VR economics {cfg.get('name')}", author="OCVR, Olds College Smart Farm")

    def foot(c, d_):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor(MUTED))
        c.drawString(0.75 * inch, 0.4 * inch, f"OCVR - Olds College Smart Farm (OFPE) - {cfg.get('name')}")
        c.drawRightString(7.75 * inch, 0.4 * inch, f"page {d_.page}")
        c.restoreState()
    doc.build(story, onFirstPage=foot, onLaterPages=foot)
    st.report_path = path
    return path


def methods(cfg, R):
    """The Method paragraphs of the report with the zoning and the assignment rule of THIS run (R = st.res). A static
    text said 'tertiles', '95 %' and '10 m' whatever k, min_share and trim the run used (Trial 22 has 5 zones).
    The zones sentence follows R['zones'] (zones.py: source, k, method, min_patch_ha), the assignment sentence
    R['join'] (mode, the min_share the join stored) and cfg['join'] (the trims)."""
    Z, J, cj, cz = R.get("zones") or {}, R.get("join") or {}, cfg.get("join") or {}, cfg.get("zones") or {}
    zsrc = str(Z.get("source") or "")
    if zsrc == "zone map":
        zone_cl = ("lightly smoothed and classified into zones, with small patches merged (Taylor, McBratney &amp; Whelan "
                   "2007). Seasons of the same crop are preferred when the crop history is known (Georgi et al. 2018). "
                   "In this run the grower's zone map was used as the zones (section 3)")
    elif zsrc.startswith("none"):
        zone_cl = ("lightly smoothed and classified into zones, with small patches merged (Taylor, McBratney &amp; Whelan "
                   "2007). Seasons of the same crop are preferred when the crop history is known (Georgi et al. 2018). "
                   "In this run no zones were made (one zone)")
    else:
        k = Z.get("k") or cz.get("k")
        quant = str(Z.get("method") or cz.get("method") or "quantile") == "quantile"
        mp = Z.get("min_patch_ha", cz.get("min_patch_ha"))
        # zones.py stores the zones LEFT after sieving in k and the requested number in k_requested when a class
        # vanished: the label follows the requested classes, the count the zones left
        kr = Z.get("k_requested") or k
        how = "tertiles" if quant and kr == 3 else ("equal-area quantile classes" if quant else "k-means")
        if k != kr:
            how = f"{kr} {'quantile classes' if quant else 'k-means classes'}, {k} zones left after merging"
        zone_cl = (f"lightly smoothed and classified into {k} zones ({how}), with patches under {mp} ha merged (Taylor, "
                   "McBratney &amp; Whelan 2007). Seasons of the same crop are preferred when the crop history is known "
                   "(Georgi et al. 2018). When the grower supplies a zone map, its zones are used instead (section 3 says "
                   "which source this report used)")
    end = float(cj.get("end_trim_m", 10.0) or 0.0)
    side = float(cj.get("side_trim_m", 0.0) or 0.0)
    trim = f"trimming {end:g} m at each plot end" + (f" and {side:g} m at each side" if side > 0 else "")
    ms = (J.get("attrition") or {}).get("min_share")
    ms = float(cj.get("min_share", 0.95) if ms is None else ms)
    if J.get("mode") == "point":
        # assign_yield keeps whole points inside a plot (no swath: no footprint)
        asg = f"Each cleaned yield point is kept when it falls inside one plot after {trim}"
    elif ms > 0:
        asg = (f"Each cleaned yield record becomes a rectangle (swath x distance along the heading); records with at least "
               f"{100 * ms:.0f} % of their footprint inside one plot, after {trim}, are kept")
    else:
        asg = (f"Each cleaned yield record becomes a rectangle (swath x distance along the heading); records are clipped "
               f"to the trimmed plots ({trim}) and weighted by the clipped area")
    return [
        "<b>Zones.</b> Peak-season Sentinel-2 NDVI of past seasons (the trial year excluded), each season the median of its clearest "
        "dates near the peak (the window follows the Prairie NDVI-yield timing of Mkhabela et al. 2011), normalised to the field "
        "mean and averaged per pixel (the temporal-mean method of Blackmore 2000 applied to Sentinel-2 NDVI as in Lenoir et al. 2024; "
        f"Basso et al. 2019), {zone_cl}.",
        f"<b>Assignment.</b> {asg}; the plot is split by zone into analysis units "
        "(footprint overlay as in dos Santos &amp; Miguez 2025; the ~10 m buffers of DIFM, Bullock et al. 2019; yield-monitor delays of "
        "10-23 s, Chung, Sudduth &amp; Drummond 2002). Applied rates are the median of the as-applied points of the unit. "
        "An input with no column in any design file (the urea of a seed x urea trial logged only by the drill) can take its design rates "
        "from its as-applied map (section 2 says when it did): the median of the points inside each trimmed plot, grouped into the levels "
        "the drill delivered; a plot with mixed rates gets no design rate (it keeps its applied rate, which enters the response when the "
        "rate in the models is the applied one).",
    ] + METHOD


METHOD = [
    "<b>Response.</b> Response to the rate in the configured form (quadratic by default, or quadratic-plateau; a straight line when "
    "the field has only 2 design rates, see the Model notes), with the other functional forms (quadratic, quadratic-plateau, straight line) for sensitivity, per "
    "zone, along the continuous NDVI, and as one curve; weighted by area, with combine shares, a trend surface and the other tested inputs as controls "
    "when the data hold them (each input's section lists the controls its model used), with standard errors clustered by plot, the "
    "randomisation unit, and Wald tests judged against F(h, plots - 1) (Cameron &amp; Miller 2015; Abadie et al. 2023). The quadratic "
    "overestimates the optimum when the true response plateaus (Cerrato &amp; Blackmer 1990; Bullock &amp; Bullock 1994), which is why the "
    "plateau form is always reported beside it.",
    "<b>Economics.</b> Partial net return = crop price x yield - input price x rate; optima inside the trial range; value of VR = zone "
    "optima vs the best flat rate on the field's zone areas (Swinton &amp; Lowenberg-DeBoer 1998; Bullock, Lowenberg-DeBoer &amp; Swinton "
    "2002). The in-sample value of VR is biased upward by construction (Mieno, Li &amp; Bullock 2024), so the decision uses a plot "
    "cross-fit (sample splitting: Chernozhukov et al. 2018; Athey &amp; Imbens 2016), a plot bootstrap for the intervals (Hernandez &amp; "
    "Mulla 2008; Jaynes 2011), a minimum worthwhile gain because payoff functions are flat near the optimum (Pannell 2006; Pannell, "
    "Gandorfer &amp; Weersink 2019) and a price grid.",
]
REFS = [
    "Blackmore S. 2000. The interpretation of trends from multiple yield maps. Computers and Electronics in Agriculture 26(1):37–51. doi:10.1016/S0168-1699(99)00075-7",
    "Lenoir A et al. 2024. Advancing Blackmore's methodology to delineate management zones from Sentinel 2 images. Precision Agriculture 25:1360–1385. doi:10.1007/s11119-024-10115-2",
    "Basso B et al. 2019. Yield stability analysis reveals sources of large-scale nitrogen loss from the US Midwest. Scientific Reports 9:5774. doi:10.1038/s41598-019-42271-1",
    "Georgi C et al. 2018. Automatic delineation algorithm for site-specific management zones based on satellite remote sensing data. Precision Agriculture 19:684–707. doi:10.1007/s11119-017-9549-y",
    "Mkhabela MS et al. 2011. Crop yield forecasting on the Canadian Prairies using MODIS NDVI data. Agricultural and Forest Meteorology 151(3):385-393. doi:10.1016/j.agrformet.2010.11.012",
    "Taylor JA, McBratney AB, Whelan BM. 2007. Establishing management classes for broadacre agricultural production. Agronomy Journal 99(5):1366–1376. doi:10.2134/agronj2007.0070",
    "Bullock DS et al. 2019. The Data-Intensive Farm Management project: changing agronomic research through on-farm precision experimentation. Agronomy Journal 111(6):2736–2746. doi:10.2134/agronj2019.03.0165",
    "dos Santos CL, Miguez FE. 2025. A comparative study of yield monitor data processing methods for on-farm agronomic trials. Agronomy Journal 117(5):e70168. doi:10.1002/agj2.70168",
    "Chung SO, Sudduth KA, Drummond ST. 2002. Determining yield monitoring system delay time with geostatistical and data segmentation approaches. Transactions of the ASAE 45(4):915-926. doi:10.13031/2013.9938",
    "Trevisan RG, Bullock DS, Martin NF. 2021. Spatial variability of crop responses to agronomic inputs in on-farm precision experimentation. Precision Agriculture 22:342–363. doi:10.1007/s11119-020-09720-8",
    "Cerrato ME, Blackmer AM. 1990. Comparison of models for describing corn yield response to nitrogen fertilizer. Agronomy Journal 82(1):138–143. doi:10.2134/agronj1990.00021962008200010030x",
    "Bullock DG, Bullock DS. 1994. Quadratic and quadratic-plus-plateau models for predicting optimal nitrogen rate of corn: a comparison. Agronomy Journal 86(1):191–195. doi:10.2134/agronj1994.00021962008600010033x",
    "Miguez FE, Poffenbarger H. 2022. How can we estimate optimum fertilizer rates with accuracy and precision?. Agricultural & Environmental Letters 7(1):e20075. doi:10.1002/ael2.20075",
    "Cameron AC, Miller DL. 2015. A practitioner's guide to cluster-robust inference. Journal of Human Resources 50(2):317-372. doi:10.3368/jhr.50.2.317",
    "Abadie A et al. 2023. When should you adjust standard errors for clustering?. Quarterly Journal of Economics 138(1):1-35. doi:10.1093/qje/qjac038",
    "Anselin L, Bongiovanni R, Lowenberg-DeBoer J. 2004. A spatial econometric approach to the economics of site-specific nitrogen management in corn production. American Journal of Agricultural Economics 86(3):675-687 (errata 87(1):261, 2005,. doi:10.1111/j.0002-9092.2004.00610.x",
    "Scharf PC et al. 2006. Spatially variable corn yield is a weak predictor of optimal nitrogen rate. Soil Science Society of America Journal 70(6):2154-2160. doi:10.2136/sssaj2005.0244",
    "Swinton SM, Lowenberg-DeBoer J. 1998. Evaluating the profitability of site-specific farming. Journal of Production Agriculture 11(4):439-446. doi:10.2134/jpa1998.0439",
    "Bullock DS, Lowenberg-DeBoer J, Swinton SM. 2002. Adding value to spatially managed inputs by understanding site-specific yield response. Agricultural Economics 27(3):233-245. doi:10.1111/j.1574-0862.2002.tb00119.x",
    "Mieno T, Li X, Bullock DS. 2024. Bias in economic evaluation of variable rate application based on geographically weighted regression. Journal of the Agricultural and Applied Economics Association 3(1):135–151. doi:10.1002/jaa2.102",
    "Chernozhukov V et al. 2018. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal 21(1):C1-C68. doi:10.1111/ectj.12097",
    "Athey S, Imbens G. 2016. Recursive partitioning for heterogeneous causal effects. Proceedings of the National Academy of Sciences 113(27):7353-7360. doi:10.1073/pnas.1510489113",
    "Hernandez JA, Mulla DJ. 2008. Estimating uncertainty of economically optimum fertilizer rates. Agronomy Journal 100(5):1221-1229. doi:10.2134/agronj2007.0273",
    "Jaynes DB. 2011. Confidence bands for measured economically optimal nitrogen rates. Precision Agriculture 12(2):196-213. doi:10.1007/s11119-010-9168-3",
    "Pannell DJ. 2006. Flat earth economics: the far-reaching consequences of flat payoff functions in economic decision making. Review of Agricultural Economics 28(4):553–566. doi:10.1111/j.1467-9353.2006.00322.x",
    "Pannell DJ, Gandorfer M, Weersink A. 2019. How flat is flat? Measuring payoff functions and the implications for site-specific crop management. Computers and Electronics in Agriculture 162:459-465. doi:10.1016/j.compag.2019.04.011",
    "Rajsic P, Weersink A. 2008. Do farmers waste fertilizer? A comparison of ex post optimal nitrogen rates and ex ante recommendations by model, site and year. Agricultural Systems 97(1–2):56–67. doi:10.1016/j.agsy.2007.12.001",
    "Babcock BA. 1992. The effects of uncertainty on optimal nitrogen applications. Review of Agricultural Economics 14(2):271-280. doi:10.2307/1349506",
    "Puntel LA, Thompson LJ, Mieno T. 2024. Leveraging digital agriculture for on-farm testing of technologies. Frontiers in Agronomy 6:1234232. doi:10.3389/fagro.2024.1234232",
    "Tukey JW. 1949. Comparing individual means in the analysis of variance. Biometrics 5(2):99-114. doi:10.2307/3001913",
    "Khakbazan M et al. 2013. An economic study of controlled release urea and split applications of nitrogen as compared with non-coated urea under conventional and reduced tillage management. Canadian Journal of Plant Science 93(3):523-534. doi:10.4141/cjps2012-107",
    "Full list, with how each paper was read and checked: docs/literature/OCVR_method_and_references.pdf",
]
