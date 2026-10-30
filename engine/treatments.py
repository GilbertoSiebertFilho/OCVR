# -*- coding: utf-8 -*-
"""Strip-treatment comparison: when the design labels its plots by strategy (VR / Augmenta / Flat / None, or any
labels) instead of, or in addition to, a rate, the treatments are compared directly on yield and on partial net return,
plot by plot (the plot is the experimental unit), with a randomised-complete-block ANOVA when blocks exist and
Tukey pairwise comparisons between treatments (alpha 0.05). The cost of each plot is the applied rate x the input
price (plus the VR cost per acre for treatments marked as variable rate). A plot whose rate of a priced input is
unknown (no design rate and no as-applied value) has an UNKNOWN input cost and net return (NaN), not a cost of 0: a
strip trial run without its as-applied map used to charge every Flat plot nothing, so the net-return comparison left
the input out entirely and only repeated the yield comparison shifted by the VR cost. Such plots are left out of the
net-return means and of the net-return ANOVA, and the result says how many there were (cost_note).

Pairwise comparisons follow the ANOVA's model. Without blocks they are the classic one-way Tukey HSD. With blocks they
are Tukey-Kramer comparisons of the treatment effects of the RCBD model (treatment + block): each difference, its
standard error and the studentized-range p-value use that model's residual mean square and residual degrees of
freedom. A one-way Tukey on a blocked trial would keep the block-to-block variance in its error term, so on strip
trials with large block effects it found no difference at all while the RCBD ANOVA found a strong one."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import studentized_range
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA = 0.05


def _tukey_rcbd(m, levels, alpha=ALPHA):
    """Tukey-Kramer pairwise comparisons of the C(treatment) effects of a fitted OLS model `m` that also holds the
    block term. The reference level of C(treatment) has effect 0; every other level is its coefficient. Returns the
    same columns as statsmodels' pairwise_tukeyhsd summary (meandiff = group2 - group1, block-adjusted)."""
    names = {lev: f"C(treatment)[T.{lev}]" for lev in levels}
    params, cov = m.params, m.cov_params()
    k, df = len(levels), float(m.df_resid)
    q_crit = float(studentized_range.ppf(1 - alpha, k, df))
    rows = []
    for g1, g2 in itertools.combinations(levels, 2):
        L = pd.Series(0.0, index=params.index)
        if names[g2] in L.index:
            L[names[g2]] += 1.0
        if names[g1] in L.index:
            L[names[g1]] -= 1.0
        est = float(L @ params)
        se = float(np.sqrt(L.values @ cov.values @ L.values))
        q = abs(est) / se * np.sqrt(2) if se > 0 else np.inf
        p = float(studentized_range.sf(q, k, df)) if np.isfinite(q) else 0.0
        half = q_crit / np.sqrt(2) * se
        rows.append({"group1": g1, "group2": g2, "meandiff": round(est, 4), "p-adj": round(p, 4),
                     "lower": round(float(est - half), 4), "upper": round(float(est + half), 4), "reject": bool(p < alpha)})
    return rows


def compare(plot_tbl, pc, input_prices, vr_labels=("vr", "variable", "augmenta", "a_vr", "b_augmenta"), vr_cost=0.0):
    """plot_tbl: join.plot_table output with 'treatment' (and optionally 'block'); input_prices: {name: $/unit}."""
    # an empty plot table (no yield observation in any plot: project.join now refuses that, but a caller may still
    # pass one) has no columns at all, so 'treatment' must be checked before it is used
    if plot_tbl is None or len(plot_tbl) == 0 or "treatment" not in plot_tbl.columns:
        return None
    d = plot_tbl.dropna(subset=["treatment"]).copy()
    if d.treatment.nunique() < 2:
        return None
    d["treatment"] = d.treatment.astype(str)
    cost = pd.Series(0.0, index=d.index)
    unknown = pd.Series(False, index=d.index)
    unknown_by_input = {}
    left_out = []                                      # inputs with no rate on any plot (not in this comparison)
    for n, p in input_prices.items():
        p = float(p or 0.0)
        if p == 0.0:
            continue                                   # a free input costs 0 whatever its rate
        c = f"rate_{n}"
        r = pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
        bad = ~np.isfinite(r.astype(float))
        if bad.all():
            # an input with no rate on any plot (listed, but neither designed nor mapped in this trial) is not part
            # of the comparison: leaving it out keeps the comparison of the other inputs' costs, rather than making
            # every plot's cost unknown
            left_out.append(n)
            continue
        if bad.any():
            unknown_by_input[n] = int(bad.sum())
        unknown |= bad
        cost = cost + r.fillna(0.0) * p
    isvr = d.treatment.str.lower().apply(lambda s: any(v in s for v in vr_labels))
    d["input_cost"] = (cost + np.where(isvr, vr_cost, 0.0)).where(~unknown, np.nan)
    d["net_return"] = pc * d["yield"] - d.input_cost
    d["cost_known"] = ~unknown
    # input cost and net return are averaged over the plots whose cost is known (all-unknown -> NaN, printed '-');
    # n_cost_unknown says how many plots of the treatment were left out of those two means
    out = dict(table=d.groupby("treatment").agg(n_plots=("plot_id", "count"), yield_mean=("yield", "mean"),
                                                yield_sd=("yield", "std"), input_cost=("input_cost", "mean"),
                                                net_return=("net_return", "mean"),
                                                net_sd=("net_return", "std"),
                                                n_cost_unknown=("cost_known", lambda s: int((~s).sum())))
               .reset_index())
    n_unknown = int(unknown.sum())
    out["n_cost_unknown"] = n_unknown
    out["rate_unknown_by_input"] = unknown_by_input
    out["inputs_left_out"] = left_out
    if left_out:
        out["left_out_note"] = (", ".join(left_out) + ": no rate on any plot, so left out of the input cost of the "
                                "treatment comparison")
    if n_unknown:
        which = ", ".join(f"{k}: {v} plot(s)" for k, v in unknown_by_input.items())
        out["cost_note"] = (f"{n_unknown} of {len(d)} plot(s) have an unknown rate ({which}; no design rate and no "
                            "as-applied value): their input cost and net return are unknown, so they are left out of "
                            "the input cost and net return means and of the net-return ANOVA"
                            + (" (no plot has a known cost, so net return is not compared)" if n_unknown == len(d)
                               else ""))
    blocks = "block" in d and d.block.notna().all() and d.block.nunique() > 1
    for var in ("yield", "net_return"):
        dv = d[np.isfinite(d[var].astype(float))]
        levels = sorted(dv.treatment.unique())
        if dv.treatment.nunique() < 2:
            out[var] = dict(error=f"fewer than 2 treatments with a known {var.replace('_', ' ')}")
            continue
        bl = blocks and dv.block.nunique() > 1
        frm = f"Q('{var}') ~ C(treatment)" + (" + C(block)" if bl else "")
        try:
            m = smf.ols(frm, data=dv).fit()
            a = anova_lm(m, typ=2)
            p = float(a.loc["C(treatment)", "PR(>F)"])
            if bl:
                tukey = _tukey_rcbd(m, levels)
                how = ("Tukey-Kramer on the RCBD model (treatment + block): block-adjusted differences, residual mean "
                       f"square and {int(m.df_resid)} residual df of that model")
            else:
                tk = pairwise_tukeyhsd(dv[var], dv.treatment, alpha=ALPHA)
                tukey = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0]).to_dict("records")
                how = "Tukey HSD (one-way, no blocks)"
            # 'blocked' is per variable: the plots with a known value of this variable may all sit in one block (e.g.
            # net return with the as-applied map missing elsewhere), and then the model is one-way although the trial
            # has blocks (out['blocks']); the report labels the ANOVA from this flag, not the trial-level one
            out[var] = dict(model=frm, p_treatment=p, tukey=tukey, pairwise_method=how, blocked=bool(bl))
        except Exception as exc:                                     # pragma: no cover
            out[var] = dict(error=str(exc))
    out["plots"] = d
    out["blocks"] = bool(blocks)
    return out
