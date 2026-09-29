"""
Statistics used on the per-patient / per-phase summaries.

Within patients (phase contrasts)
  * Wilcoxon signed-rank on patient medians, effect size = matched-pairs
    rank-biserial correlation, Hodges-Lehmann shift
  * Holm correction within a metric, then Benjamini-Hochberg across metrics
  * Linear mixed model  metric ~ phase + (1 | patient)

Between groups (randomised arms)
  * Mann-Whitney U per phase + bootstrap CI of the median difference and of Cliff's delta
  * Mixed model  metric ~ group * phase + (1 | patient), unadjusted (primary),
    adjusted for pre-randomisation covariates, and adjusted for drug targets
    (post-randomisation, i.e. potential mediators -> secondary)

`fit_mixed` guards against a known statsmodels pitfall: the default L-BFGS
optimiser can stop on a boundary optimum (random-intercept variance = 0) while
another optimiser finds a higher REML likelihood.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def holm(p) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float); out = np.full_like(p, np.nan); m = np.isfinite(p)
    if m.any():
        out[m] = multipletests(p[m], method="holm")[1]
    return out


def bh(p) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float); out = np.full_like(p, np.nan); m = np.isfinite(p)
    if m.any():
        out[m] = multipletests(p[m], method="fdr_bh")[1]
    return out


def hodges_lehmann_paired(d) -> float:
    d = np.asarray(d, float)
    w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))


def rank_biserial_paired(diff) -> float:
    """(R+ - R-) / (R+ + R-), zero differences excluded; bounded in [-1, 1]."""
    nz = np.asarray(diff, float)
    nz = nz[nz != 0]
    if not len(nz):
        return np.nan
    rk = stats.rankdata(np.abs(nz))
    return float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())


def wilcoxon_contrast(a, b) -> dict:
    """Paired comparison b vs a (same patients)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b - a
    w = stats.wilcoxon(b, a, zero_method="wilcox")
    return {"n": len(a), "median_a": np.median(a), "median_b": np.median(b),
            "hodges_lehmann": hodges_lehmann_paired(diff), "W": w.statistic, "p": w.pvalue,
            "rank_biserial": rank_biserial_paired(diff)}


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def mann_whitney_bootstrap(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mann-Whitney U with bootstrap 95 % CIs of the median difference and of Cliff's delta."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    md, cd = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        ra, rb = rng.choice(a, len(a)), rng.choice(b, len(b))
        md[i] = np.median(ra) - np.median(rb)
        cd[i] = cliffs_delta(ra, rb)
    return {"n_a": len(a), "n_b": len(b), "diff_median": np.median(a) - np.median(b),
            "diff_ci95": (np.percentile(md, 2.5), np.percentile(md, 97.5)),
            "cliffs_delta": cliffs_delta(a, b),
            "cliffs_ci95": (np.percentile(cd, 2.5), np.percentile(cd, 97.5)),
            "U": u.statistic, "p": u.pvalue}


def fit_mixed(formula: str, d: pd.DataFrame, group: str = "patient"):
    """
    Random-intercept linear mixed model (REML), fitted robustly:
      1. try 4 optimisers (lbfgs, bfgs, powell, nm) and keep the converged fit with
         the highest finite REML log-likelihood, preferring a positive-definite Hessian;
      2. if the retained random-intercept variance is ~0 (boundary) or nothing
         converges, fall back to OLS with cluster-robust (by patient) standard errors.
    Returns (fit, note).
    """
    import statsmodels.formula.api as smf
    best, best_llf, best_m, best_hess_bad = None, -np.inf, None, True
    for method in ("lbfgs", "bfgs", "powell", "nm"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                fit = smf.mixedlm(formula, d, groups=d[group]).fit(reml=True, method=method, maxiter=5000)
            except Exception:
                continue
        hess_bad = any("Hessian" in str(x.message) for x in w)
        se = fit.bse[fit.fe_params.index]
        if not fit.converged or not np.all(np.isfinite(se)) or not np.isfinite(fit.llf):
            continue
        better = fit.llf > best_llf + 1e-6
        tie_but_cleaner = best is not None and abs(fit.llf - best_llf) <= 1e-6 and best_hess_bad and not hess_bad
        if better or tie_but_cleaner:
            best, best_llf, best_m, best_hess_bad = fit, fit.llf, method, hess_bad
    if best is not None and float(best.cov_re.iloc[0, 0]) > 1e-8 * max(best.scale, 1e-12) and not best_hess_bad:
        return best, f"mixed ({best_m})"
    ols = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d[group]})
    return ols, "OLS + cluster-robust SE (random-intercept variance ~ 0 or no convergence)"
