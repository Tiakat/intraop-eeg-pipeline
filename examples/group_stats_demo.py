"""
Statistics workflow on a SIMULATED cohort (no real data, no real results).

    python examples/group_stats_demo.py

The per-patient pipeline produces one median per metric x phase x patient.
This script simulates such a table for 34 patients randomised 1:1 to two arms,
then runs the analyses used in the study:

  1. within-patient phase contrasts - Wilcoxon signed-rank, rank-biserial r,
     Hodges-Lehmann shift, Holm within metric then Benjamini-Hochberg
  2. linear mixed model  metric ~ phase + (1 | patient)
  3. arm comparison per phase - Mann-Whitney U + bootstrap CIs (median difference, Cliff's delta)
  4. arm x phase mixed model (unadjusted = primary; + age/sex/BMI = sensitivity)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegpipe.stats import bh, fit_mixed, holm, mann_whitney_bootstrap, wilcoxon_contrast  # noqa: E402

PHASES = ["baseline", "maint_early", "maint_late", "post_infusion", "emergence"]
CONTRASTS = [("baseline", "maint_early"), ("maint_early", "maint_late"),
             ("maint_late", "post_infusion"), ("post_infusion", "emergence")]


def simulate_cohort(n: int = 34, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pts = pd.DataFrame({"patient": np.arange(1, n + 1),
                        "arm": rng.permutation(np.repeat(["A", "B"], n // 2)),
                        "age": rng.normal(55, 12, n).round(), "male": rng.integers(0, 2, n),
                        "bmi": rng.normal(28, 4, n).round(1)})
    mean = {"alpha_rel": [0.015, 0.08, 0.09, 0.09, 0.05], "ap_exponent": [1.8, 2.3, 2.3, 2.2, 1.7]}
    # no true difference between arms is simulated: the demo shows what a null result looks like
    arm_effect = {"alpha_rel": [0] * 5, "ap_exponent": [0] * 5}
    rows = []
    for _, p in pts.iterrows():
        u = {m: rng.normal(0, 0.02 if m == "alpha_rel" else 0.2) for m in mean}        # random intercepts
        for k, ph in enumerate(PHASES):
            r = {"patient": p["patient"], "arm": p["arm"], "phase": ph}
            for m in mean:
                sd = 0.012 if m == "alpha_rel" else 0.15
                r[m] = mean[m][k] + u[m] + (arm_effect[m][k] if p["arm"] == "A" else 0) + rng.normal(0, sd)
            rows.append(r)
    return pd.DataFrame(rows), pts


def main():
    summ, pts = simulate_cohort()
    metrics = ["alpha_rel", "ap_exponent"]

    # 1. within-patient contrasts
    rows = []
    for m in metrics:
        piv = summ.pivot(index="patient", columns="phase", values=m)
        for a, b in CONTRASTS:
            d = piv[[a, b]].dropna()
            rows.append({"metric": m, "contrast": f"{a} -> {b}", **wilcoxon_contrast(d[a], d[b])})
    w = pd.DataFrame(rows)
    w["p_holm"] = w.groupby("metric")["p"].transform(holm)
    w["q_bh"] = bh(w["p_holm"])
    print("1. Within-patient contrasts\n", w[["metric", "contrast", "n", "median_a", "median_b",
                                              "hodges_lehmann", "rank_biserial", "p_holm", "q_bh"]]
          .round(4).to_string(index=False))

    # 2. mixed model metric ~ phase
    for m in metrics:
        d = summ[["patient", "phase", m]].rename(columns={m: "y"})
        d["phase"] = pd.Categorical(d["phase"], ["maint_early"] + [p for p in PHASES if p != "maint_early"])
        fit, note = fit_mixed("y ~ C(phase)", d)
        print(f"\n2. Mixed model {m} ~ phase (ref = maint_early) [{note}]")
        fe = fit.fe_params.index if hasattr(fit, "fe_params") else fit.params.index
        print(pd.DataFrame({"beta": fit.params[fe], "p": fit.pvalues[fe]}).round(4).to_string())

    # 3. arm comparison per phase
    rows = []
    for m in metrics:
        for ph in PHASES:
            s = summ[summ["phase"] == ph]
            r = mann_whitney_bootstrap(s.loc[s["arm"] == "A", m], s.loc[s["arm"] == "B", m])
            rows.append({"metric": m, "phase": ph, "diff_median": r["diff_median"],
                         "ci95": np.round(r["diff_ci95"], 4), "cliffs_delta": r["cliffs_delta"], "p": r["p"]})
    g = pd.DataFrame(rows)
    g["q_bh"] = bh(g["p"])
    print("\n3. Arm A vs B per phase (between patients)\n", g.round(4).to_string(index=False))

    # 4. arm x phase mixed models (maintenance phases)
    mm = summ[summ["phase"].isin(["maint_early", "maint_late", "post_infusion"])].merge(
        pts[["patient", "age", "male", "bmi"]], on="patient")
    for m in metrics:
        d = mm.rename(columns={m: "y"})
        for name, extra in (("M1 unadjusted", ""), ("M2 + age, sex, BMI", " + age + male + bmi")):
            fit, note = fit_mixed("y ~ C(arm, Treatment('B')) * C(phase)" + extra, d)
            terms = [t for t in fit.params.index if "arm" in t]
            print(f"\n4. {m}, {name} [{note}]")
            print(pd.DataFrame({"beta": fit.params[terms], "p": fit.pvalues[terms]}).round(4).to_string())


if __name__ == "__main__":
    main()
