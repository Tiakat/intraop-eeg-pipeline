"""
Anesthetic phases defined from clinical landmarks (seconds from recording start).

In the study, landmarks come from time-stamped annotations entered on the
nociception monitor during surgery, with the paper/electronic anesthesia record
as a fallback (shifted onto the monitor clock by the median offset between the
two sources for that patient).

Phases
------
baseline      : last 5 min awake, ending 30 s before induction / study-drug start
induction     : induction -> intubation + 5 min
maint_early   : first 30 min of maintenance
maint_late    : last 30 min before the end of the study-drug infusion
post_infusion : end of study drug -> propofol stop (or end of surgery)
emergence     : propofol stop -> extubation
If maintenance lasts < 60 min, early/late are its first and second halves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LANDMARKS = ["induction", "drug_start", "intubation", "drug_stop", "propofol_stop",
             "surgery_end", "extubation"]


def define_phases(lm: dict, rec_end_s: float) -> pd.DataFrame:
    g = lambda k: lm.get(k, np.nan)
    ok = lambda t: t is not None and np.isfinite(t)
    starts = [t for t in (g("induction"), g("drug_start")) if ok(t)]
    awake_end = min(starts) - 30 if starts else np.nan
    intub = g("intubation") if ok(g("intubation")) else g("induction") + 300
    m0 = intub + 300
    ends = [t for t in (g("drug_stop"), g("propofol_stop"), g("surgery_end")) if ok(t) and t > m0]
    m1 = min(ends) if ends else np.nan
    ph = []
    if ok(awake_end) and awake_end > 0:
        ph.append(("baseline", max(0.0, awake_end - 300), awake_end))
    if ok(g("induction")) and ok(m0):
        ph.append(("induction", g("induction"), m0))
    if ok(m0) and ok(m1):
        if m1 - m0 >= 3600:
            ph += [("maint_early", m0, m0 + 1800), ("maint_late", m1 - 1800, m1)]
        else:
            mid = (m0 + m1) / 2
            ph += [("maint_early", m0, mid), ("maint_late", mid, m1)]
    end_post = next((t for t in (g("propofol_stop"), g("surgery_end"))
                     if ok(t) and ok(g("drug_stop")) and t > g("drug_stop")), np.nan)
    if ok(end_post):
        ph.append(("post_infusion", g("drug_stop"), end_post))
    em0 = g("propofol_stop") if ok(g("propofol_stop")) else g("surgery_end")
    if ok(em0) and ok(g("extubation")) and g("extubation") > em0:
        ph.append(("emergence", em0, g("extubation")))
    out = pd.DataFrame(ph, columns=["phase", "start_s", "end_s"])
    out["start_s"] = out["start_s"].clip(lower=0)
    out["end_s"] = out["end_s"].clip(upper=rec_end_s)
    return out[out["end_s"] > out["start_s"]].reset_index(drop=True)


def phase_of(t_s: np.ndarray, phases: pd.DataFrame) -> np.ndarray:
    """Phase label of each time point ('other' outside every phase)."""
    t = np.asarray(t_s, float)
    out = np.full(len(t), "other", dtype=object)
    for _, r in phases.iterrows():
        m = (t >= r["start_s"]) & (t < r["end_s"]) & (out == "other")
        out[m] = r["phase"]
    return out
