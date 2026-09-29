"""
End-to-end processing of one recording:

    raw EEG (uV) --> filter + artifact rejection --> 32-s spectral metrics
                                                 --> 128-s phase-amplitude coupling
                 --> phase labels from clinical landmarks
                 --> per-phase summaries (median over windows, >= 2 windows required)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pac import pac_metrics
from .params import PARAMS, PHASE_ORDER
from .phases import define_phases, phase_of
from .preprocessing import clean_recording
from .spectral import window_metrics


@dataclass
class Result:
    filtered: np.ndarray
    cleaned: np.ndarray
    rejection: pd.DataFrame
    metrics: pd.DataFrame
    pac: pd.DataFrame
    phases: pd.DataFrame
    psd_by_phase: dict
    fs: int

    def phase_summary(self, min_windows: int = 2) -> pd.DataFrame:
        """Median of every metric per phase (a phase needs >= `min_windows` windows)."""
        cols = [c for c in self.metrics if c.startswith("mean_")]
        g = self.metrics[self.metrics["phase"] != "other"].groupby("phase")
        s = g[cols].median().where(g[cols].count() >= min_windows)
        pac = self.pac.groupby(["phase"])[["pac_mi", "pac_z"]].median()
        return s.join(pac).reindex([p for p in PHASE_ORDER if p in s.index])


def process_recording(eeg_uv: np.ndarray, landmarks_s: dict, fs: int = 128,
                      raw_ad: np.ndarray | None = None, sqi: np.ndarray | None = None,
                      seed: int = 0, fooof: bool = True, p: dict = PARAMS) -> Result:
    filtered, cleaned, rej = clean_recording(eeg_uv, fs, raw_ad, sqi, p)
    met, psds = window_metrics(filtered, rej, fs, p, fooof=fooof)
    pac = pac_metrics(cleaned, fs, seed, p)
    phases = define_phases(landmarks_s, len(eeg_uv) / fs)
    met["phase"] = phase_of(met["t_mid_s"], phases)
    pac["phase"] = phase_of(pac["t_mid_s"], phases)
    # median PSD per phase (channel 1) for the spectra figure
    psd_by_phase = {"f": psds["f"]}
    for ph in PHASE_ORDER:
        arr = [psds[i][1] for i in met.index[met["phase"] == ph] if 1 in psds.get(i, {})]
        if arr:
            psd_by_phase[ph] = np.median(np.array(arr), axis=0)
    return Result(filtered, cleaned, rej, met, pac, phases, psd_by_phase, fs)
