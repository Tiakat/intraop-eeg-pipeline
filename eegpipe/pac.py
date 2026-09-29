"""
Slow(delta)-alpha phase-amplitude coupling with the Tort et al. (2010)
modulation index, and a surrogate z-score from circularly shifted amplitudes.

Phase and amplitude are extracted from the CLEANED EEG (rejected epochs = NaN),
one clean segment at a time, so the band-pass filters and the Hilbert transform
never run across an artifact. Segment edges (filter transients) are discarded.

Reference: Tort ABL, Komorowski R, Eichenbaum H, Kopell N. J Neurophysiol 2010;104:1195-1210.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from .params import PARAMS
from .preprocessing import contiguous_segments


def tort_mi(phase: np.ndarray, amp: np.ndarray, nbins: int = 18) -> tuple[float, float]:
    """MI = (log N - H(P)) / log N; returns (MI, preferred phase in rad)."""
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    mean_amp = np.bincount(idx, weights=amp, minlength=nbins) / np.maximum(np.bincount(idx, minlength=nbins), 1)
    P = mean_amp / mean_amp.sum()
    H = -np.sum(P[P > 0] * np.log(P[P > 0]))
    centers = (edges[:-1] + edges[1:]) / 2
    return float((np.log(nbins) - H) / np.log(nbins)), float(np.angle(np.sum(P * np.exp(1j * centers))))


def phase_amplitude_histogram(phase: np.ndarray, amp: np.ndarray, nbins: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """Normalised mean amplitude per phase bin (for plotting)."""
    m = np.isfinite(phase) & np.isfinite(amp)
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    idx = np.clip(np.digitize(phase[m], edges) - 1, 0, nbins - 1)
    a = np.bincount(idx, weights=amp[m], minlength=nbins) / np.maximum(np.bincount(idx, minlength=nbins), 1)
    return (edges[:-1] + edges[1:]) / 2, a / a.sum()


def pac_signals(y: np.ndarray, fs: int = 128, p: dict = PARAMS) -> tuple[np.ndarray, np.ndarray]:
    """Slow-wave phase and alpha envelope, computed per clean segment (NaN elsewhere)."""
    ph = np.full(len(y), np.nan)
    am = np.full(len(y), np.nan)
    sp = butter(p["pac_filter_order"], p["pac_phase_band"], btype="band", fs=fs, output="sos")
    sa = butter(p["pac_filter_order"], p["pac_amp_band"], btype="band", fs=fs, output="sos")
    mg = int(p["pac_margin_s"] * fs)
    for a, b in contiguous_segments(np.isfinite(y)):
        if b - a < p["pac_min_segment_s"] * fs:
            continue
        pa = np.angle(hilbert(sosfiltfilt(sp, y[a:b])))
        aa = np.abs(hilbert(sosfiltfilt(sa, y[a:b])))
        ph[a + mg:b - mg] = pa[mg:len(pa) - mg]
        am[a + mg:b - mg] = aa[mg:len(aa) - mg]
    return ph, am


def pac_window(ph: np.ndarray, am: np.ndarray, rng: np.random.Generator,
               fs: int = 128, p: dict = PARAMS) -> dict:
    """MI, surrogate z and preferred phase for one window (NaN if < 60 s clean)."""
    m = np.isfinite(ph) & np.isfinite(am)
    n = int(m.sum())
    if n < p["pac_min_clean_s"] * fs:
        return {"pac_mi": np.nan, "pac_z": np.nan, "pac_pref_phase": np.nan, "pac_clean_s": n / fs}
    phv, amv = ph[m], am[m]
    nb = p["pac_nbins"]
    mi, pref = tort_mi(phv, amv, nb)
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    idx = np.clip(np.digitize(phv, edges) - 1, 0, nb - 1)
    cnt = np.maximum(np.bincount(idx, minlength=nb), 1)
    sur = np.empty(p["pac_n_surrogates"])
    for k, sft in enumerate(rng.integers(fs, n - fs, p["pac_n_surrogates"])):   # shifts >= 1 s
        P = np.bincount(idx, weights=np.roll(amv, sft), minlength=nb) / cnt
        P = P / P.sum()
        sur[k] = (np.log(nb) + np.sum(P[P > 0] * np.log(P[P > 0]))) / np.log(nb)
    z = (mi - sur.mean()) / sur.std() if sur.std() > 0 else np.nan
    return {"pac_mi": mi, "pac_z": z, "pac_pref_phase": pref, "pac_clean_s": n / fs}


def pac_metrics(cleaned: np.ndarray, fs: int = 128, seed: int = 0, p: dict = PARAMS) -> pd.DataFrame:
    """PAC for consecutive 128-s windows of every channel of a cleaned recording."""
    rng = np.random.default_rng(seed)
    wlen = int(p["pac_window_s"] * fs)
    rows = []
    for c in range(cleaned.shape[1]):
        ph, am = pac_signals(cleaned[:, c], fs, p)
        for a in range(0, len(ph) - wlen + 1, wlen):
            rows.append({"channel": c + 1, "t_start_s": a / fs, "t_mid_s": (a + wlen / 2) / fs,
                         **pac_window(ph[a:a + wlen], am[a:a + wlen], rng, fs, p)})
    return pd.DataFrame(rows)
