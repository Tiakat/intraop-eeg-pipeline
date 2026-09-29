"""
Spectral metrics computed on 32-s windows of clean 4-s epochs.

Per window and channel:
  * absolute band power (uV^2), relative power (/ 0.5-45 Hz) and dB
  * peak frequency (local maximum of the PSD in each band; NaN when the maximum
    sits on a band edge, because that is a slope, not a peak)
  * spectral parameterisation (FOOOF): aperiodic offset and exponent (1/f slope)
    and the oscillatory theta / alpha / beta peaks above the aperiodic component
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import get_window

from .params import PARAMS


def epoch_psd(e: np.ndarray, fs: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Modified periodogram (Hann, demeaned) of each epoch; one-sided, uV^2/Hz."""
    L = e.shape[1]
    w = get_window("hann", L)
    X = np.fft.rfft((e - e.mean(1, keepdims=True)) * w, axis=1)
    psd = np.abs(X) ** 2 / (fs * np.sum(w ** 2))
    psd[:, 1:-1] *= 2
    return np.fft.rfftfreq(L, 1 / fs), psd


def band_power(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Absolute power (uV^2) = sum of PSD * df over [lo, hi)."""
    m = (f >= lo) & (f < hi)
    return psd[..., m].sum(-1) * (f[1] - f[0])


def peak_frequency(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Frequency of the PSD maximum in [lo, hi); NaN if the maximum is on a band edge."""
    m = np.where((f >= lo) & (f < hi))[0]
    i = m[np.argmax(psd[m])]
    return np.nan if i in (m[0], m[-1]) else float(f[i])


def fit_fooof(f: np.ndarray, psd: np.ndarray, p: dict = PARAMS) -> dict:
    """Aperiodic offset/exponent and strongest oscillatory peak per band."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fooof import FOOOF
    out = {"ap_offset": np.nan, "ap_exponent": np.nan, "fooof_r2": np.nan}
    for b in p["fooof_peak_bands"]:
        out.update({f"{b}_cf": np.nan, f"{b}_pw": np.nan})
    fm = FOOOF(**p["fooof_settings"])
    try:
        fm.fit(f, psd, list(p["fooof_range"]))
    except Exception:
        return out
    out["ap_offset"], out["ap_exponent"] = fm.aperiodic_params_
    out["fooof_r2"] = fm.r_squared_
    pk = fm.peak_params_
    for b, (lo, hi) in p["fooof_peak_bands"].items():
        cand = pk[(pk[:, 0] >= lo) & (pk[:, 0] < hi)] if len(pk) else pk
        if len(cand):
            best = cand[np.argmax(cand[:, 1])]
            out[f"{b}_cf"], out[f"{b}_pw"] = best[0], best[1]
    return out


def window_metrics(filtered: np.ndarray, rejection: pd.DataFrame, fs: int = 128,
                   p: dict = PARAMS, fooof: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Metrics for consecutive 32-s windows (8 epochs). A window is used for a
    channel only if >= 50 % of its epochs are clean; the PSD is the mean of the
    clean epochs' periodograms.

    Returns (metrics DataFrame, {window index: {channel: PSD}}) plus the
    frequency vector under key 'f'.
    """
    L = int(p["epoch_s"] * fs)
    epw = int(p["window_s"] / p["epoch_s"])
    n, nch = filtered.shape
    n_ep = n // L
    rows, psds = [], {"f": np.fft.rfftfreq(L, 1 / fs)}
    for wi, w0 in enumerate(range(0, n_ep - epw + 1, epw)):
        r = {"window": wi, "t_start_s": w0 * p["epoch_s"], "t_mid_s": w0 * p["epoch_s"] + p["window_s"] / 2}
        psds[wi] = {}
        for c in range(nch):
            rj = rejection[rejection["channel"] == c + 1]
            ok = ~rj["reject"].values[w0:w0 + epw]
            ch = f"ch{c + 1}"
            r[f"{ch}_n_clean_epochs"] = int(ok.sum())
            if ok.mean() < p["min_clean_frac"]:
                continue
            ep = filtered[w0 * L:(w0 + epw) * L, c].reshape(epw, L)[ok]
            f, P = epoch_psd(ep, fs)
            psd = P.mean(0)
            psds[wi][c + 1] = psd
            tot = band_power(f, psd, *p["total_band"])
            r[f"{ch}_total_power_uv2"] = tot
            for b, (lo, hi) in p["bands"].items():
                bp = band_power(f, psd, lo, hi)
                r[f"{ch}_{b}_abs_uv2"] = bp
                r[f"{ch}_{b}_db"] = 10 * np.log10(bp)
                r[f"{ch}_{b}_rel"] = bp / tot
                r[f"{ch}_{b}_peak_hz"] = peak_frequency(f, psd, lo, hi)
            if fooof:
                for k, v in fit_fooof(f, psd, p).items():
                    r[f"{ch}_{k}"] = v
        rows.append(r)
    met = pd.DataFrame(rows)
    # channel average = primary metric
    bases = sorted({c.split("_", 1)[1] for c in met.columns if c.startswith("ch1_")} - {"n_clean_epochs"})
    for bc in bases:
        cols = [f"ch{k + 1}_{bc}" for k in range(nch) if f"ch{k + 1}_{bc}" in met]
        met[f"mean_{bc}"] = met[cols].astype(float).mean(axis=1)
    return met, psds
