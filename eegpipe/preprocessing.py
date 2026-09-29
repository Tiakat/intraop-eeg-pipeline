"""
Filtering and epoch-wise artifact rejection.

Design choices
--------------
* Filters are applied **separately to each continuous segment**: recording gaps
  stay NaN and a filter never runs across a discontinuity.
* Rejection works on 4-s epochs with 8 independent criteria, so every rejected
  epoch can be traced back to the reason it was removed.
* Variance / jump outliers use a **local** robust z-score (median/MAD over
  +/- 5 min), so awake EEG (large, eye movements) is not compared with EEG under
  anesthesia, and vice versa.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from .params import PARAMS

REJECT_CODES = ["gap_or_edge", "saturation", "amplitude", "flat", "impedance_check",
                "variance_outlier", "jump_outlier", "low_sqi"]


def contiguous_segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """Half-open intervals [a, b) where `valid` is True."""
    v = np.r_[False, np.asarray(valid, bool), False].astype(np.int8)
    d = np.diff(v)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def filter_eeg(x: np.ndarray, fs: int = 128, p: dict = PARAMS) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-phase Butterworth band-pass (0.3-45 Hz, order 4) + 60-Hz notch, applied
    segment by segment. Returns (filtered signal, edge mask) where the edge mask
    flags the first/last `edge_s` seconds of every segment (filter transients).
    """
    y = np.full(len(x), np.nan)
    edge = np.zeros(len(x), dtype=bool)
    sos = butter(p["filter_order"], [p["hp_hz"], p["lp_hz"]], btype="band", fs=fs, output="sos")
    bn, an = iirnotch(p["notch_hz"], p["notch_q"], fs)
    ne = int(p["edge_s"] * fs)
    for a, b in contiguous_segments(np.isfinite(x)):
        if b - a < p["min_segment_s"] * fs:
            edge[a:b] = True
            continue
        seg = filtfilt(bn, an, sosfiltfilt(sos, x[a:b].astype(np.float64)))
        y[a:b] = seg
        edge[a:a + ne] = True
        edge[b - ne:b] = True
    return y, edge


def _robust_z_local(v: np.ndarray, half_width: int) -> np.ndarray:
    """Robust z-score against a centred rolling median / MAD (+/- half_width epochs)."""
    s = pd.Series(v)
    w = 2 * half_width + 1
    mp = max(10, half_width // 2)
    med = s.rolling(w, center=True, min_periods=mp).median()
    mad = (s - med).abs().rolling(w, center=True, min_periods=mp).median() * 1.4826
    return ((s - med) / mad.where(mad > 0)).values


def reject_epochs(y: np.ndarray, edge: np.ndarray, raw_ad: np.ndarray | None = None,
                  sqi: np.ndarray | None = None, fs: int = 128, p: dict = PARAMS) -> pd.DataFrame:
    """
    One row per 4-s epoch of ONE filtered channel, one boolean column per
    rejection criterion, and `reject` = any criterion.

    raw_ad : raw A/D values of the channel (saturation test), optional
    sqi    : monitor signal-quality index resampled to one value per sample, optional
    """
    L = int(p["epoch_s"] * fs)
    n = len(y) // L
    e = y[: n * L].reshape(n, L)
    out = pd.DataFrame({"epoch": np.arange(n), "start_s": np.arange(n) * p["epoch_s"]})
    out["gap_or_edge"] = (~np.isfinite(e)).any(1) | edge[: n * L].reshape(n, L).any(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(e, 1)
        mx = np.nanmax(np.abs(e), 1)
        jump = np.nanmax(np.abs(np.diff(e, axis=1)), 1)
        still = np.nanmean(np.abs(np.diff(e, axis=1)) < p["flat_diff_uv"], axis=1)
    out["saturation"] = False
    if raw_ad is not None:
        r = np.nan_to_num(raw_ad[: n * L].astype(float)).reshape(n, L)
        out["saturation"] = (np.abs(r) >= p["saturation_ad"]).any(1)
    out["amplitude"] = mx > p["abs_max_uv"]
    out["flat"] = sd < p["flat_sd_uv"]
    # the monitor's periodic impedance check = near-flat line with regular spikes
    out["impedance_check"] = still > p["flat_frac_max"]
    ok = ~out["gap_or_edge"].values
    hw = int(round(p["robust_z_half_window_s"] / p["epoch_s"]))
    zsd = np.full(n, np.nan)
    zj = np.full(n, np.nan)
    zsd[ok] = _robust_z_local(np.log10(sd[ok] + 1e-9), hw)
    zj[ok] = _robust_z_local(np.log10(jump[ok] + 1e-9), hw)
    out["variance_outlier"] = np.nan_to_num(zsd) > p["robust_z"]
    out["jump_outlier"] = np.nan_to_num(zj) > p["robust_z"]
    out["low_sqi"] = False
    if sqi is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["low_sqi"] = np.nanmin(sqi[: n * L].reshape(n, L), 1) < p["sqi_min"]
    out["sd_uv"] = sd
    out["max_abs_uv"] = mx
    out["reject"] = out[REJECT_CODES].any(axis=1)
    return out


def clean_recording(eeg_uv: np.ndarray, fs: int = 128, raw_ad: np.ndarray | None = None,
                    sqi: np.ndarray | None = None, p: dict = PARAMS):
    """
    Filter + reject every channel of a (n_samples, n_channels) recording.

    The impedance check hits all electrodes at once, so an epoch flagged on one
    channel is rejected on every channel.

    Returns
    -------
    filtered : (n, n_ch) filtered EEG (NaN in gaps)
    cleaned  : (n, n_ch) filtered EEG with rejected epochs set to NaN
    rejection: long DataFrame, one row per epoch x channel
    """
    n, nch = eeg_uv.shape
    L = int(p["epoch_s"] * fs)
    filtered = np.full((n, nch), np.nan)
    tables = []
    for c in range(nch):
        y, edge = filter_eeg(eeg_uv[:, c], fs, p)
        filtered[:, c] = y
        rj = reject_epochs(y, edge, None if raw_ad is None else raw_ad[:, c],
                           None if sqi is None else sqi[:, c], fs, p)
        rj.insert(0, "channel", c + 1)
        tables.append(rj)
    imp_any = np.logical_or.reduce([t["impedance_check"].values for t in tables])
    cleaned = filtered.copy()
    for c, rj in enumerate(tables):
        rj["impedance_check"] = imp_any
        rj["reject"] = rj[REJECT_CODES].any(axis=1)
        bad = np.repeat(rj["reject"].values, L)
        cleaned[: len(bad), c][bad] = np.nan
        cleaned[len(bad):, c] = np.nan          # incomplete last epoch
    return filtered, cleaned, pd.concat(tables, ignore_index=True)
