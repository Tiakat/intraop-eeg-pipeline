"""Figures of one processed recording (matplotlib, static PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pac import pac_signals, phase_amplitude_histogram
from .params import PARAMS, PHASE_ORDER
from .preprocessing import REJECT_CODES

# categorical order, fixed (never cycled): one colour per phase
PHASE_COLORS = dict(zip(PHASE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
PHASE_LABELS = {"baseline": "Baseline (awake)", "induction": "Induction", "maint_early": "Maintenance, early",
                "maint_late": "Maintenance, late", "post_infusion": "Post-infusion", "emergence": "Emergence"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAW, CLEAN = "#8a8984", "#2a78d6"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.0,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


SHORT = {"baseline": "Baseline", "induction": "Induction", "maint_early": "Maint. early",
         "maint_late": "Maint. late", "post_infusion": "Post-infusion", "emergence": "Emergence"}


def _shade_phases(ax, phases, unit=60.0, labels=False):
    for k, (_, r) in enumerate(phases.iterrows()):
        ax.axvspan(r["start_s"] / unit, r["end_s"] / unit, color=PHASE_COLORS[r["phase"]], alpha=0.10, lw=0)
        if labels:   # staggered so that short neighbouring phases do not collide
            ax.text((r["start_s"] + r["end_s"]) / 2 / unit, 1.02 + 0.09 * (k % 2), SHORT[r["phase"]],
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK2)


def fig_overview(res, raw_uv, path, ch=0):
    """Whole recording: raw vs filtered + cleaned, and which criterion rejected each epoch."""
    fs = res.fs
    t = np.arange(len(raw_uv)) / fs / 60
    dec = 4                                                   # plot decimation (display only)
    fig, ax = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [2, 2, 1.3], "hspace": 0.3})
    ax[0].plot(t[::dec], raw_uv[::dec, ch], color=RAW, lw=0.3)
    ax[0].set_ylim(-400, 400); ax[0].set_ylabel("Raw (µV)")
    ax[0].set_title("Channel %d - raw export vs filtered and cleaned signal" % (ch + 1), loc="left", pad=26)
    ax[1].plot(t[::dec], res.cleaned[::dec, ch], color=CLEAN, lw=0.3)
    ax[1].set_ylim(-250, 250); ax[1].set_ylabel("Filtered + cleaned (µV)")
    rj = res.rejection[res.rejection["channel"] == ch + 1]
    for k, code in enumerate(REJECT_CODES):
        x = rj.loc[rj[code], "start_s"].values / 60
        ax[2].scatter(x, np.full(len(x), k), marker="|", s=40, color=INK2, lw=0.8)
    ax[2].set_yticks(range(len(REJECT_CODES)), [c.replace("_", " ") for c in REJECT_CODES], fontsize=7)
    ax[2].set_ylim(-0.7, len(REJECT_CODES) - 0.3); ax[2].invert_yaxis(); ax[2].grid(axis="y", visible=False)
    ax[2].set_xlabel("Time from recording start (min)")
    ax[2].set_ylabel("Rejected epochs", labelpad=6)
    pct = rj["reject"].mean() * 100
    ax[2].set_title(f"Rejection criteria ({pct:.1f} % of 4-s epochs rejected)", loc="left", fontsize=9)
    for a in ax:
        _shade_phases(a, res.phases, labels=a is ax[0])
    fig.savefig(path); plt.close(fig)


def fig_zoom_artifact(res, raw_uv, t0_s, dur_s, path, ch=0, title=""):
    """Zoom on an artifact: raw signal and what the pipeline keeps."""
    fs = res.fs
    a, b = int(t0_s * fs), int((t0_s + dur_s) * fs)
    t = np.arange(a, b) / fs - t0_s
    fig, ax = plt.subplots(2, 1, figsize=(11, 3.6), sharex=True, gridspec_kw={"hspace": 0.15})
    ax[0].plot(t, raw_uv[a:b, ch], color=RAW, lw=0.5); ax[0].set_ylabel("Raw (µV)")
    ax[0].set_title(title, loc="left")
    ax[1].plot(t, res.filtered[a:b, ch], color=RAW, lw=0.5, alpha=0.5, label="filtered (rejected epochs)")
    ax[1].plot(t, res.cleaned[a:b, ch], color=CLEAN, lw=0.6, label="kept for analysis")
    ax[1].set_ylim(-150, 150); ax[1].set_ylabel("µV"); ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.savefig(path); plt.close(fig)


def fig_traces_by_phase(res, path, ch=0, seconds=10.0):
    """10 s of clean, filtered EEG from the middle of each phase, same scale."""
    fs = res.fs
    ph = res.phases.set_index("phase")
    names = [p for p in PHASE_ORDER if p in ph.index]
    fig, ax = plt.subplots(len(names), 1, figsize=(11, 1.25 * len(names) + 0.6), sharex=True, sharey=True,
                           gridspec_kw={"hspace": 0.12})
    n_seg = int(seconds * fs)
    for a, p in zip(ax, names):
        y = res.cleaned[:, ch]
        mid = int((ph.loc[p, "start_s"] + ph.loc[p, "end_s"]) / 2 * fs)
        # nearest fully clean stretch to the middle of the phase
        best = None
        for off in range(0, int((ph.loc[p, "end_s"] - ph.loc[p, "start_s"]) / 2 * fs), fs):
            for s in (mid + off, mid - off):
                if s >= int(ph.loc[p, "start_s"] * fs) and s + n_seg <= int(ph.loc[p, "end_s"] * fs) \
                        and np.isfinite(y[s:s + n_seg]).all():
                    best = s; break
            if best is not None:
                break
        if best is None:
            continue
        a.plot(np.arange(n_seg) / fs, y[best:best + n_seg], color=PHASE_COLORS[p], lw=0.7)
        a.text(0.005, 0.95, PHASE_LABELS[p], transform=a.transAxes, va="top", fontsize=8, color=INK,
               bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        a.set_ylim(-160, 160); a.set_ylabel("µV")
    ax[0].set_title(f"Filtered, cleaned EEG - {seconds:.0f} s from the middle of each phase (channel {ch + 1})",
                    loc="left")
    ax[-1].set_xlabel("Time (s)")
    fig.savefig(path); plt.close(fig)


def fig_spectra(res, path):
    """Median power spectrum of each phase."""
    f = res.psd_by_phase["f"]
    m = (f >= 0.5) & (f <= 40)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for p in PHASE_ORDER:
        if p in res.psd_by_phase:
            y = 10 * np.log10(res.psd_by_phase[p][m])
            ax.plot(f[m], y, color=PHASE_COLORS[p], lw=1.6, label=PHASE_LABELS[p])
    for b, (lo, hi) in PARAMS["bands"].items():
        if hi <= 40:
            ax.axvline(hi, color=GRID, lw=0.8, zorder=0)
        ax.text((lo + min(hi, 40)) / 2, 1.01, b, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=7, color=INK2)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB µV²/Hz)")
    ax.set_xlim(0.5, 40)
    ax.set_title("Median spectrum per phase (32-s windows, channel 1)", loc="left", pad=16)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path); plt.close(fig)


def fig_evolution(res, path):
    """Time course of the main metrics (32-s windows; PAC on 128-s windows)."""
    m = res.metrics
    t = m["t_mid_s"] / 60
    pac = res.pac.groupby("t_mid_s")["pac_z"].mean()
    panels = [("mean_delta_rel", "Relative δ power", None), ("mean_alpha_rel", "Relative α power", None),
              ("mean_beta_rel", "Relative β power", None), ("mean_alpha_cf", "α peak (Hz, FOOOF)", (7, 14)),
              ("mean_ap_exponent", "Aperiodic exponent", None), ("pac_z", "δ-α PAC (z)", None)]
    fig, ax = plt.subplots(len(panels), 1, figsize=(11, 9.2), sharex=True, gridspec_kw={"hspace": 0.16})
    for a, (col, lab, lim) in zip(ax, panels):
        if col == "pac_z":
            a.plot(pac.index / 60, pac.values, color=INK, lw=1.2, marker="o", ms=2.5)
            a.axhline(1.96, color=INK2, lw=0.8, ls="--")
            a.text(1.0, 1.96, " z = 1.96", transform=a.get_yaxis_transform(), fontsize=7, color=INK2, va="center")
        else:
            a.plot(t, m[col], color=INK, lw=0.9)
        a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
        if lim:
            a.set_ylim(*lim)
        _shade_phases(a, res.phases, labels=a is ax[0])
    ax[0].set_title("Evolution of the EEG metrics during the case", loc="left", pad=26)
    ax[-1].set_xlabel("Time from recording start (min)")
    fig.savefig(path); plt.close(fig)


def fig_pac(res, path, ch=0):
    """Alpha amplitude as a function of slow-wave phase, awake vs early maintenance."""
    fs = res.fs
    ph, am = pac_signals(res.cleaned[:, ch], fs)
    phases = res.phases.set_index("phase")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.25})
    for p in ("baseline", "maint_early"):
        if p not in phases.index:
            continue
        a, b = int(phases.loc[p, "start_s"] * fs), int(phases.loc[p, "end_s"] * fs)
        c, h = phase_amplitude_histogram(ph[a:b], am[a:b])
        ax[0].plot(np.degrees(np.r_[c, c + 2 * np.pi]), np.r_[h, h], color=PHASE_COLORS[p], lw=1.8,
                   marker="o", ms=3, label=PHASE_LABELS[p])
    ax[0].axhline(1 / PARAMS["pac_nbins"], color=INK2, lw=0.8, ls="--")
    ax[0].set_xticks([-180, 0, 180, 360, 540]); ax[0].set_xlabel("Slow-wave phase (°; 0 = peak)")
    ax[0].set_ylabel("Normalised α amplitude"); ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("Phase-amplitude histogram", loc="left")
    # illustrative 8 s of maintenance EEG with slow wave and alpha envelope
    if "maint_early" in phases.index:
        from scipy.signal import butter, hilbert, sosfiltfilt
        y = res.cleaned[:, ch]
        s = int((phases.loc["maint_early", "start_s"] + 300) * fs)
        while not np.isfinite(y[s:s + 8 * fs]).all():
            s += fs
        seg = y[s - 10 * fs:s + 18 * fs]
        slow = sosfiltfilt(butter(2, PARAMS["pac_phase_band"], btype="band", fs=fs, output="sos"), seg)
        alpha = sosfiltfilt(butter(2, PARAMS["pac_amp_band"], btype="band", fs=fs, output="sos"), seg)
        env = np.abs(hilbert(alpha))
        k = slice(10 * fs, 18 * fs)
        tt = np.arange(8 * fs) / fs
        ax[1].plot(tt, seg[k], color=RAW, lw=0.5, label="cleaned EEG")
        ax[1].plot(tt, slow[k], color=PHASE_COLORS["maint_early"], lw=1.6, label="slow wave 0.5-4 Hz")
        ax[1].plot(tt, 3 * env[k] - 90, color=PHASE_COLORS["induction"], lw=1.4, label="α envelope ×3 (offset)")
        ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("µV")
        ax[1].legend(fontsize=8, loc="upper right", ncol=3)
        ax[1].set_title("Early maintenance: slow wave and α envelope (8 s)", loc="left")
    fig.savefig(path); plt.close(fig)
