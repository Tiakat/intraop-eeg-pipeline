"""Unit tests of every processing step, on synthetic signals with a known answer."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from eegpipe.bis_vista import read_raw_eeg, write_synthetic_vista      # noqa: E402
from eegpipe.pac import pac_signals, pac_window, tort_mi                # noqa: E402
from eegpipe.params import PARAMS                                       # noqa: E402
from eegpipe.phases import define_phases, phase_of                      # noqa: E402
from eegpipe.simulate import _bandlimited                              # noqa: E402
from eegpipe.preprocessing import clean_recording, filter_eeg, reject_epochs  # noqa: E402
from eegpipe.spectral import band_power, epoch_psd, fit_fooof, peak_frequency  # noqa: E402
from eegpipe.stats import cliffs_delta, fit_mixed, rank_biserial_paired  # noqa: E402

FS = 128
RNG = np.random.default_rng(0)


def sine(f, dur=60, amp=10.0):
    t = np.arange(int(dur * FS)) / FS
    return amp * np.sin(2 * np.pi * f * t)


# ----------------------------------------------------------------------------- reader
def test_vista_roundtrip_is_exact(tmp_path):
    uv = np.round(RNG.normal(0, 30, (FS * 20, 2)) / 0.05) * 0.05      # on the 0.05-uV grid
    write_synthetic_vista(tmp_path / "L0000", uv, fs=FS, gain=0.05, offset=-3234)
    back, ad, hdr = read_raw_eeg(tmp_path / "L0000")
    assert hdr.n_channels == 2 and hdr.fs == FS
    assert np.max(np.abs(back - uv)) < 1e-3


# ----------------------------------------------------------------------------- filter
def test_filter_keeps_passband_and_removes_mains_and_drift():
    y, _ = filter_eeg(sine(10), FS)
    core = slice(5 * FS, -5 * FS)
    assert abs(np.nanstd(y[core]) / np.std(sine(10)[core]) - 1) < 0.01          # 10 Hz untouched
    y60, _ = filter_eeg(sine(60), FS)
    assert np.nanstd(y60[core]) < 0.01 * np.std(sine(60))                       # > 40 dB at 60 Hz
    y01, _ = filter_eeg(sine(0.05), FS)
    assert np.nanstd(y01[core]) < 0.05 * np.std(sine(0.05))                     # drift removed


def test_filter_never_crosses_a_gap():
    x = sine(10, 120)
    x[60 * FS:62 * FS] = np.nan
    y, edge = filter_eeg(x, FS)
    assert np.isnan(y[60 * FS:62 * FS]).all()
    assert edge[60 * FS - 1] and edge[62 * FS]                                   # transients flagged


# ----------------------------------------------------------------------------- rejection
def test_each_rejection_criterion_fires_on_its_artifact():
    x = RNG.normal(0, 15, FS * 600)
    ad = np.round(x / 0.05)
    L = int(PARAMS["epoch_s"] * FS)
    x[100 * L + 50] += 400                                    # amplitude spike in epoch 100
    x[40 * L:41 * L] = 0.0                                    # flat epoch 40
    ad[70 * L + 10] = 32767                                   # saturation in epoch 70
    y, edge = filter_eeg(x, FS)
    r = reject_epochs(y, edge, ad, None, FS).set_index("epoch")
    assert r.loc[100, "amplitude"] and r.loc[40, "flat"] and r.loc[70, "saturation"]
    assert r["reject"].mean() < 0.05                          # clean signal otherwise kept


def test_impedance_check_is_shared_across_channels():
    x = RNG.normal(0, 15, (FS * 300, 2))
    a = 100 * FS
    x[a:a + 6 * FS, 0] = RNG.normal(0, 0.03, 6 * FS)          # impedance test on channel 1 only
    _, cleaned, rej = clean_recording(x, FS)
    both = rej.groupby("epoch")["impedance_check"].all()
    assert both.any()
    assert np.isnan(cleaned[a + 2 * FS, 1])                   # channel 2 removed too


# ----------------------------------------------------------------------------- spectra
def test_band_power_matches_signal_variance():
    x = sine(10, 4, amp=20)[None, :]
    f, P = epoch_psd(x, FS)
    # Hann window: power of a pure tone recovered within 1 %
    assert abs(band_power(f, P, 0.5, 45)[0] - 20 ** 2 / 2) / (20 ** 2 / 2) < 0.01


def test_peak_frequency_and_edge_rule():
    f = np.arange(0, 64.25, 0.25)
    psd = np.exp(-0.5 * ((f - 10.5) / 0.7) ** 2) + 1 / (f + 1)
    assert peak_frequency(f, psd, 8, 13) == 10.5
    assert np.isnan(peak_frequency(f, 1 / (f + 1), 0.5, 4))  # monotonic -> max on edge -> NaN


def test_fooof_recovers_aperiodic_exponent():
    f = np.arange(0.25, 64, 0.25)
    psd = 10 ** 2 * f ** -2.3 * (1 + 3 * np.exp(-0.5 * ((f - 10.5) / 1.0) ** 2))
    r = fit_fooof(f, psd)
    assert abs(r["ap_exponent"] - 2.3) < 0.1
    assert abs(r["alpha_cf"] - 10.5) < 0.3


# ----------------------------------------------------------------------------- PAC
def _coupled(k, dur=300, seed=0):
    """Non-periodic slow wave (a pure sine would make circular-shift surrogates meaningless)."""
    rng = np.random.default_rng(seed)
    n = dur * FS
    slow = _bandlimited(n, FS, 0.5, 1.5, rng)
    alpha = _bandlimited(n, FS, 9.5, 11.5, rng)
    return 30 * slow + 5 * (1 + k * slow / np.abs(slow).max()) * alpha + rng.normal(0, 3, n)


def test_tort_mi_detects_coupling_and_not_its_absence():
    rng = np.random.default_rng(1)
    zs = {}
    for k in (0.0, 0.6):
        ph, am = pac_signals(_coupled(k), FS)
        zs[k] = pac_window(ph[: 128 * FS], am[: 128 * FS], rng, FS)["pac_z"]
    assert zs[0.6] > 10 and abs(zs[0.0]) < 3


def test_pac_never_uses_rejected_samples():
    y = _coupled(0.5).astype(float)
    y[100 * FS:110 * FS] = np.nan
    ph, am = pac_signals(y, FS)
    margin = int(PARAMS["pac_margin_s"] * FS)
    assert np.isnan(ph[100 * FS - margin:110 * FS + margin]).all()


def test_tort_mi_uniform_is_zero():
    ph = np.linspace(-np.pi, np.pi, 18000, endpoint=False)
    assert tort_mi(ph, np.ones_like(ph))[0] < 1e-12


# ----------------------------------------------------------------------------- phases
def test_phases_from_landmarks():
    lm = {"drug_start": 560, "induction": 600, "intubation": 780, "drug_stop": 4800,
          "surgery_end": 5350, "propofol_stop": 5400, "extubation": 5900}
    ph = define_phases(lm, 6200).set_index("phase")
    assert list(ph.index) == ["baseline", "induction", "maint_early", "maint_late", "post_infusion", "emergence"]
    assert ph.loc["baseline", "end_s"] == 530 and ph.loc["baseline", "start_s"] == 230
    assert ph.loc["maint_early", "end_s"] - ph.loc["maint_early", "start_s"] == 1800
    assert phase_of(np.array([100, 2000, 5500]), ph.reset_index()).tolist() == ["other", "maint_early", "emergence"]


# ----------------------------------------------------------------------------- statistics
def test_effect_sizes_are_bounded():
    assert rank_biserial_paired([1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0


def test_fit_mixed_recovers_fixed_effect():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(30):
        u = rng.normal(0, 1.0)
        for ph, beta in (("a", 0.0), ("b", 0.8)):
            for _ in range(3):
                rows.append({"patient": p, "phase": ph, "y": beta + u + rng.normal(0, 0.5)})
    fit, note = fit_mixed("y ~ C(phase)", pd.DataFrame(rows))
    assert note.startswith("mixed")
    assert abs(fit.params["C(phase)[T.b]"] - 0.8) < 0.15


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
