"""
Synthetic intraoperative frontal EEG, used for the demo and the tests.

NOT real patient data. The generator reproduces the textbook features of a
propofol / remifentanil anesthetic so that every step of the pipeline has
something realistic to work on:

  awake baseline : low-amplitude EEG, beta activity, small alpha, eye blinks
  maintenance    : large slow waves (0.5-1.5 Hz), frontal alpha (~10.5 Hz) whose
                   amplitude is locked to slow-wave peaks (phase-amplitude coupling),
                   steeper aperiodic (1/f) slope
  emergence      : alpha fades, beta and muscle activity return, movement artifacts

plus the artifacts met in the operating room: electrocautery bursts that saturate
the amplifier, the monitor's periodic impedance check (flat line + regular spikes),
a disconnection (gap) and movement.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

# landmark times (s from recording start) of the synthetic case
LANDMARKS = {"drug_start": 560.0, "induction": 600.0, "intubation": 780.0,
             "drug_stop": 4800.0, "surgery_end": 5350.0, "propofol_stop": 5400.0,
             "extubation": 5900.0}
DURATION_S = 6200.0


def _colored_noise(n: int, fs: int, exponent: float, rng) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    spec /= np.maximum(f, 0.5) ** (exponent / 2)   # flat below 0.5 Hz
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def _bandlimited(n: int, fs: int, lo: float, hi: float, rng) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    spec *= np.exp(-0.5 * ((f - (lo + hi) / 2) / ((hi - lo) / 2.5)) ** 2)
    x = np.fft.irfft(spec, n)
    return x / x.std()


def _ramp(t, t0, t1):
    """0 before t0, 1 after t1, smooth (cosine) in between."""
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def simulate_case(fs: int = 128, seed: int = 1, gain: float = 0.05):
    """
    Returns
    -------
    eeg_uv   : (n, 2) float32, two frontal channels in uV (NaN during the gap)
    raw_ad   : (n, 2) int16-like A/D values (saturation visible here)
    landmarks: dict of landmark times in seconds
    """
    rng = np.random.default_rng(seed)
    n = int(DURATION_S * fs)
    t = np.arange(n) / fs
    L = LANDMARKS
    anes = _ramp(t, L["induction"], L["induction"] + 90) * (1 - _ramp(t, L["propofol_stop"], L["extubation"]))
    awake = 1 - anes
    arousal = _ramp(t, L["propofol_stop"] + 120, L["extubation"])          # emergence
    late = _ramp(t, 1500, L["drug_stop"])                                    # slow drift within maintenance

    chans = []
    shared = {k: None for k in ("ap_awake", "ap_anes", "slow", "alpha", "beta", "theta")}
    shared["ap_awake"] = _colored_noise(n, fs, 1.8, rng)
    shared["ap_anes"] = _colored_noise(n, fs, 2.6, rng)
    shared["slow"] = _bandlimited(n, fs, 0.5, 1.5, rng)
    shared["alpha"] = _bandlimited(n, fs, 9.8, 11.4, rng)
    shared["beta"] = _bandlimited(n, fs, 15, 25, rng)
    shared["theta"] = _bandlimited(n, fs, 4.5, 7, rng)
    slow_phase = np.angle(hilbert(shared["slow"]))

    for ch in range(2):
        mix = lambda k: 0.85 * shared[k] + 0.15 * (
            _colored_noise(n, fs, 1.8 if k == "ap_awake" else 2.6, rng) if k.startswith("ap")
            else _bandlimited(n, fs, *{"slow": (0.5, 1.5), "alpha": (9.8, 11.4), "beta": (15, 25),
                                       "theta": (4.5, 7)}[k], rng))
        coupling = anes * (0.14 - 0.09 * late)                                 # PAC strongest early
        alpha_env = 1 + coupling * np.cos(slow_phase)                        # alpha peaks on slow-wave peaks
        x = (awake * 8 * mix("ap_awake") + anes * 12 * mix("ap_anes")
             + anes * (1 - 0.3 * late) * 30 * mix("slow")
             + (1.5 * awake + anes * (8 + 1 * late)) * alpha_env * mix("alpha")
             + (5 * awake + 2 * arousal + 4 * anes) * mix("beta")
             + anes * 3.5 * mix("theta")
             + (awake * 1.5 + anes * 0.8 + arousal * 1.5) * rng.standard_normal(n))       # muscle / white noise
        chans.append(x)
    eeg = np.stack(chans, 1)

    # ---- artifacts ------------------------------------------------------------------
    # eye blinks while awake (before induction and after extubation) (mostly channel 1, the more frontal electrode)
    blinks = np.r_[np.arange(20, L["induction"], 5.5), np.arange(L["extubation"] + 20, DURATION_S - 5, 5.5)]
    for tb in blinks + rng.uniform(-1, 1, len(blinks)):
        i = int(tb * fs)
        w = np.exp(-0.5 * ((np.arange(-fs, fs)) / (0.12 * fs)) ** 2)
        a = rng.uniform(120, 260)
        eeg[max(0, i - fs):i + fs, 0] += a * w[: min(2 * fs, n - max(0, i - fs))]
        eeg[max(0, i - fs):i + fs, 1] += 0.5 * a * w[: min(2 * fs, n - max(0, i - fs))]
    # impedance check every 10 min: near-flat line with a spike every 0.5 s, both channels
    for ti in np.arange(300, DURATION_S, 600):
        a, b = int(ti * fs), int((ti + 6) * fs)
        eeg[a:b] = rng.normal(0, 0.03, (b - a, 2))
        eeg[a:b:fs // 2] += 40
    # electrocautery: broadband bursts that saturate the amplifier
    for tc in (1250, 1900, 2600, 3400, 4100):
        a, b = int(tc * fs), int((tc + rng.uniform(8, 20)) * fs)
        eeg[a:b] += rng.normal(0, 900, (b - a, 2)) * np.sin(2 * np.pi * 60 * t[a:b])[:, None] + rng.normal(0, 700, (b - a, 2))
    # emergence: slow movement drifts and muscle bursts
    for tm in (5650, 5760, 5840):
        a, b = int(tm * fs), int((tm + 6) * fs)
        eeg[a:b] += 350 * np.sin(np.linspace(0, np.pi, b - a))[:, None] + rng.normal(0, 60, (b - a, 2))

    # ---- acquisition: quantisation and saturation of the 16-bit converter --------------
    raw_ad = np.clip(np.round(eeg / gain), -32767, 32767)
    eeg_uv = (raw_ad * gain).astype(np.float32)
    # disconnection: 40 s without data
    a, b = int(2200 * fs), int(2240 * fs)
    eeg_uv[a:b] = np.nan
    raw_ad[a:b] = 0
    return eeg_uv, raw_ad.astype(np.int16), dict(L)
