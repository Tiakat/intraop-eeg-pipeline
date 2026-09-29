"""
Run the whole pipeline on a SYNTHETIC recording and write the figures to figures/.

    python examples/run_demo.py

1. simulate a 103-min intraoperative case (2 frontal channels, 128 Hz, with artifacts)
2. write it in the BIS Vista binary format and read it back with the pure-Python reader
3. filter, reject artifacts, compute spectral / aperiodic / PAC metrics
4. assign anesthetic phases from clinical landmarks and summarise per phase
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from eegpipe import process_recording                                   # noqa: E402
from eegpipe.bis_vista import read_raw_eeg, write_synthetic_vista        # noqa: E402
from eegpipe.plots import (fig_evolution, fig_overview, fig_pac,         # noqa: E402
                           fig_spectra, fig_traces_by_phase, fig_zoom_artifact)
from eegpipe.preprocessing import REJECT_CODES                           # noqa: E402
from eegpipe.simulate import simulate_case                               # noqa: E402

FIG = ROOT / "figures"


def main():
    FIG.mkdir(exist_ok=True)
    eeg, ad, landmarks = simulate_case(seed=1)

    # round trip through the device file format (gap samples written as 0, restored below)
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "L0101000000"
        write_synthetic_vista(base, np.nan_to_num(eeg), fs=128, gain=0.05)
        uv, raw_ad, hdr = read_raw_eeg(base)
    uv = uv.astype(np.float64)
    uv[np.isnan(eeg)] = np.nan
    print(f"Read back {uv.shape[0]} samples x {hdr.n_channels} channels at {hdr.fs} Hz; "
          f"max |difference| vs simulated = {np.nanmax(np.abs(uv - eeg)):.3f} uV")

    res = process_recording(uv, landmarks, fs=hdr.fs, raw_ad=raw_ad, seed=1)

    rj = res.rejection.groupby("channel")[REJECT_CODES + ["reject"]].mean() * 100
    print("\nRejected epochs (% of 4-s epochs) by criterion:\n", rj.round(1).to_string())
    print("\nPhases:\n", res.phases.to_string(index=False))
    cols = ["mean_delta_rel", "mean_theta_rel", "mean_alpha_rel", "mean_beta_rel",
            "mean_alpha_cf", "mean_ap_exponent", "pac_z"]
    print("\nPer-phase medians:\n", res.phase_summary()[cols].round(3).to_string())

    fig_overview(res, uv, FIG / "1_raw_vs_cleaned.png")
    fig_zoom_artifact(res, uv, 1240, 40, FIG / "2_artifact_zoom.png",
                      title="Electrocautery burst: saturated epochs removed, surrounding EEG kept")
    fig_traces_by_phase(res, FIG / "3_traces_by_phase.png")
    fig_spectra(res, FIG / "4_spectra_by_phase.png")
    fig_evolution(res, FIG / "5_metric_evolution.png")
    fig_pac(res, FIG / "6_phase_amplitude_coupling.png")
    print(f"\nFigures written to {FIG}")


if __name__ == "__main__":
    main()
