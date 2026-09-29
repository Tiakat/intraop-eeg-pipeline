"""
Processing parameters.

Every constant used by the pipeline lives here so that the methods section of a
paper can be written directly from this file.
"""

PARAMS = {
    "fs": 128,                        # Hz, native sampling rate of the BIS Vista raw EEG

    # --- filtering (zero phase) -----------------------------------------------------
    "hp_hz": 0.3, "lp_hz": 45.0,      # Butterworth band-pass (-0.14 dB at 0.5 Hz)
    "filter_order": 4,                # applied forward-backward with sosfiltfilt
    "notch_hz": 60.0, "notch_q": 30.0,  # mains notch (use 50 Hz outside North America)
    "edge_s": 2.0,                    # seconds discarded at each edge of a continuous segment
    "min_segment_s": 12.0,            # shorter segments are not filtered (marked as edge)

    # --- epoching / artifact rejection ------------------------------------------------
    "epoch_s": 4.0,
    "abs_max_uv": 200.0,              # |filtered EEG| > 200 uV            -> amplitude artifact
    "flat_sd_uv": 1.0,                # epoch SD < 1 uV                     -> flat signal
    "flat_diff_uv": 0.2,              # |x[i+1] - x[i]| < 0.2 uV = "still" sample
    "flat_frac_max": 0.3,             # > 30 % still samples                -> impedance check / flat line
    "saturation_ad": 32000,           # |raw A/D value| >= 32000            -> amplifier saturation
    "robust_z": 4.0,                  # local robust z of log(SD) and log(max|dx|)
    "robust_z_half_window_s": 300.0,  # median / MAD computed over +/- 5 min around each epoch
    "sqi_min": 50.0,                  # monitor signal-quality index < 50   -> reject

    # --- spectral metrics ---------------------------------------------------------------
    "window_s": 32.0,                 # metrics aggregated over 8 epochs of 4 s
    "min_clean_frac": 0.5,            # >= 50 % clean epochs needed in a window
    "bands": {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
              "beta": (13.0, 30.0), "gamma": (30.0, 45.0)},
    "total_band": (0.5, 45.0),

    # --- spectral parameterisation (FOOOF / specparam) ----------------------------------
    "fooof_range": (1.0, 40.0),
    "fooof_settings": dict(peak_width_limits=(1.0, 8.0), max_n_peaks=4,
                           min_peak_height=0.1, peak_threshold=2.0,
                           aperiodic_mode="fixed", verbose=False),
    # peak search bands: alpha widened to 7-14 Hz because the propofol alpha peak can
    # drift below 8 Hz; theta limited to 4-7 Hz so the alpha peak is not counted twice
    "fooof_peak_bands": {"theta": (4.0, 7.0), "alpha": (7.0, 14.0), "beta": (14.0, 30.0)},

    # --- phase-amplitude coupling ----------------------------------------------------------
    "pac_window_s": 128.0,            # 32 epochs of 4 s
    "pac_phase_band": (0.5, 4.0),     # slow / delta phase
    "pac_amp_band": (8.0, 13.0),      # alpha amplitude
    "pac_filter_order": 2,
    "pac_nbins": 18,
    "pac_n_surrogates": 200,
    "pac_min_clean_s": 60.0,          # >= 60 s of clean signal per 128-s window
    "pac_margin_s": 2.0,              # filter / Hilbert transients removed at segment edges
    "pac_min_segment_s": 20.0,        # clean segments shorter than this are ignored (>= 10 cycles at 0.5 Hz)
}

PHASE_ORDER = ["baseline", "induction", "maint_early", "maint_late", "post_infusion", "emergence"]
