"""
Reader for the raw EEG exported by a BIS Vista monitor, written in pure Python
(no vendor MATLAB runtime needed).

File layout, reverse-engineered and validated sample-by-sample against the
vendor's MATLAB reader (maximum difference 0 uV):

  .r2a / .r4a   raw EEG, little-endian int16, channels interleaved (2 or 4)
                uV = (AD - offset) * gain
  .h_a          2048-byte binary header
                  @184 int16  number of channels
                  @186 int16  sampling rate (Hz)
                  @702 4 x float32  gain   (uV per A/D unit)
                  @766 4 x float32  offset (A/D units)
  .spa          processed variables at 1 Hz ('|'-separated text, 2 header lines)

Time anchoring: the .r2a holds exactly `fs` samples per row of the .spa file, so
second k of the raw EEG is time-stamped by row k of the .spa. The time stamp in
the .h_a header is the time of the *export*, not of the recording.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HEADER_SIZE = 2048
SPA_SENTINELS = (-3276.8, -327.7, -32768, -3276, -327)   # "no value" codes in the .spa


@dataclass
class VistaHeader:
    n_channels: int
    fs: int
    gain: np.ndarray      # uV per A/D unit, one per channel
    offset: np.ndarray    # A/D units, one per channel


def read_header(h_a: str | Path) -> VistaHeader:
    b = Path(h_a).read_bytes()
    nch, fs = struct.unpack_from("<hh", b, 184)
    gain = np.array(struct.unpack_from("<4f", b, 702)[:nch], dtype=np.float64)
    offset = np.array(struct.unpack_from("<4f", b, 766)[:nch], dtype=np.float64)
    return VistaHeader(int(nch), int(fs), gain, offset)


def read_raw_ad(raw: str | Path, n_channels: int) -> np.ndarray:
    """Raw A/D values, shape (n_samples, n_channels), int16."""
    ad = np.fromfile(raw, dtype="<i2")
    n = len(ad) // n_channels
    return ad[: n * n_channels].reshape(n, n_channels)


def read_raw_eeg(base: str | Path) -> tuple[np.ndarray, np.ndarray, VistaHeader]:
    """
    Read `<base>.h_a` + `<base>.r2a` (or .r4a).
    Returns (EEG in uV float32, raw A/D int16, header).
    The raw A/D values are kept because saturation is detected on them.
    """
    base = Path(base)
    hdr = read_header(base.with_name(base.name + ".h_a"))
    raw = next((base.with_name(base.name + e) for e in (".r2a", ".r4a", ".r1a")
                if base.with_name(base.name + e).exists()), None)
    if raw is None:
        raise FileNotFoundError(f"no .r2a/.r4a file next to {base}")
    ad = read_raw_ad(raw, hdr.n_channels)
    uv = ((ad.astype(np.float64) - hdr.offset) * hdr.gain).astype(np.float32)
    return uv, ad, hdr


def read_spa(spa: str | Path) -> pd.DataFrame:
    """
    1-Hz processed variables (BIS, SQI, EMG, suppression ratio...).
    The first two lines hold the channel group and the variable code; they are
    merged into column names such as 'BIS_C12' or 'SQI_C1'. Sentinel values -> NaN.
    """
    lines = Path(spa).read_text(errors="replace").splitlines()
    groups = [g.strip() for g in lines[0].split("|")]
    names = [v.strip() for v in lines[1].split("|")]
    cols, g_prev = [], ""
    for g, v in zip(groups, names):
        g_prev = g or g_prev
        cols.append(f"{v}_{g_prev.replace(' ', '')}" if g_prev else v)
    rows = [r.split("|") for r in lines[2:] if r.strip()]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])] if rows else cols)
    time_col = df.columns[0]
    out = df.drop(columns=[time_col]).apply(pd.to_numeric, errors="coerce")
    out = out.mask(out.isin(SPA_SENTINELS))
    out.insert(0, "time", df[time_col].str.strip())
    return out


def write_synthetic_vista(base: str | Path, uv: np.ndarray, fs: int = 128,
                          gain: float = 0.05, offset: float = 0.0) -> None:
    """
    Write a minimal .h_a/.r2a pair from a signal in uV (used by the tests and the
    demo to exercise the reader without any real recording).
    """
    base = Path(base)
    nch = uv.shape[1]
    hdr = bytearray(HEADER_SIZE)
    struct.pack_into("<hh", hdr, 184, nch, fs)
    struct.pack_into("<4f", hdr, 702, *([gain] * nch + [0.0] * (4 - nch)))
    struct.pack_into("<4f", hdr, 766, *([offset] * nch + [0.0] * (4 - nch)))
    base.with_name(base.name + ".h_a").write_bytes(bytes(hdr))
    ad = np.clip(np.round(uv / gain + offset), -32768, 32767).astype("<i2")
    ad.tofile(base.with_name(base.name + ".r2a"))
