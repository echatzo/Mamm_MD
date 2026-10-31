"""DDSM optical-density calibration: raw IRMA/DDSM scanner values -> CBIS-DDSM intensities.

CBIS-DDSM (Lee et al., Sci. Data 2017) converted the DDSM films with the ddsm_tools code
(github.com/fjeg/ddsm_tools): per-scanner optical density (OD) calibration, clipping to
[0.05, 3.0] OD, and a fixed linear map to 16 bits. This module reproduces that pipeline so
that the views and exams taken from the IRMA release of DDSM have exactly the CBIS-DDSM
intensities. On 53 matched CBIS-DDSM / IRMA image pairs covering the four scanners the output
equals the CBIS-DDSM pixel values exactly.

    OD    = scanner-specific calibration of the raw grey level (constants from ddsm_tools)
    OD    = clip(OD, 0.05, 3.0)
    pixel = interp(OD, (0.05, 3.0), (65535, 0)),   background (OD = 3.0) set to 0

Scanners: DBA (MGH, log formula), Howtek (MGH), Howtek (ISMD), Lumisys (WFU). The scanner of
each DDSM case is given in annotations.json (`scanner`); it was resolved from the case letter
(B, C -> Lumisys, D -> Howtek ISMD) and, for letter A, from the DIGITIZER field of the DDSM
.ics files (DBA or Howtek MGH).

Some IRMA files store the raw values left-shifted by a power of two. The shift is detected
exactly (all pixels are multiples of 2^k) and undone before calibration.
"""
from __future__ import annotations

import cv2
import numpy as np

SCANNERS = ("dba", "howtek_mgh", "howtek_ismd", "lumisys")

# Raw grey level at OD = 3.0 for each storage shift observed in the release.
_DBA_LEVELS = [37, 75, 150]
_LINEAR = {  # OD = a * GL + b, constants copied from ddsm_tools ddsm_classes._od_correct
    "howtek_mgh": (-0.00094568, 3.789, [834, 13352]),
    "howtek_ismd": (-0.00099055807612, 3.96604095240593, [975, 1951, 3902, 15608]),
    "lumisys": (-1 / 1009.01, 4096.99 / 1009.01, [1070, 8564, 17128]),
}


def _detect_shift(raw: np.ndarray, levels: list[int]) -> int:
    """Factor by which raw values are left-shifted (1 if none)."""
    base = min(levels)
    v = raw[raw > 0].astype(np.int64)
    if v.size == 0:
        return 1
    for lv in sorted(levels, reverse=True):
        f = round(lv / base)
        if f > 1 and np.mean(v % f == 0) > 0.99:
            return f
    return 1


def optical_density(raw: np.ndarray, scanner: str) -> np.ndarray:
    """Calibrated OD clipped to [0.05, 3.0]."""
    raw = raw.astype(np.float64)
    if scanner == "dba":
        od = (np.log10(raw / _detect_shift(raw, _DBA_LEVELS) + 1) - 4.80662) / -1.07553
    else:
        a, b, levels = _LINEAR[scanner]
        od = a * (raw / _detect_shift(raw, levels)) + b
    return np.clip(od, 0.05, 3.0)


# ---------------------------------------------------------------- fallback (never used in v1.0)
def _robust01(img, lo_p=1.0, hi_p=99.0):
    lo, hi = np.percentile(img, [lo_p, hi_p])
    hi = lo + 1.0 if hi <= lo else hi
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _otsu_breast_mask(raw: np.ndarray, work_h: int = 640) -> np.ndarray:
    """Otsu breast mask (largest component, preferring the one touching a side edge)."""
    small = cv2.resize(_robust01(raw), (max(1, int(round(raw.shape[1] * work_h / raw.shape[0]))), work_h),
                       interpolation=cv2.INTER_AREA)
    u8 = (small * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (7, 7), 2)
    lo, hi = np.percentile(blur, [20, 90])
    blur = ((np.clip(blur, lo, hi) - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thr = cv2.morphologyEx(cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    h, w = small.shape
    best, best_s = 0, 0.0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.01 * h * w:
            continue
        x0 = st[i, cv2.CC_STAT_LEFT]; x1 = x0 + st[i, cv2.CC_STAT_WIDTH]
        s = st[i, cv2.CC_STAT_AREA] * (2.0 if (x0 <= 1 or x1 >= w - 1) else 1.0)
        if s > best_s:
            best, best_s = i, s
    m = np.ones((h, w), bool) if best == 0 else lab == best
    return cv2.resize(m.astype(np.uint8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR) > 0


def _fallback(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = _otsu_breast_mask(raw)
    fg = raw[mask]
    if fg.size < 100:
        return np.zeros(raw.shape, np.uint16), mask
    lo, hi = np.percentile(fg, [0.5, 99.5])
    out = (np.clip((raw.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 65535).astype(np.uint16)
    out[~mask] = 0
    return out, mask


def calibrate(raw: np.ndarray, scanner: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Raw IRMA image (2-D, bright = tissue) -> (uint16 CBIS-DDSM-style image, breast mask).

    Falls back to a percentile stretch over an Otsu mask when the scanner is unknown or the
    calibrated mask is implausible (< 3% or > 95% of the image). No image of the released
    dataset takes this path; it is kept so that the conversion never fails silently.
    """
    if scanner is None:
        return _fallback(raw)
    od = optical_density(raw, scanner)
    mask = od < 3.0
    if not 0.03 <= mask.mean() <= 0.95:
        return _fallback(raw)
    out = np.clip(np.interp(od, (0.05, 3.0), (65535, 0)), 0, 65535).astype(np.uint16)
    out[~mask] = 0
    return out, mask
