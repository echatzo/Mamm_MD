"""Breast geometry: nipple-to-chest-wall depth.

Every view carries a nipple position N and a unit axis a pointing from the nipple towards the
chest wall (perpendicular to the pectoral line in MLO, to the image edge in CC). The depth of a
pixel p is its distance to the nipple along that axis, in millimetres:

    d(p) = <p - N, a> * mm_per_px

This is the posterior-nipple-line coordinate used by radiologists to relate a finding seen in CC
to the same finding in MLO: the anatomical prior of the paper is d_MLO = d_CC (sigma = 10 mm).
The plane is affine in the pixel coordinates, so only N, a and mm_per_px are stored.
"""
from __future__ import annotations

import cv2
import numpy as np

# The nipple must lie on (or within a few mm of) the breast mask, otherwise the axis is
# discarded and the view is left unpaired. Measured on CBIS-DDSM-IRMA: median 3.1 mm, p95
# 10.6 mm, then a gap to p99 29.9 mm; 15 mm flags about 2% of the views.
NIPPLE_TO_MASK_MAX_MM = 15.0


def depth_plane(shape: tuple[int, int], nx: float, ny: float, ax: float, ay: float,
                mm_per_px: float) -> np.ndarray:
    """Depth map in mm over an (H, W) grid."""
    h, w = shape
    xs = (np.arange(w, dtype=np.float32) - float(nx))[None, :]
    ys = (np.arange(h, dtype=np.float32) - float(ny))[:, None]
    return (xs * float(ax) + ys * float(ay)) * float(mm_per_px)


def depth_at_boxes(depth: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Depth at the centre pixel of each (x1, y1, x2, y2) box."""
    h, w = depth.shape
    out = np.zeros(len(boxes), np.float32)
    for i, b in enumerate(boxes):
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        out[i] = depth[min(max(cy, 0), h - 1), min(max(cx, 0), w - 1)]
    return out


def mm_per_px_of(depth: np.ndarray) -> float:
    """Pixel size read back from a depth plane (|grad d|): it follows any resize of the image."""
    if depth.shape[0] < 2 or depth.shape[1] < 2:
        return 0.0
    dy = float(np.median(depth[1:, :] - depth[:-1, :]))
    dx = float(np.median(depth[:, 1:] - depth[:, :-1]))
    return float(np.hypot(dx, dy))


def breast_mask(img2d: np.ndarray, content_w: int) -> np.ndarray:
    """Otsu threshold + largest connected component; columns beyond `content_w` are padding."""
    out = np.zeros(img2d.shape, dtype=bool)
    w = max(1, min(int(content_w), img2d.shape[1]))
    x = img2d[:, :w].astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    _, th = cv2.threshold((x * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    out[:, :w] = True if n <= 1 else (lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))))
    return out


def nipple_on_mask(mask: np.ndarray, nx: float, ny: float, mm_per_px: float,
                   max_mm: float = NIPPLE_TO_MASK_MAX_MM) -> bool:
    if not mask.any():
        return False
    ys, xs = np.nonzero(mask)
    return float(np.hypot(xs - nx, ys - ny).min()) * float(mm_per_px) <= max_mm
