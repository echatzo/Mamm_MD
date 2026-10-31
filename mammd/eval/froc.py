"""FROC with the paper's matching rule.

A detection hits a ground-truth box when IoU >= 0.2 or when the centre of the ground-truth box
lies inside the detection. Detections are visited in decreasing score order: the first hit on an
undetected lesion is a true positive, a hit on an already detected lesion is ignored, a detection
that hits nothing is a false positive. Sensitivity is read at the FPI grid point closest to each
target; the summary metric of the paper (LARGE) is the mean sensitivity on WIDE_GRID.

BI-RADS conventions (`kind`):
  "all"      every ground-truth mass is a target
  "strict"   masses with BI-RADS <= 2 are background (a detection on them is a false positive):
             the annotation scope of VinDr-Mammo, used for VinDr models tested elsewhere.
             Masses of images mixing BI-RADS <= 2 and >= 3 are ambiguous (neither TP nor FP).
"""
from __future__ import annotations

import numpy as np

WIDE_GRID = (0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4)


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return float(inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def centre_inside(det, gt) -> bool:
    cx, cy = gt[0] + (gt[2] - gt[0]) / 2, gt[1] + (gt[3] - gt[1]) / 2
    return bool(det[0] <= cx <= det[2] and det[1] <= cy <= det[3])


def hits(det, gt) -> bool:
    return iou(gt, det) >= 0.2 or centre_inside(det, gt)


def match_view(boxes, scores, gt, labels=None, gt_labels=None, status=None):
    """(TP scores, FP scores, number of target lesions) for one view.

    `status`: None (every lesion is a target, labels must agree) or a list with one of
    "target" / "excl" / "ambig" per lesion (BI-RADS conventions; "excl" lesions must already be
    removed for the strict convention)."""
    s = np.asarray(scores, np.float64)
    order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
    order = order[~np.isnan(s[order])]                 # NaN = suppressed candidate
    done = np.zeros(len(gt), bool)
    tp, fp = [], []
    for d in order:
        if status is None:
            matched = False
            for j, g in enumerate(gt):
                if not matched and hits(boxes[d], g) and labels[d] == gt_labels[j]:
                    if not done[j]:
                        tp.append(float(s[d])); done[j] = True
                    matched = True
            if not matched:
                fp.append(float(s[d]))
        else:
            hit = next((j for j, g in enumerate(gt) if hits(boxes[d], g)), None)
            if hit is None:
                fp.append(float(s[d]))
            elif status[hit] == "target" and not done[hit]:
                tp.append(float(s[d])); done[hit] = True
    n = len(gt) if status is None else sum(x == "target" for x in status)
    return tp, fp, n


def froc_curve(per_view: list):
    """FPI and sensitivity at every distinct score threshold, from [(tp, fp, n), ...]."""
    fps = np.asarray([x for v in per_view for x in v[1]], np.float64)
    tps = np.asarray([x for v in per_view for x in v[0]], np.float64)
    thr = np.unique(np.concatenate([fps, tps]))
    n_fp = np.append(len(fps) - np.searchsorted(np.sort(fps), thr, side="left"), 0)
    n_tp = np.append(len(tps) - np.searchsorted(np.sort(tps), thr, side="left"), 0)
    n_les = float(sum(v[2] for v in per_view))
    fpi = n_fp.astype(np.float64) / float(max(len(per_view), 1))
    sens = n_tp.astype(np.float64) / n_les if n_les > 0 else np.full(len(n_tp), np.nan)
    return fpi, sens


def sensitivity_at(fpi, sens, grid=WIDE_GRID) -> dict:
    return {f"sens@{g}FPI": float(sens[int(np.argmin(np.abs(np.asarray(fpi) - g)))]) for g in grid}


def large(per_view: list, grid=WIDE_GRID) -> float:
    """Mean sensitivity on the wide FPI grid (the paper's summary metric)."""
    return float(np.mean(list(sensitivity_at(*froc_curve(per_view), grid).values())))


def froc_sensitivities(recs: list, scores: list, grid=WIDE_GRID) -> dict:
    """Sensitivities on `grid` over every view of a cache (used for model selection)."""
    pv = [match_view(r["boxes"], s, r["gt"], r["labels"], r["gt_labels"]) for r, s in zip(recs, scores)]
    return sensitivity_at(*froc_curve(pv), grid)
