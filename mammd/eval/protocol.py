"""Evaluation protocol: which views are scored, and how lesions count.

Rule 1  Only paired views (CC and MLO of the same breast) are evaluated; single views carry no
        cross-view information and would only dilute the comparison.
Rule 2  The unannotated view of a positive breast is a "support" view: it is used as a partner
        but is not scored, because CBIS-DDSM annotates images, not breasts (the other view very
        likely shows the same mass without a box).
"""
from __future__ import annotations

import numpy as np

from .froc import match_view, large


def subset(recs: list, keep: list) -> list:
    """Sub-list of views with partner indices remapped (-1 when the partner is dropped)."""
    remap = {o: n for n, o in enumerate(keep)}
    out = []
    for o in keep:
        r = dict(recs[o])
        r["partner"] = remap.get(r["partner"], -1) if r["partner"] >= 0 else -1
        out.append(r)
    return out


def paired(recs: list) -> list:
    return subset(recs, [i for i, r in enumerate(recs) if r["partner"] >= 0])


def _positive_views(recs: list) -> set:
    out, i = set(), 0
    while i < len(recs):
        n = 2 if (i + 1 < len(recs) and recs[i]["partner"] == i + 1) else 1
        if any(len(recs[j]["gt"]) for j in range(i, i + n)):
            out.update(range(i, i + n))
        i += n
    return out


def evaluation_views(raw: list) -> tuple[list, list]:
    """(views, targets): paired views and the indices of the scored ones."""
    p = paired(raw)
    pos = _positive_views(p)
    return p, [i for i, r in enumerate(p) if not (i in pos and len(r["gt"]) == 0)]


def lesion_status(recs: list, targets: list, view_birads: dict) -> dict:
    """Per-lesion status for the strict convention, from the BI-RADS of each view's masses.

    view_birads: {view id: list of BI-RADS of its masses}. A lesion is a target if all masses of
    its view are BI-RADS >= 3, excluded (background) if all are <= 2, ambiguous otherwise or when
    the BI-RADS is unknown."""
    st = {}
    for i in targets:
        n, a = len(recs[i]["gt"]), view_birads.get(recs[i]["id"])
        st[i] = ([] if not n else ["ambig"] * n if not a else ["target"] * n if min(a) >= 3
                 else ["excl"] * n if max(a) <= 2 else ["ambig"] * n)
    return st


def per_view(recs: list, targets: list, scores: list, status: dict | None = None) -> dict:
    """{view index: (TP scores, FP scores, n targets)}. status=None: every mass is a target."""
    out = {}
    for i in targets:
        r = recs[i]
        if status is None:
            out[i] = match_view(r["boxes"], scores[i], r["gt"], r["labels"], r["gt_labels"])
        else:
            keep = [j for j, x in enumerate(status[i]) if x != "excl"]
            out[i] = match_view(r["boxes"], scores[i], r["gt"][keep], status=[status[i][j] for j in keep])
    return out


def large_of(pv: dict, idx) -> float:
    return large([pv[i] for i in idx])


def paired_bootstrap(pv_a: dict, pv_b: dict, recs: list, targets: list, n: int = 1000, seed: int = 0) -> dict:
    """LARGE(a) - LARGE(b) with a 95% CI and a two-sided p-value, resampling breasts (the CC and
    MLO views of a breast are drawn together), `n` resamples."""
    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed
    breasts = {}
    for i in targets:
        p = recs[i]["partner"]
        breasts.setdefault(min(i, p) if p is not None and p >= 0 else i, []).append(i)
    B = list(breasts.values())
    d0 = large_of(pv_a, targets) - large_of(pv_b, targets)
    D = []
    for _ in range(n):
        idx = [i for b in rng.integers(0, len(B), len(B)) for i in B[b]]
        try:
            D.append(large_of(pv_a, idx) - large_of(pv_b, idx))
        except ValueError:
            pass
    D = np.array(D)
    D = D[~np.isnan(D)]
    return dict(delta=d0, ci=(float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))),
                p=float(min(1.0, 2 * min((D <= 0).mean(), (D >= 0).mean()))), n_breasts=len(B))


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni: {name: (adjusted p, rejected)}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        out[name] = (running, running <= alpha)
    return out
