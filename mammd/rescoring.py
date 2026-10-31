"""Training and inference of the MAMM-MD head on a candidate cache.

A candidate cache (see mammd/candidates.py) is a list with one record per view:
    boxes (K, 4), scores (K,), labels (K,)      detector output, 2048 frame
    depth (K,), w_mm (K,), h_mm (K,)            geometry of each candidate
    roi (K, 1024)                               appearance embedding of each candidate crop
    tp (K,) bool, gt (G, 4), gt_labels (G,)     labels for training / evaluation
    is_mlo, partner (index of the other view of the breast in the list, or -1), id
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .eval.froc import froc_sensitivities, WIDE_GRID
from .head import ANATOMICAL_PRIOR, MAMMMDHead, candidate_features

ALPHA = 0.5


# ------------------------------------------------------------------- geometric product baseline
def partner_support(recs: list, prior: dict = ANATOMICAL_PRIOR, band_sigmas: float = 2.0) -> list:
    """p_i = best partner score within `band_sigmas` sigma of the expected depth (0 if none).
    Views without a partner get p = 1, which leaves their score unchanged."""
    a, b, half = prior["slope"], prior["intercept"], band_sigmas * prior["sigma"]
    out = []
    for r in recs:
        p = r["partner"]
        if p < 0 or not len(r["scores"]):
            out.append(np.ones(len(r["scores"]), np.float32))
            continue
        rp, sp = recs[p]["depth"], recs[p]["scores"]
        v = np.zeros(len(r["scores"]), np.float32)
        for j, d in enumerate(r["depth"]):
            e = (d - b) / a if r["is_mlo"] else a * d + b
            m = np.abs(rp - e) <= half if len(rp) else np.zeros(0, bool)
            v[j] = float(sp[m].max()) if m.any() else 0.0
        out.append(v)
    return out


def product_scores(recs: list, support: list) -> list:
    """Geometric product baseline s * (alpha + (1 - alpha) * p)."""
    return [r["scores"] * (ALPHA + (1 - ALPHA) * v) for r, v in zip(recs, support)]


# ----------------------------------------------------------------------------------- batching
def pack(recs, support, order, max_cand=0, device="cpu", jitter=0.0, rng=None, use_roi=False, roi_dim=None):
    """Tensors for a list of views `order`, each with its partner view. Returns (batch, tp, selected)."""
    def take(i):
        r = recs[i]
        k = np.arange(len(r["scores"]))
        if max_cand and len(k) > max_cand:
            k = np.argsort(-r["scores"])[:max_cand]
            k.sort()
        return r, k

    K = max(len(take(i)[1]) for i in order) if order else 1
    P = 1
    for i in order:
        if recs[i]["partner"] >= 0:
            P = max(P, len(take(recs[i]["partner"])[1]))
    Bn = len(order)
    z = lambda *s: torch.zeros(*s, device=device)
    feats, dep, msk = z(Bn, K, 7), z(Bn, K), torch.zeros(Bn, K, dtype=torch.bool, device=device)
    pf, pd, pm = z(Bn, P, 7), z(Bn, P), torch.zeros(Bn, P, dtype=torch.bool, device=device)
    logs, bl, tp = z(Bn, K), z(Bn, K), z(Bn, K)
    is_mlo, has = z(Bn), z(Bn)
    if use_roi and roi_dim is None:
        roi_dim = next((r["roi"].shape[1] for r in recs if len(r.get("roi", []))), 0)
    roi = z(Bn, K, roi_dim) if use_roi else None
    p_roi = z(Bn, P, roi_dim) if use_roi else None
    t = lambda x: torch.as_tensor(x, device=device).float()
    sel = []
    for b, i in enumerate(order):
        r, k = take(i)
        n = len(k)
        sel.append(k)
        if n:
            d_own = r["depth"][k]
            if jitter and rng is not None:
                d_own = d_own + rng.normal(0.0, jitter, size=d_own.shape).astype(d_own.dtype)
            f = candidate_features(*[t(x).unsqueeze(0) for x in (r["scores"][k], d_own, r["w_mm"][k],
                                                                 r["h_mm"][k], support[i][k])])
            feats[b, :n] = f[0]
            dep[b, :n] = t(d_own)
            msk[b, :n] = True
            logs[b, :n] = torch.log(t(r["scores"][k]).clamp_min(1e-8))
            bl[b, :n] = t(support[i][k])
            tp[b, :n] = t(r["tp"][k])
            if use_roi:
                roi[b, :n] = t(np.ascontiguousarray(r["roi"][k].astype(np.float32)))
        is_mlo[b] = float(r["is_mlo"])
        p = r["partner"]
        if p >= 0:
            q, kq = take(p)
            nq = len(kq)
            has[b] = 1.0
            if nq:
                d_p = q["depth"][kq]
                if jitter and rng is not None:
                    d_p = d_p + rng.normal(0.0, jitter, size=d_p.shape).astype(d_p.dtype)
                pf[b, :nq] = candidate_features(*[t(x).unsqueeze(0) for x in (q["scores"][kq], d_p, q["w_mm"][kq],
                                                                             q["h_mm"][kq], support[p][kq])])[0]
                pd[b, :nq] = t(d_p)
                if use_roi:
                    p_roi[b, :nq] = t(np.ascontiguousarray(q["roi"][kq].astype(np.float32)))
                pm[b, :nq] = True
    out = dict(feats=feats, depth=dep, mask=msk, p_feats=pf, p_depth=pd, p_mask=pm,
               is_mlo=is_mlo, has_partner=has, log_score=logs, baseline=bl)
    if use_roi:
        out["roi"], out["p_roi"] = roi, p_roi
    return out, tp, sel


@torch.no_grad()
def predict(head: MAMMMDHead, recs: list, support: list, device="cpu", bs: int = 16) -> list:
    """Rescored candidates, exp(rank) (only the order matters for the FROC)."""
    head.eval()
    out = [np.asarray(r["scores"], np.float64).copy() for r in recs]
    for s in range(0, len(recs), bs):
        order = list(range(s, min(s + bs, len(recs))))
        batch, _, sel = pack(recs, support, order, 0, device, use_roi=head.uses_appearance)
        rank = head(**batch).float().cpu().numpy()
        for b, i in enumerate(order):
            if len(sel[b]):
                out[i][sel[b]] = np.exp(np.clip(rank[b, :len(sel[b])], -60, None))
    return out


def predict_ensemble(heads: list, recs: list, support: list, device="cpu") -> list:
    """Geometric mean of the rescored candidates of several heads (seeds)."""
    P = [[np.log(np.clip(x, 1e-12, None)) for x in predict(h, recs, support, device)] for h in heads]
    return [np.exp(np.mean([p[i] for p in P], axis=0)) for i in range(len(recs))]


# ----------------------------------------------------------------------------------- training
def rank_loss(rank, y, mask):
    """Pairwise logistic ranking loss over every (TP, FP) pair of the batch, across images."""
    m = mask.bool()
    r, yy = rank[m], y[m]
    pos, neg = r[yy > 0.5], r[yy < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return rank.sum() * 0.0
    return F.softplus(-(pos.unsqueeze(1) - neg.unsqueeze(0))).mean()


def batches(idx_pos, idx_neg, bs, pos_frac, rng):
    """Batches of `bs` views, round(bs * pos_frac) of them with at least one TP."""
    npos = max(1, int(round(bs * pos_frac)))
    nneg = max(1, bs - npos)
    P, N = rng.permutation(idx_pos), rng.permutation(idx_neg)
    for b in range(max(1, len(idx_pos) // npos)):
        a = list(P[b * npos:(b + 1) * npos])
        c = list(N[(b * nneg) % max(len(N), 1):][:nneg]) if len(N) else []
        if len(c) < nneg and len(N):
            c = list(rng.choice(N, nneg, replace=True))
        if a:
            yield a + c


def training_views(recs: list) -> list:
    """Paired views with candidates, except the unannotated view of a positive breast.

    CBIS-DDSM annotates images, not breasts: the other view of an annotated breast very likely
    shows the same mass without a box. It is used as a partner but never as a training example."""
    idx = [i for i, r in enumerate(recs) if r["partner"] >= 0 and len(r["scores"])]
    out = []
    for i in idx:
        p = recs[i]["partner"]
        in_positive = len(recs[i]["gt"]) or (p >= 0 and len(recs[p]["gt"]))
        if not (in_positive and not len(recs[i]["gt"])):
            out.append(i)
    return out


def train_head(train: list, val: list, appearance: bool, geometry: bool = True, pos_frac: float = 0.23,
               seed: int = 0, epochs: int = 40, batch_size: int = 16, lr: float = 1e-3, wd: float = 1e-4,
               max_cand: int = 100, jitter_depth_mm: float = 2.0, delta_l2: float = 0.01,
               prior: dict = ANATOMICAL_PRIOR, device="cuda", log=print):
    """Train one head. Model selection on the validation cache: the mean sensitivity on the wide
    FPI grid (0.025 - 4 FPI) smoothed over 3 epochs; the kept weights are the average of the
    3 epochs of the best window. Returns (head, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    s_tr, s_va = partner_support(train, prior), partner_support(val, prior)
    roi_dim = 0
    if appearance:
        dims = {r["roi"].shape[1] for r in train if len(r.get("roi", []))}
        assert len(dims) == 1, f"inconsistent appearance dimensions {dims}"
        roi_dim = dims.pop()
    head = MAMMMDHead(roi_dim=roi_dim, prior=prior, geometry=geometry).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    idx = training_views(train)
    pos = [i for i in idx if train[i]["tp"].any()]
    neg = [i for i in idx if not train[i]["tp"].any()]
    log(f"train {len(train)} views ({len(idx)} used: {len(pos)} with a TP), val {len(val)} views, "
        f"{sum(q.numel() for q in head.parameters())} parameters")
    rng = np.random.default_rng(seed)
    K = 3
    best, best_state, best_ep, hist_s, hist_w, history = -1.0, None, None, [], [], []
    for ep in range(1, epochs + 1):
        head.train()
        tot = nb = 0.0
        t0 = time.time()
        for order in batches(pos, neg, batch_size, pos_frac, rng):
            batch, y, _ = pack(train, s_tr, order, max_cand, dev, jitter=jitter_depth_mm, rng=rng,
                               use_roi=appearance, roi_dim=roi_dim or None)
            rank = head(**batch)
            mask = batch["mask"].float()
            loss = rank_loss(rank, y, mask)
            anchor = batch["log_score"]
            if head.w is not None:
                anchor = anchor + head.w * torch.log((head.alpha + (1 - head.alpha) * batch["baseline"]).clamp_min(1e-8))
            d2 = ((rank - anchor) ** 2 * mask).sum() / mask.sum().clamp_min(1)
            loss = loss + delta_l2 * d2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(order); nb += len(order)
        sched.step()
        f = froc_sensitivities(val, predict(head, val, s_va, dev), WIDE_GRID)
        score = float(np.mean(list(f.values())))
        hist_s.append(score)
        hist_w = (hist_w + [{k: v.detach().cpu().clone() for k, v in head.state_dict().items()}])[-K:]
        flag = ""
        if len(hist_s) >= K and np.mean(hist_s[-K:]) > best:
            best = float(np.mean(hist_s[-K:]))
            best_state = {k: (torch.stack([h[k].float() for h in hist_w]).mean(0).to(v.dtype)
                              if v.is_floating_point() else hist_w[K // 2][k])
                          for k, v in hist_w[K // 2].items()}
            best_ep, flag = ep - K // 2, f"  * (smoothed {best:.4f}, epoch {ep - K // 2})"
        history.append({"epoch": ep, "loss": tot / max(nb, 1), "val_wide": score})
        log(f"ep {ep:3d} loss {tot / max(nb, 1):.4f} val {score:.4f} {time.time() - t0:.0f}s{flag}")
    head.load_state_dict(best_state)
    return head.eval(), {"epochs": history, "best_epoch": best_ep, "best_val_smoothed": best}
