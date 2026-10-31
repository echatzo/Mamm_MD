"""CEN comparator (Jain et al., MICCAI 2024), official recipe on our cached candidates.

Taken from the official repository (github.com/Mammo-IITD-AIIMS/CEN, commit 8e6de48):
  model   MLP (roi_dim + 4) -> 1024 -> 512 -> 256 with ReLU, L2-normalised output, frozen encoder;
          preds = conf_0 + max_j(cos_ij * conf_1[i])  (as their code computes it: conf_1 is indexed
          by the row i, i.e. the confidence of the same-rank candidate of the other view)
  train   both directions (MLO <- CC, CC <- MLO), BCE(sigmoid(preds), target) summed and divided
          by N * N; one breast per step, fixed order; Adam lr 1e-6; 100 epochs; both views
          truncated to min(N_MLO, N_CC, 25) candidates sorted by score; selection on the lowest
          validation loss
  target  1 if the centre of the candidate lies inside a ground-truth box; an empty view gets
          one dummy candidate (zero box, confidence -1)
Differences: the encoder is our frozen patch classifier (same 1024-d embeddings as our head)
instead of their unreleased ResNet-50; boxes are (cx, cy, w, h) / 2048; the unannotated view of a
positive breast is a partner only (no loss term), as for our head.
Inference: sigmoid(preds) on the truncated candidates, sigmoid(conf) for the others.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IMG = 2048.0
MAX_PROPOSALS = 25
NEG = -1e4


class CENHead(nn.Module):
    def __init__(self, roi_dim: int = 1024, dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(roi_dim + 4, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
                                 nn.Linear(512, dim))
        self.roi_dim = roi_dim

    def embed(self, roi, box):
        return F.normalize(self.mlp(torch.cat([roi, box], dim=-1)), p=2, dim=-1)

    def forward(self, roi, box, score, p_roi, p_box, p_score):
        """(1, n) predictions for the n candidates of one view given the n of the other view."""
        cos = self.embed(roi, box) @ self.embed(p_roi, p_box).transpose(1, 2)        # (1, n, n)
        rank = torch.zeros_like(cos)
        m = min(cos.shape[1], p_score.shape[1])
        rank[:, :m, :] = p_score[:, :m].unsqueeze(2)
        return score + (cos * rank).max(dim=-1).values

    def save(self, path, **extra):
        torch.save({"model": self.state_dict(), "roi_dim": self.roi_dim, **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        h = cls(roi_dim=ck["roi_dim"])
        h.load_state_dict(ck["model"])
        return h.eval()


def _boxfeat(b):
    b = np.asarray(b, np.float32).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) / IMG


def _centre_in_gt(boxes, gt):
    if not len(gt) or not len(boxes):
        return np.zeros(len(boxes), np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
    g = np.asarray(gt, np.float32).reshape(-1, 4)
    ins = ((cx[:, None] >= g[None, :, 0]) & (cx[:, None] <= g[None, :, 2])
           & (cy[:, None] >= g[None, :, 1]) & (cy[:, None] <= g[None, :, 3]))
    return ins.any(1).astype(np.float32)


def _view(r, roi_dim):
    if len(r["scores"]) == 0:
        return dict(idx=np.zeros(0, int), roi=np.zeros((1, roi_dim), np.float32), box=np.zeros((1, 4), np.float32),
                    conf=np.array([-1.0], np.float32), y=np.zeros(1, np.float32))
    k = np.argsort(-np.asarray(r["scores"]), kind="stable")
    b = np.asarray(r["boxes"], np.float32)[k]
    return dict(idx=k, roi=np.asarray(r["roi"], np.float32)[k], box=_boxfeat(b),
                conf=np.asarray(r["scores"], np.float32)[k], y=_centre_in_gt(b, r["gt"]))


def breast_pairs(recs):
    """[(mlo, cc, positive breast)] for every paired breast of a cache."""
    out = []
    for i, r in enumerate(recs):
        p = r["partner"]
        if r["is_mlo"] or p is None or p < 0:
            continue
        out.append((p, i, len(r["gt"]) > 0 or len(recs[p]["gt"]) > 0))
    return out


def _run_pair(head, recs, m, c, dev):
    vm, vc = _view(recs[m], head.roi_dim), _view(recs[c], head.roi_dim)
    n = min(len(vm["conf"]), len(vc["conf"]), MAX_PROPOSALS)
    t = lambda v: [torch.as_tensor(v[k][:n], device=dev).unsqueeze(0) for k in ("roi", "box", "conf")]
    (rm, bm, sm), (rc, bc, sc) = t(vm), t(vc)
    return head(rm, bm, sm, rc, bc, sc)[0], head(rc, bc, sc, rm, bm, sm)[0], n, vm, vc


def pair_loss(head, recs, m, c, positive, dev):
    pm, pc, n, vm, vc = _run_pair(head, recs, m, c, dev)
    loss = torch.zeros((), device=dev)
    for pr, v, i in ((pm, vm, m), (pc, vc, c)):
        if positive and len(recs[i]["gt"]) == 0:            # support view: no loss term
            continue
        loss = loss + F.binary_cross_entropy(torch.sigmoid(pr), torch.as_tensor(v["y"][:n], device=dev))
    return loss / (n * n)


@torch.no_grad()
def predict(head, recs, dev="cpu"):
    """Scores for every candidate (cache order) and the mask of those inside the truncation."""
    head.eval()
    out = [1 / (1 + np.exp(-np.asarray(r["scores"], np.float64))) for r in recs]
    kept = [np.zeros(len(r["scores"]), bool) for r in recs]
    for m, c, _ in breast_pairs(recs):
        pm, pc, n, vm, vc = _run_pair(head, recs, m, c, dev)
        for pr, v, i in ((pm, vm, m), (pc, vc, c)):
            k = v["idx"][:n]
            if len(k):
                out[i][k] = torch.sigmoid(pr).cpu().numpy()[:len(k)]
                kept[i][k] = True
    return out, kept


def predict_ensemble(heads, recs, dev="cpu"):
    """Mean probability of several CEN heads (seeds)."""
    P = [predict(h, recs, dev)[0] for h in heads]
    return [np.mean([p[i] for p in P], axis=0) for i in range(len(recs))]


def train_cen(train, val, seed=0, epochs=100, lr=1e-6, device="cuda", out_path=None, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    roi_dim = {np.shape(r["roi"])[1] for r in train if len(r["scores"])}.pop()
    head = CENHead(roi_dim=roi_dim).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    ptr, pva = breast_pairs(train), breast_pairs(val)
    log(f"train {len(ptr)} breasts, val {len(pva)} breasts")
    best, hist, best_state = 1e9, [], None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        head.train()
        tl = 0.0
        for m, c, pos in ptr:
            opt.zero_grad()
            loss = pair_loss(head, train, m, c, pos, dev)
            loss.backward()
            opt.step()
            tl += loss.item()
        head.eval()
        with torch.no_grad():
            vl = sum(pair_loss(head, val, m, c, pos, dev).item() for m, c, pos in pva)
        star = vl < best
        if star:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if out_path:
                head.save(out_path, epoch=ep)
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        log(f"ep {ep:3d} train {tl:.4f} val {vl:.4f} {time.time() - t0:.0f}s{'  *' if star else ''}")
    head.load_state_dict(best_state)
    return head.eval(), {"epochs": hist, "best_epoch": min(hist, key=lambda h: h["val_loss"])["epoch"]}
