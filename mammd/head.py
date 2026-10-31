"""MAMM-MD: multiview CC <-> MLO rescoring head on the candidates of a frozen detector.

For each candidate i of a view, the head attends to the candidates j of the other view of the
same breast and predicts a log-space correction delta_i. The rank score is

    r_i = log s_i + w * log(alpha + (1 - alpha) * p_i) + delta_i,      alpha = 0.5,

where s_i is the detector score and p_i the best partner score within 2 sigma of the expected
depth (the geometric product baseline). w is initialised to 1 and the output layer to zero, so
that at initialisation the head ranks candidates exactly like the geometric product.

Tokens (d = 64): an MLP of 7 geometric features (score logit, depth, width, height, aspect, rank,
log product term), plus a learned linear projection of the L2-normalised 1024-d appearance
embedding of the candidate crop. Self-attention within the view, then cross-attention to the
partner view with
  - a hard depth mask: partners farther than 3 sigma from the expected depth are not attended,
  - a Huber depth bias -lam * huber((d_expected - d_j) / sigma) added to the attention logits,
  - a learned null slot ("no credible partner").
delta = tanh(.) * 3 is zeroed for views without a partner.

Depth prior (anatomical, no data): d_MLO = d_CC, sigma = 10 mm.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ANATOMICAL_PRIOR = dict(slope=1.0, intercept=0.0, sigma=10.0)
N_FEATURES = 7


def expected_partner_depth(depth, is_mlo, slope, intercept):
    """Depth expected in the other view: d_MLO = slope * d_CC + intercept."""
    fwd = slope * depth + intercept
    inv = (depth - intercept) / slope
    return torch.where(is_mlo.view(-1, 1).bool(), inv, fwd)


def candidate_features(scores: Tensor, depth: Tensor, w_mm: Tensor, h_mm: Tensor,
                       baseline: Tensor) -> Tensor:
    """(B, K, 7) token features from (B, K) inputs, each scaled to O(1)."""
    eps = 1e-6
    s = scores.clamp(eps, 1 - eps)
    rank = torch.arange(scores.shape[1], device=scores.device, dtype=scores.dtype).expand_as(scores)
    return torch.stack([
        (torch.log(s) - torch.log1p(-s)) / 4.0,       # detector logit
        depth / 100.0,                                # depth (mm)
        w_mm / 50.0,
        h_mm / 50.0,
        torch.log((w_mm + eps) / (h_mm + eps)),       # aspect
        rank / 50.0,                                  # rank of the candidate in its view
        torch.log(baseline.clamp_min(eps)),           # log best partner score in the 2 sigma band
    ], dim=-1)


class MAMMMDHead(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 4, lam: float = 1.0, alpha: float = 0.5,
                 max_delta: float = 3.0, roi_dim: int = 0, prior: dict = ANATOMICAL_PRIOR,
                 band_sigmas: float = 3.0, geometry: bool = True):
        """`roi_dim` = 0 for the geometry-only head, 1024 with appearance.
        `geometry=False` is the "w/o geometry" ablation: depth and product features set to zero,
        no depth mask or bias, no product anchor."""
        super().__init__()
        self.slope, self.intercept, self.sigma = prior["slope"], prior["intercept"], prior["sigma"]
        self.dim, self.heads, self.alpha, self.max_delta = dim, heads, alpha, max_delta
        self.geometry = bool(geometry)
        self.lam = lam if geometry else 0.0
        self.band_mm = band_sigmas * self.sigma if geometry else None
        # module creation order is part of the recipe: it fixes the initial weights for a seed
        self.embed = nn.Sequential(nn.Linear(N_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        self.roi_proj = nn.Linear(roi_dim, dim) if roi_dim else None
        if self.roi_proj is not None:
            nn.init.normal_(self.roi_proj.weight, std=0.02)
            nn.init.zeros_(self.roi_proj.bias)
        self.norm_s = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q, self.norm_k = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.null_k = nn.Parameter(torch.zeros(dim))
        self.null_v = nn.Parameter(torch.zeros(dim))
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.out = nn.Linear(dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.w = nn.Parameter(torch.ones(())) if geometry else None

    @property
    def uses_appearance(self) -> bool:
        return self.roi_proj is not None

    def forward(self, feats, depth, mask, p_feats, p_depth, p_mask, is_mlo, has_partner,
                log_score, baseline, roi=None, p_roi=None) -> Tensor:
        """Rank score (B, K). feats (B, K, 7), mask (B, K) True for real candidates; p_* for the
        partner view (B, Kp, ...); is_mlo, has_partner (B,); roi / p_roi (B, K, 1024) or None."""
        B, K, _ = feats.shape
        if not self.geometry:
            feats = feats.clone(); feats[..., 1] = 0; feats[..., 6] = 0
            p_feats = p_feats.clone(); p_feats[..., 1] = 0; p_feats[..., 6] = 0
        x = self.embed(feats)
        if self.roi_proj is not None and roi is not None:
            x = x + self.roi_proj(F.normalize(roi, p=2, dim=-1))
        h = self.norm_s(x)
        pad = ~mask
        pad[:, 0] = False                       # a fully padded row would make the softmax NaN
        a, _ = self.self_attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a * mask.unsqueeze(-1)

        y = self.embed(p_feats)
        if self.roi_proj is not None and p_roi is not None:
            y = y + self.roi_proj(F.normalize(p_roi, p=2, dim=-1))
        dh = self.dim // self.heads
        q = self.q(self.norm_q(x)).view(B, K, self.heads, -1).transpose(1, 2)
        kk = self.k(self.norm_k(y)).view(B, -1, self.heads, dh).transpose(1, 2)
        vv = self.v(self.norm_k(y)).view(B, -1, self.heads, dh).transpose(1, 2)

        r_exp = expected_partner_depth(depth, is_mlo, self.slope, self.intercept)       # (B, K)
        gap = r_exp.unsqueeze(2) - p_depth.unsqueeze(1)                                 # (B, K, Kp)
        d = gap / self.sigma
        bias = -self.lam * torch.where(d.abs() <= 1.0, 0.5 * d * d, d.abs() - 0.5)    # Huber
        band = (gap.abs() <= self.band_mm) if self.geometry else torch.ones_like(gap, dtype=torch.bool)
        valid = p_mask.unsqueeze(1) & band & has_partner.view(-1, 1, 1).bool()
        bias = bias.masked_fill(~valid, float("-inf"))

        scale = 1.0 / math.sqrt(dh)
        logits = (q @ kk.transpose(-1, -2)) * scale + bias.unsqueeze(1)
        null_k = self.null_k.view(1, self.heads, 1, -1).expand(B, -1, 1, -1)
        null_v = self.null_v.view(1, self.heads, 1, -1).expand(B, -1, 1, -1)
        logits = torch.cat([logits, (q * null_k).sum(-1, keepdim=True) * scale], dim=-1)
        msg = (logits.softmax(-1) @ torch.cat([vv, null_v], dim=2)).transpose(1, 2).reshape(B, K, self.dim)

        delta = torch.tanh(self.out(self.ffn(torch.cat([self.norm_f(x), msg], dim=-1))).squeeze(-1))
        delta = delta * self.max_delta * mask * has_partner.view(-1, 1).to(delta.dtype)
        rank = log_score
        if self.w is not None:
            rank = rank + self.w * torch.log((self.alpha + (1 - self.alpha) * baseline).clamp_min(1e-8))
        return rank + delta

    # ------------------------------------------------------------------ checkpoint
    def config(self) -> dict:
        return dict(dim=self.dim, heads=self.heads, lam=self.lam if self.geometry else 1.0, alpha=self.alpha,
                    max_delta=self.max_delta, roi_dim=self.roi_proj.in_features if self.roi_proj else 0,
                    prior=dict(slope=self.slope, intercept=self.intercept, sigma=self.sigma),
                    band_sigmas=(self.band_mm / self.sigma) if self.geometry else 3.0, geometry=self.geometry)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "MAMMMDHead":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        h = cls(**ck["config"])
        h.load_state_dict(ck["model"])
        return h.eval()

    def save(self, path: str, **extra) -> None:
        torch.save({"config": self.config(), "model": self.state_dict(), **extra}, path)
