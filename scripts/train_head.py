"""Train one MAMM-MD head on candidate caches.

    python scripts/train_head.py --train caches/retinanet_cbis/train.pt --val caches/retinanet_cbis/val.pt \
        --variant geoapp --pos-frac 0.23 --seed 0 --out checkpoints/heads/retinanet_cbis/geoapp_s0.pth

Variants: geoapp (MAMM-MD), geo (w/o appearance), appo (w/o geometry).
--pos-frac: fraction of views with a true positive in each batch of 16, set to the training
prevalence (0.23 CBIS-DDSM-IRMA, 0.195 its BI-RADS >= 3 version, 0.0625 VinDr-Mammo).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd.rescoring import train_head  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--variant", choices=("geoapp", "geo", "appo"), required=True)
    ap.add_argument("--pos-frac", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log_f = open(os.path.splitext(a.out)[0] + ".log", "w", buffering=1)
    log = lambda s: (print(s, flush=True), log_f.write(s + "\n"))
    tr, va = torch.load(a.train, weights_only=False), torch.load(a.val, weights_only=False)
    head, hist = train_head(tr, va, appearance=a.variant != "geo", geometry=a.variant != "appo",
                            pos_frac=a.pos_frac, seed=a.seed, epochs=a.epochs, device=a.device, log=log)
    head.cpu().save(a.out, train_args=vars(a), history=hist)
    json.dump(hist, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    log(f"-> {a.out} (epoch {hist['best_epoch']}, smoothed val {hist['best_val_smoothed']:.4f})")


if __name__ == "__main__":
    main()
