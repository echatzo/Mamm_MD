"""Train the CEN comparator (official recipe, see mammd/cen.py) on candidate caches.

    python scripts/train_cen.py --train caches/retinanet_cbis/train.pt --val caches/retinanet_cbis/val.pt \
        --seed 0 --out checkpoints/cen/retinanet_cbis/s0.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd.cen import train_cen  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log_f = open(os.path.splitext(a.out)[0] + ".log", "w", buffering=1)
    log = lambda s: (print(s, flush=True), log_f.write(s + "\n"))
    tr, va = torch.load(a.train, weights_only=False), torch.load(a.val, weights_only=False)
    _, hist = train_cen(tr, va, seed=a.seed, epochs=a.epochs, device=a.device, out_path=a.out, log=log)
    json.dump(hist, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    log(f"-> {a.out} (epoch {hist['best_epoch']})")


if __name__ == "__main__":
    main()
