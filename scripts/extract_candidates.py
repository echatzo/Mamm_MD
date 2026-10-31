"""Build a candidate cache for one split of MV-DDSM.

RetinaNet (detector run here):
    python scripts/extract_candidates.py --split test --images /data/MV-DDSM/images_2048 \
        --detector checkpoints/detectors/retinanet_cbis.pt --encoder checkpoints/encoders/patch_classifier_cbis.pt \
        --out caches/retinanet_cbis/test_cbis.pt
Deformable DETR (detections computed with detrex, scripts/detr_inference.py):
    ... --detections detr_cbis_detections.pt (instead of --detector)
--birads-min 3: BI-RADS >= 3 annotation scope (the *_b3 pipelines).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd.candidates import build_patch_encoder, extract  # noqa: E402
from mammd.data.mv_ddsm import MultiviewBreasts, collate_breasts  # noqa: E402
from mammd.detectors.retinanet import RetinaNetDetector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=os.path.join(HERE, "..", "data", "mv_ddsm", "annotations.json"))
    ap.add_argument("--images", required=True, help="images_2048/ folder of the converted dataset")
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--birads-min", type=int, default=None)
    ap.add_argument("--corrected", action="store_true",
                    help="annotation corrections made after the paper (dataset card, Known issues)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--detector", help="RetinaNet checkpoint")
    g.add_argument("--detections", help="precomputed detections {image_id: {boxes, scores}} (2048 frame)")
    ap.add_argument("--encoder", required=True, help="patch classifier checkpoint (appearance embeddings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4, help="breasts per batch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N batches (testing)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ds = MultiviewBreasts(a.annotations, a.images, a.split, birads_min=a.birads_min,
                          corrected=a.corrected)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, collate_fn=collate_breasts,
                    pin_memory=True)
    if a.limit:
        dl = [b for i, b in zip(range(a.limit), dl)]
    det = RetinaNetDetector.from_checkpoint(a.detector).to(dev) if a.detector else None
    pre = None
    if a.detections:
        pre = torch.load(a.detections, weights_only=False)
        pre.pop("_meta", None)
    recs = extract(dl, det, build_patch_encoder(a.encoder, dev), dev, precomputed=pre)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(recs, a.out)
    print(f"{a.split}: {len(recs)} views, {sum(r['partner'] >= 0 for r in recs)} paired, "
          f"{sum(len(r['gt']) for r in recs)} lesions, {sum(len(r['scores']) for r in recs)} candidates -> {a.out}")


if __name__ == "__main__":
    main()
