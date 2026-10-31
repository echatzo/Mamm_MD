"""Candidate caches: run the frozen detector once and store what the head needs per candidate.

For every view: the detections (top 10 after NMS), the depth of each candidate centre (mm), its
width and height (mm), whether it is a true positive (paper matching rule, detection order), and
the appearance embedding of its crop. The head then trains in minutes on this cache.

Appearance embedding: the detection box itself is cropped from the detector input (2048 frame,
normalised to [-1, 1]), resized to 256 x 256 (bilinear, aspect not preserved) and encoded by the
frozen patch classifier (DINOv3 ConvNeXt-B fine-tuned to classify 256 x 256 mass / background
patches); the feature is the pre-logits vector (global average pool + LayerNorm), 1024-d.
"""
from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn.functional as F

from .eval.froc import hits
from .geometry import depth_at_boxes, mm_per_px_of


def build_patch_encoder(ckpt: str, device):
    """Frozen patch classifier -> 1024-d pre-logits features."""
    m = timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=False, num_classes=2)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    r = m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(f"incompatible patch classifier: {len(r.missing_keys)} missing, "
                           f"{len(r.unexpected_keys)} unexpected keys")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return lambda x: m.forward_head(m.forward_features(x), pre_logits=True)


@torch.inference_mode()
def crop_embeddings(encoder, image, boxes, device, size: int = 256, dim: int = 1024) -> np.ndarray:
    """(K, dim) float16 embeddings of the box crops of one (3, H, W) image."""
    if not len(boxes):
        return np.zeros((0, dim), np.float16)
    _, H, W = image.shape
    crops = []
    for b in boxes:
        x1 = max(0, min(int(b[0]), W - 2)); y1 = max(0, min(int(b[1]), H - 2))
        x2 = max(x1 + 1, min(int(b[2]), W - 1)); y2 = max(y1 + 1, min(int(b[3]), H - 1))
        crops.append(F.interpolate(image[:, y1:y2 + 1, x1:x2 + 1].unsqueeze(0), size=(size, size),
                                   mode="bilinear", align_corners=False))
    out = []
    for i in range(0, len(crops), 64):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = encoder(torch.cat(crops[i:i + 64]).to(device))
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def tp_flags(boxes, labels, gt, gt_labels) -> np.ndarray:
    """True positive flag of each detection, in detector order (paper matching rule)."""
    tp = np.zeros(len(boxes), dtype=bool)
    taken = np.zeros(len(gt), dtype=bool)
    for i, det in enumerate(boxes):
        matched = False
        for j, g in enumerate(gt):
            if not matched and hits(det, g) and labels[i] == gt_labels[j]:
                if not taken[j]:
                    tp[i] = True
                    taken[j] = True
                matched = True
    return tp


def view_record(det: dict, view_boxes, view_labels, depth: np.ndarray, is_mlo: bool, partner: int,
                image_id: str) -> dict:
    """Cache record of one view from its detections and its depth plane (2048 frame)."""
    mpp = mm_per_px_of(depth)
    db = det["boxes"].float().cpu().numpy()
    dl = det["labels"].cpu().numpy()
    gt = view_boxes.numpy()
    gl = view_labels.cpu().numpy().astype(np.int64)
    return dict(boxes=db.astype(np.float32), scores=det["scores"].float().cpu().numpy().astype(np.float32),
                labels=dl,
                depth=depth_at_boxes(depth, db).astype(np.float32) if len(db) else np.zeros(0, np.float32),
                w_mm=((db[:, 2] - db[:, 0]) * mpp).astype(np.float32) if len(db) else np.zeros(0, np.float32),
                h_mm=((db[:, 3] - db[:, 1]) * mpp).astype(np.float32) if len(db) else np.zeros(0, np.float32),
                tp=tp_flags(db, dl, gt, gl), gt=gt.astype(np.float32), gt_labels=gl,
                mm_per_px=mpp, is_mlo=bool(is_mlo), partner=partner, id=image_id)


def extract(loader, detector, encoder, device, precomputed: dict | None = None, topk: int = 10, log=print) -> list:
    """Cache records for every view of a loader of collate_breasts batches.

    `detector`: a RetinaNetDetector, or None with `precomputed` = {image_id: {"boxes", "scores"}}
    (2048 frame; e.g. Deformable DETR outputs), of which the `topk` best are kept."""
    recs = []
    with torch.inference_mode():
        for bi, (images, targets, ctx) in enumerate(loader):
            ims = [i.to(device, non_blocking=True) for i in images]
            if precomputed is not None:
                dets = []
                for iid in ctx.image_ids:
                    e = precomputed[str(iid)]
                    k = np.argsort(-e["scores"])[:topk]
                    dets.append({"boxes": torch.as_tensor(e["boxes"][k]), "scores": torch.as_tensor(e["scores"][k]),
                                 "labels": torch.ones(len(k), dtype=torch.int64)})
            else:
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    dets = detector(ims)
            base = len(recs)
            for i in range(len(ims)):
                p = int(ctx.partner[i])
                r = view_record(dets[i], targets[i]["boxes"], targets[i]["labels"], ctx.depth[i, 0].numpy(),
                                bool(ctx.is_mlo[i]), base + p if p >= 0 else -1, str(ctx.image_ids[i]))
                if encoder is not None:
                    r["roi"] = crop_embeddings(encoder, ims[i], r["boxes"], device)
                recs.append(r)
            if (bi + 1) % 50 == 0:
                log(f"  {bi + 1}/{len(loader)} batches, {len(recs)} views")
    return recs
