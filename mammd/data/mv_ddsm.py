"""MV-DDSM: mass detection views and their CC <-> MLO links.

Reads data/mv_ddsm/annotations.json and the images_2048/ folder written by
scripts/convert_mv_ddsm.py.

Mass-detection population of the paper (per split):
  positive  CBIS-DDSM image with at least one mass (boxes = its mass lesions)
  negative  DDSM view verified to contain no mass (role "negative": normal exams, or a DDSM
            overlay check), or a CBIS-DDSM calcification image verified mass-free
  excluded  "support" views (a mass is present but was not outlined to CBIS-DDSM quality),
            unverifiable views
With `birads_min=3` an image whose masses are all BI-RADS <= 2 loses its boxes and becomes a
negative: the annotation scope of VinDr-Mammo, used for the CBIS -> VinDr transfer.

Views are grouped into breasts (exam, laterality). A breast with exactly one CC and one MLO,
both with a usable depth axis, is a pair; any other view is a single-view item.

    ds = MultiviewBreasts("annotations.json", "/data/MV-DDSM/images_2048", "test")
    views = ds[0]                      # [cc, mlo] or [view]
    images, targets, ctx = collate_breasts([ds[0], ds[1]])
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from ..geometry import breast_mask, depth_plane, nipple_on_mask

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------------------- annotations
def load_annotations(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)["images"]


def _ioa(a: torch.Tensor, b: torch.Tensor):
    iw = torch.clamp(torch.min(a[2], b[2]).float() - torch.max(a[0], b[0]).float(), min=0)
    ih = torch.clamp(torch.min(a[3], b[3]).float() - torch.max(a[1], b[1]).float(), min=0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return iw * ih / area_a, area_a <= (b[2] - b[0]) * (b[3] - b[1])


def merge_nested_boxes(boxes: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
    """A box lying more than `threshold` inside a larger one is merged into it (union box).

    CBIS-DDSM outlines some masses with several overlapping ROIs; they count as one lesion."""
    keep = torch.ones(len(boxes), dtype=torch.bool)
    for i in range(len(boxes)):
        if keep[i]:
            for j in range(len(boxes)):
                if i != j:
                    ioa, smaller = _ioa(boxes[j], boxes[i])
                    if ioa > threshold and smaller:
                        keep[j] = False
                        boxes[i][0] = torch.min(boxes[i][0], boxes[j][0])
                        boxes[i][1] = torch.min(boxes[i][1], boxes[j][1])
                        boxes[i][2] = torch.max(boxes[i][2], boxes[j][2])
                        boxes[i][3] = torch.max(boxes[i][3], boxes[j][3])
    return boxes[keep]


def mass_boxes(entry: dict, scale: float = 1.0, corrected: bool = False) -> np.ndarray:
    """Mass boxes of one image (N, 4), after merging nested ROIs; native pixels times `scale`."""
    b = [le["bbox"] for le in entry["lesions"] if le["type"] == "MASS" and not le.get("uncertain")
         and (corrected or le["box_source"] != "restored_original")]
    if not b:
        return np.zeros((0, 4), np.float32)
    return (merge_nested_boxes(torch.tensor(b, dtype=torch.float32)) * float(scale)).numpy()


def _breast_key(e: dict) -> tuple:
    return (e.get("ddsm_case") or e.get("cbis_patient_id") or e["image_id"], e["laterality"])


def mass_detection_views(images: list[dict], split: str, birads_min: int | None = None,
                         corrected: bool = False) -> list[dict]:
    """The views of the mass-detection population for one split, in release order.

    Each returned dict is the annotation entry plus `positive`, `support` (bool) and `boxes_2048`.
    The default is the dataset of the paper. `corrected=True` applies the corrections made after
    the paper (dataset card):
      - a view is positive when it carries a mass box, not only when CBIS-DDSM lists a mass for it;
      - the 5 boxes lost by the annotation tool are restored;
      - a view whose only mass is flagged `uncertain` is left out;
      - support views (a DDSM view without any mass box in the same breast as a mass, or a view
        whose mass is not outlined) are kept as partners of a breast with a positive view, never
        as targets or negatives (`support=True`, no box); other support views are left out.
    """
    assert split in SPLITS, split
    status = {}
    for i, e in enumerate(images):
        if e["split"] != split:
            continue
        positive = bool(e.get("cbis_mass_birads"))
        if corrected:
            masses = [le for le in e["lesions"] if le["type"] == "MASS"]
            if masses and all(le.get("uncertain") for le in masses):
                continue                     # neither a clean positive nor a clean negative
            positive = positive or bool(masses)
        dropped = False
        if positive and birads_min is not None and e.get("cbis_mass_birads") and max(e["cbis_mass_birads"]) < birads_min:
            positive, dropped = False, True
        role = e["role"] if corrected else e.get("role_paper", e["role"])
        negative = (role == "negative"
                    or (role == "annotated" and e["mass_verification"] == "verified_clean") or dropped)
        if positive or negative:
            status[i] = "positive" if positive else "negative"
        elif corrected and role == "support":
            status[i] = "support"
    if corrected:                            # a support view is kept only next to a positive view
        positive_breasts = {_breast_key(images[i]) for i, st in status.items() if st == "positive"}
        status = {i: st for i, st in status.items()
                  if st != "support" or _breast_key(images[i]) in positive_breasts}
    out = []
    for i in sorted(status):
        e, st = images[i], status[i]
        boxes = mass_boxes(e, e["scale_2048"], corrected) if st == "positive" else np.zeros((0, 4), np.float32)
        out.append({**e, "positive": st == "positive", "support": st == "support",
                    "boxes_2048": boxes.astype(np.float32)})
    return out


def _has_axis(e: dict) -> bool:
    g = e.get("geometry")
    return bool(g and g["has_axis"])


def group_breasts(views: list[dict]) -> list[tuple[int, int | None]]:
    """Breast items as (cc_index, mlo_index) pairs or (index, None) single views.

    The exam key is the DDSM case (shared by the CBIS-DDSM and IRMA views of an exam), else the
    CBIS-DDSM patient. Breasts are sorted by (exam, laterality); this order defines the order of
    the candidate caches."""
    keys = []
    for i, e in enumerate(views):
        k = e.get("ddsm_case") or e.get("cbis_patient_id") or f"__unpaired_row_{i}"
        keys.append((str(k), e["laterality"]))
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    items = []
    for k in sorted(groups):
        idx = groups[k]
        cc = [i for i in idx if views[i]["view"].upper() == "CC"]
        mlo = [i for i in idx if views[i]["view"].upper() == "MLO"]
        if len(cc) == 1 and len(mlo) == 1 and _has_axis(views[cc[0]]) and _has_axis(views[mlo[0]]):
            items.append((cc[0], mlo[0]))
        else:
            items.extend((i, None) for i in idx)
    return items


# ------------------------------------------------------------------------------------ dataset
def load_png16(path: str) -> torch.Tensor:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr.astype(np.float32))


def _minmax(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - x.amin()) / (x.amax() - x.amin()).clamp_min(1e-6)
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


class MultiviewBreasts(torch.utils.data.Dataset):
    """One item = the views of one breast, each with its depth plane and breast mask.

    A view is a dict with
      image     (3, 2048, 2048) float, min-max to [0, 1] then normalised with mean = std = 0.5
                (the detectors' input convention)
      boxes     (N, 4) mass boxes in the 2048 frame;  labels (N,) ones
      depth     (1, 2048, 2048) depth in mm (zeros when the view has no axis)
      mask      (1, 2048, 2048) breast mask
      is_mlo, has_axis, image_id, mm_per_px (2048 frame)
    `has_axis` is re-checked on the loaded image: a nipple more than 15 mm away from the breast
    mask invalidates the axis and the view is then left unpaired by `collate_breasts`.
    """

    def __init__(self, annotations: str | list, image_root: str, split: str,
                 birads_min: int | None = None, pairs_only: bool = False, corrected: bool = False):
        images = load_annotations(annotations) if isinstance(annotations, str) else annotations
        self.views = mass_detection_views(images, split, birads_min, corrected)
        self.items = group_breasts(self.views)
        if pairs_only:
            self.items = [it for it in self.items if it[1] is not None]
        self.image_root = image_root
        self.split = split

    def __len__(self):
        return len(self.items)

    def view(self, i: int) -> dict:
        e = self.views[i]
        img = load_png16(os.path.join(self.image_root, e["file_name"]))
        H, W = img.shape
        content_w = int(round(e["width"] * e["scale_2048"]))
        mask = breast_mask(img.numpy(), content_w)
        g, s = e.get("geometry"), e["scale_2048"]
        has_axis = _has_axis(e)
        if has_axis:
            nx, ny = g["nipple_xy"][0] * s, g["nipple_xy"][1] * s
            mpp = g["mm_per_px"] / s
            has_axis = nipple_on_mask(mask, nx, ny, mpp)
        depth = (depth_plane((H, W), nx, ny, g["axis"][0], g["axis"][1], mpp) if has_axis
                 else np.zeros((H, W), np.float32))
        x = _minmax(img).unsqueeze(0).repeat(3, 1, 1)
        x = TF.normalize(_minmax(x), mean=(0.5,) * 3, std=(0.5,) * 3)
        boxes = torch.as_tensor(e["boxes_2048"], dtype=torch.float32).reshape(-1, 4)
        return dict(image=x, boxes=boxes, labels=torch.ones(len(boxes), dtype=torch.int64),
                    depth=torch.from_numpy(np.ascontiguousarray(depth))[None].float(),
                    mask=torch.from_numpy(mask)[None].float(),
                    is_mlo=e["view"].upper() == "MLO", has_axis=bool(has_axis), support=e["support"],
                    mm_per_px=(mpp if has_axis else 0.0), image_id=e["image_id"])

    def __getitem__(self, b: int) -> list[dict]:
        i, j = self.items[b]
        return [self.view(i)] if j is None else [self.view(i), self.view(j)]


@dataclass
class BreastBatch:
    partner: torch.Tensor   # (N,) index of the other view of the same breast in the batch, or -1
    is_mlo: torch.Tensor    # (N,) bool
    depth: torch.Tensor     # (N, 1, H, W) mm
    mask: torch.Tensor      # (N, 1, H, W)
    image_ids: list


def collate_breasts(batch: list[list[dict]]):
    """Flatten breasts into views. Returns (images, targets, BreastBatch)."""
    flat, partner = [], []
    for views in batch:
        base = len(flat)
        flat.extend(views)
        paired = len(views) == 2 and views[0]["has_axis"] and views[1]["has_axis"]
        partner.extend([base + 1, base] if paired else [-1] * len(views))
    ctx = BreastBatch(partner=torch.tensor(partner, dtype=torch.long),
                      is_mlo=torch.tensor([v["is_mlo"] for v in flat], dtype=torch.bool),
                      depth=torch.stack([v["depth"] for v in flat]),
                      mask=torch.stack([v["mask"] for v in flat]),
                      image_ids=[v["image_id"] for v in flat])
    return ([v["image"] for v in flat], [{"boxes": v["boxes"], "labels": v["labels"]} for v in flat], ctx)
