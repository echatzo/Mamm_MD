"""RetinaNet detector of the paper (inference): DINOv3 ConvNeXt-B + FPN + RetinaNet head.

  backbone  timm convnext_base.dinov3_lvd1689m, initialised from the patch classifier trained on
            the source dataset (stages 0/1 of the training recipe), then fine-tuned for detection
  neck      torchvision FPN on stages 1-3 (256, 512, 1024 channels) -> 64 channels, + max-pool level
  head      RetinaNet classification and regression heads with GroupNorm, 9 anchors per location
            (sizes 32 / 64 / 128 / 256 at 3 scales, aspect ratios 0.5 / 1 / 2)
  output    at most 10 detections per image after NMS (IoU 0.3), score threshold 0.05

Input: list of (3, 2048, 2048) tensors normalised to [-1, 1] (see mammd.data.mv_ddsm); no
resizing is done by the model.

    det = RetinaNetDetector.from_checkpoint("checkpoints/detectors/retinanet_cbis.pt").cuda()
    out = det([img])      # [{"boxes": (n, 4), "scores": (n,), "labels": (n,)}]
"""
from __future__ import annotations

from collections import OrderedDict
from functools import partial

import timm
import torch
from torch import Tensor, nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

ANCHOR_SIZES = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in (32, 64, 128, 256))
ASPECT_RATIOS = (0.5, 1.0, 2.0)


def dinov3_convnext_fpn(timm_name: str = "convnext_base.dinov3_lvd1689m", pretrained: bool = False) -> BackboneWithFPN:
    m = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="")
    features = nn.Sequential(OrderedDict([("0", nn.Sequential(m.stem, m.stages[0])), ("1", m.stages[1]),
                                          ("2", m.stages[2]), ("3", m.stages[3])]))
    return BackboneWithFPN(features, {"1": "0", "2": "1", "3": "2"}, [256, 512, 1024], 64,
                           extra_blocks=LastLevelMaxPool())


def _cell_anchors(sizes, ratios, dtype, device):
    """Zero-centred anchors, computed in float32 then cast (fp16 under autocast)."""
    scales = torch.as_tensor(sizes, dtype=torch.float32)
    ar = torch.as_tensor(ratios, dtype=torch.float32)
    h = torch.sqrt(ar); w = 1 / h
    ws = (w[:, None] * scales[None, :]).view(-1); hs = (h[:, None] * scales[None, :]).view(-1)
    return (torch.stack([-ws, -hs, ws, hs], dim=1) / 2).round().to(dtype=dtype, device=device)


def anchors_for(image_hw, feature_maps) -> Tensor:
    """All anchors (A, 4) of one image, level by level, in the order of the head outputs."""
    dtype, device = feature_maps[0].dtype, feature_maps[0].device
    out = []
    for fm, sizes in zip(feature_maps, ANCHOR_SIZES):
        gh, gw = fm.shape[-2:]
        sy, sx = image_hw[0] // gh, image_hw[1] // gw
        base = _cell_anchors(sizes, ASPECT_RATIOS, dtype, device)
        xs = torch.arange(0, gw, dtype=torch.int32, device=device) * sx
        ys = torch.arange(0, gh, dtype=torch.int32, device=device) * sy
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx, yy = xx.reshape(-1), yy.reshape(-1)
        shifts = torch.stack((xx, yy, xx, yy), dim=1)
        out.append((shifts.view(-1, 1, 4) + base.view(1, -1, 4)).reshape(-1, 4))
    return torch.cat(out)


class RetinaNetDetector(nn.Module):
    def __init__(self, num_classes: int = 2, score_thresh: float = 0.05, nms_thresh: float = 0.3,
                 detections_per_img: int = 10, topk_candidates: int = 1000):
        super().__init__()
        self.backbone = dinov3_convnext_fpn()
        self.head = RetinaNetHead(self.backbone.out_channels, len(ANCHOR_SIZES[0]) * len(ASPECT_RATIOS),
                                  num_classes, norm_layer=partial(nn.GroupNorm, 32))
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.score_thresh, self.nms_thresh = score_thresh, nms_thresh
        self.detections_per_img, self.topk_candidates = detections_per_img, topk_candidates

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu", **kw) -> "RetinaNetDetector":
        m = cls(**kw)
        sd = torch.load(path, map_location=map_location, weights_only=False)
        sd = sd.get("model", sd)
        m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=True)
        return m.eval()

    @torch.no_grad()
    def forward(self, images: list[Tensor]) -> list[dict]:
        x = torch.stack(images)
        feats = list(self.backbone(x.contiguous()).values())
        out = self.head(feats)
        anchors = anchors_for(x.shape[-2:], feats)
        per_level = [f.size(2) * f.size(3) * (out["cls_logits"].size(1) // sum(g.size(2) * g.size(3) for g in feats))
                     for f in feats]
        logits = out["cls_logits"].split(per_level, dim=1)
        regs = out["bbox_regression"].split(per_level, dim=1)
        anchors = anchors.split(per_level)
        hw = tuple(x.shape[-2:])
        dets = []
        for i in range(len(images)):
            boxes, scores, labels = [], [], []
            for lg, rg, an in zip(logits, regs, anchors):
                k = lg.shape[-1]
                s = torch.sigmoid(lg[i]).flatten()
                keep = s > self.score_thresh
                s, idx = s[keep], torch.where(keep)[0]
                s, top = s.topk(det_utils._topk_min(idx, self.topk_candidates, 0))
                idx = idx[top]
                a_idx = torch.div(idx, k, rounding_mode="floor")
                b = box_ops.clip_boxes_to_image(self.box_coder.decode_single(rg[i][a_idx], an[a_idx]), hw)
                boxes.append(b); scores.append(s); labels.append(idx % k)
            boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)[: self.detections_per_img]
            dets.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return dets
