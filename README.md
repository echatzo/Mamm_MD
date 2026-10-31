# MAMM-MD: multiview CC <-> MLO rescoring of mammography mass detections

Code and data for the paper. A small attention head rescores the candidates of a frozen
single-view mass detector, using the other view of the same breast: anatomical depth (distance to
the nipple along the nipple - chest wall axis) and the appearance of the candidate crops.

```
code_paper/
  data/mv_ddsm/               MV-DDSM: correspondence CSVs, annotations.json, dataset card
  mammd/                      the package
    data/ddsm_calibration.py    DDSM optical-density calibration (IRMA -> CBIS-DDSM intensities)
    data/mv_ddsm.py             mass-detection views, CC <-> MLO pairing, multiview dataloader
    geometry.py                 nipple-axis depth
    detectors/retinanet.py      DINOv3 ConvNeXt-B + FPN RetinaNet (inference)
    candidates.py               candidate caches (detections, depth, size, TP labels, crop embeddings)
    head.py                     the MAMM-MD head
    rescoring.py                product baseline, training and inference of the head
    cen.py                      CEN comparator (official recipe on our candidates)
    eval/froc.py, protocol.py   matching rule, FROC, LARGE, protocol, breast-level bootstrap, Holm
  scripts/                    convert_mv_ddsm, extract_candidates, train_head, train_cen, evaluate
  configs/paper.yaml          the 6 pipelines and 12 evaluations of the paper
  tools/                      provenance only: how the released files were exported from our runs
```

## Setup
```
pip install -r requirements.txt
```

## 1. Data
- **MV-DDSM**: download CBIS-DDSM (TCIA) and the DDSM release of RWTH Aachen, then
  `python scripts/convert_mv_ddsm.py --cbis-root ... --irma-root ... --out <D>`.
  See `data/mv_ddsm/README.md`.
- **VinDr-Mammo** and **INbreast**: obtain them from their providers (PhysioNet credentialed access,
  INbreast on request). *[preprocessing and geometry scripts: to be added]*

## 2. Candidate caches
For each pipeline (detector × training set) and split:
```
python scripts/extract_candidates.py --split train --images <D>/images_2048 \
    --detector checkpoints/detectors/retinanet_cbis.pt --encoder checkpoints/encoders/patch_classifier_cbis.pt \
    --out caches/retinanet_cbis/train.pt
```
- `--birads-min 3` gives the BI-RADS >= 3 annotation scope of the `*_b3` pipelines.
- `--detections` takes precomputed Deformable DETR outputs instead of `--detector`.
- Every pipeline uses the same appearance encoder: the patch classifier trained on MV-DDSM.

## 3. Training
```
python scripts/train_head.py --train caches/retinanet_cbis/train.pt --val caches/retinanet_cbis/val.pt \
    --variant geoapp --pos-frac 0.23 --seed 0 --out checkpoints/heads/retinanet_cbis/geoapp_s0.pth
python scripts/train_cen.py  --train caches/retinanet_cbis/train.pt --val caches/retinanet_cbis/val.pt \
    --seed 0 --out checkpoints/cen/retinanet_cbis/s0.pth
```
- Variants: `geoapp` (MAMM-MD), `geo` (w/o appearance), `appo` (w/o geometry).
- Seeds 0, 1, 2.
- `--pos-frac` per pipeline is in `configs/paper.yaml`.
- A head trains in about 2 minutes on one GPU, CEN in about 10.

## 4. Evaluation
```
python scripts/evaluate.py --caches caches --checkpoints checkpoints \
    --inbreast-csv "<INbreast>/INbreast.csv" --out results/paper_results.json
```
For each setting, the script reports:
- the methods: single view, geometric product, CEN, MAMM-MD w/o appearance, MAMM-MD, and MAMM-MD
  w/o geometry in domain;
- LARGE, the mean sensitivity at 0.025-4 FPI;
- the paired breast-level bootstrap and the Holm correction.

Protocol:
- only paired views are evaluated;
- the unannotated view of a positive breast is used as a partner but not scored;
- VinDr models tested on MV-DDSM / INbreast use the strict BI-RADS convention: BI-RADS <= 2
  masses are background, as in VinDr annotations.

## Reproducibility checks (done before release)
- Converter: bit-identical images to our build on 125 images (4 DDSM scanners + CBIS-DDSM).
- Dataloader: same views, breast pairs and order, and the same boxes, as the code of the paper, on
  all splits with and without the BI-RADS >= 3 filter. Pixels, depth planes and axes identical on
  sampled breasts.
- Detector: identical detections to the paper's RetinaNet implementation (CBIS and VinDr checkpoints).
- Candidate extraction: records identical to the paper caches (detections, depth, sizes, TP flags,
  embeddings).
- Training: retraining a head with this code gives bit-identical weights to the paper checkpoints
  (RetinaNet CBIS geoapp seed 0, DETR VinDr appo seed 1); the same holds for CEN (RetinaNet CBIS
  seed 0).
- Evaluation: all 60 LARGE values of the paper (12 settings × 5 methods) reproduced exactly from the
  released caches and checkpoints.

## Checkpoints
*[hosting to be decided]* detectors (3 RetinaNet, 3 Deformable DETR), patch classifier, heads
(48 = 6 pipelines × variants × 3 seeds), CEN (18).
