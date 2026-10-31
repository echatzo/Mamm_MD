# CBIS-DDSM-IRMA

CBIS-DDSM [1] gives, for each curated patient, only the views of the breast that carries the
annotated finding. It has no normal exams, and for most patients one of the two views (CC or MLO)
of the breast is missing. CBIS-DDSM-IRMA completes it with the views and normal exams of the
original DDSM [2], taken from the release distributed by RWTH Aachen (IRMA) [3]. All the views are
converted to the CBIS-DDSM intensity pipeline.

| | images |
|---|---|
| CBIS-DDSM full mammograms (1,566 patients), with their lesion boxes | 3,032 |
| DDSM views added to CBIS-DDSM patients (other view, other breast) | 2,356 |
| DDSM views of other exams (normal volumes and non-curated cases) | 3,584 |
| **total** | **8,972** |

This folder only holds annotations. The images are rebuilt from the two public sources with
`scripts/convert_cbis_ddsm_irma.py`.

## Files

| file | content |
|---|---|
| `correspondence_cases.csv` | one row per CBIS-DDSM patient: matched DDSM case and volume, scanner, match method, pixel verification, same dimensions |
| `correspondence_images.csv` | one row per CBIS-DDSM full mammogram: TCIA identifiers (PatientID, Study/Series UID), matched IRMA file |
| `annotations.json` | every image of CBIS-DDSM-IRMA: source, role, split, lesions, breast geometry |

### `annotations.json`

`{"info": {...}, "images": [...]}`. Each image has:

- `image_id`, `file_name` (output file of the converter), `source` (`cbis-ddsm` or `ddsm-irma`),
  `cbis_patient_id`, `ddsm_case`, `laterality`, `view`, `height`, `width` (native pixels), `scale_2048`
  (native to `images_2048/` coordinates: multiply, no offset).
- `role` (`role_paper`: the role in the paper, when it differs):
  - `annotated`: a CBIS-DDSM image with its boxes;
  - `negative`: verified to contain no mass;
  - `support`: usable as the context view of a pair, never as a training or evaluation target. Either
    a mass is present (DDSM overlay) but was not outlined to CBIS-DDSM quality (435 views), or the
    view has no mass box although the other view of the same breast has one (50 DDSM views,
    `role_paper = negative`: they were negatives in the paper).
- `label_source`: why the role was assigned.
  - `cbis_box`: CBIS-DDSM annotation;
  - `ddsm_normal_volume`: DDSM normal exam;
  - `mini_overlay_clean`, `mini_overlay_calc_only`, `mini_overlay_mass`: verification against the
    original DDSM `.OVERLAY` files (Mini-DDSM [4] copy).
  Views that could not be verified are not part of the dataset.
- `mass_verification` (CBIS-DDSM calcification images): `verified_clean` means checked against the
  DDSM overlay to contain no mass, hence a valid mass negative.
- `split`: `train` / `val` / `test` for mass detection, or null.
  - CBIS-DDSM mass patients follow the **official CBIS-DDSM mass split**. Validation is 15% of the
    official training patients (patient level, stratified by pathology, seed 42).
  - The other images (added negatives and verified calcification images) are split 80/20, then 15%
    of the training part goes to validation. The unit is the DDSM case, stratified by breast density ×
    scanner, seed 42.
  - `split_cbis_mass_official` keeps the official assignment.
- `lesions` (CBIS-DDSM images): `bbox` [x1, y1, x2, y2] in native pixels, `type` (`MASS`/`CALC`),
  `pathology`, `box_source`, and, when linked to a CBIS-DDSM abnormality, `cbis_abnormality_id`,
  `birads`, `subtlety`, `breast_density`, `mass_shape`/`mass_margins` or
  `calc_type`/`calc_distribution`.
  - `roi_mask_bbox`: bounding box of the CBIS-DDSM ROI mask (3,541 boxes).
  - `manual_correction`: box corrected by hand (16).
  - `manual_addition`: mass missing from CBIS-DDSM and added by hand (36; no BI-RADS).
  - `restored_original`: CBIS-DDSM mass box lost by the annotation tool and restored (5; used only
    with `corrected=True`, see below).
- `cbis_mass_birads`: BI-RADS assessments of the image's CBIS-DDSM masses (BI-RADS filters work per
  image, because CBIS-DDSM boxes carry no lesion identifier).
- `ddsm_file` (path in the IRMA release), `ddsm_volume`, `scanner` (IRMA images).
- `geometry`: `nipple_xy` (native pixels), `axis` (unit vector from the nipple towards the chest wall),
  `mm_per_px`, `posterior_nipple_line_mm`, `has_axis`, `nipple_source`.
  - `nipple_source` is `manual` (1,622 views) or `detector` (7,350 views).
  - `has_axis = false` (482 views) when the pectoral line could not be fitted in MLO or the posterior
    nipple line falls outside 40-160 mm.
  - The depth of a point p is `<p - nipple_xy, axis> * mm_per_px`, in millimetres.

## How the two releases were matched

There is no official crosswalk between CBIS-DDSM patients and DDSM cases, so they were matched by
image content. Both releases come from the same digitised films.
1. Every view is reduced to a normalised low-resolution thumbnail:
   - percentile normalisation;
   - breast segmentation, crop and flip to a fixed orientation;
   - resizing to 256 × 192, local contrast normalisation;
   - L2 normalisation.
   A second thumbnail of the whole image (no crop, no flip) is also kept.
2. Views of the same laterality and view are compared by cosine similarity, with a bonus when the
   raw image dimensions are identical.
3. Each patient is assigned greedily one to one, from its views jointly.

**1,440 of 1,566 patients are matched** (`match_method`):
- 1,439 by this content matching (`content`);
- 1 by the pixel check below (`pixel_search`, P_00192).


The 126 unmatched patients all have *benign without callback* findings only; their DDSM volumes are
not in the IRMA release. They keep only their CBIS-DDSM images.

**Pixel-level verification.** Every match was then checked pixel by pixel:
- for each CBIS-DDSM image, the matched IRMA file is calibrated (see below) and compared inside the
  breast;
- 2,712 of the 2,714 images are identical on ≥ 98% of the breast pixels (2,574 on ≥ 99.9%), whereas
  any other IRMA image of the same size is identical on about 1% at most;
- the 2 remaining images (P_00543 and P_01497, RIGHT_MLO) have a Spearman correlation of 1.0 with
  CBIS-DDSM: the same film, stored with a different scale.

All 1,440 matches are therefore verified (`pixel_verified`, and per image `pixel_equal_frac`,
`pixel_spearman`). Every matched CBIS-DDSM image has exactly the dimensions of its IRMA file
(`same_dimensions`). The same check gives the scanner of 119 letter-A cases that have no DDSM .ics file
(`scanner_source = pixel calibration`; all Howtek MGH).

**Corrections made after the paper.** By default, `mammd.data.cbis_ddsm_irma` reproduces the dataset
of the paper exactly. `corrected=True` applies the following corrections:
- **Positive views.** In the paper, a view is positive only when CBIS-DDSM lists a mass for it. As a
  result, 7 hand-added masses on 6 views were unused: these views were left out of the
  mass-detection set. With the correction, a view is positive when it carries a mass box.
- **Restored boxes.** The annotation tool rewrote 5 CBIS-DDSM mass boxes at the display scale,
  without lesion type, so the paper's loader ignored them. The boxes are recovered exactly (the
  coordinates share one ratio) and restored.
  - The 5 views are P_00148 R-CC, P_00442 R-CC and R-MLO, P_00435 R-MLO (train) and P_01833 R-MLO
    (test).
  - In the paper they were positive views without boxes: the test view is a single view, so it was
    not scored.
  - The models of the paper were trained without these 4 training boxes.

- **Support views.** A DDSM view without any mass box, in the same breast as a view with a mass,
  cannot be trusted as a negative: the mass may be visible but not outlined. These 50 views (34
  train, 5 val, 11 test) were negatives in the paper; the RetinaNet detector and the patch
  classifier saw the training ones as negatives (the heads, CEN and the evaluation never did). They
  are now support views.
  - With `corrected=True`, every support view of a breast with a positive view is kept as its
    partner (`support=True`, no box; the evaluation does not score it).
  - Other support views are left out.
  - On the test set, this gives 175 paired breasts with a mass instead of 153.
- **Uncertain mass.** The CBIS-DDSM mass of P_00847 L-MLO (BI-RADS 2) was removed during our
  review because it is uncertain. It is kept with `uncertain: true`, never used as a box, and with
  `corrected=True` the view is left out (neither a positive nor a negative).

Still missing: the masses on the second view of P_01467 R, P_00652 L, P_00591 R (test) and P_01567 L (train).

On the CBIS-DDSM-IRMA test set, the positive-view correction alone changes LARGE by 0.1-0.5 points for
every method and none of the comparisons of the paper. The support partners were not evaluated yet.

**Known issues** (field `note` in `annotations.json`; this version keeps the breast grouping of the
dataset used in the paper):
- **P_01264** merges two DDSM cases with the same number: the left breast is A_1500 of `benign_05`,
  the right breast A_1500 of `cancer_08` (`breast_override`).
- **P_00192** was matched only by the pixel check, after the dataset build. Its RIGHT_CC is the same
  film as `C_0478_1.RIGHT_CC`, which the dataset contains as a negative of the test split although
  the film carries a benign mass. Both are single views, so the paired evaluation scores neither.

The CBIS-DDSM full mammogram and its IRMA file share the same pixel grid, so the CBIS-DDSM boxes
apply to the IRMA file without any transform.

## Intensity calibration

The IRMA files hold raw scanner grey levels. They are converted with the ddsm_tools pipeline used to
build CBIS-DDSM [1]:
- per-scanner optical density calibration (DBA, Howtek MGH, Howtek ISMD, Lumisys);
- clipping to [0.05, 3.0] OD;
- a fixed linear map to 16 bits.

The output equals the CBIS-DDSM pixel values exactly on 53 sampled matched pairs, covering the four
scanners.

One known difference concerns saturated pixels of the DBA scanner (raw value 65535), mostly a strip
along the film edge:
- CBIS-DDSM sets them to 0;
- our build keeps them white (OD clipped to 0.05).
About half of the DBA images contain such pixels, typically less than 0.6% of the image (at most
about 3%). The other three scanners have none. The CBIS-DDSM DICOMs themselves are only
format-converted.

## Building the images

```
python scripts/convert_cbis_ddsm_irma.py --cbis-root <CBIS-DDSM download> --irma-root <DDSM IRMA release> \
    --out <CBIS-DDSM-IRMA> --jobs 16
```
- Writes `images/` (full resolution, 16-bit PNG) and `images_2048/` (padded bottom/right to a square,
  resized to 2048 × 2048, the input of the detectors).
- The CBIS-DDSM DICOMs are found by SeriesInstanceUID, whatever the folder layout of the download.
- The converter reproduces our internal build bit for bit (checked on 125 images covering the four
  scanners and CBIS-DDSM).

## Sources and licences

[1] R. S. Lee et al., *A curated mammography data set for use in computer-aided detection and diagnosis
research*, Scientific Data 4, 170177 (2017). TCIA, CC BY 3.0.
[2] M. Heath et al., *The Digital Database for Screening Mammography*, IWDM 2000.
[3] DDSM release of RWTH Aachen (IRMA), 16-bit PNG, https://publications.rwth-aachen.de/record/667223
    (volumes c_01-c_15, b_01-b_14, n_01-n_12; ~10,480 images).
[4] C. D. Lekamlage et al., *Mini-DDSM*, 2020 (used only to verify labels and resolve scanners).

The annotations of this folder are derived from these sources. Using them requires complying with the
licences of CBIS-DDSM and DDSM.
