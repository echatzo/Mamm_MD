"""Export the CBIS-DDSM-IRMA release files from the internal build (provenance only).

This script is NOT needed to use the dataset. It documents how the released files
in data/cbis_ddsm_irma/ were produced from the internal build of the dataset
(breast_cancer_clean/src/cbis_ddsm_multi), and it only runs on the machine where
that build lives. Users start from the released files and data/convert_cbis_ddsm_irma.py.

Outputs (data/cbis_ddsm_irma/):
    correspondence_cases.csv   one row per CBIS-DDSM patient: matched DDSM case or none
    correspondence_images.csv  one row per CBIS-DDSM full mammogram: TCIA series + IRMA file
    annotations.json           every image of CBIS-DDSM-IRMA: source, role, split, lesions,
                               BI-RADS, pathology, breast geometry
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed

BUILD_SRC = "/home/chatzopoulos/edouard/breast_cancer_clean/src/cbis_ddsm_multi"
DATASET_ROOT = "/storage2/edouard/data/CBIS-IRMA-DDSM-multi"
CBIS_ROOT = "/storage/edouard/data/retinanet_dcm"          # CBIS-DDSM full mammograms, boxes, ROI masks
IRMA_ROOT = "/storage2/edouard/data/DDSM_IRMA"
MINI_DDSM = "/storage/edouard/data/miniddsm2/MINI-DDSM-Complete-PNG-16"
CBIS_META = os.path.join(BUILD_SRC, "cbis_meta")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cbis_ddsm_irma")

# Corrections found after the dataset build (pixel check and visual review, 2026-10-29). They are
# metadata only: annotations.json keeps the breast grouping of the dataset used in the paper.
EXTRA_MATCH = {"P_00192": ("C_0478", "benign_13")}
# CBIS-DDSM mass box removed in the annotation tool; whether it is a mass is uncertain (review 2026-10-30)
UNCERTAIN_MASS = {"P_00847_LEFT_MLO": "BENIGN 852 2232 1123 2502 MASS"}
EXTRA_IMAGE_CASE = {"P_00192_RIGHT_CC": "C_0478", "P_01264_RIGHT_MLO": "A_1500@cancer_08"}
NOTES = {
    "P_00192_RIGHT_CC": "Matched after the dataset build by pixel identity to DDSM C_0478 (benign_13). The same film "
                        "is also in the dataset as C_0478_1.RIGHT_CC (negative). Both are single views, hence not "
                        "scored by the paired evaluation.",
    "C_0478_1.RIGHT_CC.LJPEG.png": "Same film as the CBIS-DDSM image P_00192_RIGHT_CC, which carries a benign mass: "
                                   "not a true negative. Single view, not scored by the paired evaluation.",
    "P_00847_LEFT_MLO": "The CBIS-DDSM mass (BI-RADS 2, benign) was removed during the review: uncertain whether it is "
                        "a mass. Flagged `uncertain`; in the paper this view was positive without any box.",
    "P_01264_RIGHT_MLO": "CBIS-DDSM patient P_01264 merges two DDSM cases numbered A_1500: the left breast is "
                         "A_1500 of benign_05, the right breast A_1500 of cancer_08.",
}

sys.path.insert(0, BUILD_SRC)
import canonical  # noqa: E402  (scanner resolution, same code as the build)


# --------------------------------------------------------------------------- scanners
def digitizer_by_case(case_vol: dict) -> dict:
    """Mini-DDSM .ics DIGITIZER per case, gap-filled exactly as build_dataset.py does."""
    out = {}
    for cls in ("Normal", "Benign", "Cancer"):
        base = os.path.join(MINI_DDSM, cls)
        for cd in os.listdir(base):
            p = os.path.join(base, cd)
            if not os.path.isdir(p):
                continue
            ics = [f for f in os.listdir(p) if f.lower().endswith(".ics")]
            key = next((f"{m.group(1)}_{m.group(2)}" for f in os.listdir(p)
                        for m in [re.match(r"([A-D])_(\d+)_\d+\.(LEFT|RIGHT)_(CC|MLO)\.png$", f)] if m), None)
            if not ics or key is None:
                continue
            with open(os.path.join(p, ics[0]), errors="ignore") as fh:
                for line in fh:
                    if line.strip().startswith("DIGITIZER"):
                        out[key] = line.split()[1]
                        break
    vol_counts = collections.defaultdict(collections.Counter)
    for c, d in out.items():
        if case_vol.get(c):
            vol_counts[case_vol[c]][d] += 1
    vol_major = {v: c.most_common(1)[0][0] for v, c in vol_counts.items() if len(c) == 1}
    for c, v in case_vol.items():
        if c not in out and v in vol_major:
            out[c] = vol_major[v]
    let_counts = collections.defaultdict(collections.Counter)
    for c, d in out.items():
        let_counts[c.split("_")[0]][d] += 1
    let_major = {l: c.most_common(1)[0][0] for l, c in let_counts.items() if len(c) == 1}
    for c in case_vol:
        if c not in out and c.split("_")[0] in let_major:
            out[c] = let_major[c.split("_")[0]]
    return out


# --------------------------------------------------------------------------- CBIS lesions
def cbis_lesion_table() -> dict:
    """{(patient, side, view, 'MASS'|'CALC', abnormality_id): metadata} from the official CSVs."""
    out = {}
    for kind, typ in (("mass", "MASS"), ("calc", "CALC")):
        for part, split in (("train", "Training"), ("test", "Test")):
            df = pd.read_csv(os.path.join(CBIS_META, f"{kind}_case_description_{part}_set.csv"))
            dens = "breast_density" if "breast_density" in df else "breast density"
            for r in df.to_dict("records"):
                key = (r["patient_id"], r["left or right breast"], r["image view"], typ, int(r["abnormality id"]))
                m = dict(type=typ, cbis_abnormality_id=int(r["abnormality id"]),
                         pathology=r["pathology"], birads=int(r["assessment"]), subtlety=int(r["subtlety"]),
                         breast_density=int(r[dens]), cbis_split=split)
                if typ == "MASS":
                    m.update(mass_shape=r["mass shape"], mass_margins=r["mass margins"])
                else:
                    m.update(calc_type=r["calc type"], calc_distribution=r["calc distribution"])
                out[key] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items()}
    return out


_LS: dict = {}


def _listdir(d: str) -> list:
    if d not in _LS:
        _LS[d] = os.listdir(d)
    return _LS[d]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def box_file(dcm_path: str) -> tuple[str, bool]:
    d = os.path.dirname(os.path.dirname(dcm_path)); stem = os.path.splitext(os.path.basename(dcm_path))[0]
    c = os.path.join(d, "corrected_bounding_boxes", stem + ".txt")
    return (c, True) if os.path.exists(c) else (os.path.join(d, "bounding_boxes", stem + ".txt"), False)


def link_lesions(dcm_path: str, patient: str, side: str, view: str, table: dict) -> dict:
    """Box lines of one CBIS image, each linked to its CBIS abnormality through the ROI masks.

    The boxes were made as the bounding box of each ROI mask (x2, y2 exclusive); a few were
    corrected by hand afterwards (corrected_bounding_boxes/). Each box line is attached to the
    abnormality whose mask (or a connected component of it) overlaps it best."""
    path, corrected = box_file(dcm_path)
    split_dir = os.path.dirname(os.path.dirname(dcm_path)); stem = os.path.splitext(os.path.basename(dcm_path))[0]
    masks = []
    mdir = os.path.join(split_dir, "masks")
    # abnormality ids are shared by masses and calcifications of an image (mass_1, calcification_2, ...)
    pat_re = re.compile(re.escape(stem) + r"_(mass|calcification)_(\d+)\.png$")
    for fn in sorted(f for f in _listdir(mdir) if f.startswith(stem + "_")):
        mm = pat_re.match(fn)
        if mm:
            typ, k = ("MASS" if mm.group(1) == "mass" else "CALC"), int(mm.group(2))
            m = cv2.imread(os.path.join(mdir, fn), cv2.IMREAD_GRAYSCALE) > 0
            cand = []
            if m.any():
                ys, xs = np.nonzero(m); cand.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
                n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
                cand += [(s[0], s[1], s[0] + s[2], s[1] + s[3]) for s in st[1:]]
            masks.append((typ, k, cand))
    lesions, problems = [], []
    for line in open(path):
        it = line.split()
        if not it:
            continue
        if len(it) < 6:
            # Malformed line written by the annotation tool (ignored by the paper's loader): the original
            # CBIS-DDSM box multiplied by the display scale, without lesion type. The original box is
            # recovered exactly (the four coordinates share one ratio) and restored, unless a valid
            # corrected box already covers it.
            # (a coordinate moved by hand before the bug does not share the ratio: >= 3 of 4 must)
            raw = [float(v) for v in it[1:5]]
            orig = [l.split() for l in open(path.replace("corrected_bounding_boxes", "bounding_boxes")) if l.split()]
            hit = []
            for o in orig:
                r = [raw[k] / float(o[k + 1]) for k in range(4) if float(o[k + 1]) > 0]
                f = [x for x in r if sum(abs(x - y) < 1e-3 for y in r) >= 3]
                if len(r) == 4 and f:
                    hit.append((o, f[0]))
            valid = [[float(v) for v in l.split()[1:5]] for l in open(path) if len(l.split()) >= 6]
            box_ = [v / hit[0][1] for v in raw] if len(hit) == 1 else None
            if box_ is None or any(_iou(box_, b) > 0.3 for b in valid):
                problems.append(f"{stem}: dropped malformed line {line.strip()!r} "
                                f"({'covered by a corrected box' if box_ else 'no original match'})")
                continue
            o = hit[0][0]
            it, restored = [o[0]] + [f"{v:.2f}" for v in box_] + [o[5]], True
            problems.append(f"{stem}: restored {' '.join(it)} (original {' '.join(o[1:5])}, scale {hit[0][1]:.4f}) "
                            f"from malformed line {line.strip()!r}")
        else:
            restored = False
        box = [float(v) for v in it[1:5]]
        typ = it[5]
        best, bk = None, 0.0
        for mt, k, cand in masks:
            if typ is not None and mt != typ:
                continue
            s = max((_iou(box, c) for c in cand), default=0.0)
            if s > bk:
                best, bk = (mt, k), s
        meta = table.get((patient, side, view, best[0], best[1])) if best else None
        if meta is None or bk < 0.1:
            # no CBIS ROI under this box: a lesion added by hand in the annotation tool
            meta = None
            if not corrected:
                problems.append(f"{stem}: box {box} {typ} -> {best} iou {bk:.2f} (not a corrected file)")
        # the annotation tool rewrites every box of an image it touches: a box equal to a CBIS-DDSM
        # box (within 0.5 px) is the original one, not a correction
        unchanged = corrected and not restored and any(
            np.allclose(box, [float(v) for v in o.split()[1:5]], atol=0.5)
            for o in open(path.replace("corrected_bounding_boxes", "bounding_boxes")) if o.split())
        src = ("restored_original" if restored else "roi_mask_bbox" if (not corrected or unchanged)
               else ("manual_correction" if meta else "manual_addition"))
        le = {"bbox": box, "type": typ, "pathology": it[0], "box_source": src,
              "link_iou": round(bk, 4) if meta else None}
        if meta:
            le.update({k: v for k, v in meta.items() if k != "pathology"})
            le["pathology_cbis"] = meta["pathology"]
        lesions.append(le)
    if stem in UNCERTAIN_MASS:      # CBIS-DDSM mass removed by hand, kept and flagged as uncertain
        it = UNCERTAIN_MASS[stem].split()
        box = [float(v) for v in it[1:5]]
        best, bk = None, 0.0
        for mt, k, cand in masks:
            sc = max((_iou(box, c) for c in cand), default=0.0) if mt == "MASS" else 0.0
            if sc > bk:
                best, bk = (mt, k), sc
        meta = table.get((patient, side, view, best[0], best[1])) if best else None
        le = {"bbox": box, "type": "MASS", "pathology": it[0], "box_source": "roi_mask_bbox",
              "link_iou": round(bk, 4) if meta else None, "uncertain": True}
        if meta:
            le.update({k: v for k, v in meta.items() if k != "pathology"})
            le["pathology_cbis"] = meta["pathology"]
        lesions.append(le)
    return {"lesions": lesions, "problems": problems, "corrected": corrected}


def tcia_ids(dcm_path: str) -> dict:
    d = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    return {"tcia_patient_id": str(d.PatientID), "study_instance_uid": str(d.StudyInstanceUID),
            "series_instance_uid": str(d.SeriesInstanceUID)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = pd.read_csv(os.path.join(DATASET_ROOT, "manifest.csv"))
    geo = pd.read_csv(os.path.join(DATASET_ROOT, "geometry_cache.csv")).set_index("image_id")
    pat = pd.read_csv(os.path.join(BUILD_SRC, "out_excl", "cbis_to_ddsm_patients.csv"))
    img_map = pd.read_csv(os.path.join(BUILD_SRC, "out_excl", "cbis_to_ddsm.csv"))
    z = np.load(os.path.join(BUILD_SRC, "caches", "irma.npz"), allow_pickle=False)
    case_vol = {str(c).split("@")[0]: str(v) for c, v in zip(z["meta_case"], z["meta_volume"])}
    vol_path = {str(c).split("@")[0]: str(p) for c, p in zip(z["meta_case"], z["meta_pathology"])}
    dig = digitizer_by_case(case_vol)
    table = cbis_lesion_table()

    # ---------------- case-level correspondence
    final_case = man[(man.source == "cbis_dicom") & man.case.notna()].groupby("patient").case.first()
    all_patients = sorted(man[man.source == "cbis_dicom"].patient.unique())
    P = pat.set_index("patient")
    rows = []
    for p in all_patients:
        c = final_case.get(p)
        r = P.loc[p] if p in P.index else None
        rows.append(dict(
            cbis_patient_id=p,
            matched=c is not None,
            ddsm_case=c, ddsm_volume=case_vol.get(c) if c else None,
            ddsm_volume_pathology=vol_path.get(c) if c else None,
            scanner=canonical.resolve_scanner(c, case_vol.get(c), dig) if c else None,
            match_method="content" if c is not None else None,
            n_views_matched=int(r["n_views"]) if (c is not None and r is not None) else 0))
    cases = pd.DataFrame(rows)
    assert cases.matched.sum() == len(final_case)
    assert (cases[cases.matched].ddsm_case.values == final_case.reindex(cases[cases.matched].cbis_patient_id).values).all()
    cases["scanner_source"] = np.where(cases.scanner.notna(), "case letter / volume / DDSM .ics", None)

    # pixel-level verification (tools/pixel_check.py) and corrections found after the dataset build
    pix = pd.read_csv(os.path.join(BUILD_SRC, "out_excl", "pixel_check.csv"))
    pix["patient"] = pix.cbis_image.str.rsplit("_", n=2).str[0]
    pix["identical"] = (pix.status == "ok") & ((pix.pixel_equal_frac >= 0.98) | (pix.spearman.fillna(0) >= 0.999))
    for p_, (c_, v_) in EXTRA_MATCH.items():          # unmatched by the content matcher, found by pixels
        k = cases.cbis_patient_id == p_
        assert not cases.loc[k, "matched"].any()
        cases.loc[k, ["matched", "ddsm_case", "ddsm_volume", "ddsm_volume_pathology", "match_method", "n_views_matched"]] = \
            [True, c_, v_, vol_path.get(c_), "pixel_search", 1]
    pp = pix.groupby("patient").identical.all()
    cases["pixel_verified"] = cases.cbis_patient_id.map(pp).fillna(False).astype(bool) & cases.matched
    sd = (pix.status != "different size").groupby(pix.patient).all()
    cases["same_dimensions"] = cases.cbis_patient_id.map(sd).where(cases.matched)
    found = pix[pix.scanner.notna()].groupby("patient").scanner.agg(lambda x: x.mode()[0])
    k = cases.matched & cases.scanner.isna()
    cases.loc[k, "scanner"] = cases.loc[k, "cbis_patient_id"].map(found)
    cases.loc[k & cases.scanner.notna(), "scanner_source"] = "pixel calibration"
    ovr = pd.read_csv(os.path.join(BUILD_SRC, "out_excl", "breast_overrides.csv"))
    cases["breast_override"] = cases.cbis_patient_id.map(
        {r.patient: f"{r.laterality}: {r.ddsm_case}" for r in ovr.itertuples()})
    cases.to_csv(os.path.join(a.out, "correspondence_cases.csv"), index=False)
    pixd = pix.set_index("cbis_image")

    # ---------------- image-level correspondence (every CBIS full mammogram)
    cb = man[man.source == "cbis_dicom"].copy()
    ids = Parallel(n_jobs=a.jobs)(delayed(tcia_ids)(p) for p in cb.path)
    irma_of = dict(zip(img_map.cbis_image, img_map.ddsm_png))
    im_rows = []
    for (_, r), t in zip(cb.iterrows(), ids):
        x = pixd.loc[r.image_id] if r.image_id in pixd.index else None
        case_ = EXTRA_IMAGE_CASE.get(r.image_id, r.case if isinstance(r.case, str) else None)
        im_rows.append(dict(cbis_image_id=r.image_id, cbis_patient_id=r.patient, laterality=r.lat, view=r.view,
                            **t, ddsm_case=case_, ddsm_file=x.ddsm_file if x is not None else None,
                            height=int(r.orig_h), width=int(r.orig_w),
                            same_dimensions=(x.status != "different size") if x is not None else None,
                            pixel_equal_frac=x.pixel_equal_frac if x is not None else None,
                            pixel_spearman=x.spearman if x is not None else None,
                            pixel_identical=bool(x.identical) if x is not None else None))
    images = pd.DataFrame(im_rows)
    images.to_csv(os.path.join(a.out, "correspondence_images.csv"), index=False)

    # ---------------- annotations.json
    links = dict(zip(cb.image_id, Parallel(n_jobs=a.jobs)(
        delayed(link_lesions)(r.path, r.patient, r.lat, r.view, table) for r in cb.itertuples())))
    tids = dict(zip(cb.image_id, ids))
    problems = [p for v in links.values() for p in v["problems"]]
    man["split"] = man.split_cbis_mass_localval.where(man.split_cbis_mass_localval.notna(), man.split_added_negatives)
    nn = lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
    out_images = []
    for r in man.itertuples():
        fn = r.image_id if r.image_id.lower().endswith(".png") else r.image_id + ".png"
        e = dict(image_id=r.image_id, file_name=fn,
                 source="cbis-ddsm" if r.source == "cbis_dicom" else "ddsm-irma",
                 cbis_patient_id=nn(r.patient), ddsm_case=nn(r.case),
                 laterality=r.lat, view=r.view, height=int(r.orig_h), width=int(r.orig_w),
                 scale_2048=float(r.scale_2048),
                 role=r.role, label_source=r.mass_status_source, mass_verification=nn(r.mass_verification),
                 split=nn(r.split), split_cbis_mass_official=nn(r.split_cbis_mass))
        if r.source == "cbis_dicom":
            e["cbis"] = tids[r.image_id]
            e["lesions"] = links[r.image_id]["lesions"]
            e["cbis_mass_birads"] = sorted({v["birads"] for k, v in table.items()
                                            if k[:3] == (r.patient, r.lat, r.view) and k[3] == "MASS"})
            dens = {le.get("breast_density") for le in e["lesions"] if le.get("breast_density") is not None}
            e["breast_density"] = dens.pop() if len(dens) == 1 else None
            x = pixd.loc[r.image_id] if r.image_id in pixd.index else None
            e["ddsm_file"] = x.ddsm_file if x is not None else None
            if x is not None and x.status == "ok":
                e["pixel_check"] = dict(equal_frac=float(x.pixel_equal_frac),
                                        spearman=None if np.isnan(x.spearman) else float(x.spearman))
        else:
            e["ddsm_file"] = os.path.relpath(r.path, IRMA_ROOT)
            e["ddsm_volume"] = case_vol.get(r.case)
            e["scanner"] = canonical.resolve_scanner(r.case, case_vol.get(r.case), dig)
            e["lesions"] = []
        if r.image_id in geo.index:
            g = geo.loc[r.image_id]; s = float(r.scale_2048)
            e["geometry"] = dict(has_axis=bool(int(g.has_axis)), nipple_source=g.nipple_src,
                                 nipple_xy=None if np.isnan(g.nx_2048) else [float(g.nx_2048) / s, float(g.ny_2048) / s],
                                 axis=None if np.isnan(g.ax) else [float(g.ax), float(g.ay)],
                                 mm_per_px=None if np.isnan(g.mm_per_px_2048) else float(g.mm_per_px_2048) * s,
                                 posterior_nipple_line_mm=None if np.isnan(g.pnl_mm) else float(g.pnl_mm))
        else:
            e["geometry"] = None
        if r.image_id in NOTES:
            e["note"] = NOTES[r.image_id]
        out_images.append(e)
    # A view without any mass box, in the same breast as a view with a mass, cannot be trusted as a
    # negative: the mass may be visible but not outlined. Such DDSM views become support views
    # (`role_paper` keeps the role used in the paper). Review 2026-10-30.
    breast = collections.defaultdict(list)
    for e in out_images:
        breast[(e["ddsm_case"] or e["cbis_patient_id"] or e["image_id"], e["laterality"])].append(e)
    n_support = 0
    for es in breast.values():
        masses = [x for x in es if any(l["type"] == "MASS" and not l.get("uncertain") for l in x["lesions"])]
        for e in es:
            if e["role"] == "negative" and masses and all(m["view"] != e["view"] for m in masses):
                e["role_paper"], e["role"] = "negative", "support"
                n_support += 1
    print(f"{n_support} DDSM negative views ipsilateral to a mass -> support")
    info = dict(name="CBIS-DDSM-IRMA", version="1.0", date=str(datetime.date.today()),
                description="CBIS-DDSM full mammograms completed with the missing views and normal exams of the "
                            "DDSM release distributed by RWTH Aachen (IRMA), converted to the CBIS-DDSM intensity "
                            "pipeline. Coordinates are in native pixels of each image.",
                n_images=len(out_images))
    with open(os.path.join(a.out, "annotations.json"), "w") as fh:
        json.dump({"info": info, "images": out_images}, fh, indent=1)
    print(f"cases {len(cases)} ({int(cases.matched.sum())} matched), images {len(images)}, "
          f"annotations {len(out_images)} images, {sum(len(e['lesions']) for e in out_images)} lesions")
    print(cases.match_method.value_counts(dropna=False).to_string(), "\npixel verified", int(cases.pixel_verified.sum()))
    print(f"{len(problems)} box-lesion link problems")
    for p in problems[:40]:
        print("  ", p)


if __name__ == "__main__":
    main()
