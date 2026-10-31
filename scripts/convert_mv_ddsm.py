"""Build the MV-DDSM images from CBIS-DDSM (TCIA) and the IRMA release of DDSM.

Inputs
  --cbis-root   CBIS-DDSM as downloaded from TCIA (any folder layout: DICOM files are found by
                their SeriesInstanceUID, listed in annotations.json)
  --irma-root   DDSM release of RWTH Aachen / IRMA (folders cancer_*/benign_*/normal_*, 16-bit PNG)
  --annotations data/mv_ddsm/annotations.json

Outputs (--out)
  images/       full-resolution 16-bit PNG
                  CBIS-DDSM images: DICOM pixel data, format conversion only
                  IRMA images: optical-density calibration of CBIS-DDSM (mammd.data.ddsm_calibration)
  images_2048/  the same images padded on the bottom/right to a square and resized to 2048 x 2048
                (INTER_AREA), so native coordinates map to it by `scale_2048` with no offset.
                This is the input of the paper's detectors.

    python scripts/convert_mv_ddsm.py --cbis-root /data/CBIS-DDSM --irma-root /data/DDSM_IRMA \
        --out /data/MV-DDSM --jobs 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pydicom
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd.data.ddsm_calibration import calibrate  # noqa: E402

TARGET = 2048
HERE = os.path.dirname(os.path.abspath(__file__))


def _series_uid(path: str) -> tuple[str, str | None]:
    try:
        return path, str(pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
                         .SeriesInstanceUID)
    except Exception:
        return path, None


def index_cbis(root: str, wanted: set[str], jobs: int) -> dict[str, str]:
    """{SeriesInstanceUID: DICOM path} for the full mammograms listed in annotations.json."""
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(".dcm")]
    found = {}
    for p, uid in Parallel(n_jobs=jobs, batch_size=64)(delayed(_series_uid)(f) for f in files):
        if uid in wanted:
            found[uid] = p
    return found


def load_cbis(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def load_irma(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"cannot read {path}")
    return (arr[..., 0] if arr.ndim == 3 else arr).astype(np.float32)


def square_2048(img: np.ndarray) -> np.ndarray:
    """Pad bottom/right to a square (origin unchanged), then resize to 2048 x 2048."""
    h, w = img.shape
    s = max(h, w)
    sq = cv2.copyMakeBorder(img, 0, s - h, 0, s - w, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(sq, (TARGET, TARGET), interpolation=cv2.INTER_AREA if s > TARGET else cv2.INTER_LANCZOS4)


def write16(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, np.clip(img, 0, 65535).astype(np.uint16)):
        raise IOError(f"cannot write {path}")


def convert(e: dict, dcm: str | None, irma_root: str, out: str, full: bool) -> dict:
    try:
        if e["source"] == "cbis-ddsm":
            if dcm is None:
                return {"image_id": e["image_id"], "error": "DICOM not found in --cbis-root"}
            img = load_cbis(dcm)
        else:
            img = calibrate(load_irma(os.path.join(irma_root, e["ddsm_file"])), e["scanner"])[0].astype(np.float32)
        if img.shape != (e["height"], e["width"]):
            return {"image_id": e["image_id"], "error": f"size {img.shape} != {(e['height'], e['width'])}"}
        if full:
            write16(os.path.join(out, "images", e["file_name"]), img)
        write16(os.path.join(out, "images_2048", e["file_name"]), square_2048(img))
        return {"image_id": e["image_id"]}
    except Exception as ex:  # noqa: BLE001
        return {"image_id": e["image_id"], "error": f"{type(ex).__name__}: {ex}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cbis-root", required=True)
    ap.add_argument("--irma-root", required=True)
    ap.add_argument("--annotations", default=os.path.join(HERE, "..", "data", "mv_ddsm", "annotations.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-full", action="store_true", help="only write images_2048/")
    ap.add_argument("--only", nargs="*", default=None, help="convert only these image_ids (for testing)")
    a = ap.parse_args()

    images = json.load(open(a.annotations))["images"]
    if a.only:
        keep = set(a.only)
        images = [e for e in images if e["image_id"] in keep]
    wanted = {e["cbis"]["series_instance_uid"] for e in images if e["source"] == "cbis-ddsm"}
    dcm = index_cbis(a.cbis_root, wanted, a.jobs) if wanted else {}
    print(f"{len(images)} images; CBIS-DDSM DICOMs found {len(dcm)}/{len(wanted)}", flush=True)

    res = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(convert)(e, dcm.get(e["cbis"]["series_instance_uid"]) if e["source"] == "cbis-ddsm" else None,
                         a.irma_root, a.out, not a.no_full) for e in images)
    err = [r for r in res if "error" in r]
    print(f"done: {len(res) - len(err)} images, {len(err)} errors")
    if err:
        with open(os.path.join(a.out, "conversion_errors.json"), "w") as fh:
            json.dump(err, fh, indent=1)
        print(f"see {os.path.join(a.out, 'conversion_errors.json')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
