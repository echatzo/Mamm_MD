"""Pixel-level verification of the CBIS-DDSM <-> IRMA correspondence (provenance only).

For every CBIS-DDSM full mammogram with an IRMA counterpart, the IRMA file is calibrated
(mammd.data.ddsm_calibration) and compared to the CBIS-DDSM pixels inside the breast (CBIS > 0,
saturated scanner pixels excluded). A true pair is identical on >= 98% of these pixels; any other
image of the same size is identical on <= 1%. When the equality is low, the Spearman correlation
in the breast tells whether it is the same film with a different storage scale.

Letter-A cases whose scanner is unknown (no DDSM .ics) are calibrated with both MGH formulas
(DBA, Howtek); the one reproducing CBIS-DDSM gives the scanner.

Output: <build>/out_excl/pixel_check.csv, read by export_cbis_ddsm_irma.py.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd.data.ddsm_calibration import calibrate  # noqa: E402

BUILD = "/home/chatzopoulos/edouard/breast_cancer_clean/src/cbis_ddsm_multi/out_excl"
IRMA = "/storage2/edouard/data/DDSM_IRMA"
CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cbis_ddsm_irma", "correspondence_cases.csv")
# found by this check or by visual review, not by the content matcher
EXTRA = {"P_00192_RIGHT_CC": ("C_0478", "benign_13/case0478/C_0478_1.RIGHT_CC.LJPEG.png", "lumisys"),
         "P_01264_RIGHT_MLO": ("A_1500@cancer_08", "cancer_08/case1500/A_1500_1.RIGHT_MLO.LJPEG.png", None)}


def check(iid, dcm, rel, scanner):
    cb = pydicom.dcmread(dcm).pixel_array
    raw = cv2.imread(os.path.join(IRMA, rel), -1)
    if cb.shape != raw.shape:
        return dict(cbis_image=iid, ddsm_file=rel, status="different size")
    t = (cb > 0) & (raw != 65535)
    best = (-1.0, None)
    for s in ([scanner] if scanner else ["dba", "howtek_mgh"]):
        e = float((calibrate(raw.astype(np.float32), s)[0][t] == cb[t]).mean())
        best = max(best, (e, s))
    out = dict(cbis_image=iid, ddsm_file=rel, status="ok", pixel_equal_frac=round(best[0], 6), scanner=best[1])
    if best[0] < 0.98:
        idx = np.random.default_rng(0).choice(int(t.sum()), min(300000, int(t.sum())), replace=False)
        out["spearman"] = round(float(spearmanr(raw[t][idx], cb[t][idx]).correlation), 6)
    return out


def main():
    im = pd.read_csv(os.path.join(BUILD, "cbis_to_ddsm.csv"))
    sc = pd.read_csv(CASES).set_index("cbis_patient_id")["scanner"].to_dict()
    jobs = []
    for r in im.itertuples():
        if r.cbis_image in EXTRA:
            continue
        s = sc.get(r.patient)
        jobs.append((r.cbis_image, r.cbis_dcm, os.path.relpath(r.ddsm_png, IRMA), s if isinstance(s, str) else None))
    dcm = {os.path.basename(p)[:-4]: p for p in im.cbis_dcm}
    for iid, (case, rel, s) in EXTRA.items():
        p = dcm.get(iid) or [os.path.join(d, "images", iid + ".dcm") for d in
                             ("/storage/edouard/data/retinanet_dcm/train", "/storage/edouard/data/retinanet_dcm/val",
                              "/storage/edouard/data/retinanet_dcm/test")
                             if os.path.exists(os.path.join(d, "images", iid + ".dcm"))][0]
        jobs.append((iid, p, rel, s))
    res = pd.DataFrame(Parallel(n_jobs=24, batch_size=4)(delayed(check)(*j) for j in jobs))
    res.to_csv(os.path.join(BUILD, "pixel_check.csv"), index=False)
    ok = res[res.status == "ok"]
    print(f"{len(res)} images: {int((ok.pixel_equal_frac >= 0.98).sum())} identical (>= 98% of breast pixels), "
          f"{int((ok.pixel_equal_frac < 0.98).sum())} not, {int((res.status != 'ok').sum())} different size")
    print(res[(res.status != "ok") | (res.pixel_equal_frac < 0.98)].to_string(index=False))


if __name__ == "__main__":
    main()
