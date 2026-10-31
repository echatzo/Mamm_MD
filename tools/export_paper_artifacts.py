"""Export the paper's candidate caches and trained heads into the release layout (provenance only).

    <root>/caches/<pipeline>/{train,val,test_cbis,test_vindr,test_inbreast}.pt   (symlinks)
    <root>/checkpoints/heads/<pipeline>/<variant>_s<seed>.pth                   (converted)
    <root>/checkpoints/cen/<pipeline>/s<seed>.pth                               (copied)

The heads are converted from the training runs' best.pth to MAMMMDHead checkpoints after checking
that every run used the recipe implemented in mammd/ (assertions below).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd.head import MAMMMDHead  # noqa: E402

C = "/storage2/edouard/cache"
RUNS = "/storage2/edouard/ckp/retinanet_rescoring"

PIPELINES = {
    "retinanet_cbis": dict(
        train=f"{C}/mv_cbis_clean/train_pemb.pt", val=f"{C}/mv_cbis_clean/val_pemb.pt",
        tests={"cbis": f"{C}/mv_cbis_clean/test_pemb.pt", "inbreast": f"{C}/mv_inbreast_v2/cbisdet.pt"},
        heads="CBWI", cen="CBO_cen", variants=("geoapp", "geo", "appo")),
    "retinanet_cbis_b3": dict(
        train=f"{C}/mv_cbisb3/train_pemb.pt", val=f"{C}/mv_cbisb3/val_pemb.pt",
        tests={"vindr": f"{C}/mv_cbisb3_transfer/vindr_test.pt"},
        heads="CB3", cen="CB3O_cen", variants=("geoapp", "geo")),
    "retinanet_vindr": dict(
        train=f"{C}/mv_vindr_clean/train_ALL.pt", val=f"{C}/mv_vindr_clean/val_ALL.pt",
        tests={"vindr": f"{C}/mv_vindr_clean/test_full.pt", "cbis": f"{C}/mv_vindr_clean/cbis_test.pt",
               "inbreast": f"{C}/mv_vindr_clean/inbreast.pt"},
        heads="ALC", cen="ALCO_cen", variants=("geoapp", "geo", "appo")),
    "detr_cbis": dict(
        train=f"{C}/mv_cbis_detr/train.pt", val=f"{C}/mv_cbis_detr/val.pt",
        tests={"cbis": f"{C}/mv_cbis_detr/test.pt", "inbreast": f"{C}/mv_detr_transfer/cbisdetr_inbreast.pt"},
        heads="DCBI", cen="DCBO_cen", variants=("geoapp", "geo", "appo")),
    "detr_cbis_b3": dict(
        train=f"{C}/mv_cbis_detr_b3/train.pt", val=f"{C}/mv_cbis_detr_b3/val.pt",
        tests={"vindr": f"{C}/mv_cbis_detr_b3/vindr_test.pt"},
        heads="DCB3", cen="DCB3O_cen", variants=("geoapp", "geo")),
    "detr_vindr": dict(
        train=f"{C}/mv_vindr_detr/train_ALL.pt", val=f"{C}/mv_vindr_detr/val_ALL.pt",
        tests={"vindr": f"{C}/mv_vindr_detr/test_full.pt", "cbis": f"{C}/mv_detr_transfer/cbis_test.pt",
               "inbreast": f"{C}/mv_detr_transfer/inbreast.pt"},
        heads="DTRI", cen="DCO_cen", variants=("geoapp", "geo", "appo")),
}


def convert_head(run: str, out: str) -> dict:
    ck = torch.load(os.path.join(RUNS, run, "best.pth"), map_location="cpu", weights_only=False)
    a, sd = ck["args"], ck["model"]
    # the recipe of mammd/head.py and mammd/rescoring.py
    geometry = not a.get("no_geometry", False)
    # --no-geometry rewrites band_mode / lam / anchor in the saved arguments
    assert a["prior"] == "identite" and a["n_features"] == 7, run
    assert (a["band_mode"], a["lam"]) == (("hard", 1.0) if geometry else ("none", 0.0)), run
    assert a["sim_bias"] == 0.0 and not a.get("no_sim_additive") and a.get("max_down") is None, run
    assert not a["no_self_attn"] and a["dim"] == 64 and a["heads"] == 4 and a["max_delta"] == 3.0, run
    assert a["loss"] == "rank" and a["epochs"] == 40 and a["select"] == "wide" and a["select_smooth"] == 3, run
    assert a["select_avg"] and a["support_unannotated"] and a["paired_only"] == 1 and a["max_cand"] == 100, run
    assert a["jitter_depth_mm"] == 2.0 and a["delta_l2"] == 0.01 and a["lr"] == 1e-3 and a.get("band_sigma", 3.0) == 3.0, run
    assert not any(a.get(k) for k in ("jitter_size", "jitter_scale", "jitter_nipple_mm", "positives_only",
                                      "both_annotated")), run
    assert bool(a["anchor"]) == geometry, run
    roi_dim = sd["roi_proj.weight"].shape[1] if "roi_proj.weight" in sd else 0
    assert bool(roi_dim) == bool(a["use_roi"]), run
    h = MAMMMDHead(roi_dim=roi_dim, geometry=geometry)
    h.load_state_dict(sd)
    h.save(out, source_run=run, train_args=a, epochs=ck.get("epochs"))
    return dict(run=run, pos_frac=a["pos_frac"], seed=a["seed"], train=a["train"], val=a["val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/storage2/edouard/release_mammd")
    a = ap.parse_args()
    for name, p in PIPELINES.items():
        cdir = os.path.join(a.root, "caches", name); os.makedirs(cdir, exist_ok=True)
        for k, src in [("train", p["train"]), ("val", p["val"])] + [(f"test_{t}", f) for t, f in p["tests"].items()]:
            dst = os.path.join(cdir, f"{k}.pt")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        hdir = os.path.join(a.root, "checkpoints", "heads", name); os.makedirs(hdir, exist_ok=True)
        for v in p["variants"]:
            for s in range(3):
                info = convert_head(f"{p['heads']}_{v}_s{s}", os.path.join(hdir, f"{v}_s{s}.pth"))
                assert info["train"] == p["train"] and info["val"] == p["val"] and info["seed"] == s, (name, v, s, info)
        edir = os.path.join(a.root, "checkpoints", "cen", name); os.makedirs(edir, exist_ok=True)
        for s in range(3):
            run = os.path.join(RUNS, f"{p['cen']}_s{s}")
            ck = torch.load(os.path.join(run, "best.pth"), map_location="cpu", weights_only=False)
            assert ck["args"]["train"] == p["train"] and ck["args"]["val"] == p["val"] and ck["args"]["seed"] == s
            shutil.copy(os.path.join(run, "best.pth"), os.path.join(edir, f"s{s}.pth"))
        print(f"{name}: caches {sorted(os.listdir(cdir))}, heads {len(os.listdir(hdir))}, cen 3")


if __name__ == "__main__":
    main()
