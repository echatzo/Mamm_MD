"""Reproduce the paper's evaluation from candidate caches and trained heads.

For every evaluation of configs/paper.yaml: single view, geometric product, CEN, MAMM-MD w/o
appearance (geo), MAMM-MD (geoapp) and, in-domain, MAMM-MD w/o geometry (appo). Heads and CEN are
ensembles of 3 seeds. Metric: LARGE = mean sensitivity at 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4 FPI.
Significance: paired breast-level bootstrap (1000 resamples); Holm correction over the 12
"MAMM-MD vs product" comparisons.

    python scripts/evaluate.py --caches <root>/caches --checkpoints <root>/checkpoints \
        --annotations data/cbis_ddsm_irma/annotations.json --inbreast-csv "<INbreast>/INbreast.csv" \
        --out results/paper_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd import cen  # noqa: E402
from mammd.data.cbis_ddsm_irma import load_annotations  # noqa: E402
from mammd.eval.protocol import evaluation_views, holm, large_of, lesion_status, paired_bootstrap, per_view  # noqa: E402
from mammd.head import MAMMMDHead  # noqa: E402
from mammd.rescoring import partner_support, predict, predict_ensemble, product_scores  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARISONS = [("geoapp", "single_view"), ("geoapp", "product"), ("geoapp", "cen"), ("geo", "product"),
               ("appo", "geoapp"), ("geo", "geoapp")]


def view_birads(test: str, recs: list, annotations: str, inbreast_csv: str | None) -> dict:
    """{view id: BI-RADS of its masses} for the strict convention."""
    if test == "cbis":
        return {e["image_id"]: e.get("cbis_mass_birads") or None for e in load_annotations(annotations)}
    if test == "inbreast":
        df = pd.read_csv(inbreast_csv, sep=";")
        br = dict(zip(df["File Name"].astype(str).str.strip(), df["Bi-Rads"].astype(str)))
        return {r["id"]: [int(br[str(r["id"]).split("_")[0]][0])] for r in recs if str(r["id"]).split("_")[0] in br}
    raise ValueError(f"no BI-RADS convention for test set {test}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "..", "configs", "paper.yaml"))
    ap.add_argument("--caches", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--annotations", default=os.path.join(HERE, "..", "data", "cbis_ddsm_irma", "annotations.json"))
    ap.add_argument("--inbreast-csv", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="names of evaluations to run")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    prior, seeds, nb = cfg["prior"], cfg["seeds"], cfg["bootstrap"]["resamples"]
    results = {}
    for ev in cfg["evaluations"]:
        if a.only and ev["name"] not in a.only:
            continue
        pipe = cfg["pipelines"][ev["pipeline"]]
        raw = torch.load(os.path.join(a.caches, ev["pipeline"], f"test_{ev['test']}.pt"), weights_only=False)
        R, T = evaluation_views(raw)
        status = None
        if ev["convention"] == "strict":
            status = lesion_status(R, T, view_birads(ev["test"], R, a.annotations, a.inbreast_csv))
        sup = partner_support(R, prior)
        S = {"single_view": [r["scores"] for r in R], "product": product_scores(R, sup)}
        hd = os.path.join(a.checkpoints, "heads", ev["pipeline"])
        variants = [v for v in pipe["variants"] if v != "appo" or ev["pipeline"].endswith(ev["test"])]
        per_seed = {}
        for v in variants:
            heads = [MAMMMDHead.load(os.path.join(hd, f"{v}_s{s}.pth")) for s in seeds]
            S[v] = predict_ensemble(heads, R, sup)
            per_seed[v] = [large_of(per_view(R, T, predict(h, R, sup), status), T) for h in heads]
        S["cen"] = cen.predict_ensemble([cen.CENHead.load(os.path.join(a.checkpoints, "cen", ev["pipeline"], f"s{s}.pth"))
                                         for s in seeds], R)
        PV = {k: per_view(R, T, sc, status) for k, sc in S.items()}
        res = {"n_views": len(T), "n_lesions": int(sum(v[2] for v in PV["single_view"].values())),
               "large": {k: large_of(pv, T) for k, pv in PV.items()},
               "large_per_seed": per_seed, "bootstrap": {}}
        if not a.no_bootstrap:
            for x, y in COMPARISONS:
                if x in PV and y in PV:
                    res["bootstrap"][f"{x} - {y}"] = paired_bootstrap(PV[x], PV[y], R, T, nb, cfg["bootstrap"]["seed"])
        results[ev["name"]] = res
        L = res["large"]
        print(f"\n{ev['name']}  ({res['n_views']} views, {res['n_lesions']} lesions)\n  "
              + "  ".join(f"{k} {L[k]:.3f}" for k in ("single_view", "product", "cen", "geo", "geoapp", "appo") if k in L))
        for k, b in res["bootstrap"].items():
            print(f"    {k:<22} {b['delta']:+.4f}  [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]  p {b['p']:.3f}")
    if not a.no_bootstrap and len(results) > 1:
        H = holm({n: r["bootstrap"]["geoapp - product"]["p"] for n, r in results.items()})
        print("\nHolm, MAMM-MD vs product:")
        for n, (p, rej) in H.items():
            results[n]["holm_geoapp_vs_product"] = {"p_adjusted": p, "significant": rej}
            print(f"  {n:<32} p_adj {p:.3f} {'*' if rej else ''}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
