"""All numbers of the paper for the corrected MV-DDSM (`corrected=True`) (provenance only).

- MV-DDSM test: corrected caches for every pipeline tested on it.
- Pipelines trained on MV-DDSM: heads and CEN retrained on corrected train/val caches
  (same recipe as the paper); detectors and encoder are those of the paper.
- Pipelines trained on VinDr-Mammo: the paper's heads and CEN.
- VinDr-Mammo and INbreast test sets: unchanged caches.
Bootstrap: paired, breast-level, 1000 resamples, a fresh generator (seed 0) per comparison.

    python tools/paper_numbers_corrected.py --corrected-caches /storage2/edouard/cache/corrected_v1 \
        --corrected-models /storage2/edouard/ckp/rescoring_corrected_v1 --out <dir>
"""
import argparse, json, os, sys
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mammd import cen
from mammd.data.mv_ddsm import load_annotations, mass_detection_views
from mammd.eval.froc import froc_curve, sensitivity_at
from mammd.eval.protocol import evaluation_views, holm, large_of, lesion_status, paired_bootstrap, per_view
from mammd.head import MAMMMDHead
from mammd.rescoring import partner_support, predict, predict_ensemble, product_scores

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "/storage2/edouard/release_mammd"
INB_CSV = "/storage/edouard/data/INbreast Release 1.0/INbreast.csv"
COMP = [("geoapp", "single_view"), ("geoapp", "product"), ("geoapp", "cen"), ("geo", "product"),
        ("appo", "geoapp"), ("geo", "geoapp")]
ap = argparse.ArgumentParser()
ap.add_argument("--corrected-caches", required=True)
ap.add_argument("--corrected-models", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cfg = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "paper.yaml")))
ANN = load_annotations(os.path.join(HERE, "..", "data", "mv_ddsm", "annotations.json"))


def view_birads(test, recs):
    if test == "cbis":
        vb = {e["image_id"]: e.get("cbis_mass_birads") or None for e in ANN}
        for r in recs:                     # hand-added masses take the BI-RADS of their breast
            if len(r["gt"]) and not vb.get(r["id"]) and r["partner"] >= 0:
                vb[r["id"]] = vb.get(recs[r["partner"]]["id"])
        return vb
    import pandas as pd
    df = pd.read_csv(INB_CSV, sep=";")
    br = dict(zip(df["File Name"].astype(str).str.strip(), df["Bi-Rads"].astype(str)))
    return {r["id"]: [int(br[str(r["id"]).split("_")[0]][0])] for r in recs if str(r["id"]).split("_")[0] in br}


# ---------------------------------------------------------------- dataset counts (Section 3.1)
counts = {}
for corr in (False, True):
    V = {sp: mass_detection_views(ANN, sp, corrected=corr) for sp in ("train", "val", "test")}
    allv = [v for vs in V.values() for v in vs]
    free = [v for v in allv if not v["positive"] and not v["support"]]
    counts["corrected" if corr else "paper"] = dict(
        images=len(allv), subjects=len({v.get("ddsm_case") or v.get("cbis_patient_id") for v in allv}),
        per_split={sp: dict(images=len(vs), subjects=len({v.get("ddsm_case") or v.get("cbis_patient_id") for v in vs}))
                   for sp, vs in V.items()},
        with_mass=sum(v["positive"] for v in allv), support_partners=sum(v["support"] for v in allv),
        mass_free=len(free), mass_free_irma=sum(v["source"] == "ddsm-irma" for v in free),
        mass_free_normal_exams=sum(v["label_source"] == "ddsm_normal_volume" for v in free),
        mass_free_cbis_calc=sum(v["source"] == "cbis-ddsm" for v in free))
print(json.dumps(counts, indent=1))

# ---------------------------------------------------------------- evaluations
res, curves = {}, {}
for ev in cfg["evaluations"]:
    name, pipe, test = ev["name"], ev["pipeline"], ev["test"]
    cbis_trained = cfg["pipelines"][pipe]["train_set"].startswith("cbis")
    cache = (os.path.join(a.corrected_caches, pipe, "test_cbis.pt") if test == "cbis"
             else f"{REL}/caches/{pipe}/test_{test}.pt")
    mdir = a.corrected_models if cbis_trained else f"{REL}/checkpoints"
    R, T = evaluation_views(torch.load(cache, weights_only=False))
    st = lesion_status(R, T, view_birads(test, R)) if ev["convention"] == "strict" else None
    sup = partner_support(R)
    S = {"single_view": [r["scores"] for r in R], "product": product_scores(R, sup)}
    variants = [v for v in cfg["pipelines"][pipe]["variants"] if v != "appo" or pipe.endswith(test)]
    seeds = {}
    for v in variants:
        hs = [MAMMMDHead.load(f"{mdir}/heads/{pipe}/{v}_s{s}.pth") for s in cfg["seeds"]]
        S[v] = predict_ensemble(hs, R, sup)
        seeds[v] = [large_of(per_view(R, T, predict(h, R, sup), st), T) for h in hs]
    S["cen"] = cen.predict_ensemble([cen.CENHead.load(f"{mdir}/cen/{pipe}/s{s}.pth") for s in cfg["seeds"]], R)
    PV = {k: per_view(R, T, v, st) for k, v in S.items()}
    r = {"test_cache": cache, "models": mdir, "views": len(T), "lesions": int(sum(PV["single_view"][i][2] for i in T)),
         "large": {}, "at_1fpi": {}, "bootstrap": {},
         "seeds": {v: {"values": x, "std": float(np.std(x, ddof=1))} for v, x in seeds.items()}}
    for k, pv in PV.items():
        f, se = froc_curve([pv[i] for i in T])
        r["large"][k] = large_of(pv, T)
        r["at_1fpi"][k] = sensitivity_at(f, se, (1,))["sens@1FPI"]
        if pipe.endswith(test):
            curves.setdefault(name, {})[k] = (f, se)
    for x, y in COMP:
        if x in PV and y in PV:
            r["bootstrap"][f"{x} - {y}"] = paired_bootstrap(PV[x], PV[y], R, T, cfg["bootstrap"]["resamples"], 0)
    res[name] = r
    print(f"\n{name} ({r['views']} views, {r['lesions']} lesions) models: {'CORRECTED' if cbis_trained else 'paper'}", flush=True)
    print("  " + "  ".join(f"{k} {r['large'][k]:.3f}/{r['at_1fpi'][k]:.3f}" for k in r["large"]))
    for k, b in r["bootstrap"].items():
        print(f"    {k:<20} {b['delta']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] p {b['p']:.3f}")
H = holm({n: r["bootstrap"]["geoapp - product"]["p"] for n, r in res.items()})
print("\nHolm, MAMM-MD vs product:")
for n, (p, rej) in H.items():
    res[n]["holm_geoapp_product"] = {"p_adjusted": p, "significant": rej}
    print(f"  {n:<30} raw {res[n]['bootstrap']['geoapp - product']['p']:.3f}  adj {p:.3f} {'*' if rej else ''}")
json.dump({"counts": counts, "evaluations": res}, open(os.path.join(a.out, "results_corrected.json"), "w"), indent=1)
torch.save(curves, os.path.join(a.out, "froc_curves_corrected.pt"))
