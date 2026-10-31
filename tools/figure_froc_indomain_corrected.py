"""Figure 2 of the paper (in-domain FROC, 2 x 2) for the corrected MV-DDSM (provenance only).

Same layout and style as analysis/figure_froc_indomain.py; reads froc_curves_corrected.pt written by
tools/paper_numbers_corrected.py.

    python tools/figure_froc_indomain_corrected.py <dir with froc_curves_corrected.pt>
"""
import os, sys
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager

D = sys.argv[1]
C = torch.load(os.path.join(D, "froc_curves_corrected.pt"), weights_only=False)
GRID = (0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4)
LABELED = (0.025, 0.1, 0.5, 2)
serif = "Times New Roman" if any("Times New Roman" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Serif"
# drawn at its final size (one ISBI column, ~3.34 in) and included at width=\linewidth: 8 pt labels, 7 pt ticks
plt.rcParams.update({"font.family": serif, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42})
SERIES = [("single_view", "Single view", "#8c8c8c", "-", 1.0, 1),
          ("product", "Geometric product", "#2a78d6", "-", 1.1, 2),
          ("cen", "CEN", "#eb6834", (0, (4, 1.6)), 1.1, 3),
          ("geo", "Ours (geometry)", "#199e70", (0, (5, 1.4, 1, 1.4)), 1.1, 4),
          ("geoapp", "Ours (geometry + appearance)", "#4a3aa7", "-", 1.6, 5)]
PANELS = [("RetinaNet CBIS -> CBIS", "RetinaNet", "MV-DDSM"), ("RetinaNet VinDr -> VinDr", "RetinaNet", "VinDr-Mammo"),
          ("DETR CBIS -> CBIS", "Deformable DETR", "MV-DDSM"), ("DETR VinDr -> VinDr", "Deformable DETR", "VinDr-Mammo")]
fig, axes = plt.subplots(2, 2, figsize=(3.34, 3.0), sharex=True, sharey=True)
for k, (ax, (key, det, ds)) in enumerate(zip(axes.ravel(), PANELS)):
    for s, lab, col, ls, lw, z in SERIES:
        f, se = C[key][s]
        o = np.lexsort((se, f)); f, se = f[o], se[o]
        m = (f >= 0.02) & (f <= 4.2)
        ax.plot(f[m], se[m], color=col, ls=ls, lw=lw, zorder=z, label=lab, drawstyle="steps-post")
    ax.set_xscale("log"); ax.set_xlim(0.025, 4); ax.minorticks_off()
    ax.set_xticks(GRID); ax.set_xticklabels([f"{g:g}" if g in LABELED else "" for g in GRID])
    ax.set_ylim(0.2, 1.0); ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0.2, 1.01, 0.2)])
    ax.grid(True, color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.text(0.04, 0.96, f"({'abcd'[k]})", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    if k < 2: ax.set_title(ds, pad=3)
    if k % 2 == 0: ax.set_ylabel(f"{det}\nSensitivity")
    if k >= 2: ax.set_xlabel("False positives per image")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handlelength=2.0, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25)
fig.tight_layout(rect=(0, 0.095, 1, 1), h_pad=0.3, w_pad=0.3, pad=0.15)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(D, f"paper_froc_indomain_corrected.{ext}"), dpi=300, facecolor="white", bbox_inches="tight",
                pad_inches=0.01)
print("->", os.path.join(D, "paper_froc_indomain_corrected.pdf"))
