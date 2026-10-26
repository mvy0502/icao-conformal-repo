"""Journal-size Figures 6 and 7 (image-level triage decomposition, joint coverage).
Reads results_v3/image_joint_v3.json, written by image_joint_v3.py; style from fig_style.py."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import Patch
import fig_style as S
from fig_style import INK, TEAL, AMBER, RED, GREY, BLUE

S.apply()
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "results_v3")
d = json.load(open(os.path.join(OUT, "image_joint_v3.json")))
img = d["image_level_seed0"]["mond"]

# ---------- Figure 6: image-level triage, decomposed ----------
n_ok = img["inscope_reject_correct_target"]; n_wr = img["inscope_reject_wrong_reason"]
n_rt = img["inscope_retake"]; n_acc = img["inscope_accept"]
n_zr = img["zoom_reject"]; n_zt = img["zoom_retake"]; n_za = img["zoom_accept"]

fig, ax = S.figure(S.SINGLE, 92)
cats = ["ACCEPT", "REJECT", "RETAKE"]
x = np.arange(3)
# in-scope stacks; hatching keeps the segments apart in greyscale
seg = [
    ("REJECT, correct criterion", TEAL, None, [0, n_ok, 0]),
    ("REJECT, wrong criterion", AMBER, "////", [0, n_wr, 0]),
    ("RETAKE (in scope)", BLUE, "....", [0, 0, n_rt]),
    ("ACCEPT (in scope; would be a missed violation)", RED, None, [n_acc, 0, 0]),
    ("Zoom images (out of scope)", GREY, "xxxx", [n_za, n_zr, n_zt]),
]
bottom = np.zeros(3)
for lab, col, hatch, vals in seg:
    vals = np.array(vals, float)
    ax.bar(x, vals, 0.58, bottom=bottom, color=col, hatch=hatch, edgecolor="white", lw=0, zorder=3)
    for xi, v, b in zip(x, vals, bottom):
        if v >= 300:
            ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=5)
        elif v > 0 and xi == 1:   # thin segment of the stacked REJECT bar: label beside the bar
            ax.text(xi + 0.33, b + v / 2, f"{int(v)}", ha="left", va="center", fontsize=8.5, color=INK, zorder=5)
    bottom += vals
totals = bottom
for xi, t in zip(x, totals):
    ax.text(xi, t + 12, f"{int(t)}", ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylim(0, max(totals) * 1.12)
ax.set_ylabel("Test images (partition 1, n = %d)" % int(totals.sum()))
ax.set_axisbelow(True); ax.grid(True, axis="y", zorder=0)
# explicit swatches: a stack whose first bar has zero height gives an empty legend handle
fig.legend(handles=[Patch(facecolor=col, hatch=hatch, edgecolor="white", lw=0, label=lab) for lab, col, hatch, _ in seg],
           loc="outside lower center", ncol=1)
S.save(fig, OUT, "fig_imgdec_v3")

# ---------- Figure 7: joint coverage, naive vs Bonferroni ----------
jc = d["joint_coverage"]
methods = [("lac", "Marginal LAC"), ("mond", "Mondrian")]
fig, ax = S.figure(S.SINGLE, 68)
w = 0.42; x = np.arange(2)   # wide enough for the white "RETAKE" label inside the bar
unc = [jc["uncorrected"][m]["mean"] for m, _ in methods]; unc_s = [jc["uncorrected"][m]["std"] for m, _ in methods]
bon = [jc["bonferroni"][m]["mean"] for m, _ in methods]; bon_s = [jc["bonferroni"][m]["std"] for m, _ in methods]
rt = [jc["bonf_retake"][m]["mean"] for m, _ in methods]
ax.bar(x - w / 2, unc, w, yerr=unc_s, color=GREY, capsize=3, label="Per-criterion target 0.90 (uncorrected)", zorder=3, error_kw={"zorder": 4, "lw": 0.9})
ax.bar(x + w / 2, bon, w, yerr=bon_s, color=TEAL, capsize=3, label="Bonferroni, per-criterion target 0.993", zorder=3, error_kw={"zorder": 4, "lw": 0.9})
ax.axhline(0.90, color=RED, ls="--", lw=1.0, label="0.90 joint target", zorder=2)
for xi, m, s in zip(x - w / 2, unc, unc_s):
    ax.text(xi, m + s + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
for xi, m, s, r in zip(x + w / 2, bon, bon_s, rt):
    ax.text(xi, m + s + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(xi, 0.06, f"RETAKE\n{100 * r:.0f}%", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold", zorder=5)
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in methods])
ax.set_ylim(0, 1.12); ax.set_xlim(-0.55, 1.55)
ax.set_ylabel("Empirical joint coverage\n(all 14 criteria)")
ax.set_axisbelow(True); ax.grid(True, axis="y", zorder=0)
fig.legend(loc="outside lower center", ncol=1)
S.save(fig, OUT, "fig_joint_v3")
