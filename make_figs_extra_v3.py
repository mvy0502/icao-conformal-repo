"""Figures 6 and 7 of the v3 paper (image-level triage decomposition, joint coverage).
Reads results_v3/image_joint_v3.json, written by image_joint_v3.py."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "results_v3")
INK = "#1f2933"; TEAL = "#0f9d8f"; AMBER = "#e0a100"; RED = "#c92a2a"; GREY = "#adb5bd"; BLUE = "#1c7ed6"; LIGHT = "#cfe8e5"
d = json.load(open(os.path.join(OUT, "image_joint_v3.json")))
img = d["image_level_seed0"]["mond"]

# ---------- Figure 6: image-level triage, decomposed ----------
n_ok = img["inscope_reject_correct_target"]; n_wr = img["inscope_reject_wrong_reason"]
n_rt = img["inscope_retake"]; n_acc = img["inscope_accept"]
n_zr = img["zoom_reject"]; n_zt = img["zoom_retake"]; n_za = img["zoom_accept"]
n_in = img["total_violation_images"]; n_z = img["zoom_total"]

fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)
cats = ["ACCEPT", "REJECT", "RETAKE"]
x = np.arange(3)
# in-scope stacks
seg = [
    ("Correct-target REJECT (true violation flagged)", TEAL, [0, n_ok, 0]),
    ("Wrong-reason REJECT (stopped, but on another criterion)", AMBER, [0, n_wr, 0]),
    ("In-scope RETAKE", BLUE, [0, 0, n_rt]),
    ("In-scope ACCEPT (would be a missed violation)", RED, [n_acc, 0, 0]),
    ("Out-of-scope zoom images", GREY, [n_za, n_zr, n_zt]),
]
bottom = np.zeros(3)
for lab, col, vals in seg:
    vals = np.array(vals, float)
    ax.bar(x, vals, 0.58, bottom=bottom, color=col, label=lab, zorder=3)
    for xi, v, b in zip(x, vals, bottom):
        if v >= 300:
            ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold", zorder=5)
        elif v > 0 and xi == 1:   # thin segment of the stacked REJECT bar: label beside the bar, in the segment colour
            ax.text(xi + 0.32, b + v / 2, f"{int(v)}", ha="left", va="center", fontsize=9, color=col, fontweight="bold", zorder=5)
    bottom += vals
totals = bottom
for xi, t in zip(x, totals):
    ax.text(xi, t + 12, f"{int(t)}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
ax.set_ylim(0, max(totals) * 1.18)
ax.set_ylabel("Test images (partition 1, n = %d)" % int(totals.sum()), fontsize=10.5, color=INK)
ax.set_title("Image-level triage under Mondrian: %d of %d in-scope violation images wrongly ACCEPTED;\n"
             "%.1f%% rejected for the right criterion, %.1f%% for a different criterion, %.1f%% RETAKE"
             % (n_acc, n_in, 100 * n_ok / n_in, 100 * n_wr / n_in, 100 * n_rt / n_in),
             fontsize=10.2, fontweight="bold", color=INK, loc="left", pad=10)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.set_axisbelow(True); ax.grid(True, axis="y", color="#eef1f4", zorder=0)
ax.legend(frameon=False, fontsize=8.2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.09))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_imgdec_v3.png"), bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUT, "fig_imgdec_v3.svg"), bbox_inches="tight", facecolor="white")
print("saved fig_imgdec_v3")

# ---------- Figure 7: joint coverage, naive vs Bonferroni ----------
jc = d["joint_coverage"]
methods = [("lac", "Marginal LAC"), ("mond", "Mondrian")]
fig, ax = plt.subplots(figsize=(8.0, 4.9), dpi=150)
w = 0.34; x = np.arange(2)
unc = [jc["uncorrected"][m]["mean"] for m, _ in methods]; unc_s = [jc["uncorrected"][m]["std"] for m, _ in methods]
bon = [jc["bonferroni"][m]["mean"] for m, _ in methods]; bon_s = [jc["bonferroni"][m]["std"] for m, _ in methods]
rt = [jc["bonf_retake"][m]["mean"] for m, _ in methods]
b1 = ax.bar(x - w / 2, unc, w, yerr=unc_s, color=GREY, capsize=5, label="Per-criterion target 0.90 (uncorrected)", zorder=3, error_kw={"zorder": 4})
b2 = ax.bar(x + w / 2, bon, w, yerr=bon_s, color=TEAL, capsize=5, label="Bonferroni per-criterion target 0.993", zorder=3, error_kw={"zorder": 4})
ax.axhline(0.90, color=RED, ls="--", lw=1.3, label="0.90 joint target", zorder=2)
for xi, m, s in zip(x - w / 2, unc, unc_s):
    ax.text(xi, m + s + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
for xi, m, s, r in zip(x + w / 2, bon, bon_s, rt):
    ax.text(xi, m + s + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(xi, 0.06, f"RETAKE\n{100 * r:.0f}%", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold", zorder=5)
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in methods], fontsize=10.5)
ax.set_ylim(0, 1.12); ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Empirical joint coverage (all 14 criteria)", fontsize=10.5, color=INK)
ax.set_title("A joint image-level guarantee needs a Bonferroni correction, at a conservative cost:\n"
             "%.0f%% (LAC) or %.0f%% (Mondrian, infinite corrected threshold) of images go to RETAKE"
             % (100 * rt[0], 100 * rt[1]), fontsize=10.4, fontweight="bold", color=INK, loc="left", pad=10)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.set_axisbelow(True); ax.grid(True, axis="y", color="#eef1f4", zorder=0)
ax.legend(frameon=False, fontsize=8.6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.10))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_joint_v3.png"), bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUT, "fig_joint_v3.svg"), bbox_inches="tight", facecolor="white")
print("saved fig_joint_v3")
