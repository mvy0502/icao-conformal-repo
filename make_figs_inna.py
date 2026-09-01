"""Figures for the supervisor pre-freeze checks (reads results_v3/inna_protocol.json)."""
import json, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "results_v3")
INK="#1f2933"; TEAL="#0f9d8f"; AMBER="#e0a100"; RED="#c92a2a"; GREY="#adb5bd"; PURPLE="#6741d9"; BLUE="#1c7ed6"
d = json.load(open(os.path.join(OUT, "inna_protocol.json")))
cur = d["risk_abstention_curves"]

def xy(m, key):
    pts = sorted([p for p in cur[m] if p.get(key) is not None], key=lambda z: z["abst"])
    return [p["abst"] for p in pts], [p[key] for p in pts]

# ---------- risk-abstention: range of operating points + Mondrian achieved vs nominal ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=150)
for m, lab, col, mk in [("chow", "Chow confidence rejection", PURPLE, "^"), ("asym", "Asymmetric two-threshold", BLUE, "s")]:
    x, y = xy(m, "fpr_decided"); ax1.plot(x, y, "-", marker=mk, color=col, lw=1.7, ms=5, label=lab)
# conformal: abstention is not monotone in alpha, so plot operating points without a connecting line
for m, lab, col, mk in [("lac", "LAC conformal (alpha sweep)", AMBER, "D"), ("mond", "Mondrian conformal (alpha sweep)", TEAL, "o")]:
    x, y = xy(m, "fpr_decided"); ax1.scatter(x, y, marker=mk, color=col, s=42, label=lab, zorder=3)
ax1.axvline(0.10, color=GREY, ls=":", lw=1.2)
ax1.text(0.106, 0.0008, "0.10 abstention (design target)", fontsize=7.8, color="#6b7280", va="bottom", ha="left")
ax1.set_xlim(0, 0.50); ax1.set_ylim(0, 0.05); ax1.set_xlabel("Abstention / RETAKE rate", fontsize=10.5, color=INK)
ax1.set_ylabel("False-positive rate among decided cases", fontsize=10.5, color=INK)
ax1.set_title("Risk-abstention across operating points:\nno method dominates; conventional rules reach lower FPR", fontsize=10.8, fontweight="bold", color=INK, loc="left")
ax1.set_axisbelow(True); ax1.grid(True, color="#eef1f4")
ax1.legend(frameon=False, fontsize=8.3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14))
for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)
# right: Mondrian achieved vs nominal
pts = sorted(cur["mond"], key=lambda z: z["param"], reverse=True)
nom = [1 - p["param"] for p in pts]; ach = [p["coverage_viol"] for p in pts]
ax2.plot([0.68, 1.0], [0.68, 1.0], ls="--", color=GREY, lw=1.4, label="nominal (y = x)")
ax2.plot(nom, ach, "-o", color=TEAL, lw=2, ms=6, label="Mondrian achieved violation coverage")
ax2.set_xlabel("Nominal target (1 - alpha)", fontsize=10.5, color=INK); ax2.set_ylabel("Achieved violation-class coverage", fontsize=10.5, color=INK)
ax2.set_title("Mondrian: achieved coverage tracks the class-conditional\ntarget (slightly below nominal for alpha >= 0.10)", fontsize=10.8, fontweight="bold", color=INK, loc="left")
ax2.set_axisbelow(True); ax2.grid(True, color="#eef1f4"); ax2.legend(frameon=False, fontsize=8.8, loc="upper left")
for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_risk_abstention.png"), bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUT, "fig_risk_abstention.svg"), bbox_inches="tight", facecolor="white")
print("saved fig_risk_abstention")

# ---------- cluster-aware sensitivity (separating n from clustering) ----------
c = d["cluster_aware_mondrian"]
labs = ["Full\ncalibration", "Random image\nsubsample\n(equal n)", "Subject-capped\n(<=3 / subject,\nsame n)", "Subject\nbootstrap\n(mean, 5-95%)"]
means = [c["full"]["mean"], c["random_equal_n"]["mean"], c["subject_cap3"]["mean"], c["subject_boot_mean"]["mean"]]
sds = [c["full"]["sd"], c["random_equal_n"]["sd"], c["subject_cap3"]["sd"], 0]
lo, hi = c["subject_boot_p05"]["mean"], c["subject_boot_p95"]["mean"]
fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
cols = [TEAL, GREY, GREY, AMBER]
ax.bar(range(4), means, yerr=[[0, 0, 0, means[3] - lo], [0, 0, 0, hi - means[3]]], capsize=6, color=cols, width=0.62, zorder=3, error_kw={"zorder": 4, "lw": 1.2})
ax.errorbar([0, 1, 2], means[:3], yerr=sds[:3], fmt="none", ecolor=INK, capsize=5, lw=1.2, zorder=4)
ax.axhline(0.90, color=RED, ls="--", lw=1.3, label="0.90 target", zorder=2)
tops = [means[0] + sds[0], means[1] + sds[1], means[2] + sds[2], hi]
for i, m in enumerate(means): ax.text(i, tops[i] + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold", zorder=5)
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8.6); ax.set_ylim(0.80, 1.0)
ax.set_axisbelow(True); ax.grid(True, axis="y", color="#eef1f4", zorder=0)
ax.set_ylabel("Mondrian violation-class coverage", fontsize=10.5, color=INK)
ax.set_title("Cluster-aware sensitivity: subject-capping and random subsampling at equal n agree,\nso no clustering effect is detected beyond sample size; subject bootstrap contains 0.90",
             fontsize=10.2, fontweight="bold", color=INK, loc="left", pad=10)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_subject_sensitivity.png"), bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUT, "fig_subject_sensitivity.svg"), bbox_inches="tight", facecolor="white")
print("saved fig_subject_sensitivity")
