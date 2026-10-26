"""Journal-size figures for the supervisor pre-freeze checks: risk-abstention (Fig 5) and subject sensitivity (Fig 8).
Reads results_v3/inna_protocol.json; style from fig_style.py."""
import json, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import fig_style as S
from fig_style import INK, TEAL, AMBER, RED, GREY, PURPLE, BLUE, NOTE

S.apply()
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "results_v3")
d = json.load(open(os.path.join(OUT, "inna_protocol.json")))
cur = d["risk_abstention_curves"]

def xy(m, key):
    pts = sorted([p for p in cur[m] if p.get(key) is not None], key=lambda z: z["abst"])
    return [p["abst"] for p in pts], [p[key] for p in pts]

# ---------- Fig 5: range of operating points + Mondrian achieved vs nominal ----------
fig, (ax1, ax2) = S.figure(S.DOUBLE, 72, ncols=2)
for m, lab, col, mk in [("chow", "Chow confidence rejection", PURPLE, "^"), ("asym", "Asymmetric two-threshold", BLUE, "s")]:
    x, y = xy(m, "fpr_decided"); ax1.plot(x, y, "-", marker=mk, color=col, lw=1.2, ms=3.8, label=lab)
# conformal: abstention is not monotone in alpha, so plot operating points without a connecting line
for m, lab, col, mk in [("lac", "LAC conformal (α sweep)", AMBER, "D"), ("mond", "Mondrian conformal (α sweep)", TEAL, "o")]:
    x, y = xy(m, "fpr_decided"); ax1.scatter(x, y, marker=mk, color=col, s=16, label=lab, zorder=3)
ax1.axvline(0.10, color=GREY, ls=":", lw=1.0)
ax1.text(0.106, 0.0008, "0.10 design target", fontsize=8.5, color=NOTE, va="bottom", ha="left")
ax1.set_xlim(0, 0.50); ax1.set_ylim(0, 0.05); ax1.set_xlabel("Abstention (RETAKE) rate")
ax1.set_ylabel("FPR among decided cases")
ax1.set_axisbelow(True); ax1.grid(True)
ax1.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
# right: Mondrian achieved vs nominal
pts = sorted(cur["mond"], key=lambda z: z["param"], reverse=True)
nom = [1 - p["param"] for p in pts]; ach = [p["coverage_viol"] for p in pts]
ax2.plot([0.68, 1.0], [0.68, 1.0], ls="--", color=GREY, lw=1.0, label="Nominal (y = x)")
ax2.plot(nom, ach, "-o", color=TEAL, lw=1.4, ms=3.8, label="Mondrian, achieved")
ax2.set_xlabel("Nominal target 1 − α"); ax2.set_ylabel("Achieved violation-class coverage")
ax2.set_axisbelow(True); ax2.grid(True); ax2.legend(loc="lower right")
S.save(fig, OUT, "fig_risk_abstention")

# ---------- Fig 8: cluster-aware sensitivity (separating n from clustering) ----------
c = d["cluster_aware_mondrian"]
labs = ["Full\ncalibration", "Random\nsubsample\n(equal n)", "Subject-\ncapped (≤3,\nequal n)", "Subject\nbootstrap\n(5-95%)"]
means = [c["full"]["mean"], c["random_equal_n"]["mean"], c["subject_cap3"]["mean"], c["subject_boot_mean"]["mean"]]
sds = [c["full"]["sd"], c["random_equal_n"]["sd"], c["subject_cap3"]["sd"], 0]
lo, hi = c["subject_boot_p05"]["mean"], c["subject_boot_p95"]["mean"]
fig, ax = S.figure(S.SINGLE, 70)
cols = [TEAL, GREY, GREY, AMBER]
ax.bar(range(4), means, yerr=[[0, 0, 0, means[3] - lo], [0, 0, 0, hi - means[3]]], capsize=3, color=cols, width=0.62, zorder=3, error_kw={"zorder": 4, "lw": 0.9})
ax.errorbar([0, 1, 2], means[:3], yerr=sds[:3], fmt="none", ecolor=INK, capsize=3, lw=0.9, zorder=4)
ax.axhline(0.90, color=RED, ls="--", lw=1.0, label="0.90 target", zorder=2)
tops = [means[0] + sds[0], means[1] + sds[1], means[2] + sds[2], hi]
for i, m in enumerate(means): ax.text(i, tops[i] + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold", zorder=5)
ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylim(0.80, 1.0)
ax.set_axisbelow(True); ax.grid(True, axis="y", zorder=0)
ax.set_ylabel("Mondrian violation-class coverage")
ax.legend(loc="upper right")
S.save(fig, OUT, "fig_subject_sensitivity")
