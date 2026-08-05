import sys
sys.path.insert(0, "/home/claude")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import conformal_analysis as ca

CSV = "/home/claude/tono_full_pred.csv"
alphas = [0.20, 0.15, 0.10, 0.05, 0.02]
nominal, emp, lo, hi = [], [], [], []
for a in alphas:
    s = ca.run(CSV, f"/tmp/cal_{int(a*100)}", alpha=a, seed=0)
    covs = [v["coverage"] for v in s["criteria"].values() if not v.get("skip", True)]
    nominal.append(1 - a)
    emp.append(float(np.mean(covs)))
    lo.append(float(np.min(covs)))
    hi.append(float(np.max(covs)))

INK = "#1f2933"; TEAL = "#0f9d8f"; GREY = "#adb5bd"
fig, ax = plt.subplots(figsize=(6.2, 6), dpi=150)
ax.plot([0.75, 1.0], [0.75, 1.0], ls="--", color=GREY, lw=1.5, label="perfect calibration (y = x)")
nominal = np.array(nominal); emp = np.array(emp)
ax.fill_between(nominal, lo, hi, color=TEAL, alpha=0.15, label="per-criterion min–max")
ax.plot(nominal, emp, "-o", color=TEAL, lw=2, ms=7, label="mean empirical coverage")
for xn, ye in zip(nominal, emp):
    ax.annotate(f"{ye:.3f}", (xn, ye), textcoords="offset points", xytext=(6, -12),
                fontsize=8.5, color=INK)
ax.set_xlabel("Nominal target coverage  (1 − α)", fontsize=11, color=INK)
ax.set_ylabel("Empirical coverage (test set)", fontsize=11, color=INK)
ax.set_title("Empirical coverage tracks the nominal guarantee\n(14 ICAO criteria, TONO)",
             fontsize=12.5, fontweight="bold", color=INK, loc="left")
ax.set_xlim(0.78, 1.0); ax.set_ylim(0.78, 1.0)
ax.grid(True, color="#e3e8ee")
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/results/calibration.png", bbox_inches="tight", facecolor="white")
plt.savefig("/home/claude/results/calibration.svg", bbox_inches="tight", facecolor="white")
print("calibration saved:", list(zip([round(x,2) for x in nominal], [round(x,3) for x in emp])))
