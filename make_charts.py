import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/home/claude/results"
df = pd.read_csv(os.path.join(OUT, "coverage_table.csv"))
summ = json.load(open(os.path.join(OUT, "summary.json")))
df = df[df["coverage"].notna()].copy().sort_values("coverage")

target = summ["target_coverage"]
labels_tr = {
    "Head_without_covering": "Head uncovered", "Eyes_open": "Eyes open",
    "No_sunglasses": "No sunglasses", "No_posterization": "No posterization",
    "Gaze_in_camera": "Gaze in camera", "Neutral_expression": "Neutral expression",
    "In_focus": "In focus", "Correct_exposure": "Correct exposure",
    "No_or_light_makeup": "No/light makeup", "No_pixelation": "No pixelation",
    "Frontal_pose": "Frontal pose", "Correct_saturation": "Correct saturation",
    "Uniform_background": "Uniform background", "Uniform_face_lighting": "Uniform face lighting",
}
names = [labels_tr.get(c, c) for c in df["criterion"]]
cov = df["coverage"].values

# renkler (erişilebilir): hedefi tutan = mavi-yeşil, altında kalan = amber
INK = "#1f2933"; OK = "#0f9d8f"; UNDER = "#e0a100"; GRID = "#e3e8ee"; TARGET = "#d64545"
colors = [OK if v >= target - 0.005 else UNDER for v in cov]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
bars = ax.barh(names, cov, color=colors, height=0.62, zorder=3)
ax.axvline(target, color=TARGET, lw=2, ls="--", zorder=4)
ax.text(target, len(names)-0.3, f"  target {target*100:.0f}%", color=TARGET,
        va="center", ha="left", fontsize=10, fontweight="bold")

for b, v in zip(bars, cov):
    ax.text(v + 0.006, b.get_y()+b.get_height()/2, f"{v:.3f}",
            va="center", ha="left", fontsize=10, color=INK, fontweight="bold")

ax.set_xlim(0.75, 1.0)
ax.set_xlabel("Empirical coverage (test set)", fontsize=11, color=INK)
ax.set_title("Conformal Prediction — Per-Criterion Coverage\n"
             f"(TONO dataset, {summ['n_valid']} images; target guarantee {target*100:.0f}%)",
             fontsize=13, color=INK, fontweight="bold", loc="left")
ax.xaxis.grid(True, color=GRID, zorder=0)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0, labelsize=10)
leg = [Patch(color=OK, label="Meets target"),
       Patch(color=UNDER, label="Just below target (sampling noise)")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          ncol=2, frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/results/coverage_chart.png", bbox_inches="tight", facecolor="white")
plt.savefig("/home/claude/results/coverage_chart.svg", bbox_inches="tight", facecolor="white")
print("kaydedildi: coverage_chart.png + .svg |", len(names), "kriter")
