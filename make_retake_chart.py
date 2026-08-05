import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/home/claude/results"
s = json.load(open(os.path.join(OUT, "summary.json")))
crit = s["criteria"]

labels = {
    "Head_without_covering": "Head uncovered", "Eyes_open": "Eyes open",
    "No_sunglasses": "No sunglasses", "No_posterization": "No posterization",
    "Gaze_in_camera": "Gaze in camera", "Neutral_expression": "Neutral expression",
    "In_focus": "In focus", "Correct_exposure": "Correct exposure",
    "No_or_light_makeup": "No/light makeup", "No_pixelation": "No pixelation",
    "Frontal_pose": "Frontal pose", "Correct_saturation": "Correct saturation",
    "Uniform_background": "Uniform background", "Uniform_face_lighting": "Uniform face lighting",
}
rows = [(labels.get(k, k), v.get("biogaze_fp", 0), v.get("conformal_fp", 0))
        for k, v in crit.items() if not v.get("skip", True)]
rows.sort(key=lambda r: r[1])            # BioGaze false-flag sayısına göre
names = [r[0] for r in rows]
bio = np.array([r[1] for r in rows])
con = np.array([r[2] for r in rows])

INK = "#1f2933"; TEAL = "#0f9d8f"; AMBER = "#e0a100"; GRID = "#e3e8ee"
y = np.arange(len(names)); h = 0.38

fig, ax = plt.subplots(figsize=(9.2, 6), dpi=150)
ax.barh(y + h/2, bio, height=h, color=AMBER, zorder=3, label="BioGaze (rule-based)")
ax.barh(y - h/2, con, height=h, color=TEAL, zorder=3, label="Conformal layer")
for yi, b, c in zip(y, bio, con):
    ax.text(b + 1, yi + h/2, str(b), va="center", fontsize=8.5, color=INK)
    ax.text(c + 1, yi - h/2, str(c), va="center", fontsize=8.5, color=INK)

ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
red = s.get("false_flag_reduction", 0); pct = s.get("false_flag_reduction_pct", 0)
tot_b = s.get("biogaze_false_flags_total", 0); tot_c = s.get("conformal_false_flags_total", 0)
ax.set_xlabel("Unnecessary rejections (per-criterion false flags, test set)", fontsize=11, color=INK)
ax.set_title("Conformal layer cuts unnecessary rejections\n"
             f"Total false flags {tot_b} → {tot_c}   (−{pct}%, at 90% coverage guarantee)",
             fontsize=13, color=INK, fontweight="bold", loc="left")
ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
ax.legend(loc="lower right", frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "retake_chart.png"), bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUT, "retake_chart.svg"), bbox_inches="tight", facecolor="white")
print("kaydedildi: retake_chart.png + .svg | reduction %", pct)
