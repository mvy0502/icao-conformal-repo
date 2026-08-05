import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#1f2933"; TEAL = "#0f9d8f"; AMBER = "#e0a100"; RED = "#c92a2a"
LIGHT = "#eaf4f2"; GREY = "#f1f3f5"

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

def box(x, y, w, h, title, sub, fc=LIGHT, ec=TEAL, tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                 fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=tcol, zorder=4)
    if sub:
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=8.2,
                color="#4b5563", zorder=4)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                 lw=1.8, color=INK, zorder=2))

y = 24; h = 12
box(1,  y, 15, h, "Input\nface image", "passport / ID photo", fc=GREY, ec="#adb5bd")
box(19, y, 20, h, "BioGaze analyzer", "rule-based · 14 ICAO criteria\n+ continuous metrics", fc=GREY, ec="#adb5bd")
box(42, y, 18, h, "Feature vector", "per image:\n14 decisions + metrics")
box(63, y, 19, h, "Split conformal", "per-criterion classifier\n+ LAC calibration")
box(85, y, 14, h, "Prediction sets", "{ok} / {viol} / {both}")
for x1, x2 in [(16,19),(39,42),(60,63),(82,85)]:
    arrow(x1, y+h/2, x2, y+h/2)

# guarantee banner under conformal
ax.text(72.5, y-2.5, "distribution-free 90% coverage guarantee",
        ha="center", va="top", fontsize=8.6, style="italic", color=TEAL)

# decision box
dy = 2
ax.add_patch(FancyBboxPatch((84, dy), 16, 13, boxstyle="round,pad=0.6,rounding_size=1.5",
             fc="#ffffff", ec=INK, lw=1.8, zorder=3))
ax.text(92, dy+11, "Decision", ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK, zorder=4)
ax.text(92, dy+7.6, "ACCEPT", ha="center", fontsize=9.5, color=TEAL, fontweight="bold", zorder=4)
ax.text(92, dy+5.0, "REJECT", ha="center", fontsize=9.5, color=RED, fontweight="bold", zorder=4)
ax.text(92, dy+2.4, "RETAKE", ha="center", fontsize=9.5, color=AMBER, fontweight="bold", zorder=4)
arrow(92, y, 92, dy+13)

ax.text(50, 40.5, "Model-agnostic conformal layer on a rule-based ICAO compliance analyzer",
        ha="center", fontsize=12.5, fontweight="bold", color=INK)

plt.tight_layout()
plt.savefig("/home/claude/results/pipeline.png", bbox_inches="tight", facecolor="white")
plt.savefig("/home/claude/results/pipeline.svg", bbox_inches="tight", facecolor="white")
print("pipeline saved")
