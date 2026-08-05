import sys, os, glob
sys.path.insert(0, "/home/claude/BioGaze")
import torch
_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from batch_predict import BioGazeRunner

# (folder, overlay_type, criterion, human label)
CASES = [
    ("sun", "parsing", "No_sunglasses", "Sunglasses"),
    ("bkg", "parsing", "Uniform_background", "Non-uniform background"),
    ("cap", "parsing", "Head_without_covering", "Head covering"),
    ("ce",  "landmark", "Eyes_open", "Closed eyes"),
    ("sm",  "landmark", "Neutral_expression", "Non-neutral expression"),
]
ROOT = "/home/claude/TONO/release"
OUT = "/home/claude/results/xai"
os.makedirs(OUT, exist_ok=True)

runner = BioGazeRunner()  # loads all sub-models once

rows = []
for folder, otype, crit, label in CASES:
    img = sorted(glob.glob(f"{ROOT}/{folder}/*.png"))[0]
    try:
        res = runner.analyze(img)
        dec = res.get(crit)
    except Exception:
        dec = None
    biog = "FAIL" if dec is False else ("PASS" if dec is True else "-")
    stem = os.path.splitext(os.path.basename(img))[0]
    ext = os.path.splitext(img)[1]
    if otype == "parsing":
        runner.face_parser.parse_and_save_faces(img, OUT)
        ov = os.path.join(OUT, stem + "_parsing" + ext)
        oname = "Segmentation"
    else:
        runner.landmark_recognizer.detect_and_draw_landmarks(img, OUT)
        ov = os.path.join(OUT, stem + "_landmark" + ext)
        oname = "Landmarks"
    rows.append((img, ov, crit, label, oname, biog))

n = len(rows)
fig, axes = plt.subplots(n, 2, figsize=(6.6, 3.15 * n), dpi=140)
for i, (img, ov, crit, label, oname, biog) in enumerate(rows):
    a0, a1 = axes[i]
    a0.imshow(Image.open(img).convert("RGB")); a0.set_xticks([]); a0.set_yticks([])
    a1.imshow(Image.open(ov).convert("RGB")); a1.set_xticks([]); a1.set_yticks([])
    a0.set_title("Input", fontsize=10, color="#1f2933")
    a1.set_title(f"{oname} explanation", fontsize=10, color="#0f9d8f")
    col = "#c92a2a" if biog == "FAIL" else "#1f2933"
    a0.set_ylabel(f"{label}", fontsize=10.5, color="#1f2933", fontweight="bold")
    a1.set_xlabel(f"Criterion: {crit.replace('_',' ')}   ·   BioGaze: {biog}",
                  fontsize=9, color=col)
fig.suptitle("Region-level explanations of ICAO compliance decisions (TONO examples)",
             fontsize=13, fontweight="bold", color="#1f2933")
plt.tight_layout(rect=[0.02, 0, 1, 0.985])
plt.savefig(f"{OUT}/xai_panel.png", bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT}/xai_panel.svg", bbox_inches="tight", facecolor="white")
print("panel saved:", n, "cases ->", f"{OUT}/xai_panel.png")
