#!/usr/bin/env bash
# Reproduce the full hardened (v3) analysis end to end from tono_full_pred.csv.
set -e
CSV=${1:-tono_full_pred.csv}
echo "[1/6] v2 base analysis";        python3 analysis_v2.py --csv "$CSV" --out results_v2 --alpha 0.1 --seed 0
echo "[2/6] v2 strengthen";           python3 strengthen.py
echo "[3/6] v3 hardened analysis";    python3 analysis_v3.py
echo "[4/6] v3 image-level + joint";  python3 image_joint_v3.py
echo "[5/6] v3 method + calibration"; python3 method_calib_v3.py
echo "[6/6] figures";                 python3 make_figs_v3.py
echo "DONE -> results_v3/"
