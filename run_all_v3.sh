#!/usr/bin/env bash
# Reproduce the full hardened (v3) analysis end to end from tono_full_pred.csv.
set -e
CSV=${1:-tono_full_pred.csv}
echo "[1/9] v2 base analysis";                 python3 analysis_v2.py --csv "$CSV" --out results_v2 --alpha 0.1 --seed 0
echo "[2/9] v2 strengthen";                    python3 strengthen.py "$CSV" results_v2
echo "[3/9] v3 hardened analysis";             python3 analysis_v3.py "$CSV" results_v3
echo "[4/9] v3 image-level + joint";           python3 image_joint_v3.py "$CSV" results_v3
echo "[5/9] v3 method + calibration";          python3 method_calib_v3.py
echo "[6/10] primary CIs (fixed partition + subject-cluster bootstrap)"; python3 primary_ci_v3.py "$CSV" results_v3
echo "[6/9] supervisor protocol (fair table, abstention, cluster-aware)"; python3 inna_protocol.py "$CSV" results_v3
echo "[7/9] figures (ablation, per-criterion CI, risk-coverage)"; python3 make_figs_v3.py "$CSV" results_v3
echo "[8/9] figures (image-level decomposition, joint coverage)";  python3 make_figs_extra_v3.py results_v3
echo "[9/9] figures (risk-abstention, subject sensitivity)";       python3 make_figs_inna.py results_v3
echo "DONE -> results_v3/   (paper: node build_paper_v3.js)"
