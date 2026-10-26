#!/usr/bin/env bash
# Reproduce the full v4 analysis end to end from tono_full_pred.csv.
# Usage: ./run_all_v3.sh [tono_full_pred.csv]
# On Windows, point PY at the venv interpreter: PY=.venv/Scripts/python.exe ./run_all_v3.sh
set -e
CSV=${1:-tono_full_pred.csv}
PY=${PY:-python3}
echo "[1/14] v2 base analysis";                 $PY analysis_v2.py --csv "$CSV" --out results_v2 --alpha 0.1 --seed 0
echo "[2/14] v2 strengthen";                    $PY strengthen.py "$CSV" results_v2
echo "[3/14] v3 hardened analysis";             $PY analysis_v3.py "$CSV" results_v3
echo "[4/14] v3 image-level + joint";           $PY image_joint_v3.py "$CSV" results_v3
echo "[5/14] v3 method + calibration";          $PY method_calib_v3.py "$CSV" results_v3
echo "[6/14] primary CIs (fixed partition + subject-cluster bootstrap)"; $PY primary_ci_v3.py "$CSV" results_v3
echo "[7/14] supervisor protocol (fair table, abstention, cluster-aware)"; $PY inna_protocol.py "$CSV" results_v3
echo "[8/14] abstention target 0.05 (Table 4)"; $PY inna_protocol.py "$CSV" results_v3 0.05
echo "[9/14] abstention target 0.15 (Table 4)"; $PY inna_protocol.py "$CSV" results_v3 0.15
echo "[10/14] de-duplicated CIs (secondary sensitivity, not in the paper)"; $PY dedup_ci_v3.py "$CSV" results_v3
echo "[11/14] cross-checks (overall LAC coverage, agreement with MAPIE)"; $PY crosscheck_v3.py "$CSV" results_v3
echo "[12/14] figures (ablation, per-criterion CI, risk-coverage)"; $PY make_figs_v3.py "$CSV" results_v3
echo "[13/14] figures (image-level decomposition, joint coverage)";  $PY make_figs_extra_v3.py results_v3
echo "[14/14] figures (risk-abstention, subject sensitivity)";       $PY make_figs_inna.py results_v3
echo "DONE -> results_v3/"
