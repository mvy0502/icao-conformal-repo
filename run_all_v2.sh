#!/usr/bin/env bash
# Reproduce the full revised (subject-disjoint) analysis end to end.
# Prerequisite: tono_full_pred.csv (BioGaze predictions on TONO). To regenerate it:
#   python batch_predict.py -i <TONO/release> -o tono_full_pred.csv   # needs BioGaze stack
set -e
CSV=${1:-tono_full_pred.csv}
echo "[1/5] primary subject-disjoint analysis (seed 0, alpha 0.10)"
python3 analysis_v2.py --csv "$CSV" --out results_v2 --alpha 0.1 --seed 0
echo "[2/5] 5-split robustness"
python3 robustness.py
echo "[3/5] image-level decisions + joint coverage"
python3 image_level.py
echo "[4/5] TONO->ICAO->BioGaze mapping table"
python3 make_mapping.py
echo "[5/5] figures"
python3 make_figs_v2.py
echo "DONE -> results_v2/"
