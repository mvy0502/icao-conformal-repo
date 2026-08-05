# Conformal Prediction for Reliable ICAO Face-Image Compliance Verification

A conformal-prediction layer on top of a rule-based ICAO/ISO face-image compliance
analyzer (**BioGaze**), giving each per-criterion compliance decision a statistical
**coverage guarantee** and an **ACCEPT / REJECT / RETAKE** decision framework aimed at
reducing unnecessary photo retakes for machine-readable travel documents.

**Author:** Mustafa Vedat Yıldırım — Oxford Brookes University, BSc Artificial Intelligence
**Supervisor:** Dr. Inna Skarga-Bandurova

---

## Key result

Across **all 14 ICAO criteria**, empirical coverage on the held-out test set clusters
tightly around the 90 % target — **mean = 0.9024** (range 0.880–0.922). The per-criterion
scatter around 90 % is the expected finite-sample behaviour of marginal conformal
prediction, confirming the distribution-free guarantee holds on real data.

![Per-criterion coverage](results/coverage_chart.png)

**Decision framework** (test set, n = 913): ACCEPT 7 · REJECT 797 · RETAKE 109.
A naive BioGaze policy would send 907 images to retake, 63 of which are actually fully
compliant; the current (strict) conformal ACCEPT rule recovers 7 of these.

---

## Data

[TONO synthetic dataset](https://miatbiolab.csr.unibo.it/tono-synthetic-dataset/)
(MI@BioLab, University of Bologna; CC BY-NC 4.0), single-violation partition:
4,568 images organised into folders by violation type — the folder name is the
per-criterion ground truth. The dataset itself is **not** redistributed here.

## Method

1. **BioGaze** ([repo](https://github.com/Maphoz/BioGaze)) is run on every image, producing
   14 binary ICAO criterion decisions + continuous metrics (head pose, inter-eye distance,
   eye/mouth openness). See `batch_predict.py`.
2. Treating BioGaze as a black box, a probabilistic classifier is fit per criterion on these
   features; **MAPIE** split conformal prediction (LAC score, 60/20/20 train/calibration/test)
   yields prediction sets at a target 90 % coverage. See `conformal_analysis.py`.
3. Figures via `make_charts.py`.

## Reproduce

```bash
# 1. Analysis stack
pip install -r requirements.txt

# 2. (BioGaze predictions — needs the BioGaze stack: torch, dlib, mediapipe, ultralytics, rmn)
#    python batch_predict.py -i <TONO/release> -o tono_full_pred.csv

# 3. Conformal analysis + figures
python conformal_analysis.py --csv tono_full_pred.csv --out results
python make_charts.py
```

## Files

- `batch_predict.py` — BioGaze batch runner → per-criterion CSV (resume + per-image timeout).
- `conformal_analysis.py` — MAPIE split conformal + ACCEPT/REJECT/RETAKE + retake metric.
- `make_charts.py` — coverage figure (PNG + SVG).
- `results/` — `coverage_table.csv`, `summary.json`, `coverage_chart.png`, `coverage_chart.svg`.

## Known points / next steps

- Violation-class (conditional) coverage is below the marginal 90 % for some criteria
  (class imbalance) → motivates class-conditional / Mondrian conformal.
- The retake-savings metric depends on how “unnecessary retake” is defined operationally.
- Planned: refine decision/retake metric → Grad-CAM explainability → Paper 1.

## License / attribution

Research code, non-commercial. BioGaze and TONO retain their own licenses; cite TONO
(Borghi et al., ECCV 2024 Workshops) and BioGaze if you use them.
