# Uncertainty-Aware Triage for Rule-Based ICAO Face-Image Compliance

Label-conditional conformal guarantees and a trade-off study on top of a rule-based
ICAO/ISO face-image compliance analyzer (**BioGaze**), evaluated on the **TONO** synthetic
benchmark under a strictly **subject-disjoint** protocol.

**Author:** Mustafa Vedat Yildirim, Oxford Brookes University, BSc Artificial Intelligence
**Supervisor:** Dr Inna Skarga-Bandurova

**Current version: v3 (hardened, pre-freeze).** The authoritative paper is
`results_v3/Paper1_v3_draft.pdf`. Everything in `results_v3/` is reproduced by `./run_all_v3.sh`.
The `results_v2/` folder is kept for the record; where a v2 number differs from v3, v3 is
authoritative.

---

## What the study shows (v3)

Each of BioGaze's 14 per-criterion decisions is wrapped with a per-criterion supervised
meta-classifier (class-balanced logistic regression on the 14 decisions plus 7 continuous
metrics) followed by split conformal prediction. Identities, not images, are partitioned
60/20/20 (247 of 275 TONO identities recur across violation folders, so a folder split would
leak). Zoom images (head-size violation, not checked by BioGaze) are out of scope.

1. **Marginal conformal under-covers the violation class; Mondrian restores it.** Marginal
   LAC holds its overall 0.90 target, but its violation-class coverage is below 0.90 on 11 of
   14 criteria and, on 5 of them (6 with the small-sample Frontal pose), the whole
   de-duplicated 95% interval lies below 0.90 (as low as 0.40 on No posterization).
   Label-conditional (Mondrian) conformal brings each of these back to 0.88 to 0.91, with
   intervals disjoint from the marginal method. Intervals are computed on unique test images
   over ten seeds (`dedup_ci_v3.py`).
2. **The guarantee is feature-agnostic; the accuracy is a dataset artifact.** A feature
   ablation shows coverage unchanged across feature sets while the false-flag reduction
   collapses from 66% to 26% without cross-criterion features. TONO is single-violation, so
   accuracy numbers should not be expected to transfer.
3. **Seven-strategy trade-off with calibration-only tuning.** Raw BioGaze, logistic only,
   Chow confidence rejection, an asymmetric two-threshold rule (cost 10 per miss), marginal
   LAC, Mondrian and randomized APS on the same five partitions, with recall over all cases
   and among decided cases, RETAKE rate, decision coverage, selective error and false-flag
   reduction. No method dominates. The calibration-tuned abstention rules reach a lower
   false-positive rate than Mondrian at a similar retake rate (Table 3, and Table 4 at
   abstention targets 0.05 / 0.10 / 0.15). Mondrian's contribution is the formal
   class-conditional coverage property (thresholds fixed as calibration quantiles, not
   searched against a cost objective), not a win on FPR or recall. Non-randomized APS
   degenerates on a binary problem; randomized APS is competitive.
4. **Image-level triage and the multiple-comparison problem.** Aggregating 14 Mondrian sets
   gives 0 ACCEPT / 958 REJECT / 36 RETAKE on 994 test images; on the 925 in-scope violation
   images, 822 are rejected for the right criterion, 67 for a different one, 36 go to RETAKE.
   ACCEPT is structurally empty on TONO. Naive joint coverage is about 0.27; Bonferroni
   restores about 0.91 but routes 96% (LAC) or 100% (Mondrian, whose corrected violation
   threshold is infinite at this calibration size) of images to RETAKE.
5. **Exchangeability and cluster-aware sensitivity.** The guarantee is claimed at the image
   level under exchangeability; subject-disjoint splitting protects against identity leakage.
   Subject-capped and random equal-n calibration subsamples agree (0.94 both), a subject
   bootstrap gives 0.89 (5th to 95th: 0.86 to 0.91), and the between-partition SD (0.010)
   matches the binomial value, so no clustering effect is detected within the power of the
   checks. A formal clustered guarantee is future work.

Figures (all regenerated from scripts): `results_v3/fig_ablation.png`,
`fig_percrit_ci.png`, `fig_riskcoverage.png`, `fig_risk_abstention.png`,
`fig_imgdec_v3.png`, `fig_joint_v3.png`, `fig_subject_sensitivity.png`.

## Data and scope

[TONO synthetic dataset](https://miatbiolab.csr.unibo.it/tono-synthetic-dataset/) (MI@BioLab,
University of Bologna; CC BY-NC 4.0), single-violation partition. The TONO to ICAO to BioGaze
mapping (16 in-scope folders to 14 criteria) and the out-of-scope treatment of `zoom` are in
`results_v2/mapping_and_scope.md` and `results_v2/mapping_table.csv`. The dataset itself is
**not** redistributed here.

## Pipeline

1. **BioGaze** ([repo](https://github.com/Maphoz/BioGaze)) produces per-image 14 binary ICAO
   decisions plus continuous metrics: `batch_predict.py` (resume, per-image timeout).
2. **Subject-disjoint** 60/20/20 split by parsed identity: `analysis_v2.py` (also the shared
   loader and split helpers).
3. **Hardened analysis** `analysis_v3.py`: exact finite-sample quantile, feature ablation,
   ten-seed per-criterion coverage, randomized APS, isotonic risk-coverage baseline.
4. `dedup_ci_v3.py`: de-duplicated Clopper-Pearson intervals (unit = unique test image).
5. `image_joint_v3.py`: image-level decisions with the correct-target / wrong-reason
   decomposition, Bonferroni joint coverage.
6. `method_calib_v3.py`: achieved versus nominal coverage.
7. `inna_protocol.py`: the supervisor pre-freeze checks (images per subject for every split,
   seven-method fair table with calibration-only tuning, risk-abstention curves,
   cluster-aware sensitivity). Optional third argument re-runs it at another abstention target.
8. Figures: `make_figs_v3.py`, `make_figs_extra_v3.py`, `make_figs_inna.py`.
9. Paper: `node build_paper_v3.js` (docx-js), then LibreOffice for the PDF.

Split conformal is implemented from scratch and cross-checked against MAPIE for the marginal
case (agreement within 0.003).

## Reproduce

```bash
pip install -r requirements_analysis.txt
# BioGaze predictions (needs the BioGaze stack: torch, dlib, mediapipe, ultralytics, rmn):
#   python batch_predict.py -i <TONO/release> -o tono_full_pred.csv
./run_all_v3.sh tono_full_pred.csv
python3 dedup_ci_v3.py tono_full_pred.csv results_v3
python3 inna_protocol.py tono_full_pred.csv results_v3 0.05   # optional second abstention target
python3 inna_protocol.py tono_full_pred.csv results_v3 0.15
node build_paper_v3.js
```

Outputs land in `results_v3/`: `analysis_v3.json`, `dedup_ci_v3.json`, `image_joint_v3.json`,
`method_and_calib_v3.json`, `inna_protocol.json` (plus `_t0.05` / `_t0.15`), seven figures
(PNG and SVG) and the paper. Re-running the pipeline reproduces the JSON files byte for byte
(all seeds fixed).

## Explainability

Region-level explanations (BiSeNet segmentation plus 68 landmarks) of BioGaze's rule-based
decisions (`results/xai/xai_panel.png`, `make_xai_panel.py`). They explain BioGaze's rule, not
the meta-classifier's decision, and are presented as a demonstration.

## Next steps (Paper 2)

- Real, multi-attribute validation (DFIC) with a genuine, in-distribution ACCEPT class.
- Cost-calibrated RETAKE region under asymmetric miss and retake cost, and a tighter
  simultaneous (image-level) guarantee than Bonferroni.

## License and attribution

Research code, non-commercial. BioGaze and TONO retain their own licenses; cite TONO
(Borghi et al., ECCV 2024 Workshops) and BioGaze (FG 2025) if used.
