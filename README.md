# Class-Conditional Conformal Triage for ICAO Face-Image Compliance under Class Imbalance

Code, result files and figures for the paper of the same title by **Mustafa Vedat Yildirim** and
**Inna Skarga-Bandurova** (School of Engineering, Computing and Mathematics, Oxford Brookes University),
under submission to the *International Journal of Computing*.

A small per-criterion classifier and a split conformal layer are added on top of a rule-based ICAO/ISO
face-image compliance analyser (**BioGaze**), without retraining it, and evaluated on the synthetic
**TONO** benchmark under a subject-disjoint protocol. All results in the paper and Figs. 2 to 8 are
produced by `run_all_v3.sh` from one input file, `data/tono_full_pred.csv`.

---

## Main results

Identities, not images, are split 60/20/20 (247 of 275 TONO identities appear in more than one
violation folder, so a folder split would leak identities). Zoom images (head size, not checked by
BioGaze) are out of scope. alpha = 0.10 throughout.

1. **Marginal conformal under-covers the violation class.** Marginal LAC meets its overall 0.90 target,
   but on the primary partition the whole 95% subject-cluster bootstrap interval lies below 0.90 for
   3 of 14 criteria (No posterization 0.39, interval 0.25 to 0.53). Over ten partitions its mean is below
   0.90 on 11 of 14 criteria (aggregate 0.81). By the same single-partition test, Mondrian's interval lies
   below 0.90 for 2 criteria; over ten partitions every Mondrian mean lies between 0.89 and 0.92
   (aggregate 0.91). Single-partition intervals hold the calibration set fixed, so the methods are
   compared on the ten-partition means.
2. **Coverage is stable across feature sets; accuracy is not.** Removing the cross-criterion features
   cuts the false-flag reduction from 67% to 26%, because each TONO image has exactly one violation.
   Coverage stays on target, so the coverage property is robust to feature quality within the
   evaluated distribution; accuracy on real multi-attribute data remains to be tested.
3. **Seven triage strategies, thresholds tuned on calibration data only** (raw BioGaze, logistic layer,
   Chow rejection, an asymmetric two-threshold rule, marginal LAC, Mondrian, randomised APS; five
   partitions). The logistic layer gives most of the accuracy gain; the tuned abstention rules reach a
   lower false-positive rate than Mondrian at a similar RETAKE rate. Mondrian's contribution is its
   class-conditional coverage guarantee, at a cost of 1 to 3 points of false-positive rate. Note: when
   no threshold pair on the grid meets the abstention tolerance, the asymmetric rule falls back to a
   fixed band (0.4, 0.6); at the 0.10 target this happens in 27 of 70 criterion fits.
4. **Image-level triage.** On the primary test set (994 images) Mondrian gives 0 ACCEPT, 942 REJECT and
   52 RETAKE. Of the 925 in-scope violation images, 817 are rejected for the right criterion, 57 for a
   different one and 51 go to RETAKE; none of the 69 out-of-scope zoom images is accepted. Joint
   coverage of all 14 criteria is about 0.27 without correction; a Bonferroni correction raises it to
   about 0.91 but sends 96% (LAC) or all (Mondrian) images to RETAKE.
5. **Exchangeability checks.** Subject-capped and random equal-n calibration subsamples agree
   (0.92, SD 0.02); a subject bootstrap gives 0.89 (5th to 95th percentile 0.86 to 0.91).

Our split conformal layer is written from scratch; MAPIE's `SplitConformalClassifier` returns
identical prediction sets on every test image (`crosscheck_v3.py`).

## Repository layout

| Path | Contents |
|---|---|
| `data/tono_full_pred.csv` | Per-image BioGaze outputs on TONO: 4,569 images, 14 decisions, 7 continuous metrics |
| `analysis_v2.py` | Loader, criterion mapping, subject-disjoint split (shared by all scripts) |
| `analysis_v3.py` | Exact finite-sample quantile, feature ablation, ten-partition coverage, randomised APS, isotonic baseline |
| `primary_ci_v3.py` | Primary per-criterion intervals (fixed partition, subject-cluster bootstrap, B = 2000) and ten-partition robustness |
| `inna_protocol.py` | Seven-method table with calibration-only tuning, risk-abstention curves, cluster-aware checks |
| `image_joint_v3.py` | Image-level decisions and joint (Bonferroni) coverage |
| `method_calib_v3.py` | Achieved against nominal coverage |
| `crosscheck_v3.py` | Overall LAC coverage per criterion and agreement with MAPIE |
| `dedup_ci_v3.py` | Secondary sensitivity analysis (not used in the paper) |
| `make_figs_v3.py`, `make_figs_extra_v3.py`, `make_figs_inna.py`, `fig_style.py` | Figs. 2 to 8 at printed size (two-column journal layout) |
| `batch_predict.py` | Runs BioGaze over TONO to produce the CSV (needs the BioGaze stack) |
| `results_v3/` | All result files (JSON) and figures (PNG, SVG) used in the paper |
| `results_v2/`, `results/` | Earlier analysis rounds, kept for the record |

## Reproduce

Python 3.11 (the results are identical with 3.13).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt      # exact versions; requirements_analysis.txt gives minimums
cp data/tono_full_pred.csv .
./run_all_v3.sh tono_full_pred.csv                  # 14 steps, about 80 seconds
```

On Windows, point the script at the venv interpreter: `PY=.venv/Scripts/python.exe ./run_all_v3.sh tono_full_pred.csv`.

All seeds are fixed, so a re-run reproduces every JSON in `results_v3/` exactly (`.gitattributes`
keeps line endings identical across platforms) and redraws the figures.

To regenerate the CSV from the images, download TONO and run BioGaze
(`python batch_predict.py -i <TONO/release> -o tono_full_pred.csv`; needs torch, dlib, mediapipe,
ultralytics and rmn).

## Known issues

- `make_xai_panel.py` (`results/xai/`) builds the region-level explanation panel from an earlier draft.
  Three of its overlays (sunglasses, background, head covering) show the same segmentation, because
  the first image in those three TONO folders has the same file name. It also uses absolute paths.
  The panel is not part of the current paper and will be fixed.
- Some v1 and v2 scripts (`conformal_analysis.py`, `make_charts.py`, `robustness.py` and others) are
  kept for the record; they are not used by `run_all_v3.sh`.

## Data and licences

- Code: MIT licence (`LICENSE`).
- `data/tono_full_pred.csv`, the result files and the figures are derived from TONO and are released
  under CC BY-NC 4.0, the licence of the TONO dataset (`DATA_LICENSE.md`).
- The TONO images are not redistributed here: [TONO synthetic dataset](https://miatbiolab.csr.unibo.it/tono-synthetic-dataset/)
  (MI@BioLab, University of Bologna). BioGaze: [github.com/Maphoz/BioGaze](https://github.com/Maphoz/BioGaze).
  Please cite TONO (Borghi et al., ECCV 2024 Workshops) and BioGaze (Elatfi et al., FG 2025) if you use these files.

## Citation

See `CITATION.cff`. A DOI will be added once the repository is archived on Zenodo.
