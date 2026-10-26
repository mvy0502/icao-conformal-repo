# Data licence

`data/tono_full_pred.csv`, the result files in `results/`, `results_v2/` and `results_v3/`, and the
figures are derived from the TONO synthetic dataset, which is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). They are therefore released under the
same licence: you may share and adapt them for non-commercial purposes, with attribution.

Please attribute:

- TONO: G. Borghi, A. Franco, N. Di Domenico, D. Maltoni, "TONO: A synthetic dataset for face image
  compliance to ISO/ICAO standard," Computer Vision - ECCV 2024 Workshops, LNCS 15642, Springer, 2025,
  https://doi.org/10.1007/978-3-031-91907-7_1.
- BioGaze, which produced the per-image outputs: O. Elatfi, N. Di Domenico, G. Borghi, A. Franco,
  D. Maltoni, "BioGaze: a framework for evaluating the photographic requirements of the ISO/IEC 39794-5
  standard," IEEE FG 2025, https://doi.org/10.1109/FG61629.2025.11099423.
- This repository (see `CITATION.cff`).

The TONO images themselves are not included; obtain them from the dataset authors.

## Columns of `data/tono_full_pred.csv`

| Column | Meaning |
|---|---|
| `rel_path`, `source_folder`, `filename` | TONO image path; the folder names the violated requirement |
| `faces_detected` | Number of faces BioGaze found (images with more than one are excluded by the loader) |
| `Compliant` | BioGaze overall verdict |
| `Head_without_covering` ... `Uniform_face_lighting` | The 14 per-criterion BioGaze decisions (True = pass) |
| `IED`, `pitch`, `yaw`, `roll`, `eyes_open_val`, `mouth_open_val`, `has_glasses` | Continuous BioGaze metrics |
| `error` | BioGaze error message, if any |
