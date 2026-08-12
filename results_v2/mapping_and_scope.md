# TONO → ICAO/ISO → BioGaze mapping and evaluation scope

This table makes explicit, for every TONO single-violation folder, (i) the ICAO/ISO
requirement that the synthetic degradation violates and (ii) whether the rule-based
**BioGaze** analyzer actually evaluates that requirement. Only *in-scope* folders can be
used as per-criterion ground truth for the conformal layer.

| TONO folder | # images | ICAO / ISO requirement | BioGaze criterion | In scope? | Notes |
|---|---|---|---|---|---|
| bkg | 269 | Uniform / plain background (ISO/IEC 39794-5; ICAO 9303) | Uniform_background | Yes | Non-uniform / cluttered background |
| cap | 288 | No head covering | Head_without_covering | Yes | Hat / cap / hood |
| ce | 244 | Eyes open and visible | Eyes_open | Yes | Closed eyes |
| ceg | 240 | Eyes open (with spectacles) | Eyes_open | Yes | Closed eyes + glasses |
| expos | 311 | Correct exposure / brightness | Correct_exposure | Yes | Over/under-exposed |
| la_1 | 308 | Gaze at camera | Gaze_in_camera | Yes | Looking away (variant 1) |
| la_2 | 303 | Gaze at camera | Gaze_in_camera | Yes | Looking away (variant 2) |
| light | 149 | Uniform face lighting | Uniform_face_lighting | Yes | Uneven / directional lighting |
| mkup | 197 | No or light makeup | No_or_light_makeup | Yes | Heavy makeup |
| oof | 311 | In focus (sharp) | In_focus | Yes | Out of focus / blur |
| pixel | 311 | No pixelation | No_pixelation | Yes | Pixelation artefact |
| poster | 311 | No posterization | No_posterization | Yes | Colour banding |
| sat | 311 | Correct colour saturation | Correct_saturation | Yes | Over/under-saturated |
| sm | 309 | Neutral expression | Neutral_expression | Yes | Smiling / non-neutral |
| sun | 245 | No sunglasses | No_sunglasses | Yes | Sunglasses |
| tq | 150 | Frontal head pose | Frontal_pose | Yes\* | Head turned (yaw) — validated below |
| **zoom** | 311 | **Correct face dimension / head-size ratio** (ISO/IEC 39794-5 head-width & eye position) | **— not checked by BioGaze** | **No** | **OUT OF SCOPE — see below** |

16 folders map to an in-scope BioGaze criterion (across 14 distinct criteria; `ce`/`ceg`
both → Eyes_open, `la_1`/`la_2` both → Gaze_in_camera). One folder (`zoom`) is out of scope.

## Validation with the recorded metrics

We verified two ambiguous folders against the continuous metrics BioGaze records, rather
than trusting the folder name alone.

**`tq` = head-pose (yaw) deviation, correctly → Frontal_pose.**
Median |yaw| in `tq` is **13.0°** versus **1.2°** across the whole dataset (and ~1° for
control folders such as `bkg`, `sm`). BioGaze marks `tq` as *non-frontal* on **99%** of its
images (Frontal_pose pass-rate 0.01). The mapping `tq → Frontal_pose` is therefore correct;
`tq` denotes a *turned* head (yaw), not roll or pitch.

**`zoom` = head-size / framing violation, genuinely OUT OF SCOPE.**
Median inter-eye distance (IED) in `zoom` is **138.5 px** (range 117–176) versus **96.5 px**
dataset-wide — the face is imaged much larger than the ICAO head-size / framing envelope
allows. Head pose is normal (median |yaw| 2.8°). BioGaze has **no explicit head-size /
eye-position gate**, so it neither reliably passes nor fails these images on a dedicated
criterion (Frontal_pose pass-rate 0.26, tripped incidentally). Consequently `zoom` images
must **not** be labelled "genuinely compliant": they violate a real ICAO requirement that
this analyzer does not check. In the analysis they are treated as *out-of-scope* — never
counted as compliant, and reported separately in the image-level decision confusion.

## Consequence for the experiments

Because TONO is single-violation by construction, the test set contains essentially **no
fully-compliant images** (every image carries exactly one degradation, and `zoom` is out of
scope rather than compliant). This is why the image-level ACCEPT class is tiny and cannot be
evaluated properly on TONO — a limitation that directly motivates acquiring **DFIC** (real,
multi-attribute, with genuinely compliant images) for the follow-up study.
