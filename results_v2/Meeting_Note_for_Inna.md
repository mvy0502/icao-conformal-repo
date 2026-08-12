# Response to review — revised experiments and open protocol questions

**Author:** Mustafa Vedat Yıldırım (BSc AI, Oxford Brookes)
**Supervisor:** Dr. Inna Skarga-Bandurova
**Purpose:** Evidence for the meeting so we can agree the final experimental protocol.

This note summarises the experiments I re-ran in response to your review, the corrections
I made to the earlier (over-stated) claims, and the methodological questions I would like
to decide together.

---

## 1. What I corrected

**Subject-disjoint evaluation.** The earlier split was stratified by violation folder, not
by identity. I parsed the synthetic subject id (`S<id>`) from each filename and found that
**247 of 275 subjects appear in more than one violation folder** (one subject in up to 17).
The old split therefore leaked identities between calibration and test. All results below
use a **subject-disjoint 60/20/20 split** (train / calibration / test), with saved manifests,
and are reported as **mean ± std over 5 random subject-disjoint splits**.

**Violation-class coverage is now the primary metric.** As you noted, the marginal average
coverage hides the quantity that actually matters — the coverage on true violations, whose
miss-cost is high.

**Honest retake metric.** I no longer describe the reduction as "image retakes." What we
reduce is the number of **per-criterion false flags** (true-OK criteria wrongly marked as
violations), and I report the detection trade-off alongside it.

---

## 2. Headline revised results (subject-disjoint, 5 splits, target 0.90)

- **Marginal coverage holds:** mean **0.903 ± 0.015** across criteria and splits — the
  distribution-free guarantee survives the honest split. (My from-scratch LAC implementation
  matches MAPIE to within 0.003, so both are cross-checked.)
- **Marginal conformal silently under-covers the violation class:** mean violation-class
  coverage **0.819 ± 0.026**, and it falls as low as **0.32** on `No_posterization`. Seven of
  fourteen criteria are below 0.90 on the violation class. This is exactly the failure mode
  your review anticipated.
- **Mondrian (label-conditional) conformal restores it:** violation-class coverage
  **0.909 ± 0.022**, i.e. the guarantee is re-established on the costly class.
- **Method comparison (the new core contribution).** On one common subject-disjoint test set:

  | Method | FPR | Recall (violations) | Per-criterion false-flag reduction vs raw |
  |---|---|---|---|
  | Raw BioGaze | 0.147 | 0.950 | — |
  | Logistic only (no conformal) | 0.050 | 0.947 | 69% |
  | Threshold + abstention | 0.034 | 0.932 | 77% |
  | LAC conformal (marginal) | 0.033 | **0.808** | 75% |
  | Mondrian conformal | 0.045 | 0.882 | 72% |

  Two honest observations: (i) most of the false-flag reduction comes from **recalibration**
  itself, not from conformal — a logistic layer already removes ~69% of false flags while
  keeping recall; (ii) **marginal LAC buys the lowest FPR by sacrificing violation recall**
  (0.81) and by under-covering the violation class. **Mondrian is the balanced operating
  point**: it keeps the violation-class guarantee, recovers recall to 0.88, and still removes
  ~72% of false flags. So the contribution is *not* "conformal reduces retakes" but "**label-
  conditional conformal gives a violation-class coverage guarantee under class imbalance,
  and we characterise the false-flag / detection trade-off of five triage strategies.**"

- **Image-level safety:** of **925** true-violation test images, **0 were wrongly ACCEPTED**;
  every violation was routed to REJECT (862) or RETAKE (63). The ACCEPT class is tiny (6, all
  `zoom`), which is a property of TONO, not of the method (see §4).
- **Joint coverage caveat:** requiring all 14 prediction sets to simultaneously contain the
  truth gives only **~0.31** empirical joint coverage — the expected multiple-comparison
  effect. A Bonferroni-style per-criterion target of 0.993 would be needed for joint 0.90.
  I report this honestly and flag it as future work rather than hiding it.

---

## 3. TONO → ICAO → BioGaze scope (attached mapping table)

I built an explicit mapping from each TONO folder to the ICAO/ISO requirement and to the
BioGaze criterion that checks it. Two points I verified against the recorded metrics rather
than the folder name:

- **`tq` → Frontal_pose is correct.** Median |yaw| in `tq` is 13.0° vs 1.2° dataset-wide, and
  BioGaze flags 99% of `tq` as non-frontal.
- **`zoom` is out of scope, not compliant.** Median inter-eye distance is 138 px vs 96 px
  dataset-wide (the face is too large for the ICAO framing envelope), a **head-size / framing**
  violation that BioGaze does not check. I therefore do **not** label `zoom` "genuinely
  compliant"; it is reported separately as out-of-scope.

---

## 4. Why we need DFIC (acquisition & evaluation plan)

TONO is **single-violation by construction**, so the test set contains essentially **no
fully-compliant images** and no images with *co-occurring* violations. Two consequences:

1. The image-level **ACCEPT** decision cannot be validated on TONO (only 6 non-violation
   images, all out-of-scope `zoom`).
2. We cannot study **multi-attribute** conformal decisions (real photos fail several criteria
   at once).

**DFIC** (real passport-style images, multi-attribute labels, including genuinely compliant
ones) fixes both. Plan for the follow-up study (Paper 2):

- Obtain DFIC via the signed data-use form (you sign as supervisor; I complete the request).
- Re-run BioGaze to produce the same feature representation, keeping the pipeline identical.
- Re-fit the per-criterion conformal layer with a **subject-disjoint** split.
- Evaluate the full **ACCEPT / REJECT / RETAKE** confusion, including a real ACCEPT class,
  and calibrate the RETAKE region against an **asymmetric cost** (miss-cost ≫ retake-cost).
- Report joint (multi-criterion) coverage with a Bonferroni / label-conditional correction.

I have kept Paper 2 as a **standalone continuation** (real-data validation + cost-sensitive
triage), so it does not entangle Paper 1's methodology.

---

## 5. Open methodological questions to agree with you

1. **Primary guarantee.** Do we commit to reporting **label-conditional (Mondrian) coverage**
   as the headline guarantee, with marginal coverage as a secondary/contrast result?
2. **Operating point.** Is Mondrian at 0.90 the operating point we present, or do we present a
   small **cost curve** (sweep α) and let an asymmetric miss/retake cost pick the point?
3. **Baselines.** Are the five methods (raw / logistic / threshold-abstain / LAC / Mondrian)
   the agreed comparison set, or do you want **APS** added (note: MAPIE's binary API restricts
   us to LAC, so APS would use my from-scratch code)?
4. **Joint coverage.** For the image-level decision, do we present per-criterion guarantees
   only, or also a **corrected joint guarantee** (Bonferroni / label-conditional)?
5. **Research question framing.** Confirm the reframing to **"uncertainty-aware triage under
   class imbalance and asymmetric cost"** rather than "average coverage."
6. **Scope of Paper 1.** Keep Paper 1 on **TONO only** (method + guarantee + trade-off) and
   defer all real-data / ACCEPT-class claims to Paper 2 with DFIC — agreed?
7. **`zoom` handling.** Report `zoom` strictly as out-of-scope (my current choice), or add a
   short note proposing a head-size check as a BioGaze extension?

I have **not** polished the Introduction / Related Work yet, per your advice — I will do that
only once we fix the protocol above.
