# Response to review: revised experiments and open protocol questions

**Author:** Mustafa Vedat Yıldırım (BSc AI, Oxford Brookes)
**Supervisor:** Dr. Inna Skarga-Bandurova
**Purpose:** Evidence for the meeting so we can agree the final experimental protocol.

This note summarises the experiments I re-ran in response to your review, the corrections
I made to the earlier (over-stated) claims, and the methodological questions I would like
to decide together.

> Update: this note has been superseded by a hardened analysis (v3) and a rewritten paper
> (Paper1_v3_draft), produced after an internal multi-agent review. The v3 paper adds
> confidence intervals, a feature ablation, a fixed conformal quantile, a corrected
> (randomized) APS, and honest subject-level and small-sample caveats. Where a number here
> differs from the v3 paper, the v3 paper is authoritative.

---

## 1. What I corrected

**Subject-disjoint evaluation.** The earlier split was stratified by violation folder, not
by identity. I parsed the synthetic subject id (`S<id>`) from each filename and found that
**247 of 275 subjects appear in more than one violation folder** (one subject in up to 17).
The old split therefore leaked identities between calibration and test. All results below
use a **subject-disjoint 60/20/20 split** (train / calibration / test), with saved manifests,
and are reported as **mean ± std over 5 random subject-disjoint splits**.

**Violation-class coverage is now the primary metric.** As you noted, the marginal average
coverage hides the quantity that actually matters: the coverage on true violations, whose
miss-cost is high.

**Honest retake metric.** I no longer describe the reduction as "image retakes." What we
reduce is the number of **per-criterion false flags** (true-OK criteria wrongly marked as
violations), and I report the detection trade-off alongside it.

---

## 2. Headline results (superseded; see the v3 paper for current numbers)

The table and the "Mondrian is the balanced operating point" reading that stood here were
withdrawn in v3. The consolidated seven-method comparison with calibration-only tuning
(Table 3 of the v3 paper) shows that no method dominates: a calibration-tuned abstention rule
reaches a lower false-positive rate than Mondrian at a similar retake rate, and Mondrian's
distinct contribution is the formal class-conditional coverage property, not an accuracy win.
The v3 numbers that replace this section are:

- Marginal LAC violation-class coverage below 0.90 on 11 of 14 criteria; on 5 of them (6 with
  Frontal pose) the whole de-duplicated 95% interval is below 0.90 (as low as 0.40).
  Mondrian brings these back to 0.88 to 0.91 with disjoint intervals.
- Seven strategies on the same five subject-disjoint partitions (FPR / recall over all /
  recall among decided / RETAKE): raw 0.144 / 0.955 / 0.955 / 0%; logistic 0.048 / 0.939 /
  0.939 / 0%; Chow 0.027 / 0.800 / 0.954 / 9.5%; asymmetric 0.013 / 0.784 / 0.940 / 7.3%;
  LAC 0.036 / 0.807 / 0.944 / 7.3%; Mondrian 0.040 / 0.878 / 0.959 / 10.3%; randomized APS
  0.026 / 0.891 / 0.958 / 7.2%.
- Image level (Mondrian): 0 ACCEPT / 958 REJECT / 36 RETAKE on 994 images; 822 correct-target
  and 67 wrong-reason rejections among the 925 in-scope violation images.
- Joint coverage about 0.27 uncorrected; Bonferroni gives about 0.91 but routes 96% (LAC) or
  100% (Mondrian) of images to RETAKE.

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
   defer all real-data / ACCEPT-class claims to Paper 2 with DFIC, agreed?
7. **`zoom` handling.** Report `zoom` strictly as out-of-scope (my current choice), or add a
   short note proposing a head-size check as a BioGaze extension?

I have **not** polished the Introduction / Related Work yet, per your advice; I will do that
only once we fix the protocol above.

---

## 6. Additional experiments completed while awaiting your reply

I used the waiting time to pre-answer several of the questions in Section 5 with actual
evidence rather than leaving them open. All use the same subject-disjoint protocol.

**APS (question 3).** [Corrected in v3.] I added Adaptive Prediction Sets as a sixth method
(from scratch, since MAPIE's binary API is LAC-only). The NON-randomized variant is
degenerate here (coverage 1.000, confident-singleton recall 0.016), but the correctly
RANDOMIZED APS of Romano et al. is competitive: false-positive rate 0.026 at recall 0.89 and
set size about 1.07. So APS is a legitimate method, not a negative control; the earlier
degeneracy was a property of the non-randomized variant only.

**Joint image-level guarantee (question 4).** Uncorrected, the 14 per-criterion sets give
only 0.29 joint coverage. Applying a Bonferroni correction (per-criterion target 0.993)
raises joint coverage to 0.91 (Mondrian) / 0.93 (LAC), i.e. a genuine image-level guarantee
is achievable. The cost is conservatism: about 95% of images are then routed to RETAKE, and
mean set size stays small (1.22). So we can offer a corrected joint guarantee, but it trades
almost all ACCEPT/REJECT decisiveness for the guarantee.

**Operating point / cost curve (question 2).** Sweeping the target from 0.80 to 0.98 for
Mondrian, violation-class coverage tracks the target, false-positive rate stays low
(about 0.03 to 0.04), and violation recall peaks around the 0.90 to 0.95 range before falling
at 0.98 (over-conservative). A target of about 0.95 looks like a strong operating point
(violation coverage 0.97, recall 0.90, FPR 0.03), which we can discuss.

**Mondrian at image level.** Aggregating Mondrian sets into the image decision gives a more
decisive and equally safe triage than marginal LAC: 933 REJECT, 61 RETAKE, 0 ACCEPT, and
again 0 of 925 true-violation images wrongly accepted (marginal LAC gave 878 / 110 / 6).

Figures for these are in the Drive folder (fig_alpha_curve, fig_joint_coverage).
