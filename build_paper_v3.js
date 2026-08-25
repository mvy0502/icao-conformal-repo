const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
        ImageRun, Table, TableRow, TableCell, WidthType, PageBreak } = require('docx');

const R3 = "/home/claude/results_v3/";
const R2 = "/home/claude/results_v2/";
const R1 = "/home/claude/results/";
const INK = "1f2933";

const h1 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 120 } });
const h2 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 170, after: 80 } });
const P  = t => new Paragraph({ children: [new TextRun(t)], spacing: { after: 140 }, alignment: AlignmentType.JUSTIFIED });
const bullet = t => new Paragraph({ text: t, bullet: { level: 0 }, spacing: { after: 50 }, alignment: AlignmentType.JUSTIFIED });
const note = t => new Paragraph({ children: [new TextRun({ text: t, italics: true, color: "8a8f98" })], spacing: { after: 140 } });

function figure(path, w, h, caption) {
  const buf = fs.readFileSync(path);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: w, height: h } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: INK })] }),
  ];
}
function cell(text, o = {}) {
  return new TableCell({ width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { fill: "0f9d8f" } : (o.alt ? { fill: "eef4f3" } : undefined),
    children: [new Paragraph({ alignment: o.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!o.head, color: o.head ? "ffffff" : INK, size: 16 })] })] });
}
function tableFrom(headers, rows, widths) {
  const hr = new TableRow({ tableHeader: true, children: headers.map((t, i) => cell(t, { w: widths[i], head: true, center: i > 0 })) });
  const trs = rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(String(t), { w: widths[i], center: i > 0, alt: ri % 2 })) }));
  return new Table({ columnWidths: widths, width: { size: widths.reduce((a, b) => a + b), type: WidthType.DXA }, rows: [hr, ...trs] });
}

const children = [];

// ---------------- title ----------------
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Uncertainty-Aware Triage for Rule-Based ICAO Face-Image Compliance:", bold: true, size: 30, color: INK })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 150 },
  children: [new TextRun({ text: "Label-Conditional Conformal Guarantees and an Honest Trade-off Study", bold: true, size: 28, color: INK })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
  children: [new TextRun({ text: "Mustafa Vedat Yildirim, Inna Skarga-Bandurova", size: 22, color: INK })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
  children: [new TextRun({ text: "Oxford Brookes University, School of Engineering, Computing and Mathematics", size: 18, color: "8a8f98" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "Draft, methodology and results. Introduction and related work positioned for supervisor review.", size: 18, italics: true, color: "8a8f98" })] }));

// ---------------- abstract ----------------
children.push(h1("Abstract"));
children.push(P("Automated verification of ICAO/ISO compliance for face images in machine-readable travel documents is typically performed by rule-based analyzers that return hard pass/fail decisions per requirement. Such decisions carry no statistical guarantee, and their false rejections translate into unnecessary photo retakes, while a missed violation is far more costly, so the operational problem is triage under class imbalance and asymmetric error cost. We wrap an existing rule-based analyzer (BioGaze) with a lightweight per-criterion supervised meta-classifier followed by conformal calibration, and study, on the TONO synthetic ICAO benchmark under a strictly subject-disjoint protocol, what distribution-free guarantees a multi-criterion compliance pipeline can and cannot deliver. Our study is empirical and diagnostic rather than a new estimator. We report four findings. First, marginal split-conformal holds its overall coverage target (0.90) but silently under-covers the minority violation class on 6 of the 14 criteria (as low as 0.37 with a 95% confidence interval of 0.33 to 0.41), and label-conditional (Mondrian) conformal restores per-class coverage to about 0.90 with confidence intervals disjoint from the marginal method on exactly those criteria. Second, through a feature ablation we show the coverage guarantee is feature-agnostic, as theory predicts, whereas most of the apparent false-flag reduction is an artifact of the single-violation dataset structure and is not, by itself, evidence for conformal prediction. Third, across six triage strategies the honest picture is a false-flag versus detection trade-off in which recalibration supplies the accuracy and conformal supplies the guarantee, and a correctly randomized adaptive prediction set (APS) is competitive rather than degenerate. Fourth, a 14-criterion image-level decision incurs a multiple-comparison problem: naive joint coverage collapses to about 0.27, and a Bonferroni-corrected image-level guarantee of about 0.91 is attainable only by routing almost every image to retake. We confine claims to what TONO can support: it is single-violation, so the ACCEPT class is empty and cannot be validated, which motivates real multi-attribute data as follow-up."));

// ---------------- 1 introduction ----------------
children.push(h1("1. Introduction"));
children.push(P("This paper asks whether the outputs of an existing rule-based ICAO compliance analyzer can be endowed with a statistical guarantee and turned into an uncertainty-aware accept/reject/retake decision, without modifying or retraining the analyzer. The task is not simply to maximise accuracy. In a document-issuance pipeline the cost of accepting a non-compliant photo greatly exceeds the cost of asking for a retake, and true violations are the minority class, so the right framing is uncertainty-aware triage under class imbalance and asymmetric cost, and the right guarantee is one that holds on the violation class, not merely on average."));
children.push(P("We use conformal prediction, a distribution-free framework that turns any predictor's scores into prediction sets with a user-chosen coverage level, and we contrast the standard marginal formulation with its label-conditional (Mondrian) variant. Because the same synthetic identities recur across violation types in our data, we adopt a strictly subject-disjoint evaluation to prevent identity leakage between calibration and test. We deliberately treat the contribution as empirical and diagnostic: the marginal-under-covers-then-Mondrian-restores phenomenon is a known property of conformal prediction, and our value is in applying and stress-testing it on a structured, multi-decision compliance pipeline, and in reporting honestly what does and does not transfer."));
children.push(new Paragraph({ children: [new TextRun({ text: "Contributions.", bold: true })], spacing: { after: 60 } }));
children.push(bullet("A model-agnostic per-criterion supervised meta-classifier with conformal calibration that endows each of 14 ICAO decisions of a rule-based analyzer with a distribution-free coverage guarantee, evaluated under a subject-disjoint protocol with confidence intervals over ten splits."));
children.push(bullet("Evidence, with disjoint Clopper-Pearson intervals, that marginal conformal under-covers the minority violation class and that label-conditional (Mondrian) conformal restores it, the cost-relevant quantity."));
children.push(bullet("A feature ablation separating what transfers (the distribution-free guarantee) from what does not (accuracy, which here rides a single-violation dataset artifact), and an honest six-method trade-off in which recalibration provides accuracy and conformal provides the guarantee."));
children.push(bullet("An image-level triage analysis that decomposes rejections into correct-target versus wrong-reason, a characterisation of the multiple-comparison problem intrinsic to a 14-criterion decision (a Bonferroni-corrected joint guarantee and its conservatism cost), and an explicit statement of what TONO cannot evaluate."));

// ---------------- 2 related work ----------------
children.push(h1("2. Related work"));
children.push(P("Conformal prediction provides distribution-free, finite-sample coverage for any base predictor under exchangeability (Vovk, Gammerman and Shafer; Angelopoulos and Bates). We use the least-ambiguous-set (LAC) non-conformity score of Sadinle, Lei and Wasserman, and compare against Adaptive Prediction Sets (Romano, Sesia and Candes). Coverage on a minority class is a known weakness of the marginal formulation; the label-conditional (Mondrian) construction of Vovk and the class-conditional treatments of Ding et al. address it by calibrating within each class. Our per-criterion problem is the binary case of this idea, replicated across 14 coupled criteria, which introduces a joint-coverage (multiple-comparison) question that single-decision studies do not face. The accept/reject/retake structure connects to selective prediction and the reject option (Chow; El-Yaniv and Wiener; Geifman and El-Yaniv), and the asymmetric miss/retake cost connects to conformal risk control (Angelopoulos et al.; Bates et al.), which we use only to reason about an operating point rather than to control a risk functional."));
children.push(P("Automated ICAO/ISO compliance verification has been studied primarily as point classification: Ferrara, Franco and Maltoni established conformance checking and the BioLab-ICAO benchmark, and face-image-quality assessment more broadly (Schlett et al.; Terhorst et al.; and systems such as OFIQ and NIST FRVT quality) produces scalar quality or per-criterion verdicts. To our knowledge none attaches a distribution-free per-criterion coverage guarantee, which is the gap we address. We build on the open-source BioGaze analyzer and the TONO synthetic dataset, and we position real multi-attribute validation (for example the DFIC dataset) as follow-up."));
children.push(note("Note. Positioning against the most recent concurrent work is intentionally left for supervisor review; specific 2026 references should be verified before submission."));

// ---------------- 3 method ----------------
children.push(h1("3. Method"));
children.push(...figure(R1 + "pipeline.png", 560, 214, "Figure 1. Model-agnostic conformal layer on a rule-based ICAO analyzer: BioGaze features, per-criterion classifier, split-conformal prediction sets, and an accept/reject/retake decision."));
children.push(h2("3.1 Features and per-criterion meta-classifier"));
children.push(P("For each image, BioGaze produces 14 binary ICAO decisions together with continuous metrics (inter-eye distance, head-pose angles, eye and mouth openness, and a glasses indicator). For each criterion we fit a class-balanced logistic regression to predict the ground-truth label, violation versus compliant, from these outputs. This is a supervised meta-classifier over all BioGaze-derived features, not a probability calibration of a single criterion's own decision, followed by conformal calibration of its scores; the analyzer itself is never retrained. Section 4.1 examines, by ablation, which features actually drive the result, since using all 14 decisions lets the classifier exploit cross-criterion structure specific to the single-violation dataset."));
children.push(h2("3.2 Subject-disjoint evaluation"));
children.push(P("The TONO subject identifier is parsed from each filename. Of 275 identities, 247 appear in more than one violation folder, so a split stratified by violation type would place the same identity in calibration and test and inflate coverage. We therefore partition identities, not images, at 60/20/20 into train, calibration and test, with zero identity overlap (verified). Every headline number is reported as a summary over multiple subject-disjoint splits (five for aggregate comparisons, ten for per-criterion coverage), and we report Clopper-Pearson confidence intervals on pooled counts. Because a subject contributes many correlated images, exchangeability holds at the subject level, so the effective calibration size is the number of calibration subjects (about 55), a point we return to in Section 4.7."));
children.push(h2("3.3 Split conformal: marginal, Mondrian, and APS"));
children.push(P("We apply split conformal prediction with the LAC score s = 1 minus the predicted probability of the true class. On the calibration split we take the exact finite-sample threshold as the k-th order statistic with k = ceil((n+1)(1 minus alpha)), returning the trivial include-all set when calibration is too small to certify. A class is included in an image's set when 1 minus its predicted probability is at most the threshold. The marginal formulation uses one threshold; the label-conditional (Mondrian) formulation estimates a separate threshold per class, which enforces coverage within the minority violation class. We also evaluate a correctly randomized APS. The layer is implemented from scratch for full control and cross-checked against the MAPIE library for the marginal case (agreement within 0.003 coverage)."));
children.push(h2("3.4 Triage strategies and baselines"));
children.push(P("On a common subject-disjoint test set we compare six strategies: the raw BioGaze rule, the logistic classifier alone, a probability threshold with abstention, marginal LAC, Mondrian, and randomized APS. As a calibrated point baseline we also fit isotonic regression on the calibration split and sweep its threshold. For the set-valued methods, a confident singleton violation maps to REJECT, a confident compliant to ACCEPT, and an ambiguous or empty set to RETAKE."));

// ---------------- 4 results ----------------
children.push(h1("4. Results"));

children.push(h2("4.1 The guarantee is feature-agnostic; the accuracy is a dataset artifact"));
children.push(P("Because TONO is single-violation, every per-criterion classifier can exploit the other 13 BioGaze decisions: when another criterion is flagged, this criterion is almost certainly compliant. Table 1 quantifies this. With the full feature set the logistic layer removes 66% of BioGaze's false flags; with only this criterion's own decision plus the continuous metrics the reduction falls to 26%; and with the continuous metrics alone the false-positive rate rises above the raw analyzer. Yet the conformal coverage is essentially unchanged across all three feature sets, exactly as the distribution-free theory predicts. The honest reading, shown in Figure 2, is that validity transfers while accuracy does not: with weak features Mondrian still reaches 0.93 violation coverage, but only by inflating its prediction sets (mean size 1.55, with 55% of sets containing both classes), which is uninformative for triage. The accuracy gains are therefore tied to the single-violation structure and should not be expected to transfer to real multi-violation data, whereas the guarantee should."));
children.push(new Paragraph({ children: [new TextRun({ text: "Table 1. ", bold: true }), new TextRun("Feature ablation (five subject-disjoint seeds). The coverage guarantee is stable across feature sets; the logistic accuracy is not, and weak-feature coverage is held only by ambiguous sets.")], spacing: { after: 80 } }));
children.push(tableFrom(
  ["Feature set", "Logistic FPR", "False-flag redn.", "Mond. viol. cov.", "Mond. set size", "Ambig. rate"],
  [["Full (14 dec. + 7 metrics)", "0.048", "66%", "0.902", "0.98", "0.04"],
   ["Own decision + metrics", "0.107", "26%", "0.906", "1.05", "0.09"],
   ["Metrics only", "0.405", "-182%", "0.935", "1.55", "0.55"]],
  [2750, 1300, 1620, 1540, 1380, 1210]));
children.push(new Paragraph({ text: "", spacing: { after: 100 } }));
children.push(...figure(R3 + "fig_ablation.png", 520, 236, "Figure 2. Left: as features are weakened, logistic accuracy (false-positive rate) collapses while LAC overall coverage and Mondrian violation coverage stay on target. Right: weak features maintain coverage only by inflating set size and ambiguity."));

children.push(h2("4.2 Marginal conformal under-covers the violation class; Mondrian restores it"));
children.push(P("With the exact quantile and zoom images excluded (Section 3), marginal LAC holds its overall target (about 0.90) but under-covers the violation class on 6 of the 14 criteria (5 excluding the small-sample Frontal pose): No posterization 0.37 (95% interval 0.33 to 0.41), No sunglasses 0.68 (0.63 to 0.72), In focus 0.73 (0.69 to 0.76), Head without covering 0.80 (0.76 to 0.83), and No pixelation 0.81 (0.77 to 0.84). Mondrian raises each of these to about 0.89 to 0.92 with confidence intervals disjoint from the marginal method, and does no measurable harm on the criteria where marginal coverage was already adequate (Figure 3, Table 2). Excluding one degenerate small-sample criterion (Frontal pose, about 11 calibration positives), the pooled Mondrian violation coverage over the 13 stable criteria is 0.90. We interpret this as restoring the intended label-conditional coverage property, that is, bringing mean violation coverage to the target, and not as a claim that every criterion exceeds 0.90 on every split (several Mondrian point estimates sit just below 0.90, which is why we report intervals and calibration counts). This is the central result: under class imbalance the guarantee that matters is class-conditional, and marginal conformal does not provide it."));
children.push(...figure(R3 + "fig_percrit_ci.png", 470, 376, "Figure 3. Per-criterion coverage on the violation class with Clopper-Pearson 95% confidence intervals, marginal LAC (grey) versus Mondrian (teal), pooled over ten subject-disjoint seeds. Frontal pose is flagged as small-sample."));
children.push(new Paragraph({ children: [new TextRun({ text: "Table 2. ", bold: true }), new TextRun("Per-criterion violation-class coverage (ten seeds, 95% CI). Marginal LAC under-covers where the classifier is weak; Mondrian restores about 0.90. Point estimates are pooled proportions, consistent with the intervals.")], spacing: { after: 80 } }));
const PC = [
  ["No posterization","0.37","0.33-0.41","0.89","0.86-0.91","59"],
  ["No sunglasses","0.68","0.63-0.72","0.90","0.87-0.92","52"],
  ["In focus","0.73","0.69-0.76","0.89","0.86-0.91","60"],
  ["Frontal pose*","0.74","0.69-0.78","0.92","0.89-0.94","11"],
  ["Head without covering","0.80","0.76-0.83","0.89","0.86-0.91","57"],
  ["No pixelation","0.81","0.77-0.84","0.91","0.88-0.93","60"],
  ["Correct saturation","0.88","0.85-0.90","0.90","0.87-0.92","60"],
  ["Neutral expression","0.88","0.85-0.90","0.90","0.87-0.92","60"],
  ["Uniform face lighting","0.88","0.83-0.91","0.92","0.88-0.95","30"],
  ["Uniform background","0.88","0.85-0.91","0.92","0.89-0.94","51"],
  ["Correct exposure","0.89","0.86-0.91","0.90","0.87-0.92","60"],
  ["Gaze in camera","0.90","0.88-0.92","0.89","0.87-0.91","117"],
  ["No or light makeup","0.91","0.88-0.94","0.89","0.85-0.92","39"],
  ["Eyes open","0.94","0.92-0.95","0.92","0.91-0.94","95"],
];
children.push(tableFrom(
  ["ICAO criterion", "LAC viol.", "LAC 95% CI", "Mond. viol.", "Mond. 95% CI", "Calib. n"],
  PC, [2700, 1200, 1600, 1450, 1600, 1150]));
children.push(new Paragraph({ children: [new TextRun({ text: "* small-sample, reported for completeness only.", italics: true, size: 16, color: "8a8f98" })], spacing: { after: 120 } }));

children.push(h2("4.3 The false-flag versus detection trade-off"));
children.push(P("Table 3 compares the six strategies on a common test set. Two observations are essential for honesty. First, most of the false-flag reduction comes from recalibration, not conformal: a plain logistic layer already lowers the false-positive rate from 0.144 to 0.048 while keeping recall at 0.94. Second, the conformal methods do not beat this on accuracy; when plotted as a risk-coverage curve (Figure 4) the conformal operating points lie at or slightly above the isotonic-calibrated baseline. Conformal's distinct contribution is therefore not accuracy but the distribution-free guarantee, and, through Mondrian, the class-conditional guarantee. Randomized APS is competitive (false-positive rate 0.026 at recall 0.89); we note that the non-randomized APS variant degenerates to trivial two-class sets on this binary problem, which is a property of that variant, not of APS. The recall in Table 3 is measured on a single common test set and is a distinct quantity from the pooled per-criterion violation coverage of Table 2."));
children.push(new Paragraph({ children: [new TextRun({ text: "Table 3. ", bold: true }), new TextRun("Six triage strategies on one consistent protocol (mean plus or minus SD over the same five subject-disjoint splits). FPR and recall are per-criterion; the abstain/RETAKE rate is the fraction of per-criterion decisions that are not confident singletons.")], spacing: { after: 80 } }));
children.push(tableFrom(
  ["Method", "FPR", "Recall (violations)", "Abstain / RETAKE", "Output"],
  [["Raw BioGaze (rules)", "0.144 ± 0.005", "0.955 ± 0.003", "0.00", "point"],
   ["Logistic only", "0.048 ± 0.004", "0.939 ± 0.012", "0.00", "point"],
   ["Threshold + abstention", "0.036 ± 0.004", "0.920 ± 0.016", "0.03", "abstain"],
   ["LAC conformal (marginal)", "0.036 ± 0.004", "0.807 ± 0.016", "0.07", "set"],
   ["Mondrian conformal", "0.040 ± 0.004", "0.878 ± 0.016", "0.10", "set"],
   ["Randomized APS", "0.026 ± 0.004", "0.891 ± 0.017", "0.07", "set"]],
  [2900, 1900, 1900, 1700, 1000]));
children.push(new Paragraph({ text: "", spacing: { after: 100 } }));
children.push(...figure(R3 + "fig_riskcoverage.png", 460, 354, "Figure 4. Risk-coverage view. Conformal operating points lie on the isotonic-calibrated baseline, so conformal matches a calibrated classifier on the accuracy trade-off; its value is the guarantee, not the curve."));

children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h2("4.4 Exploratory image-level triage on TONO"));
children.push(P("Aggregating the Mondrian per-criterion sets into an image decision yields 0 ACCEPT, 958 REJECT and 36 RETAKE on the 994-image test set (Figure 5). Because a naive REJECT count conflates rejecting for the right reason with rejecting for the wrong one, we decompose the outcome on the 925 in-scope violation images by whether the truly-violated criterion is the one flagged. We find 822 correct-target REJECTs (88.9%, the real violation is confidently flagged), 67 wrong-reason REJECTs (7.2%, the image is rejected but on a different criterion, so the real violation is missed even though the photo is still stopped), 36 RETAKEs (3.9%), and 0 ACCEPTs. So the triage not only avoids accepting any true violation, it flags the correct criterion in the large majority of cases; the 7.2% wrong-reason rejections are the honest failure mode to report. We do not over-claim a general safety property: TONO is single-violation, so essentially no genuinely compliant images exist, the ACCEPT class is empty, and image-level specificity (the risk of wrongly rejecting a compliant photo) cannot be tested here. This is a limitation of the dataset, not of the method, and it is the concrete reason a real multi-attribute dataset with a genuine ACCEPT class is needed."));
children.push(...figure(R3 + "fig_imgdec_v3.png", 430, 307, "Figure 5. Image-level triage under Mondrian. No true-violation image is wrongly accepted, and ACCEPT is empty because TONO contains no genuinely compliant images."));
children.push(h2("4.5 Joint coverage and the multiple-comparison problem"));
children.push(P("The per-criterion guarantees are marginal in the criteria. Requiring all 14 prediction sets to contain the truth simultaneously gives only about 0.27 empirical joint coverage, the expected multiple-comparison effect. A Bonferroni correction (per-criterion target 0.993) raises joint coverage to about 0.91 for both marginal and Mondrian conformal, so an image-level guarantee is attainable, but at a conservative cost: almost every image is then routed to RETAKE (Figure 6). We report this openly; a tighter simultaneous guarantee is future work."));
children.push(...figure(R3 + "fig_joint_v3.png", 430, 325, "Figure 6. Naive joint coverage over 14 criteria collapses to about 0.27; a Bonferroni correction restores about 0.91, at the cost of routing nearly all images to RETAKE."));
children.push(h2("4.6 Robustness"));
children.push(P("The conclusions are stable. Marginal overall coverage tracks the nominal target across levels (targets 0.95, 0.90 and 0.80 give empirical 0.951, 0.899 and 0.803), across split ratios (50/25/25 to 70/15/15), and across independent seed ranges. The Mondrian-restores-violation-coverage result holds under all of these. Randomized APS is stable across random draws (recall about 0.89, set size about 1.07)."));
children.push(h2("4.7 Subject-level coverage: an honest caveat"));
children.push(P("Because images cluster within subjects, exchangeability holds only at the subject level, so the effective calibration size is the number of calibration subjects (about 55), not images. We report empirical coverage of 0.918 at the image level and 0.913 averaged per subject, with a per-subject standard deviation of 0.033, and we treat this as an empirical diagnostic. A formal clustered or subject-level conformal guarantee is left as future work."));

// ---------------- 5 explainability ----------------
children.push(h1("5. Auditability"));
children.push(P("Because the underlying decisions are rule-based, each criterion can be accompanied by a faithful region-level explanation from face-parsing segmentation and 68-point landmarks that localises the evidence (for example the lens region for sunglasses or the eyelid geometry for closed eyes). These are exact by construction rather than post-hoc saliency, which makes the triage auditable: a REJECT or RETAKE can be shown to the applicant with the region responsible."));
children.push(...figure(R1 + "xai/xai_panel.png", 300, 716, "Figure 7. Region-level explanations of ICAO compliance decisions on TONO examples, faithful to the rule-based evidence."));

// ---------------- 6 limitations ----------------
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(h1("6. Limitations and discussion"));
children.push(bullet("Recalibration versus conformal. Most of the false-flag reduction is achieved by recalibration alone; conformal's added value is the distribution-free guarantee and, through Mondrian, the class-conditional guarantee, not a further accuracy gain."));
children.push(bullet("Single-violation artifact. TONO's structure lets the classifier exploit cross-criterion signals, so the accuracy numbers are upper bounds tied to the dataset and are not expected to transfer to multi-violation data. The guarantee is feature-agnostic and does transfer."));
children.push(bullet("Empty ACCEPT class. TONO has essentially no genuinely compliant images, so image-level specificity and the ACCEPT decision cannot be validated here."));
children.push(bullet("Subject-level guarantee. We provide an empirical subject-level diagnostic, not a formal clustered-conformal guarantee."));
children.push(bullet("Joint coverage. Per-criterion guarantees do not compose into an image-level guarantee without a conservative multiple-comparison correction."));
children.push(bullet("Small-sample criteria. Coverage estimates for rare criteria (for example Frontal pose) are unstable and are flagged rather than relied upon."));
children.push(P("These limitations motivate a follow-up study on a real, multi-attribute dataset (for example DFIC) with genuinely compliant images: re-running the identical pipeline under a subject-disjoint split would allow the full accept/reject/retake confusion to be evaluated, a cost-calibrated retake region to be tuned against the asymmetric miss and retake costs, and a corrected simultaneous guarantee to be reported."));

// ---------------- 7 conclusion ----------------
children.push(h1("7. Conclusion"));
children.push(P("We showed that a rule-based ICAO compliance analyzer can be wrapped with a per-criterion conformal layer that, under a strictly subject-disjoint protocol, supplies a distribution-free coverage guarantee. The central finding is that marginal conformal silently under-covers the minority violation class and that label-conditional Mondrian conformal restores it, with statistically disjoint confidence intervals on the criteria that matter. A feature ablation separates the transferable guarantee from single-violation accuracy, and a six-method comparison shows recalibration provides the accuracy while conformal provides the guarantee. We characterise the multiple-comparison problem of a 14-criterion decision and confine every claim to what the dataset can support. Future work will validate on real multi-violation data and pursue a cost-calibrated, simultaneously-valid triage."));

// ---------------- references ----------------
children.push(h1("References"));
const refs = [
  "International Civil Aviation Organization. Doc 9303, Machine Readable Travel Documents, 8th ed. ICAO, 2021.",
  "ISO/IEC 39794-5:2019, Information technology, Extensible biometric data interchange formats, Part 5: Face image data. ISO, 2019.",
  "M. Ferrara, A. Franco, D. Maio and D. Maltoni. Face image conformance to ISO/ICAO standards in machine readable travel documents. IEEE Transactions on Information Forensics and Security, 7(4):1204-1213, 2012.",
  "G. Borghi et al. TONO: a synthetic dataset for ICAO compliance of face images. ECCV 2024 Workshops. doi:10.1007/978-3-031-91907-7_1.",
  "BioGaze: an open-source ICAO/ISO face-image compliance analyzer. FG 2025 (demo).",
  "V. Vovk, A. Gammerman and G. Shafer. Algorithmic Learning in a Random World. Springer, 2005 (2nd ed. 2022).",
  "A. N. Angelopoulos and S. Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511, 2021.",
  "M. Sadinle, J. Lei and L. Wasserman. Least ambiguous set-valued classifiers with bounded error levels. Journal of the American Statistical Association, 114(525):223-234, 2019.",
  "Y. Romano, M. Sesia and E. Candes. Classification with valid and adaptive coverage. NeurIPS, 2020.",
  "T. Ding, A. N. Angelopoulos, S. Bates, M. I. Jordan and R. J. Tibshirani. Class-conditional conformal prediction with many classes. NeurIPS, 2023 (arXiv:2306.09335).",
  "A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei and T. Schuster. Conformal risk control. arXiv:2208.02814, 2022.",
  "S. Bates, A. Angelopoulos, L. Lei, J. Malik and M. I. Jordan. Distribution-free, risk-controlling prediction sets. Journal of the ACM, 68(6), 2021.",
  "C. K. Chow. On optimum recognition error and reject tradeoff. IEEE Transactions on Information Theory, 16(1):41-46, 1970.",
  "R. El-Yaniv and Y. Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11:1605-1641, 2010.",
  "Y. Geifman and R. El-Yaniv. SelectiveNet: a deep neural network with an integrated reject option. ICML, 2019.",
  "T. Schlett, C. Rathgeb, O. Henniger, J. Galbally, J. Fierrez and C. Busch. Face image quality assessment: a literature survey. ACM Computing Surveys, 54(10s):1-49, 2022.",
  "P. Terhorst et al. Pixel-level face image quality assessment for explainable face recognition. arXiv:2110.11001, 2021.",
  "V. Taquet, V. Blot, T. Morzadec, L. Lacombe and N. Brunel. MAPIE: an open-source library for distribution-free uncertainty quantification. arXiv:2207.12274, 2022.",
];
refs.forEach((r, i) => children.push(new Paragraph({ children: [new TextRun({ text: `[${i + 1}] ${r}`, size: 18 })], spacing: { after: 60 } })));

const doc = new Document({ creator: "Mustafa Vedat Yildirim",
  title: "Uncertainty-Aware Triage for Rule-Based ICAO Face-Image Compliance",
  sections: [{ properties: { page: { margin: { top: 1000, bottom: 1000, left: 1100, right: 1100 } } }, children }] });
Packer.toBuffer(doc).then(b => { fs.writeFileSync("/home/claude/results_v3/Paper1_v3_draft.docx", b);
  console.log("wrote Paper1_v3_draft.docx", b.length, "bytes"); });
