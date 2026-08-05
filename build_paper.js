const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, PageBreak
} = require('docx');

const R = "/home/claude/results/";
const INK = "1f2933", TEAL = "0f9d8f";

const h1 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 120 } });
const h2 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 90 } });
const P = t => new Paragraph({ children: [new TextRun(t)], spacing: { after: 140 }, alignment: AlignmentType.JUSTIFIED });
const bullet = t => new Paragraph({ text: t, bullet: { level: 0 }, spacing: { after: 50 }, alignment: AlignmentType.JUSTIFIED });
const note = t => new Paragraph({ children: [new TextRun({ text: t, italics: true, color: "8a8f98" })], spacing: { after: 140 } });

function figure(path, wpx, caption) {
  const buf = fs.readFileSync(path);
  // preserve aspect from known sizes handled by caller (wpx,hpx)
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx.w, height: wpx.h } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: INK })] }),
  ];
}

// ---- Coverage table data ----
const COV = [
  ["Head uncovered", "0.920", "0.965"], ["Eyes open", "0.920", "0.979"],
  ["No sunglasses", "0.921", "0.735"], ["No posterization", "0.909", "0.452"],
  ["Gaze in camera", "0.912", "0.886"], ["Neutral expression", "0.906", "0.839"],
  ["In focus", "0.890", "0.629"], ["Correct exposure", "0.906", "0.935"],
  ["No/light makeup", "0.890", "0.923"], ["No pixelation", "0.882", "0.629"],
  ["Frontal pose", "0.880", "0.966"], ["Correct saturation", "0.890", "0.857"],
  ["Uniform background", "0.922", "0.887"], ["Uniform face lighting", "0.886", "0.833"],
];
function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.w, type: WidthType.DXA },
    shading: opts.head ? { type: ShadingType.CLEAR, fill: "0f9d8f" } : undefined,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!opts.head, color: opts.head ? "ffffff" : INK, size: 18 })],
    })],
  });
}
const COLW = [3200, 1800, 2200];
function covTable() {
  const rows = [new TableRow({
    tableHeader: true,
    children: [cell("ICAO criterion", { w: COLW[0], head: true }),
      cell("Coverage", { w: COLW[1], head: true, center: true }),
      cell("Coverage | violations", { w: COLW[2], head: true, center: true })],
  })];
  for (const r of COV) rows.push(new TableRow({
    children: [cell(r[0], { w: COLW[0] }), cell(r[1], { w: COLW[1], center: true }), cell(r[2], { w: COLW[2], center: true })],
  }));
  return new Table({ columnWidths: COLW, width: { size: COLW.reduce((a, b) => a + b), type: WidthType.DXA }, rows });
}

const children = [];
// Title block
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Conformal Prediction for Reliable ICAO Face-Image Compliance Verification:", bold: true, size: 30, color: INK })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: "Coverage Guarantees and Retake Reduction", bold: true, size: 30, color: INK })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "Mustafa Vedat Yıldırım · Inna Skarga-Bandurova", size: 22, color: INK })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
  children: [new TextRun({ text: "Oxford Brookes University — School of Engineering, Computing and Mathematics", size: 18, color: "8a8f98" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "Draft — methodology & results (introduction / related work to be expanded)", size: 18, italics: true, color: "8a8f98" })] }));

// Abstract
children.push(h1("Abstract"));
children.push(P("Automated verification of ICAO/ISO compliance for face images in machine-readable travel documents is typically performed by rule-based analyzers that return hard pass/fail decisions per requirement. Such decisions carry no statistical guarantee, and their false rejections translate directly into unnecessary photo retakes. We propose a lightweight calibration layer that wraps an existing rule-based analyzer (BioGaze) with split conformal prediction, attaching a distribution-free coverage guarantee to each per-criterion compliance decision and yielding an ACCEPT / REJECT / RETAKE framework. Using the TONO synthetic dataset (4,561 valid images, single-violation partition providing per-criterion ground truth), we show that empirical coverage holds across all 14 ICAO criteria, averaging 0.9024 at a target of 0.90. Relative to the raw rule-based analyzer, the conformal layer reduces unnecessary per-criterion rejections by 76.3% (1,748 to 414) while preserving the coverage guarantee; true violations about which the layer is uncertain are routed to RETAKE rather than silently accepted. We complement the decisions with region-level explanations aligned with pixel-level face-image-quality interpretability. The approach adds statistical rigour and interpretability to any rule-based compliance tool without retraining it."));

// 1 Introduction (skeleton)
children.push(h1("1. Introduction"));
children.push(P("Face images for passports and identity documents must satisfy strict photographic and subject requirements defined by ICAO Doc 9303 and the ISO/IEC 39794-5 / 19794-5 standards. Automated compliance checkers reduce manual effort at enrolment, but they generally output deterministic pass/fail verdicts per requirement, with no notion of confidence. In deployment this has a concrete cost: whenever the checker wrongly flags a compliant photo, the applicant is asked to retake it — an unnecessary retake that wastes time at the counter and degrades user experience."));
children.push(P("This paper asks whether the outputs of an existing rule-based compliance analyzer can be endowed with a statistical guarantee and turned into a more economical accept/reject/retake decision, without modifying or retraining the analyzer. We answer affirmatively using conformal prediction, a distribution-free framework that converts any predictor's scores into prediction sets with a user-chosen coverage level."));
children.push(note("[To be expanded with supervisor: motivation, positioning, and a sharper statement of the gap.]"));
children.push(new Paragraph({ children: [new TextRun({ text: "Contributions.", bold: true })], spacing: { after: 60 } }));
children.push(bullet("A model-agnostic conformal calibration layer that wraps a rule-based ICAO analyzer (BioGaze) and gives each of 14 per-criterion decisions a 90% coverage guarantee."));
children.push(bullet("An ACCEPT / REJECT / RETAKE decision framework derived from the conformal prediction sets, which cuts unnecessary per-criterion rejections by 76.3% while preserving the guarantee."));
children.push(bullet("An evaluation protocol on the TONO synthetic dataset whose single-violation structure yields clean per-criterion ground truth, together with region-level explanations of the decisions."));

// 2 Related work (skeleton)
children.push(h1("2. Related Work"));
children.push(P("Face Image Quality Assessment (FIQA) and ICAO/ISO compliance checking. Explainable and pixel-level FIQA. Conformal prediction and uncertainty quantification in computer vision. Rule-based compliance tools (BioLab-ICAO, BioGaze) and synthetic benchmarks (TONO)."));
children.push(note("[To be expanded with supervisor into full prose with citations.]"));

// 3 Method
children.push(h1("3. Method"));
children.push(P("Figure 1 gives an overview: BioGaze analyses the input image and emits per-criterion decisions and continuous metrics; these form a feature vector on which a per-criterion split-conformal predictor is calibrated to produce guaranteed prediction sets, which are mapped to an ACCEPT / REJECT / RETAKE decision. The analyzer itself is treated as a black box and is not retrained."));
children.push(...figure(R + "pipeline.png", { w: 520, h: 198 }, "Figure 1. Model-agnostic conformal layer wrapping a rule-based ICAO compliance analyzer."));
children.push(h2("3.1 Rule-based compliance analyzer"));
children.push(P("We use BioGaze, an open-source ISO/ICAO face-image analyzer that combines segmentation, facial-landmark, head-pose, gaze and expression sub-models with computer-vision measurements. For each input image it returns a binary decision for 14 ICAO criteria (e.g. head uncovered, eyes open, no sunglasses, neutral expression, frontal pose, correct exposure/saturation, uniform background and face lighting, absence of pixelation/posterization) plus continuous metrics (head-pose angles, inter-eye distance, eye/mouth openness). We treat BioGaze strictly as a black box and do not retrain it."));
children.push(h2("3.2 Dataset and per-criterion ground truth"));
children.push(P("We evaluate on the TONO synthetic dataset (single-violation partition), which contains face images organised into folders by the single ICAO requirement each image violates. This folder structure provides exact per-criterion ground truth: an image drawn from the folder for criterion c violates c and is compliant on every other criterion. Running BioGaze over the partition yields 4,561 images with a valid single-face analysis, which we use for all experiments. Because the dataset is synthetic and single-violation, it isolates each requirement cleanly; validation on real, multi-violation data is left to future work."));
children.push(h2("3.3 Conformal calibration layer"));
children.push(P("For each criterion c we form a feature vector from BioGaze's outputs — its 14 binary decisions together with the continuous metrics — and fit a probabilistic classifier to predict the ground-truth label (violation vs. compliant) for c. We then apply split (inductive) conformal prediction with the LAC (least-ambiguous-set) non-conformity score. The data are partitioned once into training (60%), calibration (20%) and test (20%) sets, stratified by violation type; the classifier is fit on the training split, the conformal threshold is calibrated on the calibration split, and coverage is measured on the held-out test split. For a target error rate α = 0.10 the procedure returns, per image and per criterion, a prediction set over {compliant, violation} whose marginal coverage is guaranteed to be at least 1 − α = 0.90, independently of the (rule-based) analyzer's internal behaviour — this is the distribution-free property of conformal prediction."));
children.push(h2("3.4 ACCEPT / REJECT / RETAKE decision framework"));
children.push(P("The per-criterion prediction sets are mapped to an image-level decision. A criterion whose set is the singleton {violation} is a confident failure; a set that still contains {compliant} is treated as non-confident on that criterion. An image is REJECTED if any criterion is a confident failure, ACCEPTED if every criterion is confidently compliant, and otherwise sent to RETAKE. Crucially, a true violation about which the layer is uncertain is routed to RETAKE — it is never silently accepted."));
children.push(h2("3.5 Region-level explanations"));
children.push(P("Following the pixel-level interpretability line of FIQA research, and to remain faithful to the rule-based nature of the analyzer, we explain each decision by the facial region on which it is measured rather than by gradient-based saliency over a single end-to-end network. Segmentation-driven criteria (background, head covering, glasses/sunglasses, illumination, saturation) are explained by the analyzer's face-parsing map; landmark-driven criteria (eye openness, expression, pose, makeup) by the detected 68-point landmark geometry."));

// 4 Results
children.push(h1("4. Results"));
children.push(h2("4.1 Coverage guarantee"));
children.push(P("Table 1 and Figure 2 report empirical coverage on the test split. Across all 14 criteria the coverage clusters tightly around the 0.90 target, with mean 0.9024 (range 0.880–0.922). The per-criterion scatter above and below 0.90 is the expected finite-sample behaviour of marginal conformal prediction: individual criteria fluctuate around the target while their mean matches it. This confirms that the distribution-free guarantee transfers to the rule-based analyzer's decisions in practice."));
children.push(...figure(R + "coverage_chart.png", { w: 500, h: 289 }, "Figure 2. Empirical per-criterion coverage vs. the 90% target guarantee (TONO, 4,561 images)."));
children.push(new Paragraph({ children: [new TextRun({ text: "Table 1. ", bold: true }), new TextRun("Per-criterion empirical coverage (target 0.90) and coverage conditional on true violations.")], spacing: { after: 80 } }));
children.push(covTable());
children.push(P("Coverage conditional on the (rare) violation class is lower than the marginal value for several criteria (e.g. posterization, pixelation, in-focus), a direct consequence of class imbalance under marginal conformal prediction; we return to this in the discussion."));
children.push(P("Beyond the 90% operating point, the guarantee holds across confidence levels: sweeping the nominal target from 0.80 to 0.98 yields mean empirical coverage of 0.802, 0.852, 0.903, 0.954 and 0.982 respectively, tracking the nominal line almost exactly (Figure 3)."));
children.push(...figure(R + "calibration.png", { w: 340, h: 329 }, "Figure 3. Mean empirical coverage vs. nominal target across confidence levels; shaded band is the per-criterion min-max."));
children.push(h2("4.2 Reduction of unnecessary retakes"));
children.push(P("We compare the raw rule-based analyzer against the conformal layer in terms of unnecessary rejections — per-criterion false flags, i.e. cases where a criterion is reported as violated although the ground truth is compliant. On the test split BioGaze raises 1,748 such false flags; the conformal layer, which only rejects on a confident singleton {violation}, raises 414 — a reduction of 76.3% (Figure 4). Detection of true violations remains high: 823 of 850 true per-criterion violations are confidently flagged by BioGaze versus 694 by the conformal layer, with the difference routed to RETAKE rather than accepted. At the image level the framework yields 7 ACCEPT, 797 REJECT and 109 RETAKE decisions on the 913-image test split."));
children.push(...figure(R + "retake_chart.png", { w: 500, h: 326 }, "Figure 4. Per-criterion unnecessary rejections: rule-based analyzer vs. conformal layer (1,748 → 414, −76.3%)."));
children.push(h2("4.3 Explainability"));
children.push(P("Figure 5 shows region-level explanations for five representative violation types. Segmentation maps localise background, head-covering and eyewear evidence, while landmark geometry localises eye openness and expression. The explanations are faithful to the measurements that actually drive each decision."));
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(...figure(R + "xai/xai_panel.png", { w: 340, h: 685 }, "Figure 5. Region-level explanations of ICAO compliance decisions for five TONO violation types."));

// 5 Discussion
children.push(h1("5. Discussion and Limitations"));
children.push(bullet("Marginal vs. conditional coverage. The 90% guarantee is marginal; conditional coverage on the minority violation class is lower for some criteria. Class-conditional (Mondrian) conformal prediction is a natural remedy and is planned as follow-up work."));
children.push(bullet("Operational definition of a retake. The value of the RETAKE class depends on how retakes are handled operationally (human review vs. automatic recapture); the appropriate operating point is an application decision to be set with domain stakeholders."));
children.push(bullet("Synthetic, single-violation data. TONO isolates each requirement but is synthetic and single-violation; BioGaze was also tuned on TONO. The conformal guarantee is distribution-free and therefore valid regardless, but external validity on real, multi-violation images remains to be shown."));

// 6 Conclusion
children.push(h1("6. Conclusion and Future Work"));
children.push(P("We showed that split conformal prediction can wrap an existing rule-based ICAO compliance analyzer to provide per-criterion coverage guarantees (mean 0.9024 at a 0.90 target across 14 criteria) and to cut unnecessary per-criterion rejections by 76.3% within an ACCEPT/REJECT/RETAKE framework, complemented by faithful region-level explanations. Future work will (i) adopt class-conditional conformal prediction to strengthen violation-class coverage, and (ii) validate the approach on a real, multi-violation face-image dataset (e.g. DFIC) in a follow-up study."));

// References
children.push(h1("References"));
const refs = [
  "International Civil Aviation Organization. Doc 9303 — Machine Readable Travel Documents, 8th edition. ICAO, 2021.",
  "ISO/IEC 39794-5:2019. Information technology — Extensible biometric data interchange formats — Part 5: Face image data. ISO, 2019.",
  "M. Ferrara, A. Franco, D. Maltoni. The magic passport: face image conformance to ISO/ICAO standards in machine-readable travel documents. IEEE Transactions on Information Forensics and Security, 2012.",
  "M. Ferrara, A. Franco, D. Maltoni, Y. Sun. BioLab-ICAO: a new benchmark to evaluate applications assessing face image compliance to the ISO/IEC 19794-5 standard. IEEE International Conference on Image Processing (ICIP), 2009.",
  "G. Borghi et al. BioGaze: a framework for evaluating the photographic requirements of the ISO/IEC 39794-5 standard. IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2025.",
  "G. Borghi et al. TONO: a synthetic dataset for face image compliance to the ISO/ICAO standard. In European Conference on Computer Vision (ECCV) Workshops, LNCS, Springer, 2024. doi:10.1007/978-3-031-91907-7_1.",
  "T. Schlett, C. Rathgeb, O. Henniger, J. Fierrez, C. Busch. Face image quality assessment: a literature survey. ACM Computing Surveys, 54(10s):1-49, 2022. doi:10.1145/3507901.",
  "P. Terhörst, M. Ihlefeld, M. Huber, N. Damer, F. Kirchbuchner, K. Raja, A. Kuijper. Pixel-level face image quality assessment for explainable face recognition. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2023 (arXiv:2110.11001).",
  "V. Vovk, A. Gammerman, G. Shafer. Algorithmic Learning in a Random World. Springer, 2005.",
  "A. N. Angelopoulos, S. Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511, 2021.",
  "V. Taquet, V. Blot, T. Morzadec, L. Lacombe, N. Brunel. MAPIE: an open-source library for distribution-free uncertainty quantification. arXiv:2207.12274, 2022.",
];
refs.forEach((r, i) => children.push(new Paragraph({ children: [new TextRun({ text: `[${i + 1}] ${r}`, size: 18 })], spacing: { after: 60 } })));
children.push(note("[Reference list to be finalised with exact venues, authors and years during submission preparation.]"));

const doc = new Document({
  creator: "Mustafa Vedat Yıldırım",
  title: "Conformal Prediction for Reliable ICAO Face-Image Compliance Verification",
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/results/Paper1_draft.docx", buf);
  console.log("wrote Paper1_draft.docx", buf.length, "bytes");
});
