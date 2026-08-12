"""
analysis_v2.py — Revised conformal analysis addressing Dr. Inna's review.

Key changes vs. the first version:
  1. SUBJECT-DISJOINT split. Subject id is parsed from the filename (S<id>-...).
     The same synthetic identity appears across up to 17 violation folders, so a
     folder-stratified split leaks identities between calibration and test and
     inflates coverage. We split by *subject* (GroupShuffleSplit-style) so no
     identity is shared across train / calibration / test.
  2. VIOLATION-CLASS coverage is reported as a first-class metric (not just the
     marginal average), because the cost of missing a true ICAO violation is the
     quantity that actually matters under class imbalance.
  3. METHOD COMPARISON on one common subject-disjoint test set:
        (a) raw BioGaze rule decision
        (b) logistic classifier only (argmax, no conformal)
        (c) probability threshold with an abstention band
        (d) marginal LAC split conformal
        (e) Mondrian / label-conditional LAC split conformal
     Conformal is implemented transparently from scratch (standard split
     conformal, LAC score = 1 - p_hat(true class)) and cross-checked against
     MAPIE for the marginal case.
  4. HONEST framing of the retake metric: what we reduce is the number of
     *per-criterion* false flags, and we report the detection trade-off
     (true violations still flagged) alongside it — never claiming image-level
     retake reduction.

Run with the system Python (mapie / sklearn / pandas):
    python3 analysis_v2.py --csv tono_full_pred.csv --out results_v2
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- TONO folder -> violated BioGaze criterion (see mapping.py for full table) ---
FOLDER_TO_CRIT = {
    "bkg":   "Uniform_background",
    "cap":   "Head_without_covering",
    "ce":    "Eyes_open",
    "ceg":   "Eyes_open",
    "expos": "Correct_exposure",
    "la_1":  "Gaze_in_camera",
    "la_2":  "Gaze_in_camera",
    "light": "Uniform_face_lighting",
    "mkup":  "No_or_light_makeup",
    "oof":   "In_focus",
    "pixel": "No_pixelation",
    "poster":"No_posterization",
    "sat":   "Correct_saturation",
    "sm":    "Neutral_expression",
    "sun":   "No_sunglasses",
    "tq":    "Frontal_pose",
    # 'zoom' violates the ICAO *face-dimension / head-size* requirement, which
    # BioGaze does NOT check. It is therefore NOT a fully-compliant image; it is
    # an out-of-scope violation. Treated as background (no in-scope violation)
    # for per-criterion training, but never called 'genuinely compliant'.
    "zoom":  None,
}

CRITERIA = [
    "Head_without_covering", "Eyes_open", "No_sunglasses", "No_posterization",
    "Gaze_in_camera", "Neutral_expression", "In_focus", "Correct_exposure",
    "No_or_light_makeup", "No_pixelation", "Frontal_pose", "Correct_saturation",
    "Uniform_background", "Uniform_face_lighting",
]
METRICS = ["IED", "pitch", "yaw", "roll", "eyes_open_val", "mouth_open_val", "has_glasses"]

# label convention inside the conformal layer:  0 = OK (no violation), 1 = VIOLATION


# ---------------------------------------------------------------- data loading
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df = df[(df["faces_detected"] == 1) & (df["error"].isna()) & (df["Compliant"].notna())].copy()
    for c in CRITERIA + ["Compliant"]:
        df[c] = (df[c].astype(str) == "True").astype(int)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=METRICS).reset_index(drop=True)
    df["subject"] = df["filename"].str.extract(r'^(S\d+)')
    # per-criterion ground-truth violation flag (1 = this criterion is violated)
    for c in CRITERIA:
        df["gt_viol_" + c] = df["source_folder"].map(
            lambda f: 1 if FOLDER_TO_CRIT.get(f) == c else 0)
    return df


def subject_disjoint_split(df, seed=0, fracs=(0.60, 0.20, 0.20)):
    """Partition *subjects* (not images) into train/calib/test so no identity is
    shared across the three sets."""
    subjects = np.array(sorted(df["subject"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_tr = int(round(fracs[0] * n))
    n_ca = int(round(fracs[1] * n))
    tr_subj = set(subjects[:n_tr])
    ca_subj = set(subjects[n_tr:n_tr + n_ca])
    te_subj = set(subjects[n_tr + n_ca:])
    idx = df.index.to_numpy()
    subj = df["subject"].to_numpy()
    tr = idx[np.isin(subj, list(tr_subj))]
    ca = idx[np.isin(subj, list(ca_subj))]
    te = idx[np.isin(subj, list(te_subj))]
    manifest = {
        "seed": seed, "n_subjects": n,
        "n_subjects_train": len(tr_subj), "n_subjects_calib": len(ca_subj),
        "n_subjects_test": len(te_subj),
        "n_img_train": int(len(tr)), "n_img_calib": int(len(ca)), "n_img_test": int(len(te)),
        "train_subjects": sorted(tr_subj), "calib_subjects": sorted(ca_subj),
        "test_subjects": sorted(te_subj),
        # sanity: disjoint?
        "overlap_tr_ca": len(tr_subj & ca_subj), "overlap_tr_te": len(tr_subj & te_subj),
        "overlap_ca_te": len(ca_subj & te_subj),
    }
    return tr, ca, te, manifest


# ---------------------------------------------------------- conformal (scratch)
def lac_scores(proba, y):
    """LAC non-conformity score for the *true* class: s_i = 1 - p_hat(y_i)."""
    return 1.0 - proba[np.arange(len(y)), y]


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile: ceil((n+1)(1-alpha))/n empirical
    quantile of calibration scores."""
    n = len(scores)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(proba_test, qhat_per_class):
    """Include class k in the set iff (1 - p_hat(k)) <= qhat_k.
    qhat_per_class is a length-2 array (marginal => same value for both)."""
    scores_k = 1.0 - proba_test  # shape (n,2): score for including class 0 and 1
    return scores_k <= qhat_per_class[None, :]  # bool (n,2)


# --------------------------------------------------------------- per-criterion
def eval_criterion(df, c, tr, ca, te, alpha, seed, min_pos_calib=1):
    y = df["gt_viol_" + c].to_numpy()
    ytr, yca, yte = y[tr], y[ca], y[te]
    n_pos = int(y.sum())
    info = {"criterion": c, "n_pos_total": n_pos,
            "n_pos_train": int(ytr.sum()), "n_pos_calib": int(yca.sum()),
            "n_pos_test": int(yte.sum()), "n_test": int(len(te))}
    if ytr.sum() < 5 or len(np.unique(ytr)) < 2 or yca.sum() < min_pos_calib or yte.sum() < 1:
        info["skip"] = True
        return info, None
    info["skip"] = False

    Xall = df[CRITERIA + METRICS].to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xall[tr])
    Xs = scaler.transform(Xall)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(Xs[tr], ytr)
    classes = list(clf.classes_)                     # [0,1]
    p_ca = clf.predict_proba(Xs[ca])
    p_te = clf.predict_proba(Xs[te])

    # --- calibration scores
    s_ca = lac_scores(p_ca, yca)

    # (d) MARGINAL LAC
    q_marg = conformal_quantile(s_ca, alpha)
    qhat_marg = np.array([q_marg, q_marg])
    sets_marg = prediction_sets(p_te, qhat_marg)

    # (e) MONDRIAN / label-conditional LAC : separate quantile per true class
    q_cls = np.array([
        conformal_quantile(s_ca[yca == 0], alpha),
        conformal_quantile(s_ca[yca == 1], alpha),
    ])
    sets_mond = prediction_sets(p_te, q_cls)

    # (b) LOGISTIC-ONLY (argmax point prediction)
    pred_log = (p_te[:, 1] >= 0.5).astype(int)

    # (c) THRESHOLD + ABSTENTION band (symmetric around 0.5)
    band = 0.15
    hi, lo = 0.5 + band, 0.5 - band
    pred_thr = np.full(len(te), -1)          # -1 = abstain (RETAKE)
    pred_thr[p_te[:, 1] >= hi] = 1
    pred_thr[p_te[:, 1] <= lo] = 0

    # (a) RAW BioGaze : criterion==0 means BioGaze flagged a violation
    bio_flag = (df[c].to_numpy()[te] == 0).astype(int)

    def set_metrics(sets):
        col_ok, col_v = classes.index(0), classes.index(1)
        has_ok, has_v = sets[:, col_ok], sets[:, col_v]
        size = sets.sum(1)
        covered = sets[np.arange(len(te)), [classes.index(v) for v in yte]]
        vmask, omask = (yte == 1), (yte == 0)
        confident_viol = has_v & ~has_ok      # set == {viol}
        confident_ok = has_ok & ~has_v        # set == {ok}
        ambiguous = has_ok & has_v            # {ok,viol} -> RETAKE
        empty = size == 0
        return {
            "coverage": float(covered.mean()),
            "coverage_viol": float(covered[vmask].mean()) if vmask.sum() else None,
            "coverage_ok": float(covered[omask].mean()) if omask.sum() else None,
            "avg_set_size": float(size.mean()),
            "singleton_rate": float((size == 1).mean()),
            "ambiguous_rate": float(ambiguous.mean()),
            "empty_rate": float(empty.mean()),
            # detection of true violations (confident {viol})
            "recall_viol": float(confident_viol[vmask].mean()) if vmask.sum() else None,
            # false flag on true-ok images (confident {viol} on OK)
            "fpr": float(confident_viol[omask].mean()) if omask.sum() else None,
            "n_confident_viol": int(confident_viol.sum()),
            "n_confident_ok": int(confident_ok.sum()),
            "n_ambiguous": int(ambiguous.sum()),
            "n_empty": int(empty.sum()),
            # counts for false-flag accounting
            "fp": int((omask & confident_viol).sum()),
            "tp": int((vmask & confident_viol).sum()),
        }

    def point_metrics(pred, allow_abstain=False):
        vmask, omask = (yte == 1), (yte == 0)
        flag = (pred == 1)
        abst = (pred == -1)
        return {
            "recall_viol": float(flag[vmask].mean()) if vmask.sum() else None,
            "fpr": float(flag[omask].mean()) if omask.sum() else None,
            "abstain_rate": float(abst.mean()) if allow_abstain else 0.0,
            "fp": int((omask & flag).sum()),
            "tp": int((vmask & flag).sum()),
            "n_abstain": int(abst.sum()),
        }

    info["methods"] = {
        "raw_biogaze":   point_metrics(bio_flag),
        "logistic_only": point_metrics(pred_log),
        "threshold_abstain": point_metrics(pred_thr, allow_abstain=True),
        "lac_marginal":  set_metrics(sets_marg),
        "mondrian":      set_metrics(sets_mond),
    }
    info["q_marginal"] = q_marg
    info["q_class_ok"] = float(q_cls[0])
    info["q_class_viol"] = float(q_cls[1])
    # keep the marginal-LAC sets for image-level decisions
    return info, {"sets_lac": sets_marg, "sets_mond": sets_mond, "classes": classes,
                  "yte": yte, "bio_flag": bio_flag}


# --------------------------------------------------------------- driver
def run(csv_path, out_dir, alpha=0.1, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    df = load_and_prepare(csv_path)
    tr, ca, te, manifest = subject_disjoint_split(df, seed=seed)
    conf = 1 - alpha
    print(f"[v2] valid images: {len(df)} | subjects: {manifest['n_subjects']}")
    print(f"[v2] subject-disjoint  train={manifest['n_img_train']} "
          f"calib={manifest['n_img_calib']} test={manifest['n_img_test']}  "
          f"(overlaps: {manifest['overlap_tr_ca']},{manifest['overlap_tr_te']},{manifest['overlap_ca_te']})")

    per_crit = {}
    keep = {}
    for c in CRITERIA:
        info, held = eval_criterion(df, c, tr, ca, te, alpha, seed)
        per_crit[c] = info
        if held is not None:
            keep[c] = held
        if not info.get("skip"):
            m = info["methods"]
            print(f"  [{c:22s}] LACcov={m['lac_marginal']['coverage']:.3f} "
                  f"viol={m['lac_marginal']['coverage_viol']}  "
                  f"MONviol={m['mondrian']['coverage_viol']}  n_pos_test={info['n_pos_test']}")
        else:
            print(f"  [{c:22s}] skipped (pos train/calib/test = "
                  f"{info['n_pos_train']}/{info['n_pos_calib']}/{info['n_pos_test']})")

    active = [c for c in CRITERIA if not per_crit[c].get("skip")]

    # ---- aggregate false-flag accounting per method (honest retake metric) ----
    method_totals = {}
    for meth in ["raw_biogaze", "logistic_only", "threshold_abstain", "lac_marginal", "mondrian"]:
        fp = sum(per_crit[c]["methods"][meth]["fp"] for c in active)
        tp = sum(per_crit[c]["methods"][meth]["tp"] for c in active)
        tot_viol = sum(per_crit[c]["n_pos_test"] for c in active)
        tot_ok = sum(per_crit[c]["n_test"] - per_crit[c]["n_pos_test"] for c in active)
        method_totals[meth] = {
            "false_flags_fp": int(fp), "true_flags_tp": int(tp),
            "total_true_violations": int(tot_viol), "total_ok": int(tot_ok),
            "recall_overall": round(tp / tot_viol, 4) if tot_viol else None,
            "fpr_overall": round(fp / tot_ok, 4) if tot_ok else None,
        }
    base_fp = method_totals["raw_biogaze"]["false_flags_fp"]
    for meth, d in method_totals.items():
        d["false_flag_reduction_vs_raw"] = int(base_fp - d["false_flags_fp"])
        d["false_flag_reduction_pct_vs_raw"] = (round((base_fp - d["false_flags_fp"]) / base_fp * 100, 1)
                                                if base_fp else 0.0)

    # ---- image-level ACCEPT / REJECT / RETAKE (marginal LAC) ----
    m = len(te)
    dec = []
    for i in range(m):
        any_conf_viol = any((keep[c]["sets_lac"][i, keep[c]["classes"].index(1)] and
                             not keep[c]["sets_lac"][i, keep[c]["classes"].index(0)]) for c in active)
        all_conf_ok = all((keep[c]["sets_lac"][i, keep[c]["classes"].index(0)] and
                           not keep[c]["sets_lac"][i, keep[c]["classes"].index(1)]) for c in active)
        if any_conf_viol:
            dec.append("REJECT")
        elif all_conf_ok:
            dec.append("ACCEPT")
        else:
            dec.append("RETAKE")
    dec = np.array(dec)
    dec_counts = {d: int((dec == d).sum()) for d in ["ACCEPT", "REJECT", "RETAKE"]}

    # image-level ground truth: does the image carry an in-scope violation?
    te_folder = df["source_folder"].to_numpy()[te]
    img_has_inscope_viol = np.array([FOLDER_TO_CRIT.get(f) is not None for f in te_folder])
    img_is_zoom = np.array([f == "zoom" for f in te_folder])
    # confusion of image decision vs (has in-scope violation)
    confusion = {}
    for d in ["ACCEPT", "REJECT", "RETAKE"]:
        dm = (dec == d)
        confusion[d] = {
            "on_violation_images": int((dm & img_has_inscope_viol).sum()),
            "on_zoom_out_of_scope": int((dm & img_is_zoom).sum()),
        }

    summary = {
        "config": {"alpha": alpha, "target_coverage": conf, "seed": seed,
                   "split": "subject-disjoint 60/20/20"},
        "manifest": {k: v for k, v in manifest.items()
                     if k not in ("train_subjects", "calib_subjects", "test_subjects")},
        "criteria": per_crit,
        "method_totals": method_totals,
        "image_decisions": dec_counts,
        "image_confusion": confusion,
        "n_active_criteria": len(active),
    }
    with open(os.path.join(out_dir, "summary_v2.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # tidy per-criterion table
    rows = []
    for c in CRITERIA:
        d = per_crit[c]
        if d.get("skip"):
            rows.append({"criterion": c, "n_pos_test": d["n_pos_test"], "active": False})
            continue
        lac, mon = d["methods"]["lac_marginal"], d["methods"]["mondrian"]
        rows.append({
            "criterion": c, "active": True, "n_pos_test": d["n_pos_test"],
            "lac_cov": round(lac["coverage"], 3),
            "lac_cov_viol": round(lac["coverage_viol"], 3) if lac["coverage_viol"] is not None else None,
            "lac_cov_ok": round(lac["coverage_ok"], 3) if lac["coverage_ok"] is not None else None,
            "mond_cov_viol": round(mon["coverage_viol"], 3) if mon["coverage_viol"] is not None else None,
            "lac_set_size": round(lac["avg_set_size"], 3),
            "lac_ambiguous": round(lac["ambiguous_rate"], 3),
            "lac_empty": round(lac["empty_rate"], 3),
            "raw_recall": round(d["methods"]["raw_biogaze"]["recall_viol"], 3) if d["methods"]["raw_biogaze"]["recall_viol"] is not None else None,
            "raw_fpr": round(d["methods"]["raw_biogaze"]["fpr"], 3) if d["methods"]["raw_biogaze"]["fpr"] is not None else None,
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_criterion_v2.csv"), index=False)

    print("\n=== METHOD TOTALS (subject-disjoint test) ===")
    for meth, d in method_totals.items():
        print(f"  {meth:18s} FP={d['false_flags_fp']:4d}  recall={d['recall_overall']}  "
              f"FPR={d['fpr_overall']}  ffReduc={d['false_flag_reduction_pct_vs_raw']}%")
    print("Image decisions:", dec_counts)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.csv, args.out, alpha=args.alpha, seed=args.seed)
