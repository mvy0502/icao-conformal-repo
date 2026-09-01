"""
strengthen.py: additional self-contained experiments while awaiting supervisor input.

Adds, on the same subject-disjoint protocol as analysis_v2:
  (1) APS (Adaptive Prediction Sets) as a sixth triage method (from scratch, since
      MAPIE's binary API is LAC-only). Non-randomized APS (Romano et al., 2020).
  (2) Bonferroni-corrected joint coverage: run each criterion at alpha/m and measure
      whether all 14 sets simultaneously cover (image-level guarantee), with its cost.
  (3) Mondrian image-level ACCEPT/REJECT/RETAKE decisions + safety.
  (4) Alpha cost-curve: coverage / violation-coverage / FPR / recall vs confidence level.

Run: python3 strengthen.py
"""
import json, warnings
import os, sys
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import analysis_v2 as A

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "tono_full_pred.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "results_v2")
M = len(A.CRITERIA)  # 14


# ---------------- APS (non-randomized) ----------------
def aps_scores_calib(proba, y):
    n = len(y)
    order = np.argsort(-proba, axis=1)
    scores = np.zeros(n)
    for i in range(n):
        cum = 0.0
        for k in order[i]:
            cum += proba[i, k]
            if k == y[i]:
                break
        scores[i] = cum
    return scores


def aps_sets(proba, qhat):
    n, C = proba.shape
    order = np.argsort(-proba, axis=1)
    sets = np.zeros((n, C), bool)
    for i in range(n):
        cum = 0.0
        for k in order[i]:
            sets[i, k] = True
            cum += proba[i, k]
            if cum >= qhat:
                break
    return sets


# ---------------- per-criterion fit → sets for all methods at a given alpha ----------
def fit_sets(df, c, tr, ca, te, alpha):
    y = df["gt_viol_" + c].to_numpy()
    ytr, yca, yte = y[tr], y[ca], y[te]
    if ytr.sum() < 5 or len(np.unique(ytr)) < 2 or yca.sum() < 1 or yte.sum() < 1:
        return None
    Xall = df[A.CRITERIA + A.METRICS].to_numpy(float)
    sc = StandardScaler().fit(Xall[tr]); Xs = sc.transform(Xall)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs[tr], ytr)
    classes = list(clf.classes_)
    p_ca, p_te = clf.predict_proba(Xs[ca]), clf.predict_proba(Xs[te])
    s_ca = A.lac_scores(p_ca, yca)
    q = A.conformal_quantile(s_ca, alpha)
    sets_lac = A.prediction_sets(p_te, np.array([q, q]))
    qc = np.array([A.conformal_quantile(s_ca[yca == 0], alpha),
                   A.conformal_quantile(s_ca[yca == 1], alpha)])
    sets_mond = A.prediction_sets(p_te, qc)
    e_ca = aps_scores_calib(p_ca, yca)
    qa = A.conformal_quantile(e_ca, alpha)
    sets_aps = aps_sets(p_te, qa)
    bio_flag = (df[c].to_numpy()[te] == 0).astype(int)
    return {"lac": sets_lac, "mond": sets_mond, "aps": sets_aps,
            "yte": yte, "classes": classes, "bio_flag": bio_flag}


def set_metrics(sets, yte, classes):
    n = len(yte)
    ci = [classes.index(v) for v in yte]
    covered = sets[np.arange(n), ci]
    has_ok = sets[:, classes.index(0)]; has_v = sets[:, classes.index(1)]
    size = sets.sum(1)
    vmask, omask = (yte == 1), (yte == 0)
    conf_v = has_v & ~has_ok
    return {
        "coverage": float(covered.mean()),
        "coverage_viol": float(covered[vmask].mean()) if vmask.sum() else None,
        "coverage_ok": float(covered[omask].mean()) if omask.sum() else None,
        "avg_set_size": float(size.mean()),
        "ambiguous_rate": float((has_ok & has_v).mean()),
        "empty_rate": float((size == 0).mean()),
        "recall_viol": float(conf_v[vmask].mean()) if vmask.sum() else None,
        "fpr": float(conf_v[omask].mean()) if omask.sum() else None,
        "fp": int((omask & conf_v).sum()), "tp": int((vmask & conf_v).sum()),
        "n_viol": int(vmask.sum()), "n_ok": int(omask.sum()),
    }


def totals(per_meth_fp_tp, base_fp):
    fp = sum(x[0] for x in per_meth_fp_tp); tp = sum(x[1] for x in per_meth_fp_tp)
    viol = sum(x[2] for x in per_meth_fp_tp); ok = sum(x[3] for x in per_meth_fp_tp)
    return {"fp": fp, "recall": tp / viol if viol else None, "fpr": fp / ok if ok else None,
            "ffred_pct": round((base_fp - fp) / base_fp * 100, 1) if base_fp else 0.0}


# ==================== run ====================
if __name__ == "__main__":
    df = A.load_and_prepare(CSV)
    ALPHA = 0.10

    # ---- (1) APS + all methods, method totals over 5 seeds ----
    seeds = range(5)
    agg = {m: {"coverage": [], "coverage_viol": [], "fpr": [], "recall": [], "ffred": []}
           for m in ["lac", "mond", "aps"]}
    for seed in seeds:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        held = {}
        for c in A.CRITERIA:
            r = fit_sets(df, c, tr, ca, te, ALPHA)
            if r: held[c] = r
        active = list(held)
        # raw baseline fp for ffred
        base_fp = 0
        for c in active:
            h = held[c]; om = (h["yte"] == 0); base_fp += int((om & (h["bio_flag"] == 1)).sum())
        for m in ["lac", "mond", "aps"]:
            covs, cvs, fptp = [], [], []
            for c in active:
                h = held[c]; mm = set_metrics(h[m], h["yte"], h["classes"])
                covs.append(mm["coverage"])
                if mm["coverage_viol"] is not None: cvs.append(mm["coverage_viol"])
                fptp.append((mm["fp"], mm["tp"], mm["n_viol"], mm["n_ok"]))
            t = totals(fptp, base_fp)
            agg[m]["coverage"].append(float(np.mean(covs)))
            agg[m]["coverage_viol"].append(float(np.mean(cvs)))
            agg[m]["fpr"].append(t["fpr"]); agg[m]["recall"].append(t["recall"])
            agg[m]["ffred"].append(t["ffred_pct"])
    method_summary = {m: {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                          for k, v in d.items()} for m, d in agg.items()}
    print("== APS + methods (5 seeds) ==")
    for m in ["lac", "mond", "aps"]:
        s = method_summary[m]
        print(f"  {m:5s} cov={s['coverage']['mean']:.3f} viol={s['coverage_viol']['mean']:.3f} "
              f"fpr={s['fpr']['mean']:.3f} recall={s['recall']['mean']:.3f} ffred={s['ffred']['mean']}%")

    # ---- (2) Bonferroni-corrected joint coverage over 5 seeds ----
    alpha_bonf = ALPHA / M
    joint = {"uncorrected": {"lac": [], "mond": []}, "bonferroni": {"lac": [], "mond": []},
             "set_size_bonf": {"lac": [], "mond": []}, "retake_pct_bonf": {"lac": [], "mond": []}}
    for seed in seeds:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        n = len(te)
        for tag, al in [("uncorrected", ALPHA), ("bonferroni", alpha_bonf)]:
            held = {}
            for c in A.CRITERIA:
                r = fit_sets(df, c, tr, ca, te, al)
                if r: held[c] = r
            active = list(held)
            for m in ["lac", "mond"]:
                jc = np.ones(n, bool); sizes = []
                retake = np.zeros(n, bool)
                for c in active:
                    h = held[c]; cls = h["classes"]
                    cov = h[m][np.arange(n), [cls.index(v) for v in h["yte"]]]
                    jc &= cov
                    sizes.append(h[m].sum(1))
                    has_ok = h[m][:, cls.index(0)]; has_v = h[m][:, cls.index(1)]
                    retake |= (has_ok & has_v) | (h[m].sum(1) == 0)  # ambiguous/empty → retake
                joint[tag][m].append(float(jc.mean()))
                if tag == "bonferroni":
                    joint["set_size_bonf"][m].append(float(np.mean(sizes)))
                    joint["retake_pct_bonf"][m].append(float(retake.mean()))
    joint_summary = {}
    for k, d in joint.items():
        joint_summary[k] = {m: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                            for m, v in d.items()}
    print("\n== Joint coverage (5 seeds) ==")
    print(f"  uncorrected LAC={joint_summary['uncorrected']['lac']['mean']} "
          f"Mond={joint_summary['uncorrected']['mond']['mean']}")
    print(f"  Bonferroni  LAC={joint_summary['bonferroni']['lac']['mean']} "
          f"Mond={joint_summary['bonferroni']['mond']['mean']} "
          f"(set size Mond={joint_summary['set_size_bonf']['mond']['mean']}, "
          f"retake% Mond={joint_summary['retake_pct_bonf']['mond']['mean']})")

    # ---- (3) Mondrian image-level decisions + safety (seed 0) ----
    tr, ca, te, _ = A.subject_disjoint_split(df, seed=0)
    n = len(te)
    held = {}
    for c in A.CRITERIA:
        r = fit_sets(df, c, tr, ca, te, ALPHA)
        if r: held[c] = r
    active = list(held)
    te_folder = df["source_folder"].to_numpy()[te]
    has_inscope = np.array([A.FOLDER_TO_CRIT.get(f) is not None for f in te_folder])
    is_zoom = np.array([f == "zoom" for f in te_folder])
    mond_img = {}
    for m in ["lac", "mond"]:
        dec = []
        for i in range(n):
            anyv = any(held[c][m][i, held[c]["classes"].index(1)] and not held[c][m][i, held[c]["classes"].index(0)] for c in active)
            allok = all(held[c][m][i, held[c]["classes"].index(0)] and not held[c][m][i, held[c]["classes"].index(1)] for c in active)
            dec.append("REJECT" if anyv else ("ACCEPT" if allok else "RETAKE"))
        dec = np.array(dec)
        mond_img[m] = {
            "ACCEPT": int((dec == "ACCEPT").sum()), "REJECT": int((dec == "REJECT").sum()),
            "RETAKE": int((dec == "RETAKE").sum()),
            "violations_wrongly_accepted": int(((dec == "ACCEPT") & has_inscope).sum()),
            "total_violation_images": int(has_inscope.sum()),
        }
    print("\n== Image-level (seed 0) ==")
    for m in ["lac", "mond"]:
        d = mond_img[m]
        print(f"  {m}: A={d['ACCEPT']} R={d['REJECT']} RT={d['RETAKE']} "
              f"wrongly_accepted={d['violations_wrongly_accepted']}/{d['total_violation_images']}")

    # ---- (4) Alpha cost-curve (mean over 3 seeds) ----
    alphas = [0.20, 0.15, 0.10, 0.05, 0.02]
    curve = {m: {"nominal": [], "coverage": [], "coverage_viol": [], "fpr": [], "recall": []}
             for m in ["lac", "mond"]}
    for al in alphas:
        per = {m: {"cov": [], "cv": [], "fpr": [], "rec": []} for m in ["lac", "mond"]}
        for seed in range(3):
            tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
            held = {}
            for c in A.CRITERIA:
                r = fit_sets(df, c, tr, ca, te, al)
                if r: held[c] = r
            active = list(held)
            base_fp = 0
            for c in active:
                h = held[c]; om = (h["yte"] == 0); base_fp += int((om & (h["bio_flag"] == 1)).sum())
            for m in ["lac", "mond"]:
                covs, cvs, fptp = [], [], []
                for c in active:
                    mm = set_metrics(held[c][m], held[c]["yte"], held[c]["classes"])
                    covs.append(mm["coverage"])
                    if mm["coverage_viol"] is not None: cvs.append(mm["coverage_viol"])
                    fptp.append((mm["fp"], mm["tp"], mm["n_viol"], mm["n_ok"]))
                t = totals(fptp, base_fp)
                per[m]["cov"].append(np.mean(covs)); per[m]["cv"].append(np.mean(cvs))
                per[m]["fpr"].append(t["fpr"]); per[m]["rec"].append(t["recall"])
        for m in ["lac", "mond"]:
            curve[m]["nominal"].append(1 - al)
            curve[m]["coverage"].append(round(float(np.mean(per[m]["cov"])), 4))
            curve[m]["coverage_viol"].append(round(float(np.mean(per[m]["cv"])), 4))
            curve[m]["fpr"].append(round(float(np.mean(per[m]["fpr"])), 4))
            curve[m]["recall"].append(round(float(np.mean(per[m]["rec"])), 4))
    print("\n== Alpha curve (Mondrian) ==")
    for i, nn in enumerate(curve["mond"]["nominal"]):
        print(f"  1-a={nn:.2f} cov={curve['mond']['coverage'][i]:.3f} "
              f"viol={curve['mond']['coverage_viol'][i]:.3f} fpr={curve['mond']['fpr'][i]:.3f} "
              f"recall={curve['mond']['recall'][i]:.3f}")

    out = {"config": {"alpha": ALPHA, "m_criteria": M, "alpha_bonferroni": round(alpha_bonf, 5),
                      "seeds_methods": 5, "seeds_curve": 3, "split": "subject-disjoint 60/20/20"},
           "method_summary_incl_aps": method_summary,
           "joint_coverage": joint_summary,
           "image_level_seed0": mond_img,
           "alpha_curve": curve}
    json.dump(out, open(f"{OUT}/strengthen.json", "w"), indent=2)
    print("\nsaved results_v2/strengthen.json")
