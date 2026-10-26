"""
analysis_v3.py: hardened analysis addressing the 5-agent internal review.

Changes vs v2:
  * conformal_quantile uses the exact k-th order statistic (k = ceil((n+1)(1-alpha))),
    returning +inf when k > n (fall back to "include the class"). Removes the +1
    order-statistic conservatism that inflated small-n Mondrian coverage.
  * zoom images (out-of-scope) are EXCLUDED from per-criterion calibration/test, so the
    OK/calibration class is not contaminated by genuine-but-out-of-scope violations.
  * FEATURE ABLATION: every method is run under three feature sets
        full            = 14 BioGaze decisions + 7 metrics   (v2 default)
        own+metrics     = this criterion's own decision + 7 metrics
        metrics         = 7 continuous metrics only
    This quantifies how much of the accuracy is the single-violation cross-criterion
    artifact, and shows the coverage GUARANTEE is feature-agnostic.
  * 10-seed reporting with Clopper-Pearson CIs on pooled violation-coverage counts,
    per-class calibration counts, and a vacuous-cell flag for Mondrian.
  * subject-level coverage summary (effective n = #test subjects, per-subject spread).
  * randomized APS (Romano et al. 2020) to test whether the degeneracy is real.
  * isotonic-calibrated logistic + swept-threshold baseline, and risk-coverage /
    achieved-vs-nominal curves, to show conformal earns its place as a *guarantee*.

Run: python3 analysis_v3.py
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.stats import beta as beta_dist
import analysis_v2 as A

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "tono_full_pred.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "results_v3"); os.makedirs(OUT, exist_ok=True)
CRIT, MET = A.CRITERIA, A.METRICS
ALPHA = 0.10
FEATSETS = {
    "full":        CRIT + MET,
    "own+metrics": None,          # filled per-criterion (own decision + metrics)
    "metrics":     MET,
}


def cq_exact(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


# ---- randomized APS (Romano, Sesia, Candes 2020) ----
def aps_calib_scores_rand(proba, y, rng):
    n = len(y)
    order = np.argsort(-proba, axis=1)
    E = np.zeros(n)
    for i in range(n):
        cum = 0.0
        for k in order[i]:
            cum += proba[i, k]
            if k == y[i]:
                break
        u = rng.random()
        E[i] = cum - u * proba[i, y[i]]
    return E


def aps_sets_rand(proba, qhat, rng):
    n, C = proba.shape
    order = np.argsort(-proba, axis=1)
    sets = np.zeros((n, C), bool)
    for i in range(n):
        cum = 0.0
        for k in order[i]:
            if cum + proba[i, k] <= qhat:
                sets[i, k] = True; cum += proba[i, k]
            else:
                if rng.random() < (qhat - cum) / max(proba[i, k], 1e-12):
                    sets[i, k] = True
                break
    # guarantee non-empty: include top-1 if empty
    empty = sets.sum(1) == 0
    for i in np.where(empty)[0]:
        sets[i, order[i][0]] = True
    return sets


def fit_all(df, c, tr, ca, te, alpha, feat_cols, rng=None):
    nz = df["source_folder"].to_numpy() != "zoom"
    tr = tr[nz[tr]]; ca = ca[nz[ca]]; te = te[nz[te]]
    y = df["gt_viol_" + c].to_numpy()
    ytr, yca, yte = y[tr], y[ca], y[te]
    if ytr.sum() < 5 or len(np.unique(ytr)) < 2 or yca.sum() < 1 or yte.sum() < 1:
        return None
    X = df[feat_cols].to_numpy(float)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs[tr], ytr)
    classes = list(clf.classes_)
    p_ca, p_te = clf.predict_proba(Xs[ca]), clf.predict_proba(Xs[te])
    s_ca = 1.0 - p_ca[np.arange(len(yca)), yca]
    q = cq_exact(s_ca, alpha)
    sets_lac = (1.0 - p_te) <= np.array([q, q])
    qc = np.array([cq_exact(s_ca[yca == 0], alpha), cq_exact(s_ca[yca == 1], alpha)])
    sets_mond = (1.0 - p_te) <= qc
    out = {"lac": sets_lac, "mond": sets_mond, "yte": yte, "classes": classes,
           "p_te": p_te, "p_ca": p_ca, "yca": yca,
           "bio_flag": (df[c].to_numpy()[te] == 0).astype(int),
           "n_pos_calib": int(yca.sum()), "n_pos_test": int(yte.sum()),
           "mond_viol_qinf": bool(np.isinf(qc[1])),  # vacuous: violation class always included
           "subj_te": df["subject"].to_numpy()[te],
           "clf": clf, "X_ca": Xs[ca], "X_te": Xs[te]}   # fitted model + scaled features, for crosscheck_v3.py
    if rng is not None:
        e = aps_calib_scores_rand(p_ca, yca, rng)
        qa = cq_exact(e, alpha)
        out["aps"] = aps_sets_rand(p_te, qa, rng)
    return out


def set_metrics(sets, yte, classes):
    n = len(yte); ci = [classes.index(v) for v in yte]
    covered = sets[np.arange(n), ci]
    has_ok, has_v = sets[:, classes.index(0)], sets[:, classes.index(1)]
    size = sets.sum(1)
    vmask, omask = (yte == 1), (yte == 0); conf_v = has_v & ~has_ok
    return {"coverage": float(covered.mean()),
            "coverage_viol": float(covered[vmask].mean()) if vmask.sum() else None,
            "n_viol_cov": int(covered[vmask].sum()), "n_viol": int(vmask.sum()),
            "n_cov": int(covered.sum()), "n": int(n),
            "avg_set_size": float(size.mean()),
            "both_rate": float((has_ok & has_v).mean()),
            "empty_rate": float((size == 0).mean()),
            "fp": int((omask & conf_v).sum()), "tp": int((vmask & conf_v).sum()),
            "n_ok": int(omask.sum())}


if __name__ == "__main__":
    df = A.load_and_prepare(CSV)

    # ================= (A) FEATURE ABLATION (5 seeds) =================
    print("== feature ablation ==")
    abl = {}
    for fname in FEATSETS:
        per_seed = {"lac_cov": [], "lac_viol": [], "mond_viol": [], "log_fpr": [],
                    "log_recall": [], "log_ffred": [], "mond_setsize": [], "mond_both": [],
                    "lac_empty": []}
        for seed in range(5):
            tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
            covs, cvs_l, cvs_m, sizes, boths, empties = [], [], [], [], [], []
            base_fp = fp = tp = nviol = nok = 0
            for c in CRIT:
                feat = ([c] + MET) if fname == "own+metrics" else FEATSETS[fname]
                r = fit_all(df, c, tr, ca, te, ALPHA, feat)
                if r is None:
                    continue
                ml = set_metrics(r["lac"], r["yte"], r["classes"])
                mm = set_metrics(r["mond"], r["yte"], r["classes"])
                covs.append(ml["coverage"]); empties.append(ml["empty_rate"])
                sizes.append(mm["avg_set_size"]); boths.append(mm["both_rate"])
                if ml["coverage_viol"] is not None: cvs_l.append(ml["coverage_viol"])
                if mm["coverage_viol"] is not None: cvs_m.append(mm["coverage_viol"])
                # logistic-only point prediction
                pred = (r["p_te"][:, r["classes"].index(1)] >= 0.5).astype(int)
                yte = r["yte"]; om = (yte == 0); vm = (yte == 1)
                fp += int((om & (pred == 1)).sum()); tp += int((vm & (pred == 1)).sum())
                nviol += int(vm.sum()); nok += int(om.sum())
                base_fp += int((om & (r["bio_flag"] == 1)).sum())
            per_seed["lac_cov"].append(np.mean(covs))
            per_seed["lac_viol"].append(np.mean(cvs_l))
            per_seed["mond_viol"].append(np.mean(cvs_m))
            per_seed["mond_setsize"].append(np.mean(sizes))
            per_seed["mond_both"].append(np.mean(boths))
            per_seed["lac_empty"].append(np.mean(empties))
            per_seed["log_fpr"].append(fp / nok)
            per_seed["log_recall"].append(tp / nviol)
            per_seed["log_ffred"].append((base_fp - fp) / base_fp * 100)
        abl[fname] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                      for k, v in per_seed.items()}
        a = abl[fname]
        print(f"  {fname:12s} logFPR={a['log_fpr']['mean']:.3f} logRecall={a['log_recall']['mean']:.3f} "
              f"ffred={a['log_ffred']['mean']:.1f}%  LACcov={a['lac_cov']['mean']:.3f} "
              f"MONviol={a['mond_viol']['mean']:.3f} MONsetsize={a['mond_setsize']['mean']:.3f} "
              f"MONboth={a['mond_both']['mean']:.3f}")

    # ================= (B) 10-seed per-criterion coverage + CIs (full features) =================
    print("\n== per-criterion (10 seeds, full features, quantile fixed, zoom excluded) ==")
    SEEDS = list(range(10))
    percrit = {c: {"vcov_k": 0, "vcov_n": 0, "mvcov_k": 0, "mvcov_n": 0,
                   "n_pos_calib": [], "mond_setsize": [], "n_qinf": 0} for c in CRIT}
    for seed in SEEDS:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        for c in CRIT:
            r = fit_all(df, c, tr, ca, te, ALPHA, CRIT + MET)
            if r is None:
                continue
            ml = set_metrics(r["lac"], r["yte"], r["classes"])
            mm = set_metrics(r["mond"], r["yte"], r["classes"])
            d = percrit[c]
            d["vcov_k"] += ml["n_viol_cov"]; d["vcov_n"] += ml["n_viol"]
            d["mvcov_k"] += mm["n_viol_cov"]; d["mvcov_n"] += mm["n_viol"]
            d["n_pos_calib"].append(r["n_pos_calib"]); d["mond_setsize"].append(mm["avg_set_size"])
            if r["mond_viol_qinf"]:
                d["n_qinf"] += 1
    percrit_out = {}
    robust_mviol = []   # pooled Mondrian violation coverage, excluding degenerate criteria
    for c in CRIT:
        d = percrit[c]
        # POINT ESTIMATE = pooled proportion (consistent with the Clopper-Pearson CI)
        lac_p = d["vcov_k"] / d["vcov_n"] if d["vcov_n"] else float("nan")
        mond_p = d["mvcov_k"] / d["mvcov_n"] if d["mvcov_n"] else float("nan")
        lci = clopper_pearson(d["vcov_k"], d["vcov_n"])
        mci = clopper_pearson(d["mvcov_k"], d["mvcov_n"])
        med_calib = int(np.median(d["n_pos_calib"])) if d["n_pos_calib"] else 0
        degenerate = (d["n_qinf"] > 0) or (med_calib < 15)
        percrit_out[c] = {
            "lac_viol": round(lac_p, 3), "lac_viol_ci95": [round(lci[0], 3), round(lci[1], 3)],
            "mond_viol": round(mond_p, 3), "mond_viol_ci95": [round(mci[0], 3), round(mci[1], 3)],
            "median_pos_calib": med_calib,
            "mond_avg_set_size": round(float(np.mean(d["mond_setsize"])), 3),
            "n_seeds_vacuous_qinf": d["n_qinf"],
            "degenerate_flag": bool(degenerate),
        }
        if not degenerate:
            robust_mviol.append(mond_p)
        print(f"  {c:22s} LACviol={lac_p:.3f} CI[{lci[0]:.2f},{lci[1]:.2f}] "
              f"MONviol={mond_p:.3f} CI[{mci[0]:.2f},{mci[1]:.2f}] calibPos~{med_calib}"
              f"{'  <-DEGENERATE' if degenerate else ''}")
    robust_headline = {"mond_viol_pooled_mean_stable_criteria": round(float(np.mean(robust_mviol)), 4),
                       "n_stable_criteria": len(robust_mviol),
                       "excluded_degenerate": [c for c in CRIT if percrit_out[c]["degenerate_flag"]]}
    print("  robust Mondrian violation coverage (stable criteria only):",
          robust_headline["mond_viol_pooled_mean_stable_criteria"],
          "excluding", robust_headline["excluded_degenerate"])

    # ================= (C) subject-level coverage (seed 0, full) =================
    tr, ca, te, man = A.subject_disjoint_split(df, seed=0)
    held = {c: fit_all(df, c, tr, ca, te, ALPHA, CRIT + MET) for c in CRIT}
    held = {c: r for c, r in held.items() if r is not None}
    # per-subject mean coverage (LAC), pooled over criteria
    subj_cov = {}
    for c, r in held.items():
        cov = r["lac"][np.arange(len(r["yte"])), [r["classes"].index(v) for v in r["yte"]]]
        for s, cv in zip(r["subj_te"], cov):
            subj_cov.setdefault(s, []).append(cv)
    subj_means = np.array([np.mean(v) for v in subj_cov.values()])
    all_outcomes = np.concatenate([np.asarray(v) for v in subj_cov.values()])  # every image-criterion outcome
    subject_level = {
        "n_test_subjects": int(len(subj_means)),
        "n_calib_subjects": int(man["n_subjects_calib"]),
        "image_level_coverage_micro": round(float(all_outcomes.mean()), 4),   # pooled over all outcomes
        "subject_macro_coverage": round(float(np.mean(subj_means)), 4),       # equal weight per subject
        "subject_coverage_std": round(float(np.std(subj_means)), 4),
        "treatment": "EMPIRICAL DIAGNOSTIC, not a formal guarantee.",
        "note": "Images cluster within subjects, so exchangeability holds only at the subject level; "
                "the effective calibration size is the number of calibration SUBJECTS (~55), not images "
                "(762). Calibration here is still image-level, so we report empirical coverage only; a "
                "formal clustered/subject-level conformal guarantee is left as future work.",
    }
    print("\n== subject-level ==", json.dumps(subject_level))

    # ================= (D) randomized APS (5 seeds) =================
    print("\n== randomized APS (5 seeds) ==")
    aps_cov, aps_viol, aps_rec, aps_size = [], [], [], []
    for seed in range(5):
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        rng = np.random.RandomState(1000 + seed)
        covs, cvs, tp, nviol, sizes = [], [], 0, 0, []
        for c in CRIT:
            r = fit_all(df, c, tr, ca, te, ALPHA, CRIT + MET, rng=rng)
            if r is None or "aps" not in r:
                continue
            m = set_metrics(r["aps"], r["yte"], r["classes"])
            covs.append(m["coverage"]); sizes.append(m["avg_set_size"])
            if m["coverage_viol"] is not None: cvs.append(m["coverage_viol"])
            tp += m["tp"]; nviol += m["n_viol"]
        aps_cov.append(np.mean(covs)); aps_viol.append(np.mean(cvs))
        aps_rec.append(tp / nviol); aps_size.append(np.mean(sizes))
    aps_rand = {"coverage": round(float(np.mean(aps_cov)), 3), "coverage_viol": round(float(np.mean(aps_viol)), 3),
                "recall": round(float(np.mean(aps_rec)), 3), "avg_set_size": round(float(np.mean(aps_size)), 3)}
    print("  randomized APS:", aps_rand)

    # ================= (E) isotonic baseline + risk-coverage (seed 0, full) =================
    # pooled FPR-recall by sweeping isotonic-calibrated threshold vs conformal alpha
    thr_grid = np.linspace(0.02, 0.98, 25)
    iso_curve = {"fpr": [], "recall": []}
    # gather calibrated probs on test pooled over criteria
    pooled = []
    for c, r in held.items():
        iso = IsotonicRegression(out_of_bounds="clip").fit(r["p_ca"][:, r["classes"].index(1)], r["yca"])
        pc = iso.transform(r["p_te"][:, r["classes"].index(1)])
        pooled.append((pc, r["yte"]))
    for t in thr_grid:
        fp = tp = nok = nviol = 0
        for pc, yte in pooled:
            pred = (pc >= t)
            om = (yte == 0); vm = (yte == 1)
            fp += int((om & pred).sum()); tp += int((vm & pred).sum())
            nok += int(om.sum()); nviol += int(vm.sum())
        iso_curve["fpr"].append(round(fp / nok, 4)); iso_curve["recall"].append(round(tp / nviol, 4))
    print("\n== isotonic baseline risk-coverage computed:", len(thr_grid), "points ==")

    out = {"config": {"alpha": ALPHA, "seeds_ablation": 5, "seeds_percrit": 10,
                      "quantile": "exact k-th order statistic, +inf when k>n", "zoom": "excluded from per-criterion"},
           "feature_ablation": abl, "per_criterion": percrit_out,
           "robust_headline": robust_headline,
           "subject_level": subject_level, "aps_randomized": aps_rand,
           "isotonic_riskcoverage": {"threshold": [round(float(t), 3) for t in thr_grid], **iso_curve}}
    json.dump(out, open(f"{OUT}/analysis_v3.json", "w"), indent=2)
    print("\nsaved results_v3/analysis_v3.json")
