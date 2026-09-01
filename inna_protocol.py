"""
inna_protocol.py (v2): supervisor pre-freeze checks, revised after internal review.

Fixes vs v1:
  * No test-set leak: the abstention target is a fixed design constant (0.10 = alpha),
    never read from test data. All tuning uses calibration only.
  * Steelman abstention baselines instead of a symmetric Youden band:
      chow  : confidence rejection (Chow 1970): decide by p>=0.5, abstain if max(p,1-p)<tau;
              tau tuned on calibration to the target abstention.
      asym  : asymmetric two-threshold rule (t_lo, t_hi) tuned on calibration to minimise
              10*FN + FP among decided cases subject to abstention within +/-0.02 of target.
  * Every method reports: FPR, recall over ALL cases, recall among DECIDED cases,
    abstention/RETAKE, decision coverage, selective error (among decided), share of
    violations abstained, and a uniform safety quantity "violation not confidently
    accepted" (1 - P(confident OK | violation)). Conformal methods additionally report
    singleton/ambiguous/empty frequencies and violation-class coverage (violation in set).
  * Cluster-aware sensitivity that separates sample size from clustering:
      full calibration; random image subsample at the SAME n as subject-capping;
      subject-capped subsample; a TRUE subject bootstrap with multiplicity (5th/95th pct);
      per-test-subject coverage spread.
  * Images-per-subject for train, calibration and test.
  * Risk-abstention curves for chow, asym, LAC, Mondrian across operating points.
Run: python3 inna_protocol.py [csv] [outdir] [abstention_target]
  The optional third argument re-runs the protocol at another abstention target (e.g. 0.05, 0.15)
  and writes inna_protocol_t<target>.json instead of inna_protocol.json.
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import analysis_v2 as A
import analysis_v3 as V3

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "tono_full_pred.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "results_v3")
CRIT, MET = A.CRITERIA, A.METRICS
ALPHA = 0.10
TARGET_ABST = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10   # design constant (= alpha); never read from test
OUT_NAME = "inna_protocol.json" if len(sys.argv) <= 3 else f"inna_protocol_t{TARGET_ABST:.2f}.json"
SEEDS = list(range(5))


def fit_core(df, c, tr, ca, te):
    nz = df["source_folder"].to_numpy() != "zoom"
    trf, caf, tef = tr[nz[tr]], ca[nz[ca]], te[nz[te]]
    y = df["gt_viol_" + c].to_numpy()
    ytr, yca, yte = y[trf], y[caf], y[tef]
    if ytr.sum() < 5 or len(np.unique(ytr)) < 2 or yca.sum() < 1 or yte.sum() < 1:
        return None
    X = df[CRIT + MET].to_numpy(float)
    sc = StandardScaler().fit(X[trf]); Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs[trf], ytr)
    subj = df["subject"].to_numpy()
    return {"p_ca": clf.predict_proba(Xs[caf])[:, 1], "yca": yca, "subj_ca": subj[caf],
            "p_te": clf.predict_proba(Xs[tef])[:, 1], "yte": yte, "subj_te": subj[tef],
            "bio_flag": (df[c].to_numpy()[tef] == 0).astype(int)}


def lac_qs(p_ca, yca, alpha, idx=None):
    if idx is not None:
        p_ca, yca = p_ca[idx], yca[idx]
    proba = np.column_stack([1 - p_ca, p_ca])
    s = 1 - proba[np.arange(len(yca)), yca]
    return V3.cq_exact(s, alpha), np.array([V3.cq_exact(s[yca == 0], alpha), V3.cq_exact(s[yca == 1], alpha)])


def sets_from_q(p, q_ok, q_v):
    proba = np.column_stack([1 - p, p])
    return (1.0 - proba) <= np.array([q_ok, q_v])


# ---------------- decision rules -> (dec_ok, dec_v) boolean arrays; abstain = neither ----------------
def rule_point(p, t=0.5):
    return p < t, p >= t

def rule_chow(p, tau):
    conf = np.maximum(p, 1 - p) >= tau
    return conf & (p < 0.5), conf & (p >= 0.5)

def rule_asym(p, t_lo, t_hi):
    return p <= t_lo, p >= t_hi

def rule_sets(sets):
    return sets[:, 0] & ~sets[:, 1], sets[:, 1] & ~sets[:, 0]


def tune_chow(p_ca, yca, target):
    best = None
    for tau in np.linspace(0.5, 0.999, 200):
        ab = (np.maximum(p_ca, 1 - p_ca) < tau).mean()
        d = abs(ab - target)
        if best is None or d < best[0]:
            best = (d, tau)
    return float(best[1])


def tune_asym(p_ca, yca, target, cost_fn=10.0, tol=0.02):
    grid = np.linspace(0.02, 0.98, 49)
    best = None
    for t_lo in grid:
        for t_hi in grid:
            if t_hi <= t_lo:
                continue
            ok, v = rule_asym(p_ca, t_lo, t_hi)
            ab = 1 - (ok | v).mean()
            if abs(ab - target) > tol:
                continue
            cost = cost_fn * (ok & (yca == 1)).sum() + (v & (yca == 0)).sum()
            if best is None or cost < best[0]:
                best = (cost, float(t_lo), float(t_hi))
    if best is None:   # fall back: widest band around 0.5 meeting tolerance loosely
        return 0.4, 0.6
    return best[1], best[2]


def metrics(dec_ok, dec_v, yte, bio_flag=None):
    vm, om = yte == 1, yte == 0
    dec = dec_ok | dec_v; ab = ~dec
    out = {
        "fp": int((om & dec_v).sum()), "tp_all": int((vm & dec_v).sum()),
        "n_ok": int(om.sum()), "n_v": int(vm.sum()), "n": int(len(yte)),
        "abst": int(ab.sum()), "dec": int(dec.sum()),
        "err_dec": int(((dec_ok & vm) | (dec_v & om)).sum()),
        "v_dec": int((vm & dec).sum()), "v_abst": int((vm & ab).sum()),
        "v_conf_ok": int((vm & dec_ok).sum()),
    }
    return out


# ============================================================ main
def run():
    df = A.load_and_prepare(CSV)
    # ---------- images per subject, all splits ----------
    ips = {}
    for seed in SEEDS:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        ips[f"seed{seed}"] = {}
        for name, idx in [("train", tr), ("calib", ca), ("test", te)]:
            _, cnt = np.unique(df["subject"].to_numpy()[idx], return_counts=True)
            ips[f"seed{seed}"][name] = {"n_subjects": int(len(cnt)), "n_images": int(len(idx)),
                                        "median": float(np.median(cnt)), "max": int(cnt.max()),
                                        "max_share_pct": round(100 * cnt.max() / len(idx), 1)}
    ips_summary = {}
    for name in ["train", "calib", "test"]:
        ips_summary[name] = {k: round(float(np.mean([ips[f"seed{s}"][name][k] for s in SEEDS])), 1)
                             for k in ["n_subjects", "n_images", "median", "max", "max_share_pct"]}
        ips_summary[name]["max_share_pct_max"] = round(max(ips[f"seed{s}"][name]["max_share_pct"] for s in SEEDS), 1)

    # ---------- fair table + curves + cluster-aware ----------
    METHODS = ["raw", "logistic", "chow", "asym", "lac", "mond", "aps"]
    agg = {m: {} for m in METHODS}   # per-seed accumulators -> lists of per-seed metrics
    per_seed = {m: [] for m in METHODS}
    settype = {m: [] for m in ["lac", "mond", "aps"]}
    covviol = {m: [] for m in ["lac", "mond", "aps"]}
    alphas = [0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02]
    chow_taus = np.linspace(0.55, 0.99, 12); asym_targets = np.linspace(0.02, 0.35, 12)
    curve_acc = {m: {} for m in ["chow", "asym", "lac", "mond"]}
    clus = {"full": [], "random_equal_n": [], "subject_cap3": [], "subject_boot_mean": [],
            "subject_boot_p05": [], "subject_boot_p95": [], "per_subject_cov_std": []}
    rng_boot = np.random.RandomState(0)

    for seed in SEEDS:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        cores = {c: fit_core(df, c, tr, ca, te) for c in CRIT}
        cores = {c: r for c, r in cores.items() if r is not None}
        active = list(cores)
        acc = {m: None for m in METHODS}
        st = {m: {"single": 0, "amb": 0, "empty": 0, "n": 0, "vin": 0, "nv": 0} for m in settype}
        rng_aps = np.random.RandomState(2000 + seed)

        def add(m, d):
            if acc[m] is None:
                acc[m] = dict(d)
            else:
                for k in d: acc[m][k] += d[k]

        for c in active:
            r = cores[c]; p, yte = r["p_te"], r["yte"]
            # baselines tuned on calibration only
            tau = tune_chow(r["p_ca"], r["yca"], TARGET_ABST)
            t_lo, t_hi = tune_asym(r["p_ca"], r["yca"], TARGET_ABST)
            q, qc = lac_qs(r["p_ca"], r["yca"], ALPHA)
            proba_ca = np.column_stack([1 - r["p_ca"], r["p_ca"]])
            e = V3.aps_calib_scores_rand(proba_ca, r["yca"], rng_aps); qa = V3.cq_exact(e, ALPHA)
            rules = {
                "raw": (r["bio_flag"] == 0, r["bio_flag"] == 1),
                "logistic": rule_point(p),
                "chow": rule_chow(p, tau),
                "asym": rule_asym(p, t_lo, t_hi),
                "lac": rule_sets(sets_from_q(p, q, q)),
                "mond": rule_sets(sets_from_q(p, qc[0], qc[1])),
                "aps": rule_sets(V3.aps_sets_rand(np.column_stack([1 - p, p]), qa, rng_aps)),
            }
            sets_by = {"lac": sets_from_q(p, q, q), "mond": sets_from_q(p, qc[0], qc[1])}
            # recompute aps sets deterministically for set-type stats (same rng state issue: reuse rule)
            for m in METHODS:
                add(m, metrics(*rules[m], yte))
            for m in ["lac", "mond"]:
                s = sets_by[m]; sz = s.sum(1); vm = yte == 1
                st[m]["single"] += int((sz == 1).sum()); st[m]["amb"] += int((sz == 2).sum())
                st[m]["empty"] += int((sz == 0).sum()); st[m]["n"] += len(yte)
                st[m]["vin"] += int(s[vm, 1].sum()); st[m]["nv"] += int(vm.sum())
            # aps set stats from its decision rule (aps forces non-empty)
            ok_a, v_a = rules["aps"]; amb_a = ~(ok_a | v_a); vm = yte == 1
            st["aps"]["single"] += int((ok_a | v_a).sum()); st["aps"]["amb"] += int(amb_a.sum())
            st["aps"]["n"] += len(yte); st["aps"]["vin"] += int((vm & (v_a | amb_a)).sum()); st["aps"]["nv"] += int(vm.sum())

            # ---- curves ----
            for i, tau_i in enumerate(chow_taus):
                d = metrics(*rule_chow(p, tau_i), yte); curve_acc["chow"].setdefault(i, {});
                for k, v in d.items(): curve_acc["chow"][i][k] = curve_acc["chow"][i].get(k, 0) + v
            for i, tg in enumerate(asym_targets):
                lo, hi = tune_asym(r["p_ca"], r["yca"], tg, tol=0.03)
                d = metrics(*rule_asym(p, lo, hi), yte); curve_acc["asym"].setdefault(i, {})
                for k, v in d.items(): curve_acc["asym"][i][k] = curve_acc["asym"][i].get(k, 0) + v
            for i, al in enumerate(alphas):
                q2, qc2 = lac_qs(r["p_ca"], r["yca"], al)
                for m, s in [("lac", sets_from_q(p, q2, q2)), ("mond", sets_from_q(p, qc2[0], qc2[1]))]:
                    d = metrics(*rule_sets(s), yte); d["vin"] = int(s[yte == 1, 1].sum())
                    curve_acc[m].setdefault(i, {})
                    for k, v in d.items(): curve_acc[m][i][k] = curve_acc[m][i].get(k, 0) + v

        # finalize per-seed metrics
        for m in METHODS:
            a = acc[m]
            per_seed[m].append({
                "fpr": a["fp"] / a["n_ok"], "recall_all": a["tp_all"] / a["n_v"],
                "recall_decided": a["tp_all"] / a["v_dec"] if a["v_dec"] else float("nan"),
                "abst": a["abst"] / a["n"], "dec_cov": a["dec"] / a["n"],
                "sel_err": a["err_dec"] / a["dec"] if a["dec"] else float("nan"),
                "viol_abstained": a["v_abst"] / a["n_v"],
                "viol_not_accepted": 1 - a["v_conf_ok"] / a["n_v"],
                "fp": a["fp"],
            })
        for m in settype:
            s = st[m]; settype[m].append({"singleton": s["single"] / s["n"], "ambiguous": s["amb"] / s["n"],
                                          "empty": s["empty"] / s["n"]}); covviol[m].append(s["vin"] / s["nv"])

        # ---- cluster-aware (Mondrian violation coverage) ----
        def mviol(sel_fn):
            kv = kt = 0
            for c in active:
                r = cores[c]; idx = sel_fn(r["subj_ca"], r["yca"])
                if idx is None or r["yca"][idx].sum() < 1 or (r["yca"][idx] == 0).sum() < 1:
                    return None
                _, qc = lac_qs(r["p_ca"], r["yca"], ALPHA, idx=idx)
                s = sets_from_q(r["p_te"], qc[0], qc[1]); vm = r["yte"] == 1
                kv += int(s[vm, 1].sum()); kt += int(vm.sum())
            return kv / kt
        clus["full"].append(mviol(lambda s, y: np.arange(len(s))))
        # subject cap 3 and random subsample of the SAME size
        rc = np.random.RandomState(100 + seed)
        def cap3(s, y, k=3):
            keep = []
            for u in np.unique(s):
                pos = np.where(s == u)[0]; rc.shuffle(pos); keep.extend(pos[:k])
            return np.array(sorted(keep))
        cap_sizes = {}
        def cap3_rec(s, y):
            idx = cap3(s, y); cap_sizes[len(s)] = len(idx); return idx
        v_cap = mviol(cap3_rec); clus["subject_cap3"].append(v_cap)
        def rand_equal(s, y):
            n_sub = cap_sizes.get(len(s), max(3, int(0.3 * len(s))))
            return np.sort(rc.choice(len(s), size=n_sub, replace=False))
        clus["random_equal_n"].append(mviol(rand_equal))
        # true subject bootstrap with multiplicity
        boots = []
        for b in range(100):
            def boot(s, y):
                us = np.unique(s); chosen = rng_boot.choice(us, size=len(us), replace=True)
                idx = np.concatenate([np.where(s == u)[0] for u in chosen])
                return idx
            v = mviol(boot)
            if v is not None: boots.append(v)
        clus["subject_boot_mean"].append(float(np.mean(boots)))
        clus["subject_boot_p05"].append(float(np.percentile(boots, 5)))
        clus["subject_boot_p95"].append(float(np.percentile(boots, 95)))
        # per-test-subject spread of Mondrian violation coverage
        subj_cov = {}
        for c in active:
            r = cores[c]; _, qc = lac_qs(r["p_ca"], r["yca"], ALPHA)
            s = sets_from_q(r["p_te"], qc[0], qc[1]); vm = r["yte"] == 1
            for sub, cov in zip(r["subj_te"][vm], s[vm, 1]):
                subj_cov.setdefault(sub, []).append(cov)
        clus["per_subject_cov_std"].append(float(np.std([np.mean(v) for v in subj_cov.values()])))

    def summ(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3)} if v else None
    base_fp = np.mean([d["fp"] for d in per_seed["raw"]])
    fair = {}
    for m in METHODS:
        fair[m] = {k: summ([d[k] for d in per_seed[m]]) for k in per_seed[m][0] if k != "fp"}
        fair[m]["ffred_pct"] = summ([100 * (base_fp - d["fp"]) / base_fp for d in per_seed[m]])
    set_type = {m: {k: round(float(np.mean([d[k] for d in settype[m]])), 3) for k in settype[m][0]} for m in settype}
    cov_viol = {m: summ(covviol[m]) for m in covviol}
    # curves
    curves = {}
    for m, params in [("chow", chow_taus), ("asym", asym_targets), ("lac", alphas), ("mond", alphas)]:
        pts = []
        for i in sorted(curve_acc[m]):
            d = curve_acc[m][i]
            pt = {"param": round(float(params[i]), 3), "abst": round(d["abst"] / d["n"], 3),
                  "fpr_decided": round(d["fp"] / (d["n_ok"] - (d["abst"] - d["v_abst"])), 3) if (d["n_ok"] - (d["abst"] - d["v_abst"])) else None,
                  "recall_decided": round(d["tp_all"] / d["v_dec"], 3) if d["v_dec"] else None,
                  "viol_not_accepted": round(1 - d["v_conf_ok"] / d["n_v"], 3)}
            if "vin" in d: pt["coverage_viol"] = round(d["vin"] / d["n_v"], 3)
            pts.append(pt)
        curves[m] = pts
    clus_out = {k: summ(v) for k, v in clus.items()}

    out = {"config": {"alpha": ALPHA, "target_abstention": TARGET_ABST, "seeds": SEEDS,
                      "tuning": "chow: tau on calibration to target abstention; asym: (t_lo,t_hi) on calibration "
                                "minimising 10*FN+FP among decided with abstention within 0.02 of target; test never used"},
           "images_per_subject": {"per_seed": ips, "summary": ips_summary},
           "fair_table": fair, "set_type_freq": set_type, "coverage_viol": cov_viol,
           "risk_abstention_curves": curves, "cluster_aware_mondrian": clus_out}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, OUT_NAME), "w"), indent=2)
    print("== fair table ==")
    for m in METHODS:
        f = fair[m]
        print(f"  {m:9s} FPR={f['fpr']['mean']:.3f}±{f['fpr']['sd']:.3f} recAll={f['recall_all']['mean']:.3f} "
              f"recDec={f['recall_decided']['mean']:.3f} abst={f['abst']['mean']:.3f} selErr={f['sel_err']['mean']:.3f} "
              f"vAbst={f['viol_abstained']['mean']:.3f} vNotAcc={f['viol_not_accepted']['mean']:.3f} ffred={f['ffred_pct']['mean']:.0f}")
    print("== coverage_viol (conformal) ==", {m: v["mean"] for m, v in cov_viol.items()})
    print("== set types ==", set_type)
    print("== cluster-aware ==", {k: (v["mean"] if v else None) for k, v in clus_out.items()})
    print("== images/subject ==", ips_summary)
    print("saved", os.path.join(OUT, OUT_NAME))


if __name__ == "__main__":
    run()
