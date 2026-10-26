"""Primary confidence intervals on ONE fixed subject-disjoint partition (seed 0), with a
subject-cluster bootstrap over the test subjects, plus a 10-partition robustness summary.

This implements the supervisor's final protocol decision:
  * PRIMARY analysis: partition 1 (seed 0). Per-criterion violation-class coverage for
    marginal LAC and Mondrian, with 95% percentile intervals from a cluster bootstrap that
    resamples the 55 TEST SUBJECTS with replacement (images stay with their subject, so
    within-subject dependence is respected by construction).
  * ROBUSTNESS: the same coverage recomputed on 10 repeated subject-disjoint partitions,
    reported as mean, SD and range per criterion.

Run: python3 primary_ci_v3.py [csv] [outdir]   -> results_v3/primary_ci_v3.json
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
import analysis_v2 as A
import analysis_v3 as V3

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "tono_full_pred.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "results_v3")
CRIT, MET = A.CRITERIA, A.METRICS
ALPHA = 0.10
PRIMARY_SEED = 0
ROBUST_SEEDS = list(range(10))
B = 2000                       # bootstrap resamples
rng = np.random.RandomState(42)


def cluster_boot_ci(cov_by_subj, B=B, rng=rng):
    """cov_by_subj: dict subject -> (k_covered, n_viol). Resample subjects with replacement."""
    subs = list(cov_by_subj)
    ks = np.array([cov_by_subj[s][0] for s in subs], float)
    ns = np.array([cov_by_subj[s][1] for s in subs], float)
    stats = []
    for _ in range(B):
        idx = rng.randint(0, len(subs), size=len(subs))
        n = ns[idx].sum()
        if n == 0:
            continue
        stats.append(ks[idx].sum() / n)
    return [round(float(np.percentile(stats, 2.5)), 3), round(float(np.percentile(stats, 97.5)), 3)], len(stats)


if __name__ == "__main__":
    df = A.load_and_prepare(CSV)

    # ---------------- primary: seed 0 with subject-cluster bootstrap ----------------
    tr, ca, te, _ = A.subject_disjoint_split(df, seed=PRIMARY_SEED)
    primary = {}
    for c in CRIT:
        r = V3.fit_all(df, c, tr, ca, te, ALPHA, CRIT + MET)
        if r is None:
            continue
        cls = r["classes"]; yte = r["yte"]; vm = yte == 1; subj = r["subj_te"]
        row = {"n_viol_test": int(vm.sum()),
               "n_subj_with_viol": int(len(np.unique(subj[vm]))),
               "n_pos_calib": int(r["n_pos_calib"])}
        for m in ["lac", "mond"]:
            cov = r[m][:, cls.index(1)][vm].astype(int)   # violation class inside the set
            sv = subj[vm]
            by = {s: (int(cov[sv == s].sum()), int((sv == s).sum())) for s in np.unique(sv)}
            ci, b_eff = cluster_boot_ci(by)
            row[m] = {"cov": round(float(cov.mean()), 3), "ci95_clusterboot": ci,
                      "boot_effective": b_eff}
        row["ci_disjoint"] = bool(row["lac"]["ci95_clusterboot"][1] < row["mond"]["ci95_clusterboot"][0])
        row["lac_upper_below_target"] = bool(row["lac"]["ci95_clusterboot"][1] < 0.90)
        primary[c] = row
        print(f"  {c:24s} LAC {row['lac']['cov']:.3f} {row['lac']['ci95_clusterboot']}  "
              f"MOND {row['mond']['cov']:.3f} {row['mond']['ci95_clusterboot']}  "
              f"nViol={row['n_viol_test']} subj={row['n_subj_with_viol']}"
              f"{'  <-below-target' if row['lac_upper_below_target'] else ''}")

    # ---------------- robustness: 10 repeated partitions ----------------
    rob = {c: {"lac": [], "mond": []} for c in CRIT}
    agg = {"lac": [], "mond": []}   # per-seed mean over criteria
    for seed in ROBUST_SEEDS:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        per_seed = {"lac": [], "mond": []}
        for c in CRIT:
            r = V3.fit_all(df, c, tr, ca, te, ALPHA, CRIT + MET)
            if r is None:
                continue
            cls = r["classes"]; vm = r["yte"] == 1
            for m in ["lac", "mond"]:
                v = float(r[m][:, cls.index(1)][vm].mean())
                rob[c][m].append(round(v, 4)); per_seed[m].append(v)
        for m in ["lac", "mond"]:
            agg[m].append(float(np.mean(per_seed[m])))
    rob_out = {}
    for c in CRIT:
        rob_out[c] = {}
        for m in ["lac", "mond"]:
            v = np.array(rob[c][m])
            if len(v) == 0:
                continue
            rob_out[c][m] = {"mean": round(float(v.mean()), 3), "sd": round(float(v.std()), 3),
                             "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
                             "n_seeds": int(len(v))}
    agg_out = {m: {"mean": round(float(np.mean(agg[m])), 4), "sd": round(float(np.std(agg[m])), 4),
                   "min": round(float(np.min(agg[m])), 4), "max": round(float(np.max(agg[m])), 4)}
               for m in ["lac", "mond"]}

    below = [c for c in CRIT if primary.get(c, {}).get("lac_upper_below_target")]
    disj = [c for c in CRIT if primary.get(c, {}).get("ci_disjoint")]
    out = {"config": {"alpha": ALPHA, "primary_seed": PRIMARY_SEED, "robust_seeds": ROBUST_SEEDS,
                      "bootstrap": {"B": B, "unit": "test subject", "rng_seed": 42,
                                    "note": "subjects resampled with replacement; images stay with their subject"}},
           "primary": primary,
           "summary": {"lac_upper_below_target": below, "n_below": len(below),
                       "ci_disjoint_lac_vs_mond": disj, "n_disjoint": len(disj)},
           "robustness_10seed": rob_out, "robustness_aggregate": agg_out}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "primary_ci_v3.json"), "w"), indent=2)
    print(json.dumps(out["summary"], indent=1))
    print("aggregate:", json.dumps(agg_out))
    print("saved", os.path.join(OUT, "primary_ci_v3.json"))
