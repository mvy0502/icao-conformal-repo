"""De-duplicated confidence intervals for the per-criterion violation-class coverage.

The ten subject-disjoint seeds of analysis_v3.py have overlapping test sets (each image is a
test image in about two of the ten seeds), so the pooled Clopper-Pearson intervals treat
correlated observations as independent and are too narrow. Here the unit is the UNIQUE test
image: for each image we average its coverage indicator over the seeds in which it was a test
violation, and then compute the interval on (sum of per-image means, number of unique images).

Run: python3 dedup_ci_v3.py [csv] [outdir]   -> results_v3/dedup_ci_v3.json
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
ALPHA = 0.10; SEEDS = list(range(10))

if __name__ == "__main__":
    df = A.load_and_prepare(CSV)
    nz_all = df["source_folder"].to_numpy() != "zoom"
    per_img = {c: {"lac": {}, "mond": {}} for c in CRIT}   # image index -> list of indicators
    pooled = {c: {"lac": [0, 0], "mond": [0, 0]} for c in CRIT}
    for seed in SEEDS:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        te_nz = te[nz_all[te]]                      # same filtering as fit_all
        for c in CRIT:
            r = V3.fit_all(df, c, tr, ca, te, ALPHA, CRIT + MET)
            if r is None:
                continue
            cls = r["classes"]; yte = r["yte"]; vm = yte == 1
            assert len(yte) == len(te_nz)
            for m in ["lac", "mond"]:
                cov = r[m][:, cls.index(1)]            # violation class inside the set
                for idx, cv in zip(te_nz[vm], cov[vm]):
                    per_img[c][m].setdefault(int(idx), []).append(int(cv))
                pooled[c][m][0] += int(cov[vm].sum()); pooled[c][m][1] += int(vm.sum())
    out = {}
    print(f"{'criterion':24s} {'method':5s} pooled_n uniq_n mult  pooled_CI          dedup_CI")
    for c in CRIT:
        out[c] = {}
        for m in ["lac", "mond"]:
            d = per_img[c][m]
            if not d:
                continue
            means = np.array([np.mean(v) for v in d.values()])
            n_u = len(means); k_u = float(means.sum())
            mult = pooled[c][m][1] / n_u
            lo, hi = V3.clopper_pearson(int(round(k_u)), n_u)
            plo, phi = V3.clopper_pearson(pooled[c][m][0], pooled[c][m][1])
            out[c][m] = {"pooled_n": pooled[c][m][1], "unique_images": n_u,
                         "appearances_per_image": round(mult, 2),
                         "coverage_unique": round(k_u / n_u, 3),
                         "ci95_pooled": [round(plo, 3), round(phi, 3)],
                         "ci95_dedup": [round(lo, 3), round(hi, 3)]}
            print(f"{c:24s} {m:5s} {pooled[c][m][1]:8d} {n_u:6d} {mult:4.2f}  [{plo:.3f},{phi:.3f}]  [{lo:.3f},{hi:.3f}]")
        if "lac" in out[c] and "mond" in out[c]:
            out[c]["disjoint_dedup"] = bool(out[c]["lac"]["ci95_dedup"][1] < out[c]["mond"]["ci95_dedup"][0])
            out[c]["lac_upper_dedup_below_0.90"] = bool(out[c]["lac"]["ci95_dedup"][1] < 0.90)
    below = [c for c in CRIT if out[c].get("lac_upper_dedup_below_0.90")]
    disj = [c for c in CRIT if out[c].get("disjoint_dedup")]
    summary = {"lac_dedup_upper_below_0.90": below, "n_below": len(below),
               "dedup_disjoint_lac_vs_mond": disj, "n_disjoint": len(disj),
               "mean_appearances_per_image": round(float(np.mean([out[c]["lac"]["appearances_per_image"] for c in CRIT if "lac" in out[c]])), 2)}
    json.dump({"per_criterion": out, "summary": summary}, open(os.path.join(OUT, "dedup_ci_v3.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))
