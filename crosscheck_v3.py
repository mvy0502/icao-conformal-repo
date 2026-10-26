"""Two checks behind sentences in the paper that had no stored evidence.

1. Overall (both-class) LAC coverage per criterion. The paper says the marginal method fails
   on the violation class "on a criterion whose overall coverage is on target"; this stores
   that overall coverage on the primary partition and over the ten robustness partitions.
2. Agreement with MAPIE. Our split conformal layer is written from scratch; here MAPIE's
   SplitConformalClassifier (LAC score, prefit model) is run on the same fitted model,
   calibration set and test set, and the prediction sets are compared one by one.

Run: python crosscheck_v3.py [csv] [outdir]   -> results_v3/crosscheck_v3.json
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from mapie.classification import SplitConformalClassifier
import analysis_v2 as A
import analysis_v3 as V3

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "tono_full_pred.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "results_v3")
CRIT, MET = A.CRITERIA, A.METRICS
ALPHA = 0.10
SEEDS = list(range(10))        # seed 0 = primary partition (partition 1)


def mapie_lac_sets(r):
    """MAPIE's LAC sets for the model fit_all already trained (no refit)."""
    m = SplitConformalClassifier(estimator=r["clf"], confidence_level=1 - ALPHA, conformity_score="lac", prefit=True)
    m.conformalize(r["X_ca"], r["yca"])
    _, sets = m.predict_set(r["X_te"])
    return sets[:, :, 0]       # one confidence level -> (n_test, n_classes)


if __name__ == "__main__":
    df = A.load_and_prepare(CSV)
    overall = {c: [] for c in CRIT}
    agree = {c: [] for c in CRIT}            # share of test images with identical sets
    cov_diff = {c: [] for c in CRIT}         # our coverage minus MAPIE coverage
    for seed in SEEDS:
        tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
        for c in CRIT:
            r = V3.fit_all(df, c, tr, ca, te, ALPHA, CRIT + MET)
            if r is None:
                continue
            ours = V3.set_metrics(r["lac"], r["yte"], r["classes"])["coverage"]
            theirs_sets = mapie_lac_sets(r)
            theirs = V3.set_metrics(theirs_sets, r["yte"], r["classes"])["coverage"]
            overall[c].append(ours)
            agree[c].append(float((theirs_sets == r["lac"]).all(axis=1).mean()))
            cov_diff[c].append(ours - theirs)

    summ = lambda v: {"mean": round(float(np.mean(v)), 3), "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3)}
    res = {
        "config": {"alpha": ALPHA, "seeds": SEEDS, "primary_seed": 0, "features": "14 decisions + 7 metrics",
                   "mapie": "SplitConformalClassifier(conformity_score='lac', prefit=True)"},
        "lac_overall_coverage": {c: {"primary": round(v[0], 3), "ten_partitions": summ(v)} for c, v in overall.items() if v},
        "mapie_agreement": {
            "per_criterion": {c: {"identical_sets": summ(agree[c]), "coverage_diff": summ(cov_diff[c])} for c in CRIT if agree[c]},
            "max_abs_coverage_diff": round(float(max(abs(d) for v in cov_diff.values() for d in v)), 4),
            "min_identical_sets": round(float(min(a for v in agree.values() for a in v)), 4),
        },
    }
    with open(os.path.join(OUT, "crosscheck_v3.json"), "w", newline="\n") as f:
        json.dump(res, f, indent=2)
    print("max |coverage diff| vs MAPIE:", res["mapie_agreement"]["max_abs_coverage_diff"],
          "| min share of identical sets:", res["mapie_agreement"]["min_identical_sets"])
    print("No_posterization overall LAC coverage:", res["lac_overall_coverage"]["No_posterization"])
    print("saved", os.path.join(OUT, "crosscheck_v3.json"))
