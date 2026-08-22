"""v3-consistent image-level decisions + Bonferroni joint coverage."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import analysis_v2 as A
import analysis_v3 as V3
OUT = "/home/claude/results_v3"
df = A.load_and_prepare("/home/claude/tono_full_pred.csv")
CRIT, MET = A.CRITERIA, A.METRICS
ALPHA = 0.10; M = len(CRIT)

def sets_on_full_test(df, c, tr, ca, te, alpha):
    """Fit per-criterion classifier on non-zoom train/calib, but return sets on the FULL
    test set (including zoom, which is out-of-scope) for image-level decisions."""
    nz = df["source_folder"].to_numpy() != "zoom"
    trf, caf = tr[nz[tr]], ca[nz[ca]]
    y = df["gt_viol_" + c].to_numpy()
    ytr, yca = y[trf], y[caf]
    if ytr.sum() < 5 or len(np.unique(ytr)) < 2 or yca.sum() < 1:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = df[CRIT + MET].to_numpy(float)
    sc = StandardScaler().fit(X[trf]); Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xs[trf], ytr)
    cls = list(clf.classes_)
    p_ca, p_te = clf.predict_proba(Xs[caf]), clf.predict_proba(Xs[te])
    s_ca = 1 - p_ca[np.arange(len(yca)), yca]
    q = V3.cq_exact(s_ca, alpha)
    qc = np.array([V3.cq_exact(s_ca[yca == 0], alpha), V3.cq_exact(s_ca[yca == 1], alpha)])
    return {"lac": (1 - p_te) <= np.array([q, q]), "mond": (1 - p_te) <= qc,
            "cls": cls, "yte": y[te]}

# ---- image-level decisions (seed 0), full test incl zoom ----
tr, ca, te, _ = A.subject_disjoint_split(df, seed=0)
held = {}
for c in CRIT:
    r = sets_on_full_test(df, c, tr, ca, te, ALPHA)
    if r is not None:
        held[c] = r
active = list(held)
n = len(te)
te_folder = df["source_folder"].to_numpy()[te]
has_inscope = np.array([A.FOLDER_TO_CRIT.get(f) is not None for f in te_folder])
is_zoom = np.array([f == "zoom" for f in te_folder])
img = {}
for m in ["lac", "mond"]:
    dec = []
    for i in range(n):
        anyv = any(held[c][m][i, held[c]["cls"].index(1)] and not held[c][m][i, held[c]["cls"].index(0)] for c in active)
        allok = all(held[c][m][i, held[c]["cls"].index(0)] and not held[c][m][i, held[c]["cls"].index(1)] for c in active)
        dec.append("REJECT" if anyv else ("ACCEPT" if allok else "RETAKE"))
    dec = np.array(dec)
    img[m] = {"ACCEPT": int((dec == "ACCEPT").sum()), "REJECT": int((dec == "REJECT").sum()),
              "RETAKE": int((dec == "RETAKE").sum()),
              "violations_wrongly_accepted": int(((dec == "ACCEPT") & has_inscope).sum()),
              "total_violation_images": int(has_inscope.sum()),
              "zoom_accept": int(((dec == "ACCEPT") & is_zoom).sum()),
              "zoom_total": int(is_zoom.sum())}

# ---- Bonferroni joint coverage over 5 seeds (non-zoom per-criterion test) ----
def joint_cov(alpha, seed):
    tr, ca, te, _ = A.subject_disjoint_split(df, seed=seed)
    res = {}
    for m in ["lac", "mond"]:
        held2 = {}
        for c in CRIT:
            r = V3.fit_all(df, c, tr, ca, te, alpha, CRIT + MET)
            if r is not None:
                held2[c] = r
        act = list(held2)
        nn = len(held2[act[0]]["yte"])
        jc = np.ones(nn, bool); retake = np.zeros(nn, bool); sizes = []
        for c in act:
            h = held2[c]; cls = h["cls"] if "cls" in h else h["classes"]
            cov = h["mond" if m == "mond" else "lac"][np.arange(nn), [cls.index(v) for v in h["yte"]]]
            jc &= cov
            st = h["mond" if m == "mond" else "lac"]
            sizes.append(st.sum(1)); has_ok = st[:, cls.index(0)]; has_v = st[:, cls.index(1)]
            retake |= (has_ok & has_v) | (st.sum(1) == 0)
        res[m] = (float(jc.mean()), float(np.mean(sizes)), float(retake.mean()))
    return res

joint = {"uncorrected": {"lac": [], "mond": []}, "bonferroni": {"lac": [], "mond": []},
         "bonf_retake": {"lac": [], "mond": []}}
for seed in range(5):
    u = joint_cov(ALPHA, seed); b = joint_cov(ALPHA / M, seed)
    for m in ["lac", "mond"]:
        joint["uncorrected"][m].append(u[m][0]); joint["bonferroni"][m].append(b[m][0])
        joint["bonf_retake"][m].append(b[m][2])
joint_summary = {k: {m: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                     for m, v in d.items()} for k, d in joint.items()}

out = {"image_level_seed0": img, "joint_coverage": joint_summary}
json.dump(out, open(f"{OUT}/image_joint_v3.json", "w"), indent=2)
print("image-level:", json.dumps(img, indent=2))
print("joint:", json.dumps(joint_summary, indent=2))
