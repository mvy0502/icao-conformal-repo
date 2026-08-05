"""
conformal_analysis.py — BioGaze tahminlerini (TONO ground-truth ile) MAPIE split
conformal prediction ile kalibre eder.

Üretilenler:
  - Kriter başına ampirik coverage (hedef 1-alpha ile karşılaştırma)
  - ACCEPT / REJECT / RETAKE karar çerçevesi
  - Önlenen gereksiz retake sayısı (BioGaze baseline'a karşı)
  - Özet CSV + grafikler

Sistem Python'u ile çalıştır (mapie/sklearn/pandas burada):
  python3 conformal_analysis.py --csv tono_full_pred.csv --out results
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mapie.classification import SplitConformalClassifier

# --- TONO klasör -> ihlal edilen BioGaze kriteri eşlemesi ---
FOLDER_TO_CRIT = {
    "bkg":  "Uniform_background",
    "cap":  "Head_without_covering",
    "ce":   "Eyes_open",
    "ceg":  "Eyes_open",
    "expos":"Correct_exposure",
    "la_1": "Gaze_in_camera",
    "la_2": "Gaze_in_camera",
    "light":"Uniform_face_lighting",
    "mkup": "No_or_light_makeup",
    "oof":  "In_focus",
    "pixel":"No_pixelation",
    "poster":"No_posterization",
    "sat":  "Correct_saturation",
    "sm":   "Neutral_expression",
    "sun":  "No_sunglasses",
    "tq":   "Frontal_pose",     # head pose (doğrulanacak)
    "zoom": None,               # yüz boyutu — BioGaze'de aktif kriter değil -> tam uyumlu
}

# 14 bireysel kriter (Compliant hariç)
CRITERIA = [
    "Head_without_covering", "Eyes_open", "No_sunglasses", "No_posterization",
    "Gaze_in_camera", "Neutral_expression", "In_focus", "Correct_exposure",
    "No_or_light_makeup", "No_pixelation", "Frontal_pose", "Correct_saturation",
    "Uniform_background", "Uniform_face_lighting",
]
METRICS = ["IED", "pitch", "yaw", "roll", "eyes_open_val", "mouth_open_val", "has_glasses"]


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    # geçerli satırlar: 1 yüz, hata yok, kriterler dolu
    df = df[(df["faces_detected"] == 1) & (df["error"].isna()) & (df["Compliant"].notna())].copy()
    # boolean string -> int
    for c in CRITERIA + ["Compliant"]:
        df[c] = (df[c].astype(str) == "True").astype(int)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.dropna(subset=METRICS).reset_index(drop=True)
    # ground-truth: her kriter için ihlal var mı? (1 = ihlal / non-compliant)
    for c in CRITERIA:
        df["gt_viol_" + c] = df["source_folder"].map(
            lambda f: 1 if FOLDER_TO_CRIT.get(f) == c else 0)
    return df


def build_features(df):
    # Özellik vektörü: 14 ikili BioGaze kararı + 7 sürekli metrik
    X = df[CRITERIA + METRICS].to_numpy(dtype=float)
    return X


def run(csv_path, out_dir, alpha=0.1, min_pos=15, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    df = load_and_prepare(csv_path)
    n = len(df)
    conf = 1 - alpha
    print(f"[analiz] geçerli görüntü: {n} | hedef coverage: {conf:.0%}")

    X = build_features(df)
    idx = np.arange(n)
    # tek seferlik ortak bölme: train/calib/test = 60/20/20 (klasöre göre stratify)
    strat = df["source_folder"].to_numpy()
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed, stratify=strat)
    ca, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=strat[tmp])
    print(f"[analiz] train={len(tr)} calib={len(ca)} test={len(te)}")

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)

    per_crit = {}
    test_sets = {}  # criterion -> array (len test) of sets over {ok=0, viol=1}
    for c in CRITERIA:
        y = df["gt_viol_" + c].to_numpy()
        n_pos = int(y.sum())
        # her split'te iki sınıf da olmalı
        if y[tr].sum() < min_pos or y[ca].sum() < 1 or len(np.unique(y[tr])) < 2:
            per_crit[c] = {"skip": True, "n_pos": n_pos}
            print(f"  [{c}] atlandı (yetersiz pozitif: {n_pos})")
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        scc = SplitConformalClassifier(
            estimator=clf, confidence_level=conf,
            conformity_score="lac", prefit=False, random_state=seed)
        scc.fit(Xs[tr], y[tr])
        scc.conformalize(Xs[ca], y[ca])
        out = scc.predict_set(Xs[te])
        # MAPIE 1.x: predict_set -> (y_pred, y_set) veya y_set
        if isinstance(out, tuple):
            y_pred, y_set = out[0], out[1]
        else:
            y_pred, y_set = None, out
        y_set = np.asarray(y_set)
        if y_set.ndim == 3:      # (n, n_classes, n_conf)
            y_set = y_set[:, :, 0]
        y_set = y_set.astype(bool)
        classes = list(scc._mapie_classifier.classes_) if hasattr(scc, "_mapie_classifier") else [0, 1]
        # sınıf indekslerini bul (0=ok, 1=viol)
        # y_set sütunları sınıf sırasına göre; sınıf etiketlerini al
        try:
            cls = scc.classes_
        except Exception:
            cls = np.array([0, 1])
        cls = np.asarray(cls)
        ok_col = int(np.where(cls == 0)[0][0]) if 0 in cls else None
        viol_col = int(np.where(cls == 1)[0][0]) if 1 in cls else None

        y_te = y[te]
        # ampirik coverage: gerçek etiket sette mi?
        covered = np.array([y_set[i, list(cls).index(y_te[i])] for i in range(len(te))])
        coverage = float(covered.mean())
        viol_mask = (y_te == 1)
        ok_mask = (y_te == 0)
        cov_viol = float(covered[viol_mask].mean()) if viol_mask.sum() > 0 else None
        cov_ok = float(covered[ok_mask].mean()) if ok_mask.sum() > 0 else None
        set_sizes = y_set.sum(axis=1)
        per_crit[c] = {
            "skip": False, "n_pos": n_pos,
            "coverage": coverage, "target": conf,
            "coverage_on_violations": cov_viol,
            "coverage_on_ok": cov_ok,
            "avg_set_size": float(set_sizes.mean()),
            "singleton_rate": float((set_sizes == 1).mean()),
            "empty_set_rate": float((set_sizes == 0).mean()),
        }
        # test görüntüleri için set bilgisini sakla (karar çerçevesi için)
        has_ok = y_set[:, ok_col] if ok_col is not None else np.zeros(len(te), bool)
        has_viol = y_set[:, viol_col] if viol_col is not None else np.zeros(len(te), bool)
        test_sets[c] = {"has_ok": has_ok, "has_viol": has_viol}
        # BioGaze vs conformal gereksiz-red (false-positive) analizi (test kümesi)
        bio_viol = (df[c].to_numpy()[te] == 0)      # BioGaze 'ihlal' dedi (compliant=0)
        conf_viol_only = has_viol & (~has_ok)        # conformal 'kesin ihlal'
        per_crit[c]["n_ok"] = int(ok_mask.sum())
        per_crit[c]["n_viol"] = int(viol_mask.sum())
        per_crit[c]["biogaze_fp"] = int((ok_mask & bio_viol).sum())
        per_crit[c]["conformal_fp"] = int((ok_mask & conf_viol_only).sum())
        per_crit[c]["biogaze_tp"] = int((viol_mask & bio_viol).sum())
        per_crit[c]["conformal_tp"] = int((viol_mask & conf_viol_only).sum())
        print(f"  [{c}] coverage={coverage:.3f} (hedef {conf:.2f}) "
              f"| ort.set={set_sizes.mean():.2f} | pozitif={n_pos}")

    # --- Karar çerçevesi: ACCEPT / REJECT / RETAKE (test kümesi) ---
    active = [c for c in CRITERIA if c in test_sets]
    m = len(te)
    decisions = []
    for i in range(m):
        any_viol_only = any(test_sets[c]["has_viol"][i] and not test_sets[c]["has_ok"][i] for c in active)
        all_ok_only = all(test_sets[c]["has_ok"][i] and not test_sets[c]["has_viol"][i] for c in active)
        if any_viol_only:
            decisions.append("REJECT")
        elif all_ok_only:
            decisions.append("ACCEPT")
        else:
            decisions.append("RETAKE")
    decisions = np.array(decisions)
    dec_counts = {d: int((decisions == d).sum()) for d in ["ACCEPT", "REJECT", "RETAKE"]}

    # --- Retake tasarrufu: BioGaze baseline (Compliant==0 -> retake) ---
    biogaze_reject = (df["Compliant"].to_numpy()[te] == 0)
    truly_compliant = np.array([FOLDER_TO_CRIT.get(f) is None for f in df["source_folder"].to_numpy()[te]])
    baseline_retakes = int(biogaze_reject.sum())
    # gereksiz retake: gerçekte tam uyumlu ama BioGaze reddetmiş
    unnecessary_baseline = int((biogaze_reject & truly_compliant).sum())
    # conformal bunların kaçını ACCEPT ediyor (kurtarıyor)?
    saved = int((biogaze_reject & truly_compliant & (decisions == "ACCEPT")).sum())

    # --- ANA retake metriği: kriter-başına gereksiz-red (false-positive) azaltımı ---
    total_bio_fp = sum(per_crit[c].get("biogaze_fp", 0) for c in active)
    total_conf_fp = sum(per_crit[c].get("conformal_fp", 0) for c in active)
    fp_reduction = int(total_bio_fp - total_conf_fp)
    fp_reduction_pct = round(fp_reduction / total_bio_fp * 100, 1) if total_bio_fp else 0.0
    tot_viol = sum(per_crit[c].get("n_viol", 0) for c in active)
    bio_tp = sum(per_crit[c].get("biogaze_tp", 0) for c in active)
    conf_tp = sum(per_crit[c].get("conformal_tp", 0) for c in active)

    summary = {
        "n_valid": n, "test_n": int(m), "target_coverage": conf,
        "criteria": per_crit,
        "decisions": dec_counts,
        "baseline_biogaze_retakes": baseline_retakes,
        "unnecessary_baseline_retakes": unnecessary_baseline,
        "unnecessary_retakes_prevented": saved,
        "biogaze_false_flags_total": int(total_bio_fp),
        "conformal_false_flags_total": int(total_conf_fp),
        "false_flag_reduction": fp_reduction,
        "false_flag_reduction_pct": fp_reduction_pct,
        "total_true_violations_test": int(tot_viol),
        "biogaze_true_flags": int(bio_tp),
        "conformal_confident_true_flags": int(conf_tp),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # coverage tablosu CSV
    rows = []
    for c in CRITERIA:
        d = per_crit.get(c, {})
        if d.get("skip", True):
            rows.append({"criterion": c, "n_pos": d.get("n_pos", 0), "coverage": None,
                         "coverage_on_violations": None, "target": conf,
                         "avg_set_size": None, "empty_set_rate": None})
        else:
            rows.append({"criterion": c, "n_pos": d["n_pos"], "coverage": round(d["coverage"], 3),
                         "coverage_on_violations": (round(d["coverage_on_violations"], 3)
                                                    if d["coverage_on_violations"] is not None else None),
                         "target": conf, "avg_set_size": round(d["avg_set_size"], 3),
                         "empty_set_rate": round(d["empty_set_rate"], 3)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "coverage_table.csv"), index=False)

    print("\n=== ÖZET ===")
    print("Kararlar:", dec_counts)
    print(f"Baseline (BioGaze) retake: {baseline_retakes} | "
          f"gereksiz: {unnecessary_baseline} | conformal ile önlenen: {saved}")
    print(f"Gereksiz-red (false flag): BioGaze={total_bio_fp} -> conformal={total_conf_fp} "
          f"| AZALMA={fp_reduction} (%{fp_reduction_pct})")
    print(f"Gercek ihlal tespiti (kesin): BioGaze={bio_tp}/{tot_viol} vs conformal={conf_tp}/{tot_viol}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.csv, args.out, alpha=args.alpha, seed=args.seed)
