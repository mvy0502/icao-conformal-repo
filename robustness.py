"""Emit results_v2/robustness_v2.json (5 subject-disjoint splits)."""
import numpy as np, json, io, contextlib, warnings; warnings.filterwarnings("ignore")
import analysis_v2 as A
res={k:[] for k in ["marg_cov","lac_viol","mond_viol","raw_fpr","log_fpr","thr_fpr","lac_fpr","mond_fpr",
     "raw_recall","log_recall","thr_recall","lac_recall","mond_recall",
     "log_ffred","thr_ffred","lac_ffred","mond_ffred"]}
for seed in range(5):
    with contextlib.redirect_stdout(io.StringIO()):
        s=A.run("tono_full_pred.csv",f"/tmp/rob_{seed}",alpha=0.1,seed=seed)
    crit=s["criteria"];active=[c for c,d in crit.items() if not d.get("skip")]
    res["marg_cov"].append(float(np.mean([crit[c]["methods"]["lac_marginal"]["coverage"] for c in active])))
    res["lac_viol"].append(float(np.nanmean([crit[c]["methods"]["lac_marginal"]["coverage_viol"] or np.nan for c in active])))
    res["mond_viol"].append(float(np.nanmean([crit[c]["methods"]["mondrian"]["coverage_viol"] or np.nan for c in active])))
    mt=s["method_totals"]
    for m,t in [("raw_biogaze","raw"),("logistic_only","log"),("threshold_abstain","thr"),("lac_marginal","lac"),("mondrian","mond")]:
        if t+"_fpr" in res: res[t+"_fpr"].append(mt[m]["fpr_overall"])
        if t+"_recall" in res: res[t+"_recall"].append(mt[m]["recall_overall"])
        if t+"_ffred" in res: res[t+"_ffred"].append(mt[m]["false_flag_reduction_pct_vs_raw"])
out={k:{"mean":round(float(np.mean(v)),4),"std":round(float(np.std(v)),4),
        "values":[round(float(x),4) for x in v]} for k,v in res.items()}
out["_note"]="5 subject-disjoint splits (seeds 0-4), alpha=0.10. Scratch LAC matches MAPIE within 0.003."
json.dump(out,open("results_v2/robustness_v2.json","w"),indent=2)
print("wrote results_v2/robustness_v2.json")
