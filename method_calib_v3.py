"""v3 canonical 6-method comparison + achieved-vs-nominal coverage -> results_v3/method_and_calib_v3.json"""
import os, sys, numpy as np, warnings, json; warnings.filterwarnings("ignore")
import analysis_v2 as A, analysis_v3 as V3
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "tono_full_pred.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "results_v3"); os.makedirs(OUT, exist_ok=True)
if __name__ == "__main__":
    df=A.load_and_prepare(CSV); CRIT,MET=A.CRITERIA,A.METRICS
    rows={m:{"fp":[],"rec":[]} for m in ["raw","logistic","threshold","lac","mond","aps"]}
    for seed in range(5):
        tr,ca,te,_=A.subject_disjoint_split(df,seed=seed); rng=np.random.RandomState(2000+seed)
        acc={m:{"fp":0,"tp":0,"nok":0,"nv":0} for m in rows}
        for c in CRIT:
            r=V3.fit_all(df,c,tr,ca,te,0.1,CRIT+MET,rng=rng)
            if r is None: continue
            yte=r["yte"]; cls=r["classes"]; om=(yte==0); vm=(yte==1); p1=r["p_te"][:,cls.index(1)]
            def add(m,f): acc[m]["fp"]+=int((om&f).sum()); acc[m]["tp"]+=int((vm&f).sum()); acc[m]["nok"]+=int(om.sum()); acc[m]["nv"]+=int(vm.sum())
            add("raw",r["bio_flag"]==1); add("logistic",p1>=0.5); add("threshold",p1>=0.65)
            for tag,key in [("lac","lac"),("mond","mond"),("aps","aps")]:
                st=r[key]; add(tag, st[:,cls.index(1)]&~st[:,cls.index(0)])
        for m in rows: rows[m]["fp"].append(acc[m]["fp"]/acc[m]["nok"]); rows[m]["rec"].append(acc[m]["tp"]/acc[m]["nv"])
    methods={m:{"fpr":round(float(np.mean(v["fp"])),3),"recall":round(float(np.mean(v["rec"])),3)} for m,v in rows.items()}
    ach={}
    for al,tgt in [(0.05,0.95),(0.10,0.90),(0.20,0.80)]:
        covs=[]
        for seed in range(5):
            tr,ca,te,_=A.subject_disjoint_split(df,seed=seed); cc=[]
            for c in CRIT:
                r=V3.fit_all(df,c,tr,ca,te,al,CRIT+MET)
                if r is None: continue
                cc.append(V3.set_metrics(r["lac"],r["yte"],r["classes"])["coverage"])
            covs.append(np.mean(cc))
        ach[str(tgt)]=round(float(np.mean(covs)),3)
    json.dump({"method_comparison_5seed":methods,"achieved_vs_nominal":ach},
              open(os.path.join(OUT,"method_and_calib_v3.json"),"w"),indent=2)
    print("saved", os.path.join(OUT,"method_and_calib_v3.json"))
