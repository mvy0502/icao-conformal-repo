"""Emit results_v2/image_level_v2.json (joint coverage + ACCEPT/REJECT/RETAKE confusion)."""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
import analysis_v2 as A
df=A.load_and_prepare("tono_full_pred.csv")
tr,ca,te,_=A.subject_disjoint_split(df,seed=0)
keep={};active=[]
for c in A.CRITERIA:
    info,held=A.eval_criterion(df,c,tr,ca,te,0.1,0)
    if held is not None: keep[c]=held;active.append(c)
n=len(te)
jl=np.ones(n,bool);jm=np.ones(n,bool);pm=[]
for c in active:
    cls=keep[c]["classes"];yte=keep[c]["yte"]
    cl=keep[c]["sets_lac"][np.arange(n),[cls.index(v) for v in yte]]
    cm=keep[c]["sets_mond"][np.arange(n),[cls.index(v) for v in yte]]
    jl&=cl;jm&=cm;pm.append(cl.mean())
te_folder=df["source_folder"].to_numpy()[te]
has_v=np.array([A.FOLDER_TO_CRIT.get(f) is not None for f in te_folder])
is_zoom=np.array([f=="zoom" for f in te_folder])
dec=[]
for i in range(n):
    anyv=any(keep[c]["sets_lac"][i,keep[c]["classes"].index(1)] and not keep[c]["sets_lac"][i,keep[c]["classes"].index(0)] for c in active)
    allok=all(keep[c]["sets_lac"][i,keep[c]["classes"].index(0)] and not keep[c]["sets_lac"][i,keep[c]["classes"].index(1)] for c in active)
    dec.append("REJECT" if anyv else ("ACCEPT" if allok else "RETAKE"))
dec=np.array(dec)
conf={d:{"total":int((dec==d).sum()),"on_violation_img":int(((dec==d)&has_v).sum()),
         "on_zoom_outofscope":int(((dec==d)&is_zoom).sum())} for d in ["ACCEPT","REJECT","RETAKE"]}
out={"n_test":n,"active_criteria":len(active),
 "mean_per_criterion_marginal_cov":round(float(np.mean(pm)),4),
 "joint_coverage_lac":round(float(jl.mean()),4),"joint_coverage_mondrian":round(float(jm.mean()),4),
 "independence_bound":round(0.9**len(active),4),"bonferroni_per_criterion_target":round(1-0.1/len(active),4),
 "image_confusion":conf,"true_violations_wrongly_accepted":int(((dec=="ACCEPT")&has_v).sum()),
 "total_violation_images":int(has_v.sum())}
json.dump(out,open("results_v2/image_level_v2.json","w"),indent=2)
print("wrote results_v2/image_level_v2.json")
