"""Figures for the hardened v3 analysis: feature ablation, per-criterion CIs, risk-coverage."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis_v2 as A
import analysis_v3 as V3

INK="#1f2933"; TEAL="#0f9d8f"; AMBER="#e0a100"; RED="#c92a2a"; GREY="#adb5bd"; PURPLE="#6741d9"
OUT="/home/claude/results_v3"
s=json.load(open(f"{OUT}/analysis_v3.json"))

# ---------- FIG 1: feature ablation (guarantee holds, accuracy + efficiency collapse) ----------
abl=s["feature_ablation"]; order=["full","own+metrics","metrics"]
labels=["Full\n(14 dec.+7 metrics)","Own dec.\n+ metrics","Metrics\nonly"]
x=np.arange(3)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,5.0),dpi=150)
# left: accuracy (logistic FPR) collapses while coverage holds
ax1.plot(x,[abl[k]["log_fpr"]["mean"] for k in order],"-o",color=RED,lw=2,ms=7,label="Logistic FPR (accuracy)")
ax1.plot(x,[abl[k]["lac_cov"]["mean"] for k in order],"-s",color=TEAL,lw=2,ms=7,label="LAC overall coverage")
ax1.plot(x,[abl[k]["mond_viol"]["mean"] for k in order],"-^",color=PURPLE,lw=2,ms=7,label="Mondrian violation coverage")
ax1.axhline(0.90,color=GREY,ls="--",lw=1.2)
ax1.set_xticks(x); ax1.set_xticklabels(labels,fontsize=8.5)
ax1.set_ylim(0,1.02); ax1.set_ylabel("Rate", fontsize=11, color=INK)
ax1.set_title("Coverage guarantee is feature-agnostic;\naccuracy is not", fontsize=11.5, fontweight="bold", color=INK, loc="left")
ax1.legend(loc="center left", frameon=False, fontsize=8.5)
for sp in ["top","right"]: ax1.spines[sp].set_visible(False)
ax1.grid(True,axis="y",color="#eef1f4")
# right: the honest catch — Mondrian set size / both-class rate inflate
w=0.38
ax2.bar(x-w/2,[abl[k]["mond_setsize"]["mean"] for k in order],w,color=TEAL,label="Mondrian avg set size")
ax2.bar(x+w/2,[abl[k]["mond_both"]["mean"] for k in order],w,color=AMBER,label="both-class (ambiguous) rate")
ax2.axhline(1.0,color=GREY,ls="--",lw=1.0)
ax2.set_xticks(x); ax2.set_xticklabels(labels,fontsize=8.5)
ax2.set_ylim(0,2.0)
ax2.set_title("But weak features hold coverage only\nby inflating set size (ambiguity)", fontsize=11.5, fontweight="bold", color=INK, loc="left")
ax2.legend(loc="upper left", frameon=False, fontsize=8.5)
for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
ax2.grid(True,axis="y",color="#eef1f4")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_ablation.png",bbox_inches="tight",facecolor="white")
plt.savefig(f"{OUT}/fig_ablation.svg",bbox_inches="tight",facecolor="white")
print("saved fig_ablation")

# ---------- FIG 2: per-criterion violation coverage with CIs (LAC vs Mondrian) ----------
pc=s["per_criterion"]
crits=[c for c in A.CRITERIA]
# sort by LAC violation coverage ascending
crits.sort(key=lambda c: pc[c]["lac_viol"])
y=np.arange(len(crits)); h=0.36
fig,ax=plt.subplots(figsize=(9.0,7.2),dpi=150)
for i,c in enumerate(crits):
    d=pc[c]
    lac=d["lac_viol"]; lci=d["lac_viol_ci95"]; mon=d["mond_viol"]; mci=d["mond_viol_ci95"]
    deg=d["degenerate_flag"]
    ax.errorbar(lac,i+h/2,xerr=[[lac-lci[0]],[lci[1]-lac]],fmt="s",color=GREY,ms=6,capsize=3,lw=1.4)
    ax.errorbar(mon,i-h/2,xerr=[[mon-mci[0]],[mci[1]-mon]],fmt="o",color=TEAL,ms=6,capsize=3,lw=1.4)
    if deg:
        ax.text(1.005,i,"small n",fontsize=7.5,color=RED,va="center")
ax.axvline(0.90,color=RED,ls="--",lw=1.4)
ax.set_yticks(y); ax.set_yticklabels([c.replace("_"," ") for c in crits],fontsize=9)
ax.set_xlabel("Coverage on the violation class (pooled over 10 subject-disjoint seeds, 95% CI)", fontsize=10.5, color=INK)
ax.set_xlim(0.25,1.06)
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=GREY,ms=8,label="Marginal LAC"),
     Line2D([0],[0],marker="o",color="w",markerfacecolor=TEAL,ms=8,label="Mondrian"),
     Line2D([0],[0],color=RED,ls="--",label="0.90 target")]
ax.legend(handles=leg,loc="lower right",frameon=False,fontsize=9.5)
ax.set_title("Marginal conformal under-covers the violation class on several criteria;\nMondrian restores it (Clopper-Pearson 95% CIs)",
             fontsize=12.0, fontweight="bold", color=INK, loc="left", pad=12)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_percrit_ci.png",bbox_inches="tight",facecolor="white")
plt.savefig(f"{OUT}/fig_percrit_ci.svg",bbox_inches="tight",facecolor="white")
print("saved fig_percrit_ci")

# ---------- FIG 3: risk-coverage — isotonic threshold vs conformal (accuracy parity) ----------
# conformal confident-singleton FPR-recall as alpha varies (seed 0, full features, zoom excluded)
df=A.load_and_prepare("/home/claude/tono_full_pred.csv")
tr,ca,te,_=A.subject_disjoint_split(df,seed=0)
alphas=[0.30,0.20,0.15,0.10,0.07,0.05,0.03]
lac_pts={"fpr":[],"rec":[]}; mon_pts={"fpr":[],"rec":[]}
for al in alphas:
    for tag,store in [("lac",lac_pts),("mond",mon_pts)]:
        fp=tp=nok=nv=0
        for c in A.CRITERIA:
            r=V3.fit_all(df,c,tr,ca,te,al,A.CRITERIA+A.METRICS)
            if r is None: continue
            sets=r[tag]; cls=r["classes"]; yte=r["yte"]
            conf_v=sets[:,cls.index(1)] & ~sets[:,cls.index(0)]
            om=(yte==0); vm=(yte==1)
            fp+=int((om&conf_v).sum()); tp+=int((vm&conf_v).sum()); nok+=int(om.sum()); nv+=int(vm.sum())
        store["fpr"].append(fp/nok); store["rec"].append(tp/nv)
iso=s["isotonic_riskcoverage"]
fig,ax=plt.subplots(figsize=(7.8,6.0),dpi=150)
ax.plot(iso["fpr"],iso["recall"],"-",color=GREY,lw=2,label="Isotonic-calibrated threshold (baseline)")
ax.plot(lac_pts["fpr"],lac_pts["rec"],"-s",color=AMBER,lw=1.8,ms=6,label="LAC conformal (confident REJECT)")
ax.plot(mon_pts["fpr"],mon_pts["rec"],"-o",color=TEAL,lw=1.8,ms=6,label="Mondrian conformal (confident REJECT)")
ax.set_xlabel("False-positive rate (true-OK flagged)", fontsize=11, color=INK)
ax.set_ylabel("Recall on true violations", fontsize=11, color=INK)
ax.set_xlim(0,0.5); ax.set_ylim(0.4,1.0)
ax.set_title("Conformal matches a calibrated baseline on accuracy;\nits added value is the distribution-free guarantee, not the curve",
             fontsize=11.8, fontweight="bold", color=INK, loc="left", pad=12)
ax.grid(True,color="#eef1f4")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_riskcoverage.png",bbox_inches="tight",facecolor="white")
plt.savefig(f"{OUT}/fig_riskcoverage.svg",bbox_inches="tight",facecolor="white")
print("saved fig_riskcoverage")
print("DONE v3 figures")
