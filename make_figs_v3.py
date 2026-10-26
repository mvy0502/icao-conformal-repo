"""Journal-size figures for the v4 analysis: feature ablation (Fig 2), per-criterion CIs (Fig 3), risk-coverage (Fig 4).
Style and sizes come from fig_style.py; claims live in the captions, not in figure titles."""
import json, os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import analysis_v2 as A
import analysis_v3 as V3
import fig_style as S
from fig_style import INK, TEAL, AMBER, RED, GREY, PURPLE

S.apply()
BASE=os.path.dirname(os.path.abspath(__file__))
OUT=sys.argv[2] if len(sys.argv)>2 else os.path.join(BASE,"results_v3")
PRED=sys.argv[1] if len(sys.argv)>1 else os.path.join(BASE,"tono_full_pred.csv")
s=json.load(open(f"{OUT}/analysis_v3.json"))

# ---------- Fig 2: feature ablation (guarantee holds, accuracy + efficiency collapse) ----------
abl=s["feature_ablation"]; order=["full","own+metrics","metrics"]
labels=["Full\n(14 dec. + 7 metrics)","Own decision\n+ metrics","Metrics\nonly"]
x=np.arange(3)
fig,(ax1,ax2)=S.figure(S.DOUBLE,62,ncols=2)
# left: accuracy (logistic FPR) collapses while coverage holds
ax1.plot(x,[abl[k]["log_fpr"]["mean"] for k in order],"-o",color=RED,label="Logistic FPR (accuracy)")
ax1.plot(x,[abl[k]["lac_cov"]["mean"] for k in order],"-s",color=TEAL,label="LAC overall coverage")
ax1.plot(x,[abl[k]["mond_viol"]["mean"] for k in order],"-^",color=PURPLE,label="Mondrian violation coverage")
ax1.axhline(0.90,color=GREY,ls="--",lw=0.9)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylim(0,1.02); ax1.set_ylabel("Rate")
ax1.legend(loc="center left",bbox_to_anchor=(0,0.55))
for xi,k,off in zip(x,order,[(6,3),(6,-10),(6,-4)]):   # offsets keep each value clear of the rising FPR line
    ax1.annotate(f"{abl[k]['log_fpr']['mean']:.2f}",(xi,abl[k]["log_fpr"]["mean"]),textcoords="offset points",xytext=off,ha="left",fontsize=8.5,color=RED)
    ax1.annotate(f"{abl[k]['mond_viol']['mean']:.2f}",(xi,abl[k]["mond_viol"]["mean"]),textcoords="offset points",xytext=(0,6),ha="center",fontsize=8.5,color=PURPLE)
ax1.set_xlim(-0.35,2.35)
ax1.set_axisbelow(True); ax1.grid(True,axis="y")
# right: the catch: Mondrian set size / both-class rate inflate (hatching keeps the pair apart in greyscale)
w=0.38
b1=ax2.bar(x-w/2,[abl[k]["mond_setsize"]["mean"] for k in order],w,color=TEAL,label="Mondrian average set size",zorder=3)
b2=ax2.bar(x+w/2,[abl[k]["mond_both"]["mean"] for k in order],w,color=AMBER,hatch="////",edgecolor="white",lw=0,label="Both-class (ambiguous) rate",zorder=3)
for bars in (b1,b2):
    for b in bars: ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.03,f"{b.get_height():.2f}",ha="center",fontsize=8.5,color=INK)
ax2.axhline(1.0,color=GREY,ls="--",lw=0.9)
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylim(0,2.0)
ax2.legend(loc="upper left")
ax2.set_axisbelow(True); ax2.grid(True,axis="y",zorder=0)
S.save(fig,OUT,"fig_ablation")

# ---------- Fig 3: per-criterion violation coverage with CIs (LAC vs Mondrian) ----------
# primary partition estimates with subject-cluster bootstrap CIs (see primary_ci_v3.py)
pj=json.load(open(f"{OUT}/primary_ci_v3.json"))
dd=pj["primary"]; rb=pj["robustness_10seed"]
crits=[c for c in A.CRITERIA if c in dd]
crits.sort(key=lambda c: dd[c]["lac"]["cov"])
small=lambda c: dd[c]["n_viol_test"]<20 or dd[c]["n_subj_with_viol"]<15   # flagged with * as in Table 2
y=np.arange(len(crits)); h=0.36
fig,ax=S.figure(S.SINGLE,118)
for i,c in enumerate(crits):
    for m,off,col,mk in [("lac",+h/2,GREY,"s"),("mond",-h/2,TEAL,"o")]:
        cov=dd[c][m]["cov"]; ci=dd[c][m]["ci95_clusterboot"]
        r=rb[c][m]
        # robustness range: thin light band with a tick at the 10-partition mean
        ax.plot([r["min"],r["max"]],[i+off]*2,color=col,alpha=0.30,lw=4.5,solid_capstyle="butt",zorder=1)
        ax.plot([r["mean"]]*2,[i+off-0.13,i+off+0.13],color=col,lw=1.1,alpha=0.9,zorder=2)
        ax.errorbar(cov,i+off,xerr=[[cov-ci[0]],[ci[1]-cov]],fmt=mk,color=col,ms=3.8,capsize=2,lw=0.9,zorder=3)
ax.axvline(0.90,color=RED,ls="--",lw=1.0,zorder=2)
ax.set_yticks(y); ax.set_yticklabels([c.replace("_"," ").capitalize()+("*" if small(c) else "") for c in crits])
ax.set_ylim(-0.6,len(crits)-0.4)
ax.set_xlabel("Coverage on the violation class")
ax.set_xlim(0.22,1.02)
leg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=GREY,ms=6,label="Marginal LAC, primary partition (95% CI)"),
     Line2D([0],[0],marker="o",color="w",markerfacecolor=TEAL,ms=6,label="Mondrian, primary partition (95% CI)"),
     Line2D([0],[0],color=GREY,alpha=0.4,lw=4.5,label="Range over ten partitions (tick: mean)"),
     Line2D([0],[0],color=RED,ls="--",lw=1.0,label="0.90 target")]
fig.legend(handles=leg,loc="outside lower center",ncol=1)
S.save(fig,OUT,"fig_percrit_ci")

# ---------- Fig 4: risk-coverage, isotonic threshold vs conformal (accuracy parity) ----------
# conformal confident-singleton FPR-recall as alpha varies (seed 0, full features, zoom excluded)
df=A.load_and_prepare(PRED)
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
fig,ax=S.figure(S.SINGLE,72)
ax.plot(iso["fpr"],iso["recall"],"-",color=GREY,lw=1.6,label="Isotonic-calibrated threshold")
ax.plot(lac_pts["fpr"],lac_pts["rec"],"-s",color=AMBER,lw=1.2,label="LAC (confident REJECT)")
ax.plot(mon_pts["fpr"],mon_pts["rec"],"-o",color=TEAL,lw=1.2,label="Mondrian (confident REJECT)")
# mark where each sweep starts; the legend title gives the range, so the hook at small alpha reads without a note
ax.annotate(f"α = {alphas[0]:.2f}",(lac_pts["fpr"][0],lac_pts["rec"][0]),textcoords="offset points",xytext=(6,-11),fontsize=8.5,color=AMBER)
ax.annotate(f"α = {alphas[0]:.2f}",(mon_pts["fpr"][0],mon_pts["rec"][0]),textcoords="offset points",xytext=(8,-4),va="top",fontsize=8.5,color=TEAL)
ax.set_xlabel("False-positive rate"); ax.set_ylabel("Recall on true violations")
ax.set_xlim(0,0.13); ax.set_ylim(0.4,1.0)
ax.set_axisbelow(True); ax.grid(True)
ax.legend(loc="lower right",title=f"Conformal sweeps: α = {alphas[0]:.2f} to {alphas[-1]:.2f}",title_fontsize=8.5)
S.save(fig,OUT,"fig_riskcoverage")
print("DONE v3 figures")
