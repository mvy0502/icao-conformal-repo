"""Regenerate paper figures from the subject-disjoint (v2) analysis."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis_v2 as A

INK="#1f2933"; TEAL="#0f9d8f"; AMBER="#e0a100"; RED="#c92a2a"; GREY="#adb5bd"; PURPLE="#6741d9"
OUT="/home/claude/results_v2"

s = json.load(open(f"{OUT}/summary_v2.json"))
rob = json.load(open(f"{OUT}/robustness_v2.json"))
crit = s["criteria"]
active = [c for c in A.CRITERIA if not crit[c].get("skip")]
short = {c: c.replace("_"," ").replace("Correct ","").replace("Uniform ","Unif.") for c in active}

# ---------- FIG 2: violation-class coverage, marginal LAC vs Mondrian ----------
lac_v = [crit[c]["methods"]["lac_marginal"]["coverage_viol"] for c in active]
mon_v = [crit[c]["methods"]["mondrian"]["coverage_viol"] for c in active]
order = np.argsort(lac_v)
acts=[active[i] for i in order]; lv=[lac_v[i] for i in order]; mv=[mon_v[i] for i in order]
y=np.arange(len(acts)); h=0.38
fig,ax=plt.subplots(figsize=(8.8,7.0),dpi=150)
ax.barh(y+h/2, lv, h, color=GREY, label="Marginal LAC")
ax.barh(y-h/2, mv, h, color=TEAL, label="Mondrian (label-conditional)")
ax.axvline(0.90, color=RED, ls="--", lw=1.5, label="0.90 target")
ax.set_yticks(y); ax.set_yticklabels([short[c] for c in acts], fontsize=9)
ax.set_xlabel("Coverage on the VIOLATION class (test)", fontsize=11, color=INK)
ax.set_xlim(0,1.05); ax.set_ylim(-0.7, len(acts)-0.3)
ax.set_title("Marginal conformal under-covers true violations;\nMondrian restores the guarantee on the costly class",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=14)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
# legend OUTSIDE the axes (below), horizontal — never overlaps the bars
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
          frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_violation_coverage.png", bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT}/fig_violation_coverage.svg", bbox_inches="tight", facecolor="white")
print("saved fig_violation_coverage")

# ---------- FIG 3: method comparison — FPR vs violation recall ----------
mt=s["method_totals"]
meths=[("raw_biogaze","Raw BioGaze",RED,"o"),
       ("logistic_only","Logistic only",AMBER,"s"),
       ("threshold_abstain","Threshold+abstain",PURPLE,"^"),
       ("lac_marginal","LAC conformal",GREY,"D"),
       ("mondrian","Mondrian conformal",TEAL,"*")]
fig,ax=plt.subplots(figsize=(8.2,6.2),dpi=150)
# label offsets tuned per-point so text never overlaps a marker or the title
off={"raw_biogaze":(-10,-18),"logistic_only":(10,8),"threshold_abstain":(10,8),
     "lac_marginal":(12,-4),"mondrian":(12,6)}
ha={"raw_biogaze":"right"}
for key,lab,col,mk in meths:
    x=mt[key]["fpr_overall"]; yv=mt[key]["recall_overall"]
    ax.scatter(x,yv,s=260 if mk=="*" else 140,color=col,marker=mk,zorder=3,edgecolor="white",lw=1.2)
    ax.annotate(lab,(x,yv),textcoords="offset points",xytext=off[key],
                fontsize=9.5,color=INK,ha=ha.get(key,"left"))
ax.set_xlabel("False-positive rate  (true-OK flagged as violation)", fontsize=11, color=INK)
ax.set_ylabel("Recall on true violations", fontsize=11, color=INK)
ax.set_xlim(0.015, 0.175); ax.set_ylim(0.80, 0.98)
ax.set_title("The false-flag / detection trade-off\n(subject-disjoint test, 14 ICAO criteria)",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=12)
ax.grid(True,color="#e3e8ee")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
# "better" indicator in the empty upper band, clear of every marker and label
ax.text(0.105, 0.975, "← better  (lower FPR, higher recall)",
        fontsize=8.8, color=GREY, style="italic", ha="left", va="center")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_method_tradeoff.png", bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT}/fig_method_tradeoff.svg", bbox_inches="tight", facecolor="white")
print("saved fig_method_tradeoff")

# ---------- FIG 4: robustness — coverage across 5 subject-disjoint splits ----------
fig,ax=plt.subplots(figsize=(7.6,5.4),dpi=150)
groups=[("marg_cov","Marginal\n(overall)",GREY),
        ("lac_viol","LAC\n(violation)",AMBER),
        ("mond_viol","Mondrian\n(violation)",TEAL)]
for i,(k,lab,col) in enumerate(groups):
    vals=rob[k]["values"]
    ax.scatter([i]*len(vals),vals,color=col,s=70,zorder=3,alpha=0.85)
    ax.errorbar(i,rob[k]["mean"],yerr=rob[k]["std"],fmt="_",color=INK,ms=34,lw=2,capsize=8,zorder=4)
    ax.annotate(f"{rob[k]['mean']:.3f}",(i,rob[k]['mean']),textcoords="offset points",
                xytext=(16,-3),fontsize=10,color=INK,fontweight="bold")
ax.axhline(0.90,color=RED,ls="--",lw=1.4,label="0.90 target")
ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in groups],fontsize=10)
ax.set_ylabel("Empirical coverage", fontsize=11, color=INK)
ax.set_ylim(0.74,0.99); ax.set_xlim(-0.5,2.5)
ax.set_title("Coverage across 5 subject-disjoint splits\nMarginal holds; only Mondrian holds on violations",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=12)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.10), ncol=1, frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_robustness.png", bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT}/fig_robustness.svg", bbox_inches="tight", facecolor="white")
print("saved fig_robustness")

# ---------- FIG 5: image-level decisions (safety) ----------
il=json.load(open(f"{OUT}/image_level_v2.json"))
conf=il["image_confusion"]
cats=["ACCEPT","REJECT","RETAKE"]
viol=[conf[c]["on_violation_img"] for c in cats]
zoom=[conf[c]["on_zoom_outofscope"] for c in cats]
x=np.arange(3); w=0.6
fig,ax=plt.subplots(figsize=(7.2,5.2),dpi=150)
ax.bar(x,viol,w,color=TEAL,label="in-scope violation images")
ax.bar(x,zoom,w,bottom=viol,color=AMBER,label="zoom (out-of-scope)")
for i in range(3):
    ax.text(i,viol[i]+zoom[i]+14,f"{viol[i]+zoom[i]}",ha="center",fontsize=10,color=INK,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cats,fontsize=11)
ax.set_ylabel("# test images", fontsize=11, color=INK)
ax.set_ylim(0, max(v+z for v,z in zip(viol,zoom))*1.16)
ax.set_title(f"Image-level triage: 0 of {il['total_violation_images']} true-violation images\nwrongly ACCEPTED — violations routed to REJECT/RETAKE",
             fontsize=12.2, fontweight="bold", color=INK, loc="left", pad=12)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.09), ncol=2, frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_image_decisions.png", bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT}/fig_image_decisions.svg", bbox_inches="tight", facecolor="white")
print("saved fig_image_decisions")
print("ALL FIGURES DONE")
