"""R07 submission-figure hardening.

Rebuilds the concept, scarcity, crossover-budget and leakage figures from the
same derived tables used by the manuscript.  The script deliberately treats
library-size labels as strings (the ``all`` row otherwise turns numeric ``S``
into an object column and silently produces an empty plot).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
DERIVED = ROOT / "derived"
FIG.mkdir(exist_ok=True, parents=True)

COL = dict(ink="#1a1a1a", grey="#7f7f7f", light="#d9d9d9", blue="#0072B2",
           sky="#56B4E9", red="#D55E00", orange="#E69F00", green="#009E73",
           purple="#CC79A7", yellow="#F0E442")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.titlesize": 8.0,
    "axes.labelsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{name}.png", dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def frame(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(ax, x, y, w, h, label, ec=COL["grey"], fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor="white", edgecolor=ec, linewidth=0.7))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, color=COL["ink"], lw=0.7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=5, color=color, lw=lw,
                                 shrinkA=1.5, shrinkB=1.5))


def concept():
    fig, axes = plt.subplots(1, 5, figsize=(8.35, 2.45))
    for ax in axes:
        frame(ax)
    # (a) hypothesis
    ax = axes[0]
    ax.text(.5, .97, "(a) coded term", ha="center", va="top", fontsize=7.8, weight="bold")
    box(ax, .06, .66, .88, .15, r"$V_c=0.4\sqrt{f'_c}\,b_w\,kd$", ec=COL["ink"], fs=7.2)
    for k, txt in enumerate(("sign   +", "exponent   1", r"domain   $a/d\geq2.5$")):
        ax.text(.10, .52 - .11*k, txt, ha="left", va="center", fontsize=6.7, color=COL["blue"])
    ax.text(.5, .08, "an omitted variable is\na claim of no dependence", ha="center", va="center",
            fontsize=6.0, color=COL["grey"], style="italic")
    # (b) clustered evidence
    ax = axes[1]
    ax.text(.5, .97, "(b) scarce unit", ha="center", va="top", fontsize=7.8, weight="bold")
    rng = np.random.default_rng(3)
    for cx, cy in rng.uniform(.16, .84, size=(6, 2)):
        yy = .43 + .25*(cy-.5)
        ax.add_patch(Circle((cx, yy), .043, color=COL["sky"], alpha=.28, ec="none"))
        for _ in range(5):
            ax.add_patch(Circle((cx+rng.normal(0,.028), yy+rng.normal(0,.035)), .0065,
                                color=COL["ink"], alpha=.85, ec="none"))
    ax.text(.5, .13, "2,158 tests  →  41–61\neffective observations", ha="center", va="center", fontsize=6.5)
    # (c) within/between
    ax = axes[2]
    ax.text(.5, .97, "(c) within", ha="center", va="top", fontsize=7.2, weight="bold")
    ax.add_patch(FancyBboxPatch((.06,.42), .38,.35, boxstyle="round,pad=.02", fc="#eef5f8", ec=COL["sky"], lw=.8))
    ax.add_patch(FancyBboxPatch((.56,.42), .38,.35, boxstyle="round,pad=.02", fc="#fbefeb", ec=COL["red"], lw=.8))
    for x0, c in ((.15, COL["green"]), (.65, COL["red"])):
        ax.add_patch(Circle((x0,.52), .012, color=COL["ink"]))
        ax.add_patch(Circle((x0+.18,.68), .012, color=COL["ink"]))
        arrow(ax, x0+.01, .53, x0+.17, .67, c)
    ax.text(.74,.49,r"$\times$",ha="center",va="center",fontsize=9,color=COL["red"])
    ax.text(.25,.29,"within: free of\nprogramme effects",ha="center",va="top",fontsize=6.1,color=COL["green"])
    ax.text(.75,.29,"between:\nconfounded",ha="center",va="top",fontsize=6.1,color=COL["red"])
    # (d) crossover budget
    ax = axes[3]
    ax.text(.5, .97, "(d) price", ha="center", va="top", fontsize=7.2, weight="bold")
    S=np.linspace(3,30,300); p,c,B,s2=6,.35,.30,.05; Ss=p*c/B
    ax.plot(S,s2+B+c/S,color=COL["blue"],lw=1.0)
    ax.plot(S,s2+(p+1)*c/S,color=COL["orange"],lw=1.0)
    ax.plot([Ss],[s2+B+c/Ss],"o",ms=3.0,color=COL["ink"])
    ax.plot([Ss,Ss],[.08,s2+B+c/Ss],ls=":",lw=.6,color=COL["grey"])
    ax.set_xlim(2.5,30); ax.set_ylim(.04,1.03); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(Ss,.13,r"$S^*=pc/B$",ha="center",va="center",fontsize=6.8)
    ax.text(.78,.78,"code",ha="center",fontsize=6.2,color=COL["blue"])
    ax.text(.78,.42,"data",ha="center",fontsize=6.2,color=COL["orange"])
    # (e) allocation, with non-overlapping boxes
    ax = axes[4]
    ax.text(.5,.97,"(e) update with new tests",ha="center",va="top",fontsize=7.8,weight="bold")
    for k,lab in enumerate(("code", "prior", "data")):
        yy=.76-.18*k; box(ax,.02,yy-.055,.23,.11,lab,fs=6.2); arrow(ax,.25,yy,.36,.58,COL["grey"])
    box(ax,.36,.45,.27,.22,"programme\nscore",ec=COL["ink"],fs=6.2)
    arrow(ax,.63,.56,.70,.56,COL["ink"])
    box(ax,.70,.45,.28,.22,"$k$ tests\nof new\nprogramme",ec=COL["blue"],fs=6.0)
    ax.text(.5,.15,"13–32% lower error at $k=3$",ha="center",va="center",fontsize=6.2,color=COL["green"])
    fig.subplots_adjust(left=.01,right=.99,top=.91,bottom=.03,wspace=.08)
    save(fig,"fig0_concept")


def _skey(v):
    return str(v).strip()


def scarcity():
    summ=pd.read_csv(DERIVED/"ladder_summary.csv")
    theory=pd.read_csv(DERIVED/"ladder_theory.csv")
    summ["S_key"]=summ["S"].map(_skey)
    NP={"FRP":41,"SFRC":85,"CARB":48,"DEEP":48}
    order=["3","6","12","24","all"]
    colors={"FRP":{"K1":COL["blue"],"K2":COL["sky"]},
            "SFRC":{"K1":COL["red"],"K2":COL["orange"]},
            "CARB":{"K1":COL["green"],"K2":COL["purple"],"K3":COL["yellow"]},
            "DEEP":{"BEAM":COL["grey"],"ARCH":COL["red"],"ARCHS":COL["orange"],"ARCH99":COL["purple"]}}
    labels={"FRP":"a  FRP-RC","SFRC":"b  SFRC","CARB":"c  carbonation","DEEP":"d  deep beams"}
    fig,axes=plt.subplots(1,4,figsize=(8.25,2.55),sharey=False)
    for ax,dom in zip(axes,("FRP","SFRC","CARB","DEEP")):
        d=summ[(summ.dataset==dom)&(summ.k==0)]
        xvals=np.array([3,6,12,24,NP[dom]],dtype=float)
        for rung,col in colors[dom].items():
            g=d[(d.rung==rung)&(d.method=="KA-FIXED")].set_index("S_key").reindex(order)["rmse"].astype(float).values
            p=d[(d.rung==rung)&(d.method=="PAKA-S")].set_index("S_key").reindex(order)["rmse"].astype(float).values
            ax.plot(xvals,g,"-o",ms=2.5,lw=.85,color=col,label=f"{rung} code")
            ax.plot(xvals,p,":",lw=.85,color=col,alpha=.9)
        gd=d[d.method=="DATA-RE"].groupby("S_key").rmse.first().reindex(order).astype(float).values
        ax.plot(xvals,gd,"--",lw=.85,color=COL["ink"],label="data only")
        tr=theory[theory.domain==dom]
        for rung,col in colors[dom].items():
            row=tr[tr.rung==rung]
            if len(row) and np.isfinite(row.S_obs.iloc[0]):
                ax.axvline(min(float(row.S_obs.iloc[0]),NP[dom]*1.02),color=col,lw=.55,ls=(0,(2,2)),alpha=.65)
        vals=np.concatenate([g[np.isfinite(g)] for g in [gd] if np.any(np.isfinite(g))])
        if vals.size:
            ax.set_ylim(max(0,vals.min()-.06),vals.max()+.07)
        # A linear axis keeps the five library labels legible; the panel is a
        # comparison of the prescribed rungs rather than a scale-law plot.
        ax.set_xlim(2.2,NP[dom]*1.08)
        ax.set_xticks(xvals); ax.set_xticklabels(order,fontsize=6.1,rotation=45,ha="right")
        ax.set_title(labels[dom],loc="left",fontsize=7.5)
        ax.set_xlabel("training programmes",fontsize=6.8)
        ax.tick_params(axis="y",labelsize=6.1)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("new-programme RMSE ($k=0$)",fontsize=6.8)
    axes[0].legend(frameon=False,fontsize=5.4,loc="upper right",handlelength=1.6)
    fig.tight_layout(pad=.45)
    save(fig,"fig3_scarcity_v2")


def budget():
    t=pd.read_csv(DERIVED/"ladder_theory.csv")
    labs=pd.read_csv(DERIVED/"ladder_theory_labs.csv")
    fig,ax=plt.subplots(figsize=(4.35,3.85))
    lo,hi=3.5,60; grid=np.logspace(np.log10(lo),np.log10(hi),100)
    ax.fill_between(grid,grid*.5,grid*2,color=COL["grey"],alpha=.16,lw=0)
    ax.plot(grid,grid,color=COL["grey"],lw=.75); ax.plot(grid,grid*2,color=COL["grey"],lw=.55,ls=":"); ax.plot(grid,grid*.5,color=COL["grey"],lw=.55,ls=":")
    domc={"FRP":COL["blue"],"SFRC":COL["orange"],"CARB":COL["green"],"DEEP":COL["red"],"DEEP-WB":COL["purple"]}
    mark={"K1":"o","K2":"s","K3":"D","BEAM":"^","ARCH":"v","ARCHS":"P","ARCH99":"X"}
    for r in t.itertuples():
        x,y=r.S_obs,r.S_star
        if np.isfinite(x) and np.isfinite(y) and y<=hi:
            ax.errorbar(x,y,xerr=[[max(0,x-r.S_obs_lo)],[max(0,r.S_obs_hi-x)]],yerr=[[max(0,y-r.S_star_lo)],[max(0,r.S_star_hi-y)]],fmt=mark.get(r.rung,"o"),ms=4.8,color=domc.get(r.domain,COL["ink"]),mec=domc.get(r.domain,COL["ink"]),capsize=1.3,elinewidth=.7,zorder=4)
        elif np.isfinite(x):
            ax.add_patch(FancyArrowPatch((x,hi*.25),(x,hi*.88),arrowstyle="-|>",mutation_scale=8,color=domc.get(r.domain,COL["ink"]),lw=.8))
    for r in labs.itertuples():
        row=t[(t.domain==r.domain)&(t.rung==r.rung)]
        if len(row) and np.isfinite(row.S_obs.iloc[0]) and np.isfinite(r.S_star):
            ax.plot(row.S_obs.iloc[0],r.S_star,"x",ms=3.5,color=domc.get(r.domain,COL["ink"]),mew=1.0)
    handles=[Line2D([],[],marker="o",ls="",color=domc[d],ms=4.8,label=d) for d in ("FRP","SFRC","CARB","DEEP","DEEP-WB")]
    handles += [Line2D([],[],marker="x",ls="",color=COL["ink"],ms=3.5,label="laboratory group")]
    ax.legend(handles=handles,frameon=False,ncol=3,loc="upper left",bbox_to_anchor=(0,-.14),fontsize=5.8,columnspacing=1.0,handletextpad=.3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ticks=[4,6,10,20,40,60]; ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_xticklabels([str(x) for x in ticks]); ax.set_yticklabels([str(x) for x in ticks])
    ax.set_xlabel("observed crossover $S_{\\mathrm{obs}}$ (programmes)"); ax.set_ylabel("predicted $S^*$ (programmes)")
    ax.set_title("First-order crossover budget",loc="left",fontsize=8)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=.16,right=.98,top=.93,bottom=.29)
    save(fig,"fig4_pred_vs_obs")


def leakage():
    pg=pd.read_csv(DERIVED/"pglo.csv"); pg["optimism"]=(pg.rmse_lopo/pg.rmse_random-1)*100
    order=["FRP","CARB","DEEP","SFRC"]; lab={"FRP":"FRP shear","CARB":"carbonation","DEEP":"deep beams","SFRC":"SFRC shear"}
    fig,ax=plt.subplots(figsize=(4.15,2.45))
    for i,dom in enumerate(order):
        g=pg[pg.domain==dom].set_index("arm")
        for j,arm in enumerate(("KA-FIXED","KA-RE","DATA-RE")):
            y=i+(j-1)*.17; r,l=float(g.loc[arm,"rmse_random"]),float(g.loc[arm,"rmse_lopo"])
            ax.plot([r,l],[y,y],color=COL["light"],lw=2.2,zorder=1,solid_capstyle="round")
            ax.plot(r,y,"o",ms=3.2,color=COL["orange"],zorder=3); ax.plot(l,y,"o",ms=3.2,color=COL["blue"],zorder=3)
            ax.text(min(max(r,l)+.015,.84),y,f"{g.loc[arm,'optimism']:.0f}%",va="center",fontsize=6.2,clip_on=False)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[d] for d in order],fontsize=6.8); ax.set_xlabel("new-programme RMSE (log units)",fontsize=6.8); ax.set_xlim(.14,.88)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.plot([],[],"o",ms=3.2,color=COL["orange"],label="random five-fold"); ax.plot([],[],"o",ms=3.2,color=COL["blue"],label="leave-one-programme-out")
    ax.legend(frameon=False,loc="lower right",fontsize=5.9); ax.set_title("Random splits overstate transfer",loc="left",fontsize=7.6)
    fig.tight_layout(pad=.4); save(fig,"fig_leakage")


if __name__ == "__main__":
    concept(); scarcity(); budget(); leakage()
    print("R07 hardened figures written")

