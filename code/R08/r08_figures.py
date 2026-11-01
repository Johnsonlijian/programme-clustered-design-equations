"""R08 submission figures.

``fig4_pred_vs_obs_r08`` reproduces the R07 crossover-budget panel and adds the
five prospective repaired anchors as star markers.  ``fig_repair`` shows the
library-size curves around each repaired anchor with the predicted crossover.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
R06 = R08.parent / "R06_mechanism_pivot_2026-10-22"
FIG = R08 / "figures"
OUT = R08 / "outputs"
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


def star(ax, x, y, col, ms=7.0, z=6):
    ax.plot([x], [y], marker="*", ms=ms + 2.2, mfc="none", mec=COL["ink"],
            mew=0.55, ls="", zorder=z - 0.1)
    ax.plot([x], [y], marker="*", ms=ms, mfc="white", mec=col, mew=1.05,
            ls="", zorder=z)


def budget():
    t = pd.read_csv(R06 / "outputs" / "ladder_theory.csv")
    labs = pd.read_csv(R06 / "outputs" / "ladder_theory_labs.csv")
    pc = pd.read_csv(OUT / "prospective_check.csv")
    fig, ax = plt.subplots(figsize=(4.35, 3.95))
    lo, hi = 3.5, 60
    yhi = 100.0
    grid = np.logspace(np.log10(lo), np.log10(yhi), 100)
    ax.fill_between(grid, grid * .5, grid * 2, color=COL["grey"], alpha=.16, lw=0)
    ax.plot(grid, grid, color=COL["grey"], lw=.75)
    ax.plot(grid, grid * 2, color=COL["grey"], lw=.55, ls=":")
    ax.plot(grid, grid * .5, color=COL["grey"], lw=.55, ls=":")
    domc = {"FRP": COL["blue"], "SFRC": COL["orange"], "CARB": COL["green"],
            "DEEP": COL["red"], "DEEP-WB": COL["purple"]}
    mark = {"K1": "o", "K2": "s", "K3": "D", "BEAM": "^", "ARCH": "v",
            "ARCHS": "P", "ARCH99": "X"}
    for r in t.itertuples():
        x, y = r.S_obs, r.S_star
        if np.isfinite(x) and np.isfinite(y) and y <= hi:
            ax.errorbar(x, y, xerr=[[max(0, x - r.S_obs_lo)], [max(0, r.S_obs_hi - x)]],
                        yerr=[[max(0, y - r.S_star_lo)], [max(0, r.S_star_hi - y)]],
                        fmt=mark.get(r.rung, "o"), ms=4.8,
                        color=domc.get(r.domain, COL["ink"]),
                        mec=domc.get(r.domain, COL["ink"]),
                        capsize=1.3, elinewidth=.7, zorder=4)
        elif np.isfinite(x):
            ax.add_patch(FancyArrowPatch((x, hi * .25), (x, yhi * .88),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=domc.get(r.domain, COL["ink"]), lw=.8))
    for r in labs.itertuples():
        row = t[(t.domain == r.domain) & (t.rung == r.rung)]
        if len(row) and np.isfinite(row.S_obs.iloc[0]) and np.isfinite(r.S_star):
            ax.plot(row.S_obs.iloc[0], r.S_star, "x", ms=3.5,
                    color=domc.get(r.domain, COL["ink"]), mew=1.0)
    # prospective repaired anchors (write-once predictions, checked here)
    for r in pc.itertuples():
        col = domc.get(r.domain, COL["ink"])
        if np.isfinite(r.S_obs):
            ax.errorbar(r.S_obs, r.S_pred,
                        xerr=[[max(0, r.S_obs - r.S_obs_lo)], [max(0, r.S_obs_hi - r.S_obs)]],
                        yerr=[[max(0, r.S_pred - r.S_pred_lo)], [max(0, r.S_pred_hi - r.S_pred)]],
                        fmt="none", ecolor=col, elinewidth=.6, capsize=1.1,
                        alpha=.85, zorder=5)
            star(ax, r.S_obs, r.S_pred, col)
        else:
            star(ax, 24, r.S_pred, col)
            # predicted interval as a vertical bar, clipped at the axis top
            # with an upward arrow when the upper bound is off-scale
            top = min(r.S_pred_hi, yhi * .97)
            ax.plot([24, 24], [r.S_pred_lo, top], color=col, lw=.6,
                    alpha=.85, zorder=5)
            if r.S_pred_hi > top:
                ax.add_patch(FancyArrowPatch((24, top), (24, yhi * .99),
                                             arrowstyle="-|>", mutation_scale=6,
                                             color=col, lw=.6, zorder=5))
            ax.add_patch(FancyArrowPatch((24 * 1.12, r.S_pred), (hi * .95, r.S_pred),
                                         arrowstyle="-|>", mutation_scale=7,
                                         color=col, lw=.7, zorder=5))
    # short labels, each immediately beside its own star
    lab = [("FRP", "K1C", (9.9, 6.5), "right"),         # below-left of star (10.5, 8.7)
           ("FRP", "K2C", (14.5, 20.0), "left"),       # below-right of star (11.8, 24.3)
           ("SFRC", "K2C", (19.8, 25.8), "right"),     # above-left of star (22.1, 23.2)
           ("DEEP", "BEAMC", (20.8, 66.3), "right")]   # left of star (24, 66.3)
    for dom, rung, (lx, ly), h in lab:
        ax.text(lx, ly, f"{dom} {rung}", fontsize=5.6,
                color=domc.get(dom, COL["ink"]), ha=h, va="center")
    # SFRC K1C sits below-right of its star, clear of the y=x/2 line
    ax.annotate("SFRC K1C", (24.9, 20.8), (30.0, 12.0), fontsize=5.6,
                color=COL["orange"], ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=.45, color=COL["grey"],
                                shrinkA=0, shrinkB=2))
    handles = [Line2D([], [], marker="o", ls="", color=domc[d], ms=4.8, label=d)
               for d in ("FRP", "SFRC", "CARB", "DEEP")]
    handles.append(Line2D([], [], marker="o", ls="", color=domc["DEEP-WB"], ms=4.8,
                          label="DEEP-WB\n(plate-adjusted\nsensitivity)"))
    handles += [Line2D([], [], marker="x", ls="", color=COL["ink"], ms=3.5,
                       label="laboratory group"),
                Line2D([], [], marker="*", ls="", ms=7, mfc="white", mec=COL["ink"],
                       label="prospective (repaired anchors)")]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -.14), fontsize=5.8, columnspacing=1.0,
              handletextpad=.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, yhi)
    ax.set_xticks([4, 6, 10, 20, 40, 60])
    ax.set_xticklabels(["4", "6", "10", "20", "40", "60"])
    ax.set_yticks([4, 6, 10, 20, 40, 60, 100])
    ax.set_yticklabels(["4", "6", "10", "20", "40", "60", "100"])
    ax.set_xlabel("observed crossover $S_{\\mathrm{obs}}$ (programmes)")
    ax.set_ylabel("predicted $S^*$ (programmes)")
    ax.set_title("First-order crossover budget", loc="left", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=.16, right=.98, top=.93, bottom=.29)
    save(fig, "fig4_pred_vs_obs_r08")


def _curve(frame, rung, meth, dom=None):
    if dom is None:
        g = frame[(frame.rung == rung) & (frame.method == meth)]
    else:
        g = frame[(frame.dataset == dom) & (frame.rung == rung) & (frame.method == meth)]
    return g.iloc[0]


def repair():
    csc = pd.read_csv(OUT / "certified_scarcity_curves.csv")
    cs = pd.read_csv(OUT / "certified_summary.csv")
    rsc = pd.read_csv(OUT / "repair_scarcity_curves.csv")
    pc = pd.read_csv(OUT / "prospective_check.csv")
    xs = np.array([3, 6, 12, 24, 48], dtype=float)
    panels = [
        ("(a) SFRC: fibre term released", "SFRC", "K1", "K1C", "K1C"),
        ("(b) FRP-RC: $a/d$ released", "FRP", "K1", "K1C", "K1C"),
        ("(c) deep beams: $a/d$ released", "DEEP", "BEAM", "BEAMC", "BEAMC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False)
    for ax, (title, dom, par, rr, prr) in zip(axes, panels):
        if dom == "SFRC":
            def val(rung, meth, col):
                if col == "all":
                    g = cs[(cs.rung == rung) & (cs.method == meth) & (cs.k == 0)]
                    return g.rmse.iloc[0] if len(g) else np.nan
                return float(_curve(csc, rung, meth)[str(col)])
        else:
            def val(rung, meth, col):
                return float(_curve(rsc, rung, meth, dom)[str(col)])
        yd = [val(rr, "DATA-RE", c) for c in (3, 6, 12, 24, "all")]
        yp = [val(par, "KA-FIXED", c) for c in (3, 6, 12, 24, "all")]
        yr = [val(rr, "KA-FIXED", c) for c in (3, 6, 12, 24, "all")]
        ax.plot(xs, yd, "--", lw=.9, color=COL["grey"], label="data-only")
        ax.plot(xs, yp, "--o", lw=.9, ms=3.0, mfc="white", mec=COL["orange"],
                color=COL["orange"], label="code imposed, parent anchor")
        ax.plot(xs, yr, "-o", lw=1.0, ms=3.0, color=COL["blue"],
                label="code imposed, repaired anchor")
        sstar = float(pc[(pc.domain == dom) & (pc.rung == prr)].S_pred.iloc[0])
        if sstar <= 48:
            ax.axvline(sstar, color=COL["blue"], lw=.7, ls=":", zorder=1)
            ax.annotate(f"predicted $S^*$ = {sstar:.1f}", (sstar * 1.07, 0.94),
                        xycoords=("data", "axes fraction"), fontsize=7.5,
                        color=COL["blue"], ha="left", va="top")
        else:
            ax.add_patch(FancyArrowPatch((44, 0.88), (57.5, 0.88),
                                         arrowstyle="-|>", mutation_scale=7,
                                         color=COL["blue"], lw=.8, zorder=1,
                                         transform=ax.get_xaxis_transform()))
            ax.text(60, 0.99, f"predicted $S^*$ = {sstar:.1f}", fontsize=7.5,
                    color=COL["blue"], ha="right", va="top",
                    transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_xlim(2.6, 60)
        ax.set_xticks(xs)
        ax.set_xticklabels(["3", "6", "12", "24", "all"], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("training programmes", fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("new-programme RMSE ($k=0$)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=.45, rect=(0, .115, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, handlelength=1.8, bbox_to_anchor=(.5, .0))
    save(fig, "fig_repair")


if __name__ == "__main__":
    budget()
    repair()
    print("R08 figures written to", FIG)
