"""R06 manuscript figures -- matplotlib only (Agg), Okabe-Ito colour-blind-safe
palette, 8pt fonts, each figure saved as PDF + SVG + preview PNG under
rounds/R06_mechanism_pivot_2026-10-22/figures/. Data only from outputs/*.csv and
outputs/ladder_signatures.csv (r06_numbers.py); no number is hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts")]
import kag_data_r05 as D  # noqa: E402

R06 = HERE.parent
OUT = R06 / "outputs"
FIG = R06 / "figures"
PRE = FIG / "preview"
PRE.mkdir(parents=True, exist_ok=True)
R05OUT = R06.parents[1] / "R05_consolidation_2026-10-21" / "outputs"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})

# Okabe-Ito palette (colour-blind safe)
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "teal": "#009E73",
     "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
     "grey": "#999999"}
DOM_C = {"FRP": C["blue"], "SFRC": C["orange"], "CARB": C["teal"], "DEEP": C["verm"],
         "DEEP-WB": C["purple"]}
RUNG_M = {"K1": "o", "K2": "s", "K3": "^", "BEAM": "o", "ARCH": "s",
          "ARCHS": "^", "ARCH99": "D"}
DLAB = {"FRP": "FRP-RC shear", "SFRC": "SFRC shear", "CARB": "Carbonation",
        "DEEP": "RC deep beams", "DEEP-WB": "RC deep beams (plate-adj.)"}
FTEX = dict(D.FEATURE_TEX, ln_wb=r"$w_{bp}/d$")
SORD = ["3", "6", "12", "24", "all"]


def load(name):
    return pd.read_csv(D.long_path(OUT / name))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(PRE / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  -> {FIG / name}.pdf/.svg + preview/{name}.png")


def sx(s_vals, all_x):
    """x positions for S = 3,6,12,24,all on a log axis."""
    return [float(s) if s != "all" else all_x for s in s_vals]


def fig2_signature(sig):
    """Observed within-programme exponent deviations dW (error bars) and the
    fainter between-programme deviations dB, against the anchor-implied
    references (lambda_s / Zsutty / Papadakis / Leemann-Moro buffer)."""
    panels = [("FRP", "K1", "a FRP-RC shear, K1 (ACI 440.1R-15 frame)"),
              ("SFRC", "K1", "b SFRC shear, K1 (Narayanan-Darwish frame)"),
              ("CARB", "K2", "c Carbonation, K2 (Papadakis $(C/a)^{1/2}$ frame)"),
              ("DEEP", "ARCH", "d RC deep beams, ARCH (Zsutty 1971 frame)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharey=False)
    for ax, (name, rung, title) in zip(axes.flat, panels):
        blk = sig[(sig.domain == name) & (sig.rung == rung)].reset_index(drop=True)
        x = np.arange(len(blk))
        coded = blk.role == "coded"
        ax.axhline(0, color=C["grey"], lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.errorbar(x[coded], blk.dW[coded], yerr=blk.dW_se[coded], fmt="o",
                    ms=4.5, color=C["blue"], capsize=2, lw=0.9, zorder=4,
                    label="within-programme $\\Delta$ (dW)")
        ax.errorbar(x[coded], blk.dB[coded], yerr=blk.dB_se[coded], fmt="o",
                    ms=4.5, mfc="none", mec=C["grey"], capsize=2, lw=0.9,
                    alpha=0.55, zorder=3, label="between-programme $\\Delta$ (dB)")
        if (~coded).any():  # free features: fitted total exponent
            ax.errorbar(x[~coded], blk.dW[~coded], yerr=blk.dW_se[~coded],
                        fmt="^", ms=4.5, mfc="none", mec=C["blue"], capsize=2,
                        lw=0.9, zorder=4, label="free feature (fitted exponent)")
        # code-implied / literature markers
        marks = []
        lam = blk.lambda_s_implied.dropna()
        if len(lam) and name in ("FRP", "SFRC", "DEEP"):
            i = blk.index[blk.feature == "ln_d"][0]
            marks.append((i, float(lam.iloc[0]),
                          "$\\lambda_s$ implied" if name != "DEEP" else "$\\lambda_s$ implied"))
        if name == "DEEP":
            i = blk.index[blk.feature == "ln_ad"][0]
            marks.append((i, 4.0 / 3.0, "beam action ($a/d$ 0 vs $-4/3$)"))
        if name == "CARB":
            i = blk.index[blk.feature == "x_cao"][0]
            marks.append((i, -0.5, "Leemann-Moro buffer (CaO $_r$ $-1$)"))
            wb = blk.index[blk.feature == "x_wb"]
            if len(wb):
                marks.append((wb[0], 1.0, "Leemann-Moro buffer ($w/b$ $+1$)"))
        for i, y, lab in marks:
            ax.plot([i - 0.28, i + 0.28], [y, y], color=C["verm"], lw=2.2,
                    solid_capstyle="butt", zorder=5)
        if marks:
            ax.plot([], [], color=C["verm"], lw=2.2, label="code-implied / literature")
        ax.set_xticks(x)
        ax.set_xticklabels([FTEX[f] for f in blk.feature], fontsize=7.5)
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.5, len(blk) - 0.5)
    axes.flat[0].legend(frameon=False, loc="lower left", ncol=1)
    for ax in (axes[1, 0], axes[0, 0]):
        ax.set_ylabel("exponent deviation from anchor")
    fig.tight_layout()
    save(fig, "fig2_signature")


def fig3_scarcity(summ, theory, nprog):
    """RMSE at k=0 vs library size S for each rung's KA-FIXED arm, the shared
    DATA-RE baseline (black dashed) and PAKA-S (grey); observed crossovers
    marked on the x-axis."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=False)
    rung_cols = {"FRP": {"K1": C["blue"], "K2": C["sky"]},
                 "SFRC": {"K1": C["verm"], "K2": C["orange"]},
                 "CARB": {"K1": C["teal"], "K2": C["purple"], "K3": C["yellow"]},
                 "DEEP": {"BEAM": C["grey"], "ARCH": C["verm"],
                          "ARCHS": C["orange"], "ARCH99": C["purple"]}}
    lab = {"FRP": "a FRP-RC", "SFRC": "b SFRC", "CARB": "c Carbonation", "DEEP": "d Deep beams"}
    for ax, name in zip(axes, ("FRP", "SFRC", "CARB", "DEEP")):
        d = summ[(summ.dataset == name) & (summ.k == 0)]
        allx = nprog[name]
        for rung, col in rung_cols[name].items():
            g = d[(d.rung == rung) & (d.method == "KA-FIXED")].set_index("S").reindex(SORD)
            ax.plot(sx(SORD, allx), g.rmse, "-o", ms=3, lw=1.0, color=col,
                    label=f"{rung} code imposed")
            p = d[(d.rung == rung) & (d.method == "PAKA-S")].set_index("S").reindex(SORD)
            ax.plot(sx(SORD, allx), p.rmse, ":", lw=1.0, color=col, alpha=0.75)
        g = d[d.method == "DATA-RE"].groupby("S").rmse.first().reindex(SORD)
        ax.plot(sx(SORD, allx), g, "--", lw=1.0, color=C["black"], label="data-only")
        ax.plot([], [], ":", lw=1.0, color=C["grey"], label="PAKA (per rung)")
        tr = theory[theory.domain == name]
        for rung, col in rung_cols[name].items():
            sobs = tr[tr.rung == rung].S_obs.iloc[0]
            if np.isfinite(sobs):
                ax.axvline(sobs, color=col, lw=0.7, ls=(0, (2, 2)), alpha=0.7)
        ax.set_xscale("log")
        ax.set_xticks([3, 6, 12, 24, allx])
        ax.set_xticklabels(["3", "6", "12", "24", "all"], fontsize=7.5)
        ax.set_title(lab[name], loc="left")
        ax.set_xlabel("library size $S$ (programmes)")
    axes[0].set_ylabel("RMSE at $k{=}0$")
    axes[0].legend(frameon=False, loc="upper right", handlelength=2.2)
    fig.tight_layout()
    save(fig, "fig3_scarcity")


def fig4_pred_vs_obs(theory, labs, chrono):
    """Predicted vs observed crossover sizes on a log-log scale; the grey band is
    the factor-two acceptance region."""
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    lo_lim, hi_lim = 3.5, 60
    grid = np.logspace(np.log10(lo_lim), np.log10(hi_lim), 50)
    ax.fill_between(grid, grid * 0.5, grid * 2, color=C["grey"], alpha=0.15, lw=0,
                    label="factor-two band")
    ax.plot(grid, grid, color=C["grey"], lw=0.8)
    ax.plot(grid, grid * 2, color=C["grey"], lw=0.6, ls=":")
    ax.plot(grid, grid * 0.5, color=C["grey"], lw=0.6, ls=":")
    t = theory.copy()
    for r in t.itertuples():
        col = DOM_C[r.domain]
        mk = RUNG_M[r.rung]
        finite_p = np.isfinite(r.S_star) and r.S_star <= 1000
        if np.isfinite(r.S_obs) and finite_p:
            xe = [[r.S_obs - r.S_obs_lo], [r.S_obs_hi - r.S_obs]]
            ye = [[r.S_star - r.S_star_lo], [r.S_star_hi - r.S_star]]
            ax.errorbar(r.S_obs, r.S_star, xerr=np.clip(xe, 0, None),
                        yerr=np.clip(ye, 0, None), fmt=mk, ms=5.5, color=col,
                        mec=col, capsize=1.6, elinewidth=0.8, zorder=4)
        else:
            # degenerate rungs drawn as arrows
            if not finite_p:  # S* -> infinity: arrow up (at S_obs or right edge)
                x0 = r.S_obs if np.isfinite(r.S_obs) else hi_lim * 0.8
                ax.add_patch(FancyArrowPatch((x0, hi_lim * 0.25), (x0, hi_lim * 0.9),
                                             arrowstyle="-|>", mutation_scale=9,
                                             color=col, lw=1.0, zorder=4))
            if not np.isfinite(r.S_obs) and not pd.isna(r.S_obs):
                pass
            if pd.isna(r.S_obs):  # crossover below S=3: left arrow at the prediction
                ax.add_patch(FancyArrowPatch((lo_lim * 1.9, min(r.S_star, hi_lim * 0.8)),
                                             (lo_lim * 1.05, min(r.S_star, hi_lim * 0.8)),
                                             arrowstyle="-|>", mutation_scale=9,
                                             color=col, lw=1.0, zorder=4))
    for r in labs.itertuples():  # laboratory-group predictions (small x)
        trow = t[(t.domain == r.domain) & (t.rung == r.rung)]
        if len(trow) and np.isfinite(trow.S_obs.iloc[0]) and np.isfinite(r.S_star) and r.S_star < 1000:
            ax.plot(trow.S_obs.iloc[0], r.S_star, "x", ms=4, color=DOM_C[r.domain],
                    mew=1.1, zorder=5)
    for r in chrono.itertuples():  # chronological library (small +)
        trow = t[(t.domain == r.dataset) & (t.rung == r.rung)]
        if len(trow) and np.isfinite(r.S_obs) and np.isfinite(trow.S_star.iloc[0]):
            ax.plot(r.S_obs, trow.S_star.iloc[0], "+", ms=5, color=DOM_C[r.dataset],
                    mew=1.1, zorder=5)
    handles = [Line2D([], [], marker=RUNG_M["K1"], ls="", color=DOM_C[n], ms=5.5,
                      label=DLAB[n]) for n in ("FRP", "SFRC", "CARB", "DEEP", "DEEP-WB")]
    handles += [Line2D([], [], marker="x", ls="", color=C["black"], ms=4, label="laboratory-group $S^*$"),
                Line2D([], [], marker="+", ls="", color=C["black"], ms=5, label="chronological $S_{obs}$")]
    ax.legend(handles=handles, frameon=False, loc="lower right", handletextpad=0.2,
              borderaxespad=0.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo_lim, hi_lim); ax.set_ylim(lo_lim, hi_lim)
    ax.set_xlabel("observed crossover $S_{\\mathrm{obs}}$ (programmes)")
    ax.set_ylabel("predicted $S^*$ (programmes)")
    fig.tight_layout()
    save(fig, "fig4_pred_vs_obs")


def fig5_paka(summ):
    """PAKA-S regret vs the best single arm (left) and the k-shot learning curves
    at S=all for FRP and SFRC (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4),
                           gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    ks = sorted(summ.k.unique())
    combos = [(n, r) for n in ("FRP", "SFRC", "CARB", "DEEP") for r in
              summ[summ.dataset == n].rung.unique()]
    xs, vals, cols = [], [], []
    for i, (name, rung) in enumerate(combos):
        d = summ[(summ.dataset == name) & (summ.rung == rung) & (summ.S == "all") & (summ.k == 0)]
        best = d[d.method.isin(["KA-FIXED", "KA-RE", "DATA-RE"])].rmse.min()
        p = d[d.method == "PAKA-S"].rmse.iloc[0]
        xs.append(i); vals.append(p - best); cols.append(DOM_C[name])
    ax.bar(xs, vals, color=cols, width=0.75)
    ax.axhline(0, color=C["grey"], lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n} {r}" for n, r in combos], rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("PAKA regret vs best arm")
    ax.set_title("a regret at $S{=}$all, $k{=}0$", loc="left")
    ax2 = axes[1]
    for name, col in (("FRP", DOM_C["FRP"]), ("SFRC", DOM_C["SFRC"])):
        for rung in summ[summ.dataset == name].rung.unique():
            g = summ[(summ.dataset == name) & (summ.rung == rung) & (summ.S == "all")
                     & (summ.method == "PAKA-S")].set_index("k").reindex(ks)
            ax2.plot(ks, g.rmse, "-o", ms=3.5, lw=1.0, color=col, alpha=0.85,
                     label=f"{name} {rung}")
        g = summ[(summ.dataset == name) & (summ.S == "all") & (summ.method == "DATA-RE")] \
            .groupby("k").rmse.first().reindex(ks)
        ax2.plot(ks, g, "--", lw=1.0, color=col, alpha=0.5)
    ax2.set_xlabel("in-library examples $k$")
    ax2.set_ylabel("RMSE, PAKA at $S{=}$all")
    ax2.set_xticks(ks)
    ax2.set_title("b few-shot gain, FRP and SFRC", loc="left")
    ax2.legend(frameon=False, ncol=2, handlelength=1.8, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig5_paka")


def figS1_robust(chs, summ, theory, labs):
    """Chronological-sub-library signature vs random sub-libraries (left) and the
    laboratory-grouping sensitivity of S* (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    for name, col in (("FRP", DOM_C["FRP"]), ("SFRC", DOM_C["SFRC"])):
        c = chs[(chs.dataset == name) & (chs.rung == "K1")]
        ax.errorbar(c.s_centre, c["diff_ka_fixed_minus_data_re"],
                    yerr=np.abs([c.lo - c["diff_ka_fixed_minus_data_re"],
                                 c.hi - c["diff_ka_fixed_minus_data_re"]]),
                    fmt="o-", ms=3.5, lw=1.0, color=col, capsize=2,
                    label=f"{name} chronological")
        d = summ[(summ.dataset == name) & (summ.rung == "K1") & (summ.k == 0)]
        for s in ("3", "6", "12", "24", "all"):
            row = d[d.S.astype(str) == s]
            if not row.empty:
                kf = row[row.method == "KA-FIXED"].rmse.iloc[0]
                dr = row[row.method == "DATA-RE"].rmse.iloc[0]
                ax.plot(float(s) if s != "all" else 40, kf - dr, "s", ms=3.5,
                        mfc="none", mec=col)
    ax.plot([], [], "s", ms=3.5, mfc="none", mec=C["black"], label="random sub-library")
    ax.axhline(0, color=C["grey"], lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("library size $S$ (programmes)")
    ax.set_ylabel("RMSE(K1) $-$ RMSE(data-only)")
    ax.set_title("a K1 scarcity signature, KA-FIXED vs DATA-RE", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    ax2 = axes[1]
    t = theory[theory.domain.isin(("FRP", "SFRC", "CARB"))]
    for i, r in enumerate(t.itertuples()):
        lr = labs[(labs.domain == r.domain) & (labs.rung == r.rung)]
        if lr.empty:
            continue
        x0 = np.log10(np.clip(r.S_star, 1e-1, 1e3)) if np.isfinite(r.S_star) else np.nan
        lab_s = np.clip(lr.iloc[0].S_star, 1e-1, 1e3)
        if not np.isfinite(x0):
            continue
        ax2.plot([x0, np.log10(lab_s)], [i, i], "-", color=C["grey"], lw=0.8)
        ax2.plot(x0, i, "o", ms=4.5, color=DOM_C[r.domain])
        ax2.plot(np.log10(lab_s), i, "s", ms=4, mfc="none", mec=DOM_C[r.domain])
    ax2.set_yticks(range(len(t)))
    ax2.set_yticklabels([f"{r.domain} {r.rung}" for r in t.itertuples()], fontsize=6.5)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(["$10^0$", "$10^1$", "$10^2$", "$10^3$"], fontsize=7.5)
    ax2.set_xlabel("predicted $S^*$ (programmes, log scale)")
    ax2.set_title("b laboratory vs public grouping", loc="left")
    ax2.plot([], [], "o", ms=4.5, color=C["black"], label="public-group $S^*$")
    ax2.plot([], [], "s", ms=4, mfc="none", mec=C["black"], label="laboratory-group $S^*$")
    ax2.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "figS1_robust")


def figS2_within_bias(within):
    """Within-programme smallest-k extrapolation bias for the shear K1/K2 pair
    and the deep-beam ARCH/ARCHS pair (KA-FIXED arm)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    style = {("FRP", "K1"): (DOM_C["FRP"], "o", "-"), ("FRP", "K2"): (DOM_C["FRP"], "o", "--"),
             ("DEEP", "ARCH"): (DOM_C["DEEP"], "s", "-"), ("DEEP", "ARCHS"): (DOM_C["DEEP"], "s", "--")}
    for (name, rung), (col, mk, ls) in style.items():
        g = within[(within.dataset == name) & (within.rung == rung)
                   & (within.method == "KA-FIXED")].sort_values("k")
        ax.errorbar(g.k, g.bias, yerr=np.abs([g.bias_lo - g.bias, g.bias_hi - g.bias]),
                    fmt=f"{mk}{ls}", ms=4, lw=1.0, color=col, capsize=2,
                    label=f"{name} {rung}")
    ax.axhline(0, color=C["grey"], lw=0.7, ls=":")
    ax.set_xlabel("in-programme specimens $k$")
    ax.set_ylabel("bias of KA-FIXED")
    ax.set_xticks([1, 2, 3])
    ax.legend(frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "figS2_within_bias")


def main():
    print("reading outputs ...")
    theory = load("ladder_theory.csv")
    summ = load("ladder_summary.csv")
    labs = load("ladder_theory_labs.csv")
    chrono = load("ladder_chrono_crossover.csv")
    chs = load("ladder_chrono_summary.csv")
    within = load("ladder_within_summary.csv")
    sig = load("ladder_signatures.csv")
    summ["S"] = summ["S"].astype(str)
    # "all" library size on the x-axis = number of programmes in each database
    # (= programmes evaluated at S == "all" in the scarcity summary)
    nprog = {n: int(summ[(summ.dataset == n) & (summ.S.astype(str) == "all")
                         & (summ.k == 0)].programmes.max())
             for n in ("FRP", "SFRC", "CARB", "DEEP")}
    print("fig2_signature"); fig2_signature(sig)
    print("fig3_scarcity"); fig3_scarcity(summ, theory, nprog)
    print("fig4_pred_vs_obs"); fig4_pred_vs_obs(theory, labs, chrono)
    print("fig5_paka"); fig5_paka(summ)
    print("figS1_robust"); figS1_robust(chs, summ, theory, labs)
    print("figS2_within_bias"); figS2_within_bias(within)
    print("figures done")


if __name__ == "__main__":
    main()
