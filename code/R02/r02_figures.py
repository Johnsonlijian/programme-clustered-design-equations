"""R02 figures: three-domain audit, knowledge-value curves, arm comparison.

Outputs (vector masters plus PNG previews):
  figures/fig1_domain_audit.{svg,pdf}
  figures/fig2_knowledge_value.{svg,pdf}
  figures/fig3_arms_over_S.{svg,pdf}
  figures/preview/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import pandas as pd

ROUND = Path(__file__).resolve().parents[1]
OUT = ROUND / "outputs"
FIG = ROUND / "figures"
PRE = FIG / "preview"
PRE.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})

C = {"FRP": "#2b6cb0", "SFRC": "#2f855a", "DEEP": "#b7791f"}
C_W, C_B = "#2b6cb0", "#dd6b20"
ORDER = ["FRP", "SFRC", "DEEP"]
TICKS = [3, 6, 12, 24]


def logx(ax):
    ax.set_xscale("log")
    ax.set_xlim(2.6, 30)
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PRE / f"{name}.png", dpi=220)
    plt.close(fig)


def fig1(audit):
    a = audit.set_index("domain")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.05), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    x = np.arange(len(ORDER))

    for axi, col, title in ((ax[0], "icc", "a  programme share of residual"),
                            (ax[1], "eta2_d", "b  between share of ln $d$")):
        vals = [a.loc[d, col] for d in ORDER]
        axi.bar(x, vals, color=[C[d] for d in ORDER], width=0.62, edgecolor="white", lw=0.4)
        axi.set_xticks(x, ORDER)
        axi.set_ylim(0, 1.0)
        axi.set_title(title, loc="left", pad=4)
        axi.axhline(0.5, color="0.65", lw=0.6, ls=":")
        for i, v in enumerate(vals):
            axi.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=6.5)

    axc = ax[2]
    y = np.arange(len(ORDER))
    for i, d in enumerate(ORDER):
        r = a.loc[d]
        axc.errorbar(r.d_within, i + 0.15, xerr=1.96 * r.d_within_se, fmt="o", ms=3.2,
                     color=C_W, capsize=0, lw=1.0)
        axc.errorbar(r.d_between, i - 0.15, xerr=1.96 * r.d_between_se, fmt="s", ms=3.2,
                     color=C_B, capsize=0, lw=1.0)
    axc.axvline(0, color="0.35", lw=0.7)
    axc.set_yticks(y, ORDER)
    axc.set_ylim(len(ORDER) - 0.45, -0.55)
    axc.set_xlabel("$d$-exponent deviation from the code")
    axc.set_title("c  where the size effect is identifiable", loc="left", pad=4)
    h = [plt.Line2D([], [], marker="o", ls="", color=C_W, ms=3.2),
         plt.Line2D([], [], marker="s", ls="", color=C_B, ms=3.2)]
    fig.legend(h, ["within programme", "between programmes"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.79, -0.08))
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig1_domain_audit")


def fig2(theory, summ):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25))
    for k, (a, d) in enumerate(zip(ax, ORDER)):
        t = theory[(theory.dataset == d) & (theory.population == "all")].iloc[0]
        blk = summ[(summ.dataset == d) & (summ.k == 0)]
        piv = blk.pivot_table(index="S_order", columns="method", values="rmse")
        lab = blk.drop_duplicates("S_order").set_index("S_order")
        S = np.array([{"3": 3, "6": 6, "12": 12}.get(lab.S[so], 24) for so in piv.index], float)
        gain = piv["DATA-RE"].values ** 2 - piv["KA-RE"].values ** 2

        a.axhline(0, color="0.6", lw=0.6)
        a.plot(S, gain, "o", ms=3.6, color=C[d], zorder=3)
        ss = np.linspace(3, 24.5, 200)
        a.plot(ss, t.A / ss - t.B, "-", lw=1.1, color="0.15")
        if np.isfinite(t.A_meas) and t.B_meas > 0:
            a.plot(ss, t.A_meas / ss - t.B_meas, "--", lw=0.9, color="0.55")
        a.axvline(t.S_star_pred, color="0.4", lw=0.7, ls=":")
        logx(a)
        a.set_title(f"{d}   $S^*$={t.S_star_pred:.0f} (measured {t.S_star_meas:.0f})", loc="left", pad=4)
        a.set_xlabel("training programmes $S$")
        if k == 0:
            a.set_ylabel("knowledge gain (MSE, $k=0$)")
    h = [plt.Line2D([], [], marker="o", ls="", color="0.35", ms=3.6),
         plt.Line2D([], [], ls="-", color="0.15", lw=1.1),
         plt.Line2D([], [], ls="--", color="0.55", lw=0.9),
         plt.Line2D([], [], ls=":", color="0.4", lw=0.8)]
    fig.legend(h, ["measured", "$A/S-B$ from variance components",
                   "$A/S-B$ fitted to the points", "$S^*$"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(w_pad=1.2, rect=(0.02, 0.10, 1, 1))
    save(fig, "fig2_knowledge_value")


def fig3(summ):
    arms = [a for a in ["ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "EAT-T"]
            if a in set(summ.method)]
    style = {"ANCHOR": dict(ls=":", color="0.55", lw=1.0, marker="o", ms=2.2),
             "KA-FIXED": dict(ls="--", color="#805ad5", lw=1.0, marker="o", ms=2.2),
             "KA-RE": dict(ls="-", color="#2b6cb0", lw=1.0, marker="o", ms=2.2),
             "DATA-RE": dict(ls="-", color="#dd6b20", lw=1.0, marker="o", ms=2.2),
             "EAT-T": dict(ls="-", color="#1a202c", lw=1.8, marker="o", ms=2.6)}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25), sharey=True)
    for a, d in zip(ax, ORDER):
        blk = summ[(summ.dataset == d) & (summ.k == 0)]
        piv = blk.pivot_table(index="S_order", columns="method", values="rmse")
        lab = blk.drop_duplicates("S_order").set_index("S_order")
        S = np.array([{"3": 3, "6": 6, "12": 12}.get(lab.S[so], 24) for so in piv.index], float)
        for arm in arms:
            if arm in piv:
                a.plot(S, piv[arm].values, label=arm, **style[arm])
        logx(a)
        a.set_title(d, loc="left", pad=4)
        a.set_xlabel("training programmes $S$")
    ax[0].set_ylabel("RMSE of ln $V$")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(w_pad=1.0, rect=(0.055, 0.08, 1, 1))
    save(fig, "fig3_arms_over_S")


def main():
    audit = pd.read_csv(OUT / "domain_audit.csv")
    theory = pd.read_csv(OUT / "theory_prediction.csv")
    summ = pd.read_csv(OUT / "scarcity2_summary.csv")
    fig1(audit)
    fig2(theory, summ)
    fig3(summ)
    print("figures written")


if __name__ == "__main__":
    main()
