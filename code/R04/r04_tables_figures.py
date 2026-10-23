"""R04 tables/macros + figures for the CARB memo."""
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
GEN = ROUND / "note" / "generated"
PRE = FIG / "preview"
for p in (FIG, GEN, PRE):
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
C = {"FRP": "#2b6cb0", "SFRC": "#2f855a", "DEEP": "#b7791f", "CARB": "#805ad5"}
R02_OUT = ROUND.parents[0] / "R02_crossdomain_theory_2026-10-20" / "outputs"


def f2(x):
    return "---" if not np.isfinite(x) else f"{x:.2f}"


def f1(x):
    return "---" if x is None or not np.isfinite(x) else f"{x:.1f}"


def main():
    a = pd.read_csv(OUT / "domain_audit_r04.csv").iloc[0]
    th = pd.read_csv(OUT / "theory_r04.csv").iloc[0]
    s = pd.read_csv(OUT / "scarcity_r04_summary.csv")
    r02 = pd.read_csv(R02_OUT / "domain_audit.csv")

    # ---- macros --------------------------------------------------------------
    M = {
        "CARBN": str(int(a.n)), "CARBP": str(int(a.programmes)),
        "CARBPge": str(int(a.programmes_ge6)), "CARBICC": f2(a.icc),
        "CARBTAU": f2(a.tau), "CARBSIG": f2(a.sigma),
        "CARBEtaExpo": f2(a.eta2_x_co2), "CARBEWB": f2(a.eta2_x_wb),
        "CARBExpoW": f2(a.x_co2_within), "CARBExpoWse": f2(a.x_co2_within_se),
        "CARBExpoB": f2(a.x_co2_between), "CARBExpoBse": f2(a.x_co2_between_se),
        "CARBExpoZ": f"{a.x_co2_z:.1f}",
        "CARBWBW": f2(a.x_wb_within), "CARBWBWse": f2(a.x_wb_within_se),
        "CARBWBB": f2(a.x_wb_between), "CARBWBBse": f2(a.x_wb_between_se),
        "CARBWBZ": f"{a.x_wb_z:.1f}",
        "CARBA": f2(th.A), "CARBB": f"{th.B:.3f}",
        "CARBSSp": f"{th.S_star_pred:.1f}", "CARBSSm": f"{th.S_star_meas:.1f}",
        "CARBSSlo": f1(th.S_star_lo), "CARBSShi": f1(th.S_star_hi),
"CARBRatio": f2(th.ratio), "CARBRsq": f2(th.gain_r2),
        "CARBSScod": f"{th.S_star_pred_codified:.0f}",
        "CARBRatioCod": f"{th.S_star_pred_codified / th.S_star_meas:.0f}",
        "CARBBcod": f"{th.B_codified_only:.4f}",
        "CARBFixAll": f2(float(s[(s.S.astype(str) == "24") & (s.k == 0) & (s.method == "KA-FIXED")].rmse.iloc[0])),
        "CARBDataAll": f2(float(s[(s.S.astype(str) == "24") & (s.k == 0) & (s.method == "DATA-RE")].rmse.iloc[0])),
        "CARBNBAR": f2(th.n_bar),
    }
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (GEN / "numbers_r04.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- table: CARB vs the three R02 domains --------------------------------
    rows = []
    for _, r in r02.iterrows():
        rows.append(f"{r.domain} & {int(r.n)} & {int(r.programmes)} & {f2(r.icc)} & "
                    f"{f2(r.eta2_d)} & {f2(r.d_within)} ({f2(r.d_within_se)}) & "
                    f"{f2(r.d_between)} ({f2(r.d_between_se)}) & {r.d_z:.1f} \\\\")
    rows.append("\\midrule")
    rows.append(f"CARB & {int(a.n)} & {int(a.programmes)} & {f2(a.icc)} & {f2(a.eta2_x_co2)} & "
                f"{f2(a.x_co2_within)} ({f2(a.x_co2_within_se)}) & "
                f"{f2(a.x_co2_between)} ({f2(a.x_co2_between_se)}) & {a.x_co2_z:.1f} \\\\")
    head = (r"域 & $n$ & 项目 & ICC & $\eta^2$ & 项目内偏差 & 项目间偏差 & $z$")
    body = "\n".join([head + r" \\", r"\midrule"] + rows)
    (GEN / "tab_r04.tex").write_text(
        "\\begin{tabular}{lccccccc}\\toprule\n" + body + "\n\\bottomrule\\end{tabular}\n",
        encoding="utf-8")

    # ---- figure: ICC across domains + CARB exponent ---------------------------
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.1), gridspec_kw=dict(width_ratios=[1.2, 1]))
    order = ["FRP", "SFRC", "DEEP", "CARB"]
    icc = [float(r02.set_index("domain").loc[d, "icc"]) for d in order[:3]] + [float(a.icc)]
    ax[0].bar(range(4), icc, color=[C[d] for d in order], width=0.6)
    ax[0].set_xticks(range(4), order)
    ax[0].set_ylim(0, 1)
    ax[0].axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax[0].set_title("a  programme share of the residual", loc="left")
    for i, v in enumerate(icc):
        ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    y = [0, 1]
    ax[1].errorbar([a.x_co2_within], [0.15], xerr=[1.96 * a.x_co2_within_se], fmt="o", ms=3.5,
                   color="#2b6cb0", capsize=0)
    ax[1].errorbar([a.x_co2_between], [-0.15], xerr=[1.96 * a.x_co2_between_se], fmt="s", ms=3.5,
                   color="#dd6b20", capsize=0)
    ax[1].axvline(0, color="0.35", lw=0.7)
    ax[1].set_yticks([0.15, -0.15], ["within", "between"])
    ax[1].set_ylim(0.6, -0.6)
    ax[1].set_xlabel("deviation of the CO$_2$ exponent from 1/2")
    ax[1].set_title("b  CARB: is the $\\sqrt{C}$ law identified?", loc="left")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIG / "fig1_carb_audit.svg")
    fig.savefig(FIG / "fig1_carb_audit.pdf")
    fig.savefig(PRE / "fig1_carb_audit.png", dpi=220)
    plt.close(fig)

    # ---- figure: knowledge value --------------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    blk = s[s.k == 0]
    piv = blk.pivot_table(index="S_order", columns="method", values="rmse")
    lab = blk.drop_duplicates("S_order").set_index("S_order").S
    Smap = {"3": 3, "6": 6, "12": 12, "24": 24}
    Sx = np.array([Smap.get(str(v), np.nan) for v in lab[list(piv.index)]])
    gain = piv["DATA-RE"].values ** 2 - piv["KA-RE"].values ** 2
    ok = np.isfinite(Sx)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.plot(Sx[ok], gain[ok], "o", ms=3.6, color=C["CARB"])
    ss = np.linspace(3, 24.5, 200)
    ax.plot(ss, th.A / ss - th.B, "-", lw=1.1, color="0.15")
    if np.isfinite(th.A_meas) and th.B_meas > 0:
        ax.plot(ss, th.A_meas / ss - th.B_meas, "--", lw=0.9, color="0.55")
    ax.axvline(th.S_star_pred, color="0.4", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(2.6, 30)
    ax.set_xticks([3, 6, 12, 24], ["3", "6", "12", "24"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("training programmes $S$")
    ax.set_ylabel("knowledge gain (MSE, $k=0$)")
    ax.set_title(f"CARB: $S^*$={th.S_star_pred:.0f} (measured {th.S_star_meas:.0f})", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_carb_value.svg")
    fig.savefig(FIG / "fig2_carb_value.pdf")
    fig.savefig(PRE / "fig2_carb_value.png", dpi=220)
    plt.close(fig)
    print("tables and figures written")


if __name__ == "__main__":
    main()
