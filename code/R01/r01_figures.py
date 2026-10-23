"""R01 figures (vector SVG + PDF) built only from the round's output tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R01 = Path(__file__).resolve().parents[1]
OUT, FIG = R01 / "outputs", R01 / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "legend.frameon": False})
OI = {"KA-FIXED": "#E69F00", "KA-RE": "#0072B2", "KA-WB": "#56B4E9", "DATA-RE": "#009E73",
      "DATA-WB": "#7FCDBB", "GBM": "#D55E00", "GBM-FEAT": "#F0A07A", "GBM-RES": "#CC79A7",
      "GP-RES": "#999999", "ANCHOR": "#000000"}
LABEL = {"KA-FIXED": "code fixed + programme effect", "KA-RE": "knowledge-anchored hierarchical",
         "KA-WB": "knowledge-anchored within-between", "DATA-RE": "data-only hierarchical power law",
         "DATA-WB": "data-only within-between", "GBM": "gradient boosting", "GBM-FEAT": "boosting + code feature",
         "GBM-RES": "boosting on code residual", "GP-RES": "GP on code residual", "ANCHOR": "code as published"}
DS_TITLE = {"FRP": "FRP-RC beams (ACI 440.1R-15 anchor)", "SFRC": "SFRC beams (Narayanan-Darwish anchor)"}
FEAT_SHORT = {"ln_d": "d", "ln_bw": "b_w", "ln_fc": "f'c", "ln_rho": "rho", "ln_Ef": "E_f", "ln_ad": "a/d",
              "ln1p_F": "F"}


def save(fig, stem):
    for ext in ("svg", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    (FIG / "preview").mkdir(exist_ok=True)
    fig.savefig(FIG / "preview" / f"{stem}.png", dpi=150, bbox_inches="tight")   # QA preview only
    plt.close(fig)


def fig_confounding():
    f = pd.read_csv(OUT / "confounding_features.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    for i, ds in enumerate(("FRP", "SFRC")):
        d = f[f.dataset == ds].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        a = ax[i, 0]
        a.barh(y, d.between_share, color="#4D4D4D", height=0.6)
        a.barh(y, 1 - d.between_share, left=d.between_share, color="#D9D9D9", height=0.6)
        for yy, r in zip(y, d.itertuples()):
            a.text(1.02, yy, f"{r.programmes_varying}/{r.programmes_total}", va="center", fontsize=7)
        a.set_yticks(y, [FEAT_SHORT[v] for v in d.feature])
        a.set_xlim(0, 1)
        a.set_xlabel("share of log-feature variance between programmes")
        a.set_title(f"{DS_TITLE[ds]}", loc="left", fontsize=8)
        b = ax[i, 1]
        for j, r in enumerate(d.itertuples()):
            for off, est, se, col, lab in ((0.15, r.anchor_resid_beta_within, r.anchor_resid_se_within, "#0072B2",
                                            "within programmes"),
                                           (-0.15, r.anchor_resid_beta_between, r.anchor_resid_se_between, "#D55E00",
                                            "between programmes")):
                b.errorbar(est, y[j] + off, xerr=1.96 * se, fmt="o", ms=3, color=col, lw=0.9,
                           label=lab if j == 0 else None)
        b.axvline(0, color="k", lw=0.5)
        b.set_yticks(y, [FEAT_SHORT[v] for v in d.feature])
        b.set_xlabel("exponent deviation from the code (95% CI)")
        b.set_xlim(-1.6, 1.6)
        if i == 0:
            b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig1_programme_confounding")


def _band(a, x, lo, hi, col):
    a.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)


def fig_fewshot():
    s = pd.read_csv(OUT / "fewshot_summary.csv")
    meths = ("KA-FIXED", "KA-RE", "DATA-RE", "GBM", "GBM-RES", "GP-RES")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for i, ds in enumerate(("FRP", "SFRC")):
        a = ax[i]
        for mm in meths:
            d = s[(s.dataset == ds) & (s.method == mm) & (s.k <= 3)].sort_values("k")
            a.plot(d.k, d.rmse, "-o", ms=3, color=OI[mm], lw=1.1, label=LABEL[mm])
            if mm in ("KA-RE", "DATA-RE", "GBM"):
                _band(a, d.k, d.rmse_lo, d.rmse_hi, OI[mm])
        an = s[(s.dataset == ds) & (s.method == "ANCHOR") & (s.k == 0)].rmse.iloc[0]
        a.set_title(f"{'a' if i == 0 else 'b'}  {ds}: 18 held-out programmes (code as published {an:.2f})",
                    loc="left", fontsize=7.5)
        a.set_xticks([0, 1, 2, 3])
        a.set_xlabel("tests k observed from the new programme")
        a.set_ylabel("RMSE of ln(V_test / V_pred)")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig2_new_programme_fewshot")


def fig_scarcity():
    s = pd.read_csv(OUT / "scarcity_summary.csv")
    order = ["3", "6", "12", "24", "all"]
    meths = ("KA-FIXED", "KA-RE", "DATA-RE", "GBM-RES", "GBM")
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.9),
                           gridspec_kw=dict(width_ratios=[1, 1, 0.95], wspace=0.5, hspace=0.85))
    for i, ds in enumerate(("FRP", "SFRC")):
        for j, k in enumerate((0, 3)):
            a = ax[i, j]
            for mm in meths:
                d = s[(s.dataset == ds) & (s.method == mm) & (s.k == k)].set_index("S").reindex(order)
                x = np.arange(len(order))
                a.plot(x, d.rmse, "-o", ms=3, color=OI[mm], lw=1.1, label=LABEL[mm])
                if mm in ("KA-RE", "DATA-RE"):
                    _band(a, x, d.rmse_lo, d.rmse_hi, OI[mm])
            a.set_ylim(0.1, 0.75)
            g = s[(s.dataset == ds) & (s.method == "GBM") & (s.k == k)].set_index("S").reindex(order).rmse
            if g.iloc[0] > 0.75:
                a.text(0.03, 0.97, f"boosting at 3 programmes: {g.iloc[0]:.2f}", transform=a.transAxes,
                       fontsize=5.8, color=OI["GBM"], ha="left", va="top")
            n_tr = s[(s.dataset == ds) & (s.method == "KA-RE") & (s.k == k)].set_index("S").reindex(order).n_train
            a.set_xticks(np.arange(len(order)), [f"{o}\n({v:.0f})" for o, v in zip(order, n_tr)], fontsize=6.5)
            a.set_xlabel("training programmes (mean specimens)", fontsize=7)
            a.set_ylabel("RMSE of ln(V_test / V_pred)", fontsize=7)
            a.set_title(f"{'abcdef'[3 * i + j]}  {ds}, k = {k}", loc="left", fontsize=7.5)
        a = ax[i, 2]
        for k, mk in ((0, "o"), (3, "s")):
            d = s[(s.dataset == ds) & (s.method == "KA-RE") & (s.k == k)].set_index("S").reindex(order)
            x = np.arange(len(order)) + (0.08 if k else -0.08)
            a.errorbar(x, d["gain_vs_DATA-RE"], yerr=[d["gain_vs_DATA-RE"] - d["gain_vs_DATA-RE_lo"],
                                                       d["gain_vs_DATA-RE_hi"] - d["gain_vs_DATA-RE"]],
                       fmt=mk, ms=3, color="#0072B2" if k == 0 else "#56B4E9", lw=0.9, label=f"k = {k}")
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks(np.arange(len(order)), order, fontsize=6.5)
        a.set_xlabel("training programmes", fontsize=7)
        a.set_ylabel("RMSE gain of the anchor", fontsize=7)
        a.set_title(f"{'abcdef'[3 * i + 2]}  {ds}: anchored minus data-only", loc="left", fontsize=7.5)
        a.legend(fontsize=6.5)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.17)
    save(fig, "fig3_knowledge_value_vs_programmes")


def fig_extrap_stress_unit():
    e = pd.read_csv(OUT / "extrapolation_summary.csv")
    st = pd.read_csv(OUT / "stress_summary.csv") if (OUT / "stress_summary.csv").exists() else None
    ev = pd.read_csv(OUT / "evidence_unit_summary.csv") if (OUT / "evidence_unit_summary.csv").exists() else None
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw=dict(wspace=0.42))
    meths = ("KA-FIXED", "KA-RE", "KA-WB", "DATA-RE", "GBM", "GBM-RES", "GP-RES")
    a = ax[0]
    w = 0.38
    for i, ds in enumerate(("FRP", "SFRC")):
        d = e[(e.dataset == ds) & (e.k == 0)].set_index("method").reindex(meths)
        x = np.arange(len(meths)) + (i - 0.5) * w
        a.bar(x, d.rmse, width=w, color=[OI[m] for m in meths], edgecolor="k", lw=0.4,
              hatch=None if i == 0 else "///")
    a.set_xticks(np.arange(len(meths)), meths, fontsize=5.8, rotation=60, ha="right")
    a.set_ylabel("RMSE, larger-size programmes (k = 0)", fontsize=7)
    a.set_title("a  size extrapolation\n    (plain FRP, hatched SFRC)", loc="left", fontsize=7.2)
    if st is not None:
        a = ax[1]
        variants = ["none", "size+0.3", "size-0.3", "missing-term"]
        for mm, mk in (("KA-FIXED", "o"), ("KA-RE", "s"), ("GBM-RES", "^")):
            for ds, ls in (("FRP", "-"), ("SFRC", "--")):
                d = st[(st.dataset == ds) & (st.method == mm) & (st.k == 0) & (st.S_label == "6")]
                d = d.set_index("variant").reindex(variants)
                a.plot(np.arange(4), d.rmse, ls, marker=mk, ms=3, color=OI[mm], lw=1,
                       label=f"{mm} ({ds})")
        a.set_xticks(np.arange(4), ["as coded", "d exponent +0.3", "d exponent -0.3", "term removed"],
                     fontsize=5.8, rotation=35, ha="right")
        a.set_ylabel("RMSE, 6 training programmes (k = 0)", fontsize=7)
        a.set_title("b  corrupted codified knowledge\n    (solid FRP, dashed SFRC)", loc="left", fontsize=7.2)
        a.legend(fontsize=5.3, ncol=1, loc="upper left")
    if ev is not None:
        a = ax[2]
        for ds, mk, sh in (("FRP", "o", -0.06), ("SFRC", "s", 0.06)):
            for mm in ("KA-RE", "DATA-RE", "GBM-RES"):
                d = ev[(ev.dataset == ds) & (ev.method == mm) & (ev.k == 0)].sort_values("N")
                xs = np.log2(d.N / 24.0) + sh
                a.errorbar(xs, d.dispersed_gain,
                           yerr=[d.dispersed_gain - d.gain_lo, d.gain_hi - d.dispersed_gain], fmt=mk, ms=3,
                           color=OI[mm], lw=0.9, label=f"{mm} ({ds})")
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks([0, 1, 2], ["24", "48", "96"])
        a.set_xlabel("training specimens N", fontsize=7)
        a.set_ylabel("RMSE gain, dispersed vs concentrated", fontsize=7)
        a.set_title("c  more programmes at fixed N\n    (k = 0)", loc="left", fontsize=7.2)
        a.legend(fontsize=5.3, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_extrapolation_stress_evidence_unit")


if __name__ == "__main__":
    fig_confounding()
    fig_fewshot()
    fig_scarcity()
    fig_extrap_stress_unit()
    print("figures written to", FIG)
