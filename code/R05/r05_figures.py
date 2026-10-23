"""R05 manuscript figures (vector SVG + PDF; PNG previews for QA only), built only from
the round's output tables."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kag_data_r05 as D  # noqa: E402

R05 = HERE.parent
OUT, FIG = R05 / "outputs", R05 / "figures"
FIG.mkdir(exist_ok=True)
(FIG / "preview").mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "legend.frameon": False, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})
COL = {"ANCHOR": "#000000", "KA-FIXED": "#E69F00", "KA-RE": "#56B4E9", "DATA-RE": "#009E73",
       "PAKA-S": "#0072B2", "PAKA": "#7F7F7F", "GBM": "#D55E00", "GBM-RES": "#CC79A7", "GBM-ME": "#8C510A",
       "GP-RES": "#999999"}
LAB = {"ANCHOR": "code as published", "KA-FIXED": "code exponents imposed", "KA-RE": "code as REML prior",
       "DATA-RE": "data-only hierarchical", "PAKA-S": "PAKA (proposed)", "PAKA": "1-SE selection",
       "GBM": "gradient boosting", "GBM-RES": "boosting on code residual", "GBM-ME": "mixed-effects boosting",
       "GP-RES": "GP on code residual"}
DOM = ["FRP", "SFRC", "CARB"]
TITLE = {"FRP": "FRP-RC shear", "SFRC": "SFRC shear", "CARB": "carbonation rate"}


def save(fig, stem):
    for ext in ("svg", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(FIG / "preview" / f"{stem}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig1():
    dom = pd.read_csv(OUT / "audit_domains.csv").set_index("domain")
    feat = pd.read_csv(OUT / "audit_features.csv")
    doms = DOM + ["DEEP"]
    short = {"FRP": "FRP", "SFRC": "SFRC", "CARB": "CARB", "DEEP": "DEEP"}
    fig = plt.figure(figsize=(7.4, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 1.1, 1.25], wspace=0.62)
    a = fig.add_subplot(gs[0])
    x = np.arange(len(doms))
    a.bar(x - 0.2, dom.loc[doms, "n"], 0.38, color="#BDBDBD", label="tests")
    a.bar(x + 0.2, dom.loc[doms, "effective_n"], 0.38, color="#0072B2", label="effective tests")
    for i, d in enumerate(doms):
        a.text(i, dom.loc[d, "n"] * 1.18, f"ICC\n{dom.loc[d, 'icc']:.2f}", ha="center", fontsize=5.8)
    a.set_yscale("log")
    a.set_ylim(10, 9000)
    a.set_xlim(-0.6, 3.6)
    a.set_xticks(x, [short[d] for d in doms], fontsize=5.6)
    a.set_ylabel("number of tests")
    a.legend(fontsize=5.8, loc="upper left", ncol=1, handlelength=1.0)
    a.set_title("a  information content", loc="left", fontsize=7.5)

    b = fig.add_subplot(gs[1])
    rows = feat[feat.domain.isin(doms)].reset_index(drop=True)
    ypos, labels = [], []
    y = 0
    for d in doms:
        blk = rows[rows.domain == d]
        for r in blk.itertuples():
            frac = r.programmes_varying / r.programmes_total
            b.scatter(r.between_share, y, s=12 + 70 * frac, color="#0072B2" if r.role == "coded" else "#E69F00",
                      edgecolor="k", lw=0.3, zorder=3)
            b.text(0.415, y, f"{r.programmes_varying}/{r.programmes_total}", va="center", ha="left", fontsize=5.4,
                   color="#555555")
            ypos.append(y)
            labels.append(f"{short[d]} {D.FEATURE_TEX.get(r.feature, r.feature)}")
            y -= 1
        y -= 0.6
    b.set_yticks(ypos, labels, fontsize=5.8)
    b.set_xlim(0.4, 1.0)
    b.axvline(0.5, color="#BDBDBD", lw=0.5, ls=":")
    b.set_xlabel("share of variance between programmes")
    b.set_title("b  where the design variables vary", loc="left", fontsize=7.5)

    c = fig.add_subplot(gs[2])
    keys = [("FRP", "ln_d"), ("FRP", "ln_Ef"), ("SFRC", "ln_d"), ("SFRC", "ln_fc"), ("CARB", "x_co2"),
            ("CARB", "x_cao"), ("DEEP", "ln_d"), ("DEEP", "ln_ad")]
    for i, (d, f) in enumerate(keys):
        r = feat[(feat.domain == d) & (feat.feature == f)].iloc[0]
        yy = -i
        c.errorbar(r.uni_within, yy + 0.14, xerr=1.96 * r.uni_within_se, fmt="o", ms=3, color="#0072B2", lw=0.9,
                   label="within programmes" if i == 0 else None)
        c.errorbar(r.uni_between, yy - 0.14, xerr=1.96 * r.uni_between_se, fmt="s", ms=3, color="#D55E00", lw=0.9,
                   label="between programmes" if i == 0 else None)
        if not pd.isna(r.get("lambda_s_implied", np.nan)):
            c.plot([r.lambda_s_implied] * 2, [yy - 0.35, yy + 0.35], color="k", lw=1.2,
                   label="ACI 318-19 size factor" if f == "ln_d" and d == "FRP" else None)
    c.axvline(0, color="k", lw=0.5)
    c.set_yticks([-i for i in range(len(keys))],
                 [f"{short[d]} {D.FEATURE_TEX.get(f, f)}" for d, f in keys], fontsize=6)
    c.set_ylim(-len(keys) + 0.4, 1.6)
    c.set_xlabel("exponent deviation from the anchor (95% CI)")
    c.legend(fontsize=5.6, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8)
    c.set_title("c  physics within vs between programmes", loc="left", fontsize=7.5)
    save(fig, "fig1_evidence_structure")


def fig2():
    s = pd.read_csv(OUT / "lopo_summary.csv")
    meths = ["KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S", "GBM", "GBM-RES", "GBM-ME", "GP-RES"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for i, d in enumerate(DOM):
        a = ax[i]
        for m in meths:
            blk = s[(s.dataset == d) & (s.method == m)].sort_values("k")
            if blk.empty:
                continue
            a.plot(blk.k, blk.rmse, "-o", ms=2.6, lw=1.4 if m == "PAKA-S" else 0.9, color=COL[m], label=LAB[m],
                   zorder=4 if m == "PAKA-S" else 2)
            if m in ("PAKA-S", "GBM-ME"):
                a.fill_between(blk.k, blk.rmse_lo, blk.rmse_hi, color=COL[m], alpha=0.12, lw=0)
        n_p = int(s[(s.dataset == d) & (s.k == 0)].programmes.iloc[0])
        a.set_title(f"{'abc'[i]}  {TITLE[d]} ({n_p} held-out programmes)", loc="left", fontsize=7.2)
        a.set_xticks([0, 1, 2, 3, 5])
        a.set_xlabel("tests k observed from the new programme")
        a.set_ylabel("RMSE of ln(observed / predicted)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_new_programme_prediction")


def fig3():
    s = pd.read_csv(OUT / "scarcity_summary.csv")
    th = pd.read_csv(OUT / "theory_crossover.csv").set_index("domain")
    pp = pd.read_csv(OUT / "scarcity_per_draw.csv.gz")
    order = ["3", "6", "12", "24", "all"]
    meths = ["ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S", "GBM-RES", "GBM-ME"]
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.0), gridspec_kw=dict(hspace=0.6, wspace=0.48))
    rng = np.random.default_rng(0)
    for i, d in enumerate(DOM):
        a = ax[0, i]
        sd = s[(s.dataset == d) & (s.k == 0)]
        n_all = sd[sd.S == "all"].n_train.iloc[0] if "n_train" in sd else np.nan
        progs_all = D.load(d).source.nunique() - 1
        xs = np.array([3, 6, 12, 24, progs_all], float)
        for m in meths:
            blk = sd[sd.method == m].set_index("S").reindex(order)
            if blk.rmse.isna().all():
                continue
            a.plot(xs, blk.rmse, marker="o", ms=2.6, lw=1.4 if m == "PAKA-S" else 0.9, color=COL[m], label=LAB[m],
                   ls="--" if m == "ANCHOR" else "-", zorder=4 if m == "PAKA-S" else 2)
            if m in ("PAKA-S", "DATA-RE", "KA-FIXED"):
                a.fill_between(xs, blk.rmse_lo, blk.rmse_hi, color=COL[m], alpha=0.10, lw=0)
        t = th.loc[d]
        a.axvspan(t.S_star_lo, t.S_star_hi, color="#0072B2", alpha=0.07, lw=0)
        a.axvline(t.S_star, color="#0072B2", lw=0.8, ls="--")
        a.axvline(t.S_obs, color="k", lw=0.8)
        a.set_xscale("log")
        a.set_xticks(xs, [f"{int(v)}" for v in xs])
        a.minorticks_off()
        top = 1.05 * sd[sd.method.isin(["KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S"])].rmse.max()
        a.set_ylim(0.8 * sd[sd.method.isin(["KA-RE", "DATA-RE", "PAKA-S"])].rmse.min(), top)
        a.set_xlabel("programmes in the training library")
        a.set_ylabel("new-programme RMSE (k = 0)")
        a.set_title(f"{'abc'[i]}  {TITLE[d]}", loc="left", fontsize=7.5)
        # theory vs observed: G(S) = MSE_data - MSE_fixed
        b = ax[1, i]
        q = pp[(pp.dataset == d) & (pp.k == 0) & (pp.method.isin(["KA-FIXED", "DATA-RE"])) & (pp.S != "all")]
        per = q.groupby(["S", "method", "source"]).rmse.mean().unstack("method")
        per["g"] = per["DATA-RE"] ** 2 - per["KA-FIXED"] ** 2
        tab = per.g.unstack("S")[["3", "6", "12", "24"]].to_numpy()
        mean = tab.mean(axis=0)
        boots = np.array([tab[rng.integers(0, len(tab), len(tab))].mean(axis=0) for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        Sx = np.array([3, 6, 12, 24], float)
        b.errorbar(Sx, mean, yerr=[mean - lo, hi - mean], fmt="o", ms=3, color="k", lw=0.9,
                   label="observed (held-out programmes)")
        grid = np.geomspace(3, 40, 200)
        b.plot(grid, t.p * t.c / grid - t.B, color="#0072B2", lw=1.2, label="predicted, p c / S - B")
        b.axhline(0, color="k", lw=0.5)
        b.set_xscale("log")
        b.set_xticks([3, 6, 12, 24, 40], ["3", "6", "12", "24", "40"])
        b.minorticks_off()
        b.set_xlabel("programmes in the training library")
        b.set_ylabel("MSE data only - MSE code imposed")
        b.set_title(f"{'def'[i]}  S* predicted {t.S_star:.1f}, observed {t.S_obs:.1f}", loc="left", fontsize=7.0)
        if i == 2:
            b.legend(fontsize=6, loc="upper right")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.15)
    save(fig, "fig3_value_of_codified_knowledge")


def fig4():
    u = pd.read_csv(OUT / "unit_summary.csv")
    e = pd.read_csv(OUT / "extrap_summary.csv")
    c = pd.read_csv(OUT / "circular_summary.csv")
    lp = pd.read_csv(OUT / "lopo_summary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.5))
    a = ax[0]
    for j, (d, mk) in enumerate((("FRP", "o"), ("SFRC", "s"), ("CARB", "^"))):
        for m, sh in (("DATA-RE", -0.12), ("KA-RE", 0.0), ("PAKA-S", 0.12)):
            blk = u[(u.dataset == d) & (u.k == 0) & (u.method == f"{m}|DISPERSED")].sort_values("N")
            if blk.empty:
                continue
            x = np.arange(len(blk)) + 4 * j + sh
            col = f"gain_vs_{m}|CONCENTRATED"
            a.errorbar(x, blk[col], yerr=[blk[col] - blk[col + "_lo"], blk[col + "_hi"] - blk[col]], fmt=mk, ms=3,
                       color=COL[m], lw=0.8, label=LAB[m] if j == 0 else None)
    a.axhline(0, color="k", lw=0.5)
    ticks, tl = [], []
    for j, d in enumerate(("FRP", "SFRC", "CARB")):
        Ns = sorted(u[u.dataset == d].N.unique())
        ticks += list(np.arange(len(Ns)) + 4 * j)
        tl += [f"{n}" for n in Ns]
    a.set_xticks(ticks, tl, fontsize=6)
    a.set_xlabel("training tests N  (FRP | SFRC | carbonation)")
    a.set_ylabel("RMSE gain, many vs few programmes")
    a.legend(fontsize=5.8, loc="upper right")
    a.set_title("a  programmes vs tests", loc="left", fontsize=7.2)

    b = ax[1]
    meths = ["KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S", "GBM", "GBM-RES", "GBM-ME", "GP-RES"]
    w = 0.38
    for i, d in enumerate(("FRP", "SFRC")):
        blk = e[(e.dataset == d) & (e.k == 0)].set_index("method").reindex(meths)
        x = np.arange(len(meths)) + (i - 0.5) * w
        b.bar(x, blk.rmse, w, color=[COL[m] for m in meths], edgecolor="k", lw=0.3, hatch=None if i == 0 else "///")
    b.set_xticks(np.arange(len(meths)), meths, rotation=60, ha="right", fontsize=5.8)
    b.set_ylabel("RMSE, larger-size programmes (k = 0)")
    b.set_title("b  size extrapolation", loc="left", fontsize=7.2)

    cc = ax[2]
    meths2 = ["KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S", "GBM-ME"]
    for i, (d, mk) in enumerate((("FRP", "o"), ("SFRC", "s"))):
        full = lp[(lp.dataset == d) & (lp.k == 0)].set_index("method").reindex(meths2).rmse.to_numpy()
        post = c[(c.dataset == d) & (c.k == 0)].set_index("method").reindex(meths2).rmse.to_numpy()
        x = np.arange(len(meths2)) + (i - 0.5) * 0.36
        for xi, f0, p0, m in zip(x, full, post, meths2):
            cc.plot([xi, xi], [f0, p0], color="#BDBDBD", lw=0.8, zorder=1)
        cc.scatter(x, full, marker=mk, color=[COL[m] for m in meths2], edgecolor="k", lw=0.3, s=16, zorder=3,
                   label=f"{d}, all programmes")
        cc.scatter(x, post, marker="x", color="k", s=16, lw=0.8, zorder=4,
                   label="after calibration year" if i == 0 else None)
    cc.set_xticks(np.arange(len(meths2)), meths2, rotation=50, ha="right", fontsize=5.8)
    cc.set_ylabel("new-programme RMSE (k = 0)")
    cc.legend(fontsize=5.4, loc="upper right")
    cc.set_title("c  anchor-calibration circularity", loc="left", fontsize=7.2)
    save(fig, "fig4_unit_extrapolation_circularity")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("figures written")
