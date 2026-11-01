"""R09 figure rebuild at print size (6.3 in wide, fonts >= 7.5 pt).

Sources: R05 audit/lopo outputs, R06 ladder outputs, R07 tost/pglo outputs.
Colours: Okabe-Ito, one scheme everywhere --
code imposed #E69F00, code as prior #56B4E9, data only #333333 dashed,
PAKA #0072B2 thick, boosting #D55E00, mixed-effects boosting #CC79A7,
boosting on code residual #8C564B, GP on code residual #999999.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROOT = R08.parent.parent
R05 = ROOT / "rounds" / "R05_consolidation_2026-10-21"
R06 = ROOT / "rounds" / "R06_mechanism_pivot_2026-10-22"
R07 = ROOT / "rounds" / "R07_merged_protocol_2026-10-22"
FIG = R08 / "figures"
FIG.mkdir(exist_ok=True, parents=True)

C = dict(imposed="#E69F00", prior="#56B4E9", data="#333333", paka="#0072B2",
         gbm="#D55E00", gbmme="#CC79A7", gbmres="#8C564B", gpres="#999999",
         ink="#1a1a1a", grey="#7f7f7f", light="#d9d9d9", green="#009E73",
         purple="#CC79A7", sky="#56B4E9", orange="#E69F00", red="#D55E00")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
DOMLAB = {"FRP": "FRP shear", "SFRC": "SFRC shear", "CARB": "carbonation",
          "DEEP": "deep beams"}
FEAT = {"ln_d": "$d$", "ln_bw": "$b_w$", "ln_fc": "$f'_c$", "ln_rho": r"$\rho$",
        "ln_Ef": "$E_f$", "ln_ad": "$a/d$", "ln1p_F": "$F$",
        "x_cao": "CaO$_r$", "x_co2": "CO$_2$", "x_wb": "$w/b$"}
ARM = {"KA-FIXED": ("code imposed", C["imposed"]),
       "KA-RE": ("code as prior", C["prior"]),
       "DATA-RE": ("data only", C["data"]),
       "PAKA-S": ("PAKA", C["paka"]),
       "GBM": ("boosting", C["gbm"]),
       "GBM-ME": ("mixed-effects boosting", C["gbmme"]),
       "GBM-RES": ("boosting on code residual", C["gbmres"]),
       "GP-RES": ("GP on code residual", C["gpres"])}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{name}.png", dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def frame(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(ax, x, y, w, h, label, ec=C["grey"], fs=7.5, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 facecolor=fc, edgecolor=ec, linewidth=0.7))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, linespacing=1.25)


def arr(ax, x0, y0, x1, y1, color=C["ink"], lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=6, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5))


def concept():
    fig = plt.figure(figsize=(6.3, 3.9))
    gs = fig.add_gridspec(2, 3, left=.02, right=.98, top=.86, bottom=.06,
                          wspace=.18, hspace=.30,
                          width_ratios=(1, 1, 1), height_ratios=(1, 1))
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs[1, 0]); axe = fig.add_subplot(gs[1, 1:3])
    for ax in (axa, axb, axc, axd, axe):
        frame(ax)
    hdr = [((0, 0), "Evidence"), ((0, 1), "Evidence"), ((0, 2), "Certify"),
           ((1, 0), "Price"), ((1, 1), "Price"), ((1, 2), "Allocate")]
    # grey stage label above each panel title
    for ax, lab in ((axa, "Evidence"), (axb, "Evidence"), (axc, "Certify"),
                    (axd, "Price"), (axe, "Allocate")):
        ax.text(.5, 1.00, lab, ha="center", va="top", fontsize=7.5,
                color=C["grey"], weight="bold", transform=ax.transAxes)
    # (a)
    axa.text(.5, .88, "(a) A design equation is a\nset of hypotheses",
             ha="center", va="top", fontsize=8)
    box(axa, .04, .42, .55, .16, r"$V_c=0.4\sqrt{f'_c}\,b_w\,kd$",
        ec=C["ink"], fs=7.5)
    for k, txt in enumerate(("sign +", "exponent 1", r"range $a/d\geq2.5$")):
        axa.text(.63, .55 - .09 * k, txt, ha="left", va="center",
                 fontsize=7.5, color=C["paka"])
    axa.text(.5, .16, "omitted variable:\nclaims no dependence", ha="center",
             va="center", fontsize=7.5, color=C["grey"], style="italic")
    # (b)
    axb.text(.5, .88, "(b) Tests cluster by\nprogramme", ha="center", va="top",
             fontsize=8)
    rng = np.random.default_rng(3)
    for cx, cy in rng.uniform(.20, .80, size=(6, 2)):
        yy = .50 + .22 * (cy - .5)
        axb.add_patch(Circle((cx, yy), .05, color=C["sky"], alpha=.28,
                             ec="none"))
        for _ in range(5):
            axb.add_patch(Circle((cx + rng.normal(0, .024),
                                  yy + rng.normal(0, .030)),
                                 .0075, color=C["ink"], alpha=.85, ec="none"))
    axb.text(.5, .10, "2,158 tests, 222 programmes\n"
             r"$\rightarrow$ 41–61 effective observations per database",
             ha="center", va="center", fontsize=7.5)
    # (c)
    axc.text(.5, .90, "(c) Within- vs between-\nprogramme contrasts",
             ha="center", va="top", fontsize=8)
    axc.add_patch(FancyBboxPatch((.06, .40), .40, .26, boxstyle="round,pad=.02",
                  fc="#eef5f8", ec=C["sky"], lw=.9))
    axc.add_patch(FancyBboxPatch((.54, .40), .40, .26, boxstyle="round,pad=.02",
                  fc="#fbefeb", ec=C["red"], lw=.9))
    for x0, cc in ((.15, C["green"]), (.63, C["red"])):
        axc.add_patch(Circle((x0, .48), .014, color=C["ink"]))
        axc.add_patch(Circle((x0 + .18, .60), .014, color=C["ink"]))
        arr(axc, x0 + .01, .49, x0 + .17, .59, cc)
    axc.plot(.83, .52, "x", ms=9, color=C["red"], mew=2)
    axc.text(.26, .32, "within: free of\nlaboratory effects", ha="center",
             va="top", fontsize=7.5, color=C["green"])
    axc.text(.74, .32, "between:\nconfounded", ha="center", va="top",
             fontsize=7.5, color=C["red"])
    # (d)
    axd.text(.5, .88, "(d) Price: when do data\novertake the equation?",
             ha="center", va="top", fontsize=8)
    S = np.linspace(3, 30, 300); p, c_, B, s2 = 6, .35, .30, .05; Ss = p * c_ / B
    xs = (S - 3) / 27 * .72 + .12
    axd.plot(xs, s2 + B + c_ / S, color=C["imposed"], lw=1.3)
    axd.plot(xs, s2 + .62 * (p + 1) * c_ / S, color=C["data"], lw=1.3, ls="--")
    sx = (Ss - 3) / 27 * .72 + .12
    axd.plot([sx], [s2 + B + c_ / Ss], "o", ms=4.5, color=C["ink"])
    axd.plot([sx, sx], [.10, s2 + B + c_ / Ss], ls=":", lw=.8, color=C["grey"])
    axd.text(sx, .10, r"$S^*=pc/B$", ha="center", va="center", fontsize=7.5)
    axd.text(.84, s2 + B + c_ / 30 + .05, "code imposed", ha="center",
             va="bottom", fontsize=7.5, color=C["imposed"])
    axd.text(.86, s2 + (p + 1) * c_ / 30 - .04, "data only", ha="left",
             va="center", fontsize=7.5, color=C["data"])
    # (e)
    axe.text(.5, .88, "(e) Allocate: re-weight\nwith the first tests",
             ha="center", va="top", fontsize=8)
    for k, lab in enumerate(("code\nimposed", "code\nas prior", "data\nonly")):
        yy = .68 - .20 * k
        box(axe, .02, yy - .07, .24, .14, lab, fs=7.5)
        arr(axe, .26, yy, .36, .50, C["grey"])
    box(axe, .36, .38, .30, .24, "weights from\nwhole-programme\nscores",
        ec=C["ink"], fs=7.5)
    arr(axe, .66, .50, .74, .50, C["ink"])
    box(axe, .74, .38, .24, .24, "update with\n$k$ tests of a\nnew programme",
        ec=C["paka"], fs=7.5)
    axe.text(.5, .14, "13–32% lower error at $k=3$", ha="center", va="center",
             fontsize=7.5, color=C["green"])
    save(fig, "fig0_concept_r09")


def evidence():
    ad = pd.read_csv(R05 / "outputs" / "audit_domains.csv").set_index("domain")
    af = pd.read_csv(R05 / "outputs" / "audit_features.csv")
    tv = pd.read_csv(R07 / "outputs" / "tost_verdicts.csv")
    fig = plt.figure(figsize=(6.3, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.25), hspace=.75,
                          wspace=.55)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[:, 1])
    axc = fig.add_subplot(gs[1, 0])
    # (a) tests vs effective observations
    ax = axa
    doms = ["FRP", "SFRC", "CARB", "DEEP"]
    x = np.arange(4)
    n = [ad.loc[d, "n"] for d in doms]
    ne = [ad.loc[d, "effective_n"] for d in doms]
    icc = [ad.loc[d, "icc"] for d in doms]
    ax.bar(x - .19, n, .36, color=C["ink"], label="tests")
    ax.bar(x + .19, ne, .36, color=C["sky"], label="effective\nobservations")
    for i, (a, b, ic) in enumerate(zip(n, ne, icc)):
        ax.text(i, max(a, b) * 1.6, f"ICC\n{ic:.2f}", ha="center", va="bottom",
                fontsize=7.5, linespacing=1.1)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_ylim(20, 9000)
    ax.set_xticklabels(doms, fontsize=7.5, rotation=90)
    ax.set_title("(a) tests (black) and effective\nobservations (blue)",
                 loc="left", fontsize=8.5)
    ax.set_ylabel("count (log scale)")
    # (b) between share per coded variable
    ax = axb
    rows = af[(af.domain.isin(doms)) & (af.role == "coded")].copy()
    rows["lab"] = rows.domain.map(DOMLAB) + " " + rows.feature.map(
        lambda s: FEAT.get(s, s))
    rows = rows.sort_values(["domain", "between_share"],
                            key=lambda s: s.map({d: i for i, d in enumerate(doms)})
                            .fillna(s), ascending=[True, False])
    y = np.arange(len(rows))
    dc = {"FRP": C["paka"], "SFRC": C["imposed"], "CARB": C["green"],
          "DEEP": C["red"]}
    ax.scatter(rows.between_share, y, s=24, c=rows.domain.map(dc), zorder=4)
    for yi, r in zip(y, rows.itertuples()):
        ax.text(1.10, yi, f"{r.programmes_varying}/{r.programmes_total}",
                fontsize=7, ha="right", va="center", color=C["grey"])
    ax.set_yticks(y); ax.set_yticklabels(rows["lab"], fontsize=7.5)
    ax.yaxis.tick_right()
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("between-programme share of log variance")
    ax.set_title("(b) design variables vary\nbetween programmes", loc="left",
                 fontsize=8.5)
    # (c) depth exponent within vs between
    ax = axc
    grp = [("FRP", "K1"), ("SFRC", "K1"), ("DEEP", "ARCH")]
    ylab = {"FRP K1": "FRP shear", "SFRC K1": "SFRC shear",
            "DEEP ARCH": "deep beams"}
    for i, (dom, rung) in enumerate(grp):
        r = tv[(tv.domain == dom) & (tv.rung == rung) & (tv.feature == "ln_d")].iloc[0]
        ax.errorbar(r.dW, i, xerr=1.645 * r.dW_se, fmt="o", ms=6,
                    color=C["paka"], capsize=3, zorder=4)
        ax.errorbar(r.dB, i, xerr=1.645 * r.dB_se, fmt="o", ms=6,
                    mfc="white", mec=C["imposed"], color=C["imposed"],
                    capsize=3, zorder=4)
        if np.isfinite(r.lambda_s_implied):
            ax.plot([r.lambda_s_implied, r.lambda_s_implied],
                    [i - .26, i + .26], color=C["ink"], lw=2.4)
    ax.axvline(0, color=C["grey"], lw=.7, ls=":")
    ax.set_yticks(range(3))
    ax.set_yticklabels([ylab[f"{d} {r_}"] for d, r_ in grp], fontsize=7.5)
    ax.set_xlabel("depth-exponent deviation")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C["paka"],
                              label="within (filled)"),
                       Line2D([], [], marker="o", ls="", mfc="white",
                              mec=C["imposed"], color=C["imposed"],
                              label="between (open)"),
                       Line2D([], [], color=C["ink"], lw=2.4,
                              label="$\\lambda_s$ implied")],
              frameon=False, fontsize=7.5, loc="upper center", ncol=3,
              bbox_to_anchor=(.5, -.30))
    ax.set_title("(c) size exponent, within\nvs between programmes",
                 loc="left", fontsize=8.5)
    for ax in (axa, axb, axc):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=.6)
    save(fig, "fig_evidence_r09")


def grading():
    tv = pd.read_csv(R07 / "outputs" / "tost_verdicts.csv")
    t = tv[tv.domain != "DEEP-WB"].copy()
    order = [("FRP", "K1"), ("FRP", "K2"), ("SFRC", "K1"), ("SFRC", "K2"),
             ("CARB", "K1"), ("CARB", "K2"), ("CARB", "K3"),
             ("DEEP", "BEAM"), ("DEEP", "ARCH"), ("DEEP", "ARCHS"),
             ("DEEP", "ARCH99")]
    vcol = {"CONFIRMED": C["green"], "ATTENUATED": C["imposed"],
            "CONTRADICTED": C["red"], "NOT-EVALUABLE": C["gpres"]}
    vlab = {"CONFIRMED": "confirmed", "ATTENUATED": "mis-scaled",
            "CONTRADICTED": "contradicted", "NOT-EVALUABLE": "not evaluable"}
    rows, ticklab, gaps = [], [], []
    for dom, rung in order:
        blk = t[(t.domain == dom) & (t.rung == rung)]
        for r in blk.itertuples():
            rows.append(r)
            ticklab.append(f"{dom} {rung} \u00b7 {FEAT[r.feature]}")
        gaps.append(len(rows) - .5)
    fig, ax = plt.subplots(figsize=(6.3, 8.2))
    X = 1.2
    # margin band per row: +-eps around 0
    for i, r in enumerate(rows):
        lo_, hi_ = -r.eps, r.eps
        ax.fill_betweenx([i - .42, i + .42], lo_, hi_, color=C["light"],
                         alpha=.55, lw=0, zorder=1)
    ax.axvline(0, color=C["grey"], lw=.7)
    for i, r in enumerate(rows):
        y = i
        lo_, hi_ = r.dW - 1.645 * r.dW_se, r.dW + 1.645 * r.dW_se
        col = vcol[r.verdict_tost]
        lo_c, hi_c = max(lo_, -X), min(hi_, X)
        dw = np.clip(r.dW, -X + .02, X - .02)
        ax.plot([lo_c, hi_c], [y, y], color=col, lw=1.4, zorder=3)
        ax.plot(dw, y, "o", ms=5.5, color=col, zorder=4)
        if lo_ < -X:
            ax.add_patch(FancyArrowPatch((-X + .03, y), (-X + .005, y),
                         arrowstyle="<|-", mutation_scale=7, color=col))
        if hi_ > X:
            ax.add_patch(FancyArrowPatch((X - .03, y), (X - .005, y),
                         arrowstyle="-|>", mutation_scale=7, color=col))
    for g in gaps[:-1]:
        ax.axhline(g, color=C["light"], lw=.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(ticklab, fontsize=7.5)
    ax.set_ylim(len(rows) - .5, -.5)  # inverted, no autoscale margin
    ax.set_xlim(-X, X)
    ax.set_xlabel("within-programme deviation from the coded exponent")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=vcol[k],
                              label=vlab[k]) for k in
                       ("CONFIRMED", "ATTENUATED", "CONTRADICTED",
                        "NOT-EVALUABLE")],
              frameon=False, fontsize=7.5, loc="lower center", ncol=4,
              bbox_to_anchor=(.5, 1.005))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=.3, rect=(0, 0, 1, .97))
    save(fig, "fig_grading_r09")


def scarcity():
    ls = pd.read_csv(R06 / "outputs" / "ladder_summary.csv", dtype={"S": str})
    th = pd.read_csv(R06 / "outputs" / "ladder_theory.csv")
    order = ["3", "6", "12", "24", "all"]
    xv = np.array([3, 6, 12, 24, 48], float)
    panels = [("FRP", ["K1", "K2"], "K2",
               {"K1": "K1 (ACI 440.1R-15)", "K2": "K2 (with $\\lambda_s$)"}),
              ("SFRC", ["K1", "K2"], "K2",
               {"K1": "K1 (Narayanan–Darwish)", "K2": "K2 (with $\\lambda_s$)"}),
              ("CARB", ["K1", "K2", "K3"], "K3",
               {"K1": "K1 ($\\sqrt{C}$)", "K2": "K2 (Papadakis)",
                "K3": "K3 (buffer capacity)"}),
              ("DEEP", ["BEAM", "ARCH", "ARCHS", "ARCH99"], "ARCHS",
               {"BEAM": "BEAM", "ARCH": "ARCH", "ARCHS": "ARCHS",
                "ARCH99": "ARCH99"})]
    rcol = [C["imposed"], C["red"], C["sky"], C["green"]]
    rmark = ["o", "s", "D", "v"]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6))
    for ax, (dom, rungs, best, rlab) in zip(axes.flat, panels):
        d = ls[(ls.dataset == dom) & (ls.k == 0)]
        for j, rung in enumerate(rungs):
            g = d[(d.rung == rung) & (d.method == "KA-FIXED")] \
                .set_index("S").reindex(order)["rmse"].astype(float).values
            ax.plot(xv, g, ls="-", ms=3.5, lw=1.0, color=rcol[j],
                    marker=rmark[j], label=rlab[rung])
        gd = d[d.method == "DATA-RE"].groupby("S").rmse.first() \
            .reindex(order).astype(float).values
        ax.plot(xv, gd, "--", lw=1.0, color=C["data"], label="data only")
        gp = d[(d.rung == best) & (d.method == "PAKA-S")] \
            .set_index("S").reindex(order)["rmse"].astype(float).values
        ax.plot(xv, gp, "-", lw=1.6, color=C["paka"], label="PAKA")
        tr = th[th.domain == dom]
        fix = d[d.method == "KA-FIXED"]
        for j, rung in enumerate(rungs):
            row = tr[tr.rung == rung]
            if not len(row) or not np.isfinite(row.S_obs.iloc[0]):
                continue
            so = float(row.S_obs.iloc[0])
            gv = fix[fix.rung == rung].set_index("S") \
                .reindex(order)["rmse"].astype(float).values
            yi = np.interp(np.log10(min(so, 48)), np.log10(xv), gv)
            if np.isfinite(yi):
                ax.plot(min(so, 48), yi, "x", ms=6, color=C["ink"], mew=1.4,
                        zorder=6)
        ax.set_xscale("log")
        ax.set_xlim(2.6, 58)
        ax.set_xticks(xv)
        ax.set_xticklabels(["3", "6", "12", "24", "all"], fontsize=7.5)
        ax.set_title("(%s) %s" % ("abcd"[panels.index((dom, rungs, best, rlab))],
                                DOMLAB[dom]), loc="left", fontsize=8.5)
        ax.set_xlabel("training programmes")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("new-programme RMSE ($k=0$)")
    fig.tight_layout(pad=.6)
    save(fig, "fig_scarcity_r09")


def predict():
    s = pd.read_csv(R05 / "outputs" / "lopo_summary.csv")
    ks = [0, 1, 2, 3, 5]
    meths = ["KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S", "GBM", "GBM-RES",
             "GBM-ME", "GP-RES"]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=False)
    for ax, dom, let in zip(axes, ("FRP", "SFRC", "CARB"), "abc"):
        d = s[s.dataset == dom]
        for m_ in meths:
            g = d[d.method == m_].set_index("k").reindex(ks)
            if g.rmse.isna().all():
                continue
            lab, col = ARM[m_]
            st = dict(ls="--" if m_ == "DATA-RE" else "-",
                      lw=1.5 if m_ == "PAKA-S" else 0.9)
            ax.plot(ks, g.rmse, marker="o", ms=3, color=col, label=lab, **st)
            if m_ in ("PAKA-S", "GBM-ME"):
                ax.fill_between(ks, g.rmse_lo, g.rmse_hi, color=col, alpha=.15,
                                lw=0)
        ax.set_title(f"({let}) {DOMLAB[dom]}", loc="left", fontsize=8.5)
        ax.set_xlabel("$k$")
        ax.set_xticks(ks)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("new-programme RMSE")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(pad=.5, rect=(0, .16, 1, 1))
    save(fig, "fig_predict_r09")


def paka():
    s = pd.read_csv(R06 / "outputs" / "ladder_summary.csv", dtype={"S": str})
    piv = s[s.k == 0].pivot_table(index=["dataset", "rung", "S"],
                                  columns="method", values="rmse")
    piv["best"] = piv[["KA-FIXED", "KA-RE", "DATA-RE"]].min(axis=1)
    piv["regret"] = piv["PAKA-S"] - piv["best"]
    regmax = piv.regret.max()
    keep = ~piv.index.get_level_values("rung").isin(["BEAM"])
    regok = piv.loc[keep, "regret"].max()
    assert abs(regmax - 0.042) < 0.001, f"regmax {regmax:.4f} != 0.042"
    assert abs(regok - 0.020) < 0.001, f"regmaxok {regok:.4f} != 0.020"
    print(f"assert ok: regret max {regmax:.3f}, excl BEAM {regok:.3f}")

    order = [("FRP", "K1"), ("FRP", "K2"), ("SFRC", "K1"), ("SFRC", "K2"),
             ("CARB", "K1"), ("CARB", "K2"), ("CARB", "K3"),
             ("DEEP", "BEAM"), ("DEEP", "ARCH"), ("DEEP", "ARCHS"),
             ("DEEP", "ARCH99"), ("DEEP-WB", "BEAM"), ("DEEP-WB", "ARCH"),
             ("DEEP-WB", "ARCHS"), ("DEEP-WB", "ARCH99")]
    dlab = {"DEEP-WB": "deep beams (plate-adj.)"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5),
                           gridspec_kw=dict(width_ratios=(1.35, 1.0)))
    ax = axes[0]
    for i, (dom, rung) in enumerate(order):
        vals = [piv.regret.get((dom, rung, S), np.nan)
                for S in ("3", "6", "12", "24", "all")]
        vals = np.asarray(vals, float)
        col = C["purple"] if dom == "DEEP-WB" else C["paka"]
        ax.plot(vals[np.isfinite(vals)], np.full(np.isfinite(vals).sum(), i),
                "o", ms=3.4, color=col, alpha=.55, mfc="none")
        w = np.nanmax(vals)
        ax.plot(w, i, "s", ms=5, color=col, mec=C["ink"], mew=.8, zorder=5)
    ax.axvline(0, color=C["grey"], lw=.7, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{dlab.get(d, DOMLAB.get(d, d))} {r_}" for d, r_ in order],
                       fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"PAKA shortfall vs best single predictor ($k=0$)")
    ax.set_xlim(-.03, .05); ax.set_xticks([-.02, 0, .02, .04])
    ax.set_title("(a) worst shortfall over library sizes", loc="left",
                 fontsize=8.5)
    # (b) k=0 -> k=3
    ax = axes[1]
    grp = [("FRP", "K2"), ("SFRC", "K2"), ("CARB", "K3"), ("DEEP", "ARCHS")]
    dcol = {"FRP": C["paka"], "SFRC": C["imposed"], "CARB": C["green"],
            "DEEP": C["purple"]}
    dmark = {"FRP": "o", "SFRC": "s", "CARB": "^", "DEEP": "D"}
    handles = []
    for dom, rung in grp:
        g = s[(s.dataset == dom) & (s.rung == rung) & (s.S == "all")
              & (s.method == "PAKA-S")].set_index("k")
        r0, r3 = g.loc[0, "rmse"], g.loc[3, "rmse"]
        red = 100 * (1 - r3 / r0)
        ax.plot([0, 3], [r0, r3], marker=dmark[dom], ms=4.5, color=dcol[dom],
                lw=1.4, label=f"{DOMLAB[dom]} ($-{red:.0f}$%)")
    ax.set_xticks([0, 3]); ax.set_xticklabels(["0", "3"])
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(0, .68)
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.02, 0.74),
              labelspacing=0.25, handlelength=1.2, borderaxespad=0.1)
    ax.set_xlabel("tests observed, $k$")
    ax.set_ylabel("new-programme RMSE")
    ax.set_title("(b) PAKA at the full library", loc="left", fontsize=8.5)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=.5)
    save(fig, "fig_paka_r09")


def leakage():
    pg = pd.read_csv(R07 / "outputs" / "pglo.csv")
    pg["opt"] = (pg.rmse_lopo / pg.rmse_random - 1) * 100
    order = ["FRP", "CARB", "DEEP", "SFRC"]
    marks = {"KA-FIXED": "o", "KA-RE": "s", "DATA-RE": "^"}
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    for i, dom in enumerate(order):
        g = pg[pg.domain == dom].set_index("arm")
        for j, arm in enumerate(("KA-FIXED", "KA-RE", "DATA-RE")):
            y = i + (j - 1) * .26
            r_, l_ = float(g.loc[arm, "rmse_random"]), float(g.loc[arm, "rmse_lopo"])
            ax.plot([r_, l_], [y, y], color=C["light"], lw=2.4, zorder=1,
                    solid_capstyle="round")
            ax.plot(r_, y, marks[arm], ms=5, color=C["imposed"], zorder=3)
            ax.plot(l_, y, marks[arm], ms=5, color=C["paka"], zorder=3,
                    mec=C["paka"])
            ax.text(max(r_, l_) + .012, y, f"{g.loc[arm,'opt']:+.0f}%",
                    va="center", fontsize=7.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DOMLAB[d] for d in order], fontsize=7.5)
    ax.set_xlabel("new-programme RMSE (log units)")
    ax.set_xlim(.14, .90)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C["imposed"],
                               label="random five-fold"),
                        Line2D([], [], marker="o", ls="", color=C["paka"],
                               label="leave-one-programme-out"),
                        Line2D([], [], marker="o", ls="", color=C["ink"],
                               label="code imposed"),
                        Line2D([], [], marker="s", ls="", color=C["ink"],
                               label="code as prior"),
                        Line2D([], [], marker="^", ls="", color=C["ink"],
                               label="data only")],
               frameon=False, fontsize=7.5, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, -.02))
    fig.tight_layout(pad=.5, rect=(0, .14, 1, 1))
    save(fig, "fig_leakage_r09")


if __name__ == "__main__":
    concept(); evidence(); grading(); scarcity(); predict(); paka(); leakage()
    print("R09 figures written to", FIG)
