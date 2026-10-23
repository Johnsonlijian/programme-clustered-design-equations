"""R06 LaTeX tables for the manuscript, generated only from the round's output
tables and outputs/ladder_signatures.csv (produced by r06_numbers.py). Same
\\tabinput/booktabs row style as r05_tables.py -- fragments contain rows only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts")]
import r06_rungs as R  # noqa: E402
import kag_data_r05 as D  # noqa: E402

R06 = HERE.parent
OUT = R06 / "outputs"
GEN = R06.parents[1] / "paper" / "manuscript" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

DOMRUNG = [("FRP", "K1"), ("FRP", "K2"), ("SFRC", "K1"), ("SFRC", "K2"),
           ("CARB", "K1"), ("CARB", "K2"), ("CARB", "K3"),
           ("DEEP", "BEAM"), ("DEEP", "ARCH"), ("DEEP", "ARCHS"), ("DEEP", "ARCH99"),
           ("DEEP-WB", "BEAM"), ("DEEP-WB", "ARCH"), ("DEEP-WB", "ARCHS"), ("DEEP-WB", "ARCH99")]
DLAB = {"FRP": "FRP-RC shear", "SFRC": "SFRC shear", "CARB": "Carbonation",
        "DEEP": "RC deep beams", "DEEP-WB": "RC deep beams (plate-adj.)"}
FEATURE_TEX = dict(D.FEATURE_TEX, ln_wb="$w_{bp}/d$")
# anchor basis, short text per task_packet_assets.md
BASIS = {("FRP", "K1"): "ACI 440.1R-15", ("FRP", "K2"): "ACI 440.1R-15 $\\times\\lambda_s$ (ACI 318-19)",
         ("SFRC", "K1"): "Narayanan--Darwish 1987",
         ("SFRC", "K2"): "Narayanan--Darwish 1987 $\\times\\lambda_s$ (ACI 318-19)",
         ("CARB", "K1"): "Fick $\\sqrt{C}$", ("CARB", "K2"): "Papadakis $(C/a)^{1/2}$",
         ("CARB", "K3"): "Leemann--Moro buffer",
         ("DEEP", "BEAM"): "ACI 318-19 beam", ("DEEP", "ARCH"): "Zsutty 1971",
         ("DEEP", "ARCHS"): "Zsutty 1971 $\\times\\lambda_s$",
         ("DEEP", "ARCH99"): "ACI 318-99 Eq.\\ 11-29"}
# coded-exponent text per rung (asserted exponents from the anchor equations;
# shear anchors are not pure power laws -- the exponent set is asserted by the
# equation form, and lambda_s adds d ln lambda_s / d ln d on the depth exponent)
CODED = {("FRP", "K1"): "$d,b_w,f'_c,\\rho,E_f$ (eq.\\ form); $a/d$ absent",
         ("FRP", "K2"): "same; $d$ exponent $1+\\mathrm{d}\\ln\\lambda_s/\\mathrm{d}\\ln d$",
         ("SFRC", "K1"): "$d,b_w,f'_c,\\rho,a/d,F$ (eq.\\ form)",
         ("SFRC", "K2"): "same; $d$ exponent $1+\\mathrm{d}\\ln\\lambda_s/\\mathrm{d}\\ln d$",
         ("CARB", "K1"): "CO$_2$ $+0.5$",
         ("CARB", "K2"): "CO$_2$ $+0.5$; CaO$_r$ $-0.5$",
         ("CARB", "K3"): "CO$_2$ $+0.5$; $w/b$ $+1$; CaO$_r$ $-1$",
         ("DEEP", "BEAM"): "$f'_c$ $+0.5$; $\\rho$ $+1/3$; $b_w$ $1$; $d$ $1+\\mathrm{d}\\ln\\lambda_s/\\mathrm{d}\\ln d$; $a/d$ $0$",
         ("DEEP", "ARCH"): "$f'_c$ $+1/3$; $\\rho$ $+1/3$; $a/d$ $-4/3$; $b_w$ $1$; $d$ $1$",
         ("DEEP", "ARCHS"): "as ARCH; $d$ exponent $1+\\mathrm{d}\\ln\\lambda_s/\\mathrm{d}\\ln d$",
         ("DEEP", "ARCH99"): "piecewise ($M_u/(V_u d)$ factor, capped)"}
FREE = {("CARB", "K1"): "CaO$_r$, $w/b$", ("CARB", "K2"): "$w/b$"}
RUNG_LAB = {"K1": "K1", "K2": "K2", "K3": "K3", "BEAM": "BEAM", "ARCH": "ARCH",
            "ARCHS": "ARCHS", "ARCH99": "ARCH99"}
METH_LAB = {"KA-FIXED": "code imposed", "KA-RE": "code as REML prior",
            "DATA-RE": "data-only", "PAKA-S": "PAKA"}

MINUS = "\\ensuremath{-}"


def _minus(s):
    return s.replace("-", MINUS) if s.startswith("-") else s


def _signed(s):
    return s.replace("-", MINUS).replace("+", "\\ensuremath{+}")


def f3(v):
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.3f}")


def f2(v):
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.2f}")


def f1(v):
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.1f}")


def s2(v):
    return "--" if pd.isna(v) else _signed(f"{v:+.2f}")


def s3(v):
    return "--" if pd.isna(v) else _signed(f"{v:+.3f}")


def sstar(v):
    if v is None or pd.isna(v):
        return "--"
    if not np.isfinite(v) or v > 1000:
        return "$\\infty$"
    return _minus(f"{v:.1f}")


def ci(lo, hi, fmt=None):
    fmt = fmt or sstar
    return f"[{fmt(lo)}, {fmt(hi)}]"


def ratio(v):
    """Predicted/observed ratio; degenerate huge values shown in sci notation."""
    if v is None or pd.isna(v):
        return "--"
    if not np.isfinite(v):
        return "$\\infty$"
    if v > 1000:
        e = int(np.floor(np.log10(v)))
        return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"
    return _minus(f"{v:.2f}")


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")
    print(f"  -> {GEN / name}")


def load(name):
    return pd.read_csv(D.long_path(OUT / name))


def row(dom, rung, df):
    return df[(df.domain == dom) & (df.rung == rung)].iloc[0]


# ---------------------------------------------------------------- tables ----
def tab_rungs():
    """tab_rungs.tex -- the 15 anchor rungs. DEEP-WB rows carry the same anchors
    as DEEP with the free plate-width covariate (variant note in the domain cell)."""
    rows = []
    for name, rung in DOMRUNG:
        dom_cell = DLAB[name] if name != "DEEP-WB" else "DEEP-WB \\emph{(variant of DEEP)}"
        free = FREE.get((name, rung), "--")
        if name == "DEEP-WB":
            free = "$w_{bp}/d$" if free == "--" else free + ", $w_{bp}/d$"
        coded = CODED.get((name, rung), CODED.get(("DEEP", rung), "--"))
        rows.append(f"{dom_cell} & {RUNG_LAB[rung]} & {BASIS.get((name, rung), BASIS.get(('DEEP', rung)))} "
                    f"& {coded} & {free} \\\\")
        if (name, rung) in (("FRP", "K2"), ("SFRC", "K2"), ("CARB", "K3"), ("DEEP", "ARCH99")):
            rows.append("\\midrule")
    write("tab_rungs.tex", "\n".join(rows) + "\n")


def tab_theory():
    """tab_theory.tex -- theory inputs and predicted crossover per rung."""
    t = load("ladder_theory_prereg.csv")
    labs = load("ladder_theory_labs.csv")
    rows = []
    prev = None
    for name, rung in DOMRUNG:
        r = row(name, rung, t)
        lab = f"{DLAB[name]} {RUNG_LAB[rung]}"
        if name != prev:
            if prev is not None:
                rows.append("\\midrule")
            prev = name
        lr = labs[(labs.domain == name) & (labs.rung == rung)]
        labstar = sstar(lr.iloc[0].S_star) if not lr.empty else "--"
        if (name, rung) == ("CARB", "K2"):
            labstar += "$^{\\dagger}$"   # laboratory bootstrap upper bound unstable
        rows.append(f"{lab} & {r.p:.0f} & {r.c:.3f} & {r.tau:.3f} & {r.sigma:.3f} & "
                    f"{f3(r.B_raw)} & {f3(r.B)} & {sstar(r.S_star)} {ci(r.S_star_lo, r.S_star_hi)} "
                    f"& {labstar} \\\\")
    rows.append("\\multicolumn{9}{l}{\\footnotesize $^{\\dagger}$laboratory-group bootstrap "
                "upper bound unstable}\\\\")
    write("tab_theory.tex", "\n".join(rows) + "\n")


def tab_crossover():
    """tab_crossover.tex -- main table: predicted vs observed crossover, no-crossing
    share, ratio, and the chronological-library crossover where available."""
    t = load("ladder_theory.csv")
    cro = load("ladder_chrono_crossover.csv")
    rows = []
    prev = None
    for name, rung in DOMRUNG:
        r = row(name, rung, t)
        if name != prev:
            if prev is not None:
                rows.append("\\midrule")
            prev = name
        lab = f"{DLAB[name]} {RUNG_LAB[rung]}"
        cr = cro[(cro.dataset == name) & (cro.rung == rung)]
        ch = f"{sstar(cr.iloc[0].S_obs)} {ci(cr.iloc[0].S_obs_lo, cr.iloc[0].S_obs_hi)}" \
            if not cr.empty else "--"
        # degenerate rows: S_obs inf (never overtaken: CARB K3), NaN (already
        # beaten at S=3: BEAM rungs), S* degenerate (B at floor: CARB K1/K3)
        if pd.isna(r.S_obs):
            obs_txt = "none ($<3$)"
            note = "anchor beaten before the smallest library"
        elif not np.isfinite(r.S_obs):
            obs_txt = "$\\infty$" + f" {ci(r.S_obs_lo, r.S_obs_hi)}"
            note = "never overtaken in library"
        else:
            obs_txt = f"{sstar(r.S_obs)} {ci(r.S_obs_lo, r.S_obs_hi)}"
            note = ""
        if r.B <= 1e-9 + 1e-12:
            note = (note + "; " if note else "") + "misfit below noise floor"
        rows.append(f"{lab} & {sstar(r.S_star)} {ci(r.S_star_lo, r.S_star_hi)} & {obs_txt} & "
                    f"{f2(r.obs_boot_no_crossing)} & {ratio(r.ratio)} "
                    f"& {ch} & {note} \\\\")
    write("tab_crossover.tex", "\n".join(rows) + "\n")


def tab_scarcity():
    """tab_scarcity.tex -- RMSE of each rung by library size S (k=0), arms
    KA-FIXED / DATA-RE / PAKA-S, four primary domains."""
    s = load("ladder_summary.csv")
    order = ["3", "6", "12", "24", "all"]
    rows = []
    for name in ("FRP", "SFRC", "CARB", "DEEP"):
        for rung in R.RUNGS[name]:
            d = s[(s.dataset == name) & (s.rung == rung) & (s.k == 0)]
            for m in ("KA-FIXED", "DATA-RE", "PAKA-S"):
                g = d[d.method == m].set_index("S").reindex(order)
                cells = " & ".join(f3(v) for v in g.rmse)
                rows.append(f"{DLAB[name]} {RUNG_LAB[rung]} & {METH_LAB[m]} & {cells} \\\\")
            rows.append("\\addlinespace[1pt]")
        rows.pop()
        rows.append("\\midrule")
    write("tab_scarcity.tex", "\n".join(rows[:-1]) + "\n")


def tab_within():
    """tab_within.tex -- within-programme exponent signatures: anchor-implied
    exponent vs observed within (dW) and between (dB) deviations, all features."""
    sig = load("ladder_signatures.csv")
    FRAME = {"FRP": "K1", "SFRC": "K1", "CARB": "K2", "DEEP": "ARCH", "DEEP-WB": "ARCH"}
    rows = []
    for name in ("FRP", "SFRC", "CARB", "DEEP", "DEEP-WB"):
        rung = FRAME[name]
        blk = sig[(sig.domain == name) & (sig.rung == rung)]
        for i, r in enumerate(blk.itertuples()):
            tex = FEATURE_TEX.get(r.feature, r.feature)
            role = r.role
            aim = f"{r.anchor_implied:+.2f}" if not pd.isna(r.anchor_implied) else "--"
            if not pd.isna(r.anchor_implied) and (r.anchor_implied_hi - r.anchor_implied_lo) > 0.02:
                aim += f" [{r.anchor_implied_lo:+.2f},{r.anchor_implied_hi:+.2f}]"
            if aim != "--":
                aim = aim.replace("-", MINUS).replace("+", "\\ensuremath{+}")
            rows.append(f"{DLAB[name]} {RUNG_LAB[rung]} & {tex} & {role} & {aim} & "
                        f"{s2(r.dW)} ({f2(r.dW_se)}) & {s2(r.dB)} ({f2(r.dB_se)}) \\\\")
        rows.append("\\midrule")
    # literature reference rows (code-implied constants; lambda_s endpoints are
    # the recomputed row-mean implied slopes, most-negative .. least-negative)
    rows.append("\\multicolumn{6}{l}{\\emph{Literature references (code-implied)}}\\\\")
    lam = sig.loc[sig.feature == "ln_d", ["domain", "lambda_s_implied"]] \
        .dropna().groupby("domain").lambda_s_implied.first()
    rows.append(f"ACI 318-19 $\\lambda_s$ implied depth slope, FRP/SFRC/DEEP & & & "
                f"{f2(lam.max())}\\ \\ldots\\ {f2(lam.min())} & & \\\\")
    rows.append("Zsutty $a/d$ exponent (arch action) & & & $-4/3$ & & \\\\")
    rows.append("Papadakis binding exponent, CaO$_r$ & & & $-0.5$ & & \\\\")
    rows.append("Leemann--Moro buffer exponents, CaO$_r$ / $w/b$ & & & $-1$ / $+1$ & & \\\\")
    write("tab_within.tex", "\n".join(rows) + "\n")


def tab_paka():
    """tab_paka.tex -- PAKA-S on the signature-aligned rung of each domain
    (K2 for the shear domains, K3 for CARB, ARCHS for DEEP; the rung whose anchor
    matches the within-programme signature). Regret vs the best single arm,
    k=3 -> k=0 RMSE reduction, predictive cover90/width90, median nlpd."""
    s = load("ladder_summary.csv")
    pd_draw = pd.read_csv(D.long_path(OUT / "ladder_per_draw.csv.gz"),
                          usecols=["dataset", "rung", "S", "k", "method", "source", "draw",
                                   "cover90", "width90", "nlpd"])
    pick = {"FRP": "K2", "SFRC": "K2", "CARB": "K3", "DEEP": "ARCHS"}
    rows = []
    for name, rung in pick.items():
        d = s[(s.dataset == name) & (s.rung == rung) & (s.S == "all") & (s.k == 0)]
        p = d[d.method == "PAKA-S"].iloc[0]
        best = d[d.method.isin(["KA-FIXED", "KA-RE", "DATA-RE"])].rmse.min()
        p3 = s[(s.dataset == name) & (s.rung == rung) & (s.S == "all") & (s.k == 3)
               & (s.method == "PAKA-S")].iloc[0]
        dr = pd_draw[(pd_draw.dataset == name) & (pd_draw.rung == rung)
                     & (pd_draw.S.astype(str) == "all")
                     & (pd_draw.k == 0) & (pd_draw.method == "PAKA-S")]
        per = dr.groupby("source").agg(cover90=("cover90", "mean"), width90=("width90", "mean"),
                                       nlpd=("nlpd", "mean"))
        red = 100 * (1 - p3.rmse / p.rmse)
        rows.append(f"{DLAB[name]} {RUNG_LAB[rung]} & {f3(p.rmse)} {ci(p.rmse_lo, p.rmse_hi, f2)} & "
                    f"{s3(p.rmse - best)} & {red:.0f}\\% & {f2(per.cover90.mean())} & "
                    f"{f2(per.width90.mean())} & {f2(per.nlpd.median())} \\\\")
    write("tab_paka.tex", "\n".join(rows) + "\n")


def tab_extrap():
    """tab_extrap.tex (supplement) -- size-extrapolation bias and RMSE.
    Between-programme extrapolation (FRP/SFRC K1 vs K2, ladder_extrap_summary)
    and within-programme smallest-k extrapolation (FRP/SFRC K1 vs K2, DEEP
    ARCH vs ARCHS, ladder_within_summary); arms KA-FIXED / KA-RE / PAKA-S."""
    e = load("ladder_extrap_summary.csv")
    w = load("ladder_within_summary.csv")
    rows = ["\\multicolumn{6}{l}{\\emph{Between programmes (train on smaller 75\\%)}}\\\\"]
    for name in ("FRP", "SFRC"):
        for rung in R.RUNGS[name]:
            for m in ("KA-FIXED", "KA-RE", "PAKA-S"):
                g = e[(e.dataset == name) & (e.rung == rung) & (e.method == m)].set_index("k")
                rows.append(f"{DLAB[name]} {RUNG_LAB[rung]} & {METH_LAB[m]} & "
                            f"{f3(g.rmse.get(0))} & {s3(g.bias.get(0))} & "
                            f"{f3(g.rmse.get(3))} & {s3(g.bias.get(3))} \\\\")
            rows.append("\\addlinespace[1pt]")
        rows.pop()
        rows.append("\\midrule")
    rows.append("\\multicolumn{6}{l}{\\emph{Within programmes (smallest $k$ specimens)}}\\\\")
    for name, rungs in (("FRP", ("K1", "K2")), ("SFRC", ("K1", "K2")),
                        ("DEEP", ("ARCH", "ARCHS"))):
        for rung in rungs:
            for m in ("KA-FIXED", "KA-RE", "PAKA-S"):
                g = w[(w.dataset == name) & (w.rung == rung) & (w.method == m)].set_index("k")
                k1, k3 = g.index.min(), g.index.max()
                rows.append(f"{DLAB[name]} {RUNG_LAB[rung]} & {METH_LAB[m]} & "
                            f"{f3(g.rmse.get(k1))} & {s3(g.bias.get(k1))} & "
                            f"{f3(g.rmse.get(k3))} & {s3(g.bias.get(k3))} \\\\")
            rows.append("\\addlinespace[1pt]")
        rows.pop()
        rows.append("\\midrule")
    write("tab_extrap.tex", "\n".join(rows[:-1]) + "\n")


def tab_chrono():
    """tab_chrono.tex (supplement) -- chronological-library crossover vs the
    random sub-library crossover (FRP/SFRC, K1/K2)."""
    cro = load("ladder_chrono_crossover.csv")
    t = load("ladder_theory.csv")
    chs = load("ladder_chrono_summary.csv")
    rows = []
    for name in ("FRP", "SFRC"):
        for rung in R.RUNGS[name]:
            c = cro[(cro.dataset == name) & (cro.rung == rung)].iloc[0]
            r = row(name, rung, t)
            n_t = int(chs[(chs.dataset == name) & (chs.rung == rung)].targets.sum())
            rows.append(f"{DLAB[name]} {RUNG_LAB[rung]} & {n_t} & "
                        f"{sstar(r.S_obs)} {ci(r.S_obs_lo, r.S_obs_hi)} & "
                        f"{sstar(c.S_obs)} {ci(c.S_obs_lo, c.S_obs_hi)} & {f2(c.obs_boot_no_crossing)} \\\\")
        rows.append("\\midrule")
    write("tab_chrono.tex", "\n".join(rows[:-1]) + "\n")


if __name__ == "__main__":
    print("writing R06 tables:")
    tab_rungs()
    tab_theory()
    tab_crossover()
    tab_scarcity()
    tab_within()
    tab_paka()
    tab_extrap()
    tab_chrono()
    print("tables done")
