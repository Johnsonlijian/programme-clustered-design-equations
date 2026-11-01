"""R08 manuscript table bodies -> paper/manuscript/generated/tab_*.tex.

Row fragments only (``\\\\`` line ends, ``\\midrule`` between domains); the
tabular environments and captions live in the manuscript/supplement sources.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROOT = R08.parent.parent
OUT = R08 / "outputs"
R06 = ROOT / "rounds" / "R06_mechanism_pivot_2026-10-22"
R07 = ROOT / "rounds" / "R07_merged_protocol_2026-10-22"
GEN = ROOT / "paper" / "manuscript" / "generated"

DOMLAB = {"FRP": "FRP-RC shear", "SFRC": "SFRC shear", "CARB": "Carbonation",
          "DEEP": "RC deep beams", "DEEP-WB": r"RC deep beams (plate-adj.)"}
FREED = {"ln1p_F": "$F$", "ln_ad": "$a/d$"}


def f(x, d):
    return f"{float(x):.{d}f}"


def sgn(x, d=1):
    return (r"\ensuremath{+}" if float(x) >= 0 else r"\ensuremath{-}") + f(abs(float(x)), d)


def s_intv(v, lo, hi, d=1):
    if not np.isfinite(v):
        return r"$\infty$"
    return f"{f(v, d)} [{f(lo, d)}, {f(hi, d)}]"


# ============================ tab_repair.tex ======================================
def tab_repair():
    pc = pd.read_csv(OUT / "prospective_check.csv")
    lt = pd.read_csv(R06 / "outputs" / "ladder_theory.csv").set_index(["domain", "rung"])
    tv = pd.read_csv(R07 / "outputs" / "tost_verdicts.csv").set_index(
        ["domain", "rung", "feature"])
    cert = pd.read_csv(OUT / "certified_prediction.csv").set_index(["domain", "rung"])
    rep = pd.read_csv(OUT / "repair_prediction.csv").set_index(["domain", "rung"])
    feat = {"SFRC": "ln1p_F", "FRP": "ln_ad", "DEEP": "ln_ad"}
    order = [("SFRC", "K1", "K1C"), ("SFRC", "K2", "K2C"),
             ("FRP", "K1", "K1C"), ("FRP", "K2", "K2C"),
             ("DEEP", "BEAM", "BEAMC")]
    rows, last = [], None
    for dom, par, rr in order:
        if last is not None and dom != last:
            rows.append(r"\midrule")
        last = dom
        g = pc[(pc.domain == dom) & (pc.rung == rr)].iloc[0]
        z = float(tv.loc[(dom, par, feat[dom]), "z"])
        b_par = float(lt.loc[(dom, par), "B"])
        src = cert if (dom, rr) in cert.index else rep
        b_rep = float(src.loc[(dom, rr), "B"])
        pred = f"{f(g.parent_pred, 1)} $\\to$ {f(g.S_pred, 1)} [{f(g.S_pred_lo, 1)}, {f(g.S_pred_hi, 1)}]"
        obs_par = "none" if not np.isfinite(g.parent_obs) else f(g.parent_obs, 1)
        if np.isfinite(g.S_obs):
            obs_rep = f"{f(g.S_obs, 1)} [{f(g.S_obs_lo, 1)}, {f(g.S_obs_hi, 1)}]"
        else:
            obs_rep = rf"$>$24 ({round(100 * g.no_cross):.0f}\%)"
        dm = ("yes" if g.direction_ok else "no") + " / " + ("yes" if g.magnitude_ok else "no")
        rows.append(
            f"{DOMLAB[dom]} {par} $\\to$ {rr} & {FREED[feat[dom]]} ({sgn(z)}) & "
            f"{f(b_par, 3)} $\\to$ {f(b_rep, 3)} & {pred} & {obs_par} $\\to$ {obs_rep} & {dm} \\\\")
    (GEN / "tab_repair.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


# ============================ tab_budget_split.tex ================================
def tab_budget_split():
    bs = pd.read_csv(OUT / "budget_split.csv")
    rows, last = [], None
    for r in bs.itertuples():
        if last is not None and r.domain != last:
            rows.append(r"\midrule")
        last = r.domain
        pre = r"$\infty$" if (not np.isfinite(r.S_prespec) or r.S_prespec > 1e6) else f(r.S_prespec, 1)
        if not np.isfinite(r.S_split) or r.S_split > 1e6:
            split = r"$\infty$"
        elif r.S_split == 0:
            split = "0.0"
        else:
            split = s_intv(r.S_split, r.S_split_lo, r.S_split_hi)
        if np.isfinite(r.S_obs):
            obs = f(r.S_obs, 1)
        elif np.isnan(r.S_obs):
            obs = "none"
        else:
            obs = r"$\infty$"
        rows.append(
            f"{DOMLAB[r.domain]} {r.rung} & {r.p} & {r.q} & {f(r.c_prespec, 3)} & "
            f"{f(r.c_F, 3)} & {f(r.c_D, 3)} & {pre} & {split} & {obs} \\\\")
    (GEN / "tab_budget_split.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


# ============================ tab_repair_lopo.tex =================================
def tab_repair_lopo():
    cs = pd.read_csv(OUT / "certified_summary.csv")
    rf = pd.read_csv(OUT / "repair_fulllib.csv")
    ARM = {"KA-FIXED": "code imposed", "PAKA-S": r"\PAKA{}",
           "GBM": "GBM (anchor-free)", "GBM-ME": "GBM-ME (anchor-free)"}

    def cell(df, rung, meth, k):
        g = df[(df.rung == rung) & (df.method == meth) & (df.k == k)]
        if not len(g):
            return "--"
        g = g.iloc[0]
        return f"{f(g.rmse, 3)} [{f(g.lo, 2)}, {f(g.hi, 2)}]"

    def block(dom, frame, rungs, arms):
        out = []
        for rung in rungs:
            for meth in arms:
                out.append(f"{DOMLAB[dom]} {rung} & {ARM[meth]} & "
                           f"{cell(frame, rung, meth, 0)} & {cell(frame, rung, meth, 3)} \\\\")
        return out

    rows = []
    rows += block("SFRC", cs, ["K1", "K1C", "K2C"], ["KA-FIXED", "PAKA-S"])
    for meth in ("GBM", "GBM-ME"):
        rows.append(f"{DOMLAB['SFRC']} -- & {ARM[meth]} & "
                    f"{cell(cs, '-', meth, 0)} & {cell(cs, '-', meth, 3)} \\\\")
    rows.append(r"\midrule")
    rows += block("FRP", rf, ["K1", "K1C", "K2", "K2C"], ["KA-FIXED", "PAKA-S"])
    rows.append(r"\midrule")
    rows += block("DEEP", rf, ["BEAM", "BEAMC"], ["KA-FIXED", "PAKA-S"])
    for meth in ("GBM", "GBM-ME"):
        rows.append(f"{DOMLAB['DEEP']} BEAMC & {ARM[meth]} & "
                    f"{cell(rf, 'BEAMC', meth, 0)} & {cell(rf, 'BEAMC', meth, 3)} \\\\")
    (GEN / "tab_repair_lopo.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    tab_repair()
    tab_budget_split()
    tab_repair_lopo()
    for n in ("tab_repair.tex", "tab_budget_split.tex", "tab_repair_lopo.tex"):
        p = GEN / n
        print(f"--- {n} ({p.stat().st_size} bytes)")
        print(p.read_text(encoding="utf-8"))
