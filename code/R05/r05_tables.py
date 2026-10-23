"""R05: LaTeX tables and number macros for the manuscript, generated only from the
round's output tables (every number in the text is traceable to a regenerated file)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kag_data_r05 as D  # noqa: E402

R05 = HERE.parent
OUT = R05 / "outputs"
GEN = R05.parents[1] / "paper" / "manuscript" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
DOM = ["FRP", "SFRC", "CARB"]
AUDIT_DOM = ["FRP", "SFRC", "CARB", "CARB-CO2", "DEEP", "DEEP-WB"]
TAG = {"FRP": "F", "SFRC": "S", "CARB": "C", "CARB-CO2": "Q", "DEEP": "D", "DEEP-WB": "W"}
NAME = {"ANCHOR": "Code as published", "KA-FIXED": "Code exponents imposed", "KA-RE": "Code as REML prior",
        "DATA-RE": "Data-only hierarchical", "PAKA": "1-SE trust selection", "PAKA-S": "PAKA (proposed)",
        "PAKA-S0": "PAKA, $a_0=0$", "PAKA-S6": "PAKA, $a_0=6$", "GBM": "Gradient boosting",
        "GBM-RES": "Boosting on code residual", "GBM-ME": "Mixed-effects boosting", "GP-RES": "GP on code residual"}
DLAB = {"FRP": "FRP-RC shear", "SFRC": "SFRC shear", "CARB": "Carbonation", "DEEP": "RC deep beams",
        "DEEP-WB": "RC deep beams (plate-adj.)"}
M = {}


def mac(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in M, name
    M[name] = value


MINUS, PLUS = "\\ensuremath{-}", "\\ensuremath{+}"     # safe in text and in math mode


def _minus(s):
    return s.replace("-", MINUS) if s.startswith("-") else s


def _signed(s):
    return s.replace("-", MINUS).replace("+", PLUS)


def f3(v):
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.3f}")


def f2(v):
    return "--" if v is None or pd.isna(v) else _minus(f"{v:.2f}")


def s2(v):
    """Signed, typographic minus (for deviations and gaps)."""
    return "--" if pd.isna(v) else _signed(f"{v:+.2f}")


def s3(v):
    return "--" if pd.isna(v) else _signed(f"{v:+.3f}")


def a3(v):
    """Magnitude for prose ("lowers the error by 0.210")."""
    return f"{abs(v):.3f}"


def s1(v):
    return "--" if pd.isna(v) else _signed(f"{v:+.1f}")


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")


def domains():
    d = pd.read_csv(OUT / "audit_domains.csv").set_index("domain")
    rows = []
    for k in AUDIT_DOM:
        r = d.loc[k]
        s = D.spec(k)
        coded = ", ".join(D.FEATURE_TEX[f] for f in s["coded"])
        free = ", ".join(D.FEATURE_TEX[f] for f in s["free"]) or "--"
        lab = DLAB.get(k, "Carbonation (CO$_2$-only anchor)")
        rows.append(f"{lab} & {r.n:.0f} & {r.programmes:.0f} & {r.programmes_ge6:.0f} & {s['anchor']} & {coded} & "
                    f"{free} & {f2(r.icc)} & {r.design_effect:.1f} & {r.effective_n:.0f} \\\\")
        t = TAG[k]
        mac(f"n{t}", f"{r.n:,.0f}")
        mac(f"prog{t}", f"{r.programmes:.0f}")
        mac(f"progsix{t}", f"{r.programmes_ge6:.0f}")
        mac(f"icc{t}", f2(r.icc))
        mac(f"deff{t}", f"{r.design_effect:.1f}")
        mac(f"neff{t}", f"{r.effective_n:.0f}")
        mac(f"tau{t}", f2(r.tau))
        mac(f"sig{t}", f2(r.sigma))
    write("tab_domains.tex", "\n".join(rows) + "\n")
    prim = DOM + ["DEEP"]
    mac("nAll", f"{d.loc[prim, 'n'].sum():,.0f}")
    mac("progAll", f"{d.loc[prim, 'programmes'].sum():.0f}")
    mac("iccMin", f2(d.loc[prim, "icc"].min()))
    mac("iccMax", f2(d.loc[prim, "icc"].max()))
    mac("neffMin", f"{d.loc[prim, 'effective_n'].min():.0f}")
    mac("neffMax", f"{d.loc[prim, 'effective_n'].max():.0f}")


def audit():
    f = pd.read_csv(OUT / "audit_features.csv")
    rows = []
    for k in AUDIT_DOM:
        for r in f[f.domain == k].itertuples():
            ls = getattr(r, "lambda_s_implied", np.nan)
            rows.append(f"{DLAB.get(k, 'Carbonation, CO$_2$-only')} & {D.FEATURE_TEX[r.feature]} & {r.role} & "
                        f"{f2(r.between_share)} & {r.programmes_varying}/{r.programmes_total} & "
                        f"{s2(r.uni_within)} ({f2(r.uni_within_se)}) & {s2(r.uni_between)} ({f2(r.uni_between_se)}) & "
                        f"{s1(r.uni_z)} & {s1(r.joint_z)} & {s2(ls) if not pd.isna(ls) else '--'} \\\\")
        rows.append("\\midrule")
    write("tab_audit.tex", "\n".join(rows[:-1]) + "\n")
    coded = f[f.domain.isin(DOM + ["DEEP"]) & (f.role == "coded")]
    mac("bsMin", f"{100 * coded.between_share.min():.0f}")
    mac("bsMax", f"{100 * coded.between_share.max():.0f}")
    for k, feat, tag in (("FRP", "ln_d", "dF"), ("SFRC", "ln_d", "dS"), ("CARB", "x_co2", "coC"),
                         ("CARB", "x_cao", "caoC"), ("CARB-CO2", "x_co2", "coQ"), ("FRP", "ln_Ef", "efF"),
                         ("SFRC", "ln_fc", "fcS"), ("DEEP", "ln_d", "dD"), ("DEEP", "ln_ad", "adD")):
        r = f[(f.domain == k) & (f.feature == feat)].iloc[0]
        mac(f"w{tag}", s2(r.uni_within))
        mac(f"wse{tag}", f2(r.uni_within_se))
        mac(f"b{tag}", s2(r.uni_between))
        mac(f"bse{tag}", f2(r.uni_between_se))
        mac(f"z{tag}", s1(r.uni_z))
        mac(f"jz{tag}", s1(r.joint_z))
        mac(f"vary{tag}", f"{r.programmes_varying}")
        mac(f"bs{tag}", f2(r.between_share))
        mac(f"aw{tag}", f"{abs(r.uni_within):.2f}")
        if feat in D.CODE_EXPONENT:
            mac(f"xw{tag}", s2(D.CODE_EXPONENT[feat] + r.uni_within))
            mac(f"xb{tag}", s2(D.CODE_EXPONENT[feat] + r.uni_between))
        if feat == "ln_d":
            mac(f"ls{tag}", s2(r.lambda_s_implied))
            mac(f"jw{tag}", s2(r.joint_within))
            mac(f"jb{tag}", s2(r.joint_between))


def lopo():
    s = pd.read_csv(OUT / "lopo_summary.csv")
    rows = []
    order = ["ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "PAKA", "PAKA-S", "GBM", "GBM-RES", "GBM-ME", "GP-RES"]
    for k in DOM:
        for m in order:
            g = s[(s.dataset == k) & (s.method == m)].set_index("k")
            if g.empty:
                continue
            d0 = g.loc[0]
            dr = "--" if m == "PAKA-S" else f"{s3(d0.minus_ref)} [{s3(d0.minus_ref_lo)}, {s3(d0.minus_ref_hi)}]"
            rows.append(f"{DLAB[k]} & {NAME[m]} & {f3(d0.rmse)} [{f2(d0.rmse_lo)}, {f2(d0.rmse_hi)}] & "
                        f"{f3(g.rmse.get(3))} & {f2(g.cover90.get(3))} & {f2(d0.unconservative)} & {dr} \\\\")
        rows.append("\\midrule")
    write("tab_lopo.tex", "\n".join(rows[:-1]) + "\n")
    for k in DOM:
        t = TAG[k]
        for m, mt in (("PAKA-S", "paka"), ("KA-FIXED", "fix"), ("KA-RE", "kare"), ("DATA-RE", "data"),
                      ("GBM", "gbm"), ("GBM-RES", "gbmres"), ("GBM-ME", "gbmme"), ("GP-RES", "gp")):
            g = s[(s.dataset == k) & (s.method == m)].set_index("k")
            if g.empty:
                continue
            mac(f"lo{mt}zero{t}", f3(g.rmse.get(0)))
            mac(f"lo{mt}three{t}", f3(g.rmse.get(3)))
        g = s[(s.dataset == k) & (s.method == "PAKA-S")].set_index("k")
        mac(f"kred{t}", f"{100 * (1 - g.rmse.get(3) / g.rmse.get(0)):.0f}")
        mac(f"covthree{t}", f2(g.cover90.get(3)))
        best_ml = s[(s.dataset == k) & (s.k == 0) & s.method.isin(["GBM", "GBM-RES", "GBM-ME", "GP-RES"])].rmse.min()
        mac(f"bestml{t}", f3(best_ml))


def scarcity():
    s = pd.read_csv(OUT / "scarcity_summary.csv")
    s["S"] = s.S.astype(str)
    order = ["3", "6", "12", "24", "all"]
    base = ["KA-FIXED", "KA-RE", "DATA-RE"]
    rows, regrets = [], {}
    for k in DOM:
        for S in order:
            g = s[(s.dataset == k) & (s.k == 0) & (s.S == S)].set_index("method")
            best = g.loc[base, "rmse"].min()
            cells = " & ".join(f3(g.rmse.get(m)) for m in ["ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S",
                                                            "GBM-RES", "GBM-ME"])
            fx = g.loc["KA-FIXED"]
            gain = f"{s3(fx['gain_vs_DATA-RE'])} [{s2(fx['gain_vs_DATA-RE_lo'])}, {s2(fx['gain_vs_DATA-RE_hi'])}]"
            reg = g.rmse.get("PAKA-S") - best
            regrets.setdefault(k, []).append(reg)
            rows.append(f"{DLAB[k]} & {S} & {g.n_train.iloc[0]:.0f} & {cells} & {gain} & {s3(reg)} \\\\")
        rows.append("\\midrule")
    write("tab_scarcity.tex", "\n".join(rows[:-1]) + "\n")
    for k in DOM:
        t = TAG[k]
        for S, st in (("3", "three"), ("6", "six"), ("12", "twelve"), ("24", "tfour"), ("all", "all")):
            g = s[(s.dataset == k) & (s.k == 0) & (s.S == S)].set_index("method")
            fx = g.loc["KA-FIXED"]
            mac(f"gfix{st}{t}", f3(fx['gain_vs_DATA-RE']))
            mac(f"gfixabs{st}{t}", a3(fx['gain_vs_DATA-RE']))
            mac(f"gfixlo{st}{t}", f3(fx['gain_vs_DATA-RE_lo']))
            mac(f"gfixhi{st}{t}", f3(fx['gain_vs_DATA-RE_hi']))
            mac(f"sc{st}fix{t}", f3(g.rmse.get("KA-FIXED")))
            mac(f"sc{st}data{t}", f3(g.rmse.get("DATA-RE")))
            mac(f"sc{st}paka{t}", f3(g.rmse.get("PAKA-S")))
            mac(f"sc{st}kare{t}", f3(g.rmse.get("KA-RE")))
            mac(f"sc{st}gbmme{t}", f3(g.rmse.get("GBM-ME")))
        mac(f"maxreg{t}", f3(max(regrets[k])))
    allreg = {}
    for m in ["KA-FIXED", "KA-RE", "DATA-RE", "PAKA", "PAKA-S", "PAKA-S0", "PAKA-S6"]:
        vals = []
        for k in DOM:
            for S in order:
                g = s[(s.dataset == k) & (s.k == 0) & (s.S == S)].set_index("method")
                vals.append(g.rmse.get(m) - g.loc[base, "rmse"].min())
        allreg[m] = (float(np.nanmax(vals)), float(np.nanmean(vals)))
    rows = [f"{NAME[m]} & {s3(mx)} & {s3(mn)} \\\\" for m, (mx, mn) in allreg.items()]
    write("tab_regret.tex", "\n".join(rows) + "\n")
    for m, mt in (("KA-FIXED", "fix"), ("KA-RE", "kare"), ("DATA-RE", "data"), ("PAKA", "sel"), ("PAKA-S", "paka"),
                  ("PAKA-S0", "pakazero"), ("PAKA-S6", "pakasix")):
        mac(f"regmax{mt}", f3(allreg[m][0]))
        mac(f"regmean{mt}", f3(allreg[m][1]))


def theory():
    t = pd.read_csv(OUT / "theory_crossover.csv").set_index("domain")
    rows = []
    for k in DOM:
        r = t.loc[k]
        rows.append(f"{DLAB[k]} & {r.p:.0f} & {r.c:.3f} & {r.B:.3f} & {r.S_star:.1f} [{r.S_star_lo:.1f}, {r.S_star_hi:.1f}] & "
                    f"{r.S_obs:.1f} [{r.S_obs_lo:.1f}, {r.S_obs_hi:.1f}] & {r.ratio:.2f} \\\\")
        tg = TAG[k]
        mac(f"spred{tg}", f"{r.S_star:.1f}")
        mac(f"spredlo{tg}", f"{r.S_star_lo:.1f}")
        mac(f"spredhi{tg}", f"{r.S_star_hi:.1f}")
        mac(f"sobs{tg}", f"{r.S_obs:.1f}")
        mac(f"sobslo{tg}", f"{r.S_obs_lo:.1f}")
        mac(f"sobshi{tg}", f"{r.S_obs_hi:.1f}")
        mac(f"sratio{tg}", f2(r.ratio))
    write("tab_theory.tex", "\n".join(rows) + "\n")
    mac("sobsMin", f"{t.loc[DOM, 'S_obs'].min():.0f}")
    mac("sobsMax", f"{t.loc[DOM, 'S_obs'].max():.0f}")


def unit():
    u = pd.read_csv(OUT / "unit_summary.csv")
    rows = []
    for k in DOM:
        for N in sorted(u[u.dataset == k].N.unique()):
            g = u[(u.dataset == k) & (u.N == N) & (u.k == 0)].set_index("method")
            cells = []
            for m in ["DATA-RE", "KA-RE", "KA-FIXED", "PAKA-S"]:
                r = g.loc[f"{m}|DISPERSED"]
                col = f"gain_vs_{m}|CONCENTRATED"
                cells.append(f"{s3(r[col])} [{s2(r[col + '_lo'])}, {s2(r[col + '_hi'])}]")
            r0 = g.iloc[0]
            rows.append(f"{DLAB[k]} & {N} & {r0.CONCENTRATED:.1f} / {r0.DISPERSED:.1f} & " + " & ".join(cells) + " \\\\")
    write("tab_unit.tex", "\n".join(rows) + "\n")
    for k in DOM:
        N = sorted(u[u.dataset == k].N.unique())[0]
        g = u[(u.dataset == k) & (u.N == N) & (u.k == 0)].set_index("method")
        mac(f"unitN{TAG[k]}", f"{N}")
        mac(f"unitdata{TAG[k]}", f3(g.loc['DATA-RE|DISPERSED', 'gain_vs_DATA-RE|CONCENTRATED']))
        mac(f"unitpaka{TAG[k]}", f3(g.loc['PAKA-S|DISPERSED', 'gain_vs_PAKA-S|CONCENTRATED']))
        mac(f"unitpc{TAG[k]}", f"{g.iloc[0].CONCENTRATED:.0f}")
        mac(f"unitpd{TAG[k]}", f"{g.iloc[0].DISPERSED:.0f}")


def conditional_icc():
    """Programme share of the error that remains after the design variables are modelled
    (data-only hierarchical model, median over the leave-one-programme-out fits), and an
    illustrative resistance-factor consequence of k programme-specific tests."""
    import json
    info = json.loads((OUT / "lopo_fit_info.json").read_text())
    for k in DOM:
        taus = [v["DATA-RE"]["tau"] for kk, v in info.items() if kk.startswith(k + "::")]
        sigs = [v["DATA-RE"]["sig"] for kk, v in info.items() if kk.startswith(k + "::")]
        tau, sig = float(np.median(taus)), float(np.median(sigs))
        t = TAG[k]
        mac(f"ctau{t}", f2(tau))
        mac(f"csig{t}", f2(sig))
        mac(f"cicc{t}", f2(tau ** 2 / (tau ** 2 + sig ** 2)))
        v0 = np.sqrt(tau ** 2 + sig ** 2)
        v3 = np.sqrt(sig ** 2 + tau ** 2 * sig ** 2 / (3 * tau ** 2 + sig ** 2))
        mac(f"vnew{t}", f2(v0))
        mac(f"vthree{t}", f2(v3))
        mac(f"rfac{t}", f"{100 * (np.exp(3.0 * (v0 - v3)) - 1):.0f}")


def labs():
    t = pd.read_csv(OUT / "audit_labs.csv").set_index("domain")
    for k in DOM:
        r = t.loc[k]
        mac(f"labg{TAG[k]}", f"{r.lab_groups:.0f}")
        mac(f"labicc{TAG[k]}", f2(r.lab_icc))
        mac(f"labneff{TAG[k]}", f"{r.lab_effective_n:.0f}")
    mac("labiccMin", f2(t.loc[DOM, "lab_icc"].min()))
    mac("labiccMax", f2(t.loc[DOM, "lab_icc"].max()))
    mac("labneffMin", f"{t.loc[DOM, 'lab_effective_n'].min():.0f}")
    mac("labneffMax", f"{t.loc[DOM, 'lab_effective_n'].max():.0f}")


def circular_scarcity():
    p = OUT / "circular_scarcity_summary.csv"
    if not p.exists():
        return
    c = pd.read_csv(p)
    c["S"] = c.S.astype(str)
    for k in ("FRP", "SFRC"):
        for S, st in (("3", "three"), ("6", "six"), ("12", "twelve")):
            r = c[(c.dataset == k) & (c.k == 0) & (c.S == S) & (c.method == "KA-FIXED")].iloc[0]
            mac(f"cg{st}{TAG[k]}", f3(r.gain_vs_data))
            mac(f"cglo{st}{TAG[k]}", f3(r.gain_lo))
            mac(f"cghi{st}{TAG[k]}", f3(r.gain_hi))


def extrap_circ():
    e = pd.read_csv(OUT / "extrap_summary.csv")
    sp = pd.read_csv(OUT / "extrap_split.csv").set_index("dataset")
    c = pd.read_csv(OUT / "circular_summary.csv")
    cs = pd.read_csv(OUT / "circular_split.csv").set_index("dataset")
    lp = pd.read_csv(OUT / "lopo_summary.csv")
    rows, rows2 = [], []
    order = ["KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S", "GBM", "GBM-RES", "GBM-ME", "GP-RES"]
    for k in ("FRP", "SFRC"):
        for m in order:
            g = e[(e.dataset == k) & (e.method == m)].set_index("k")
            if g.empty:
                continue
            rows.append(f"{DLAB[k]} & {NAME[m]} & {f3(g.rmse.get(0))} & {s3(g.bias.get(0))} & "
                        f"{f2(g.unconservative.get(0))} & {f3(g.rmse.get(3))} \\\\")
            gc = c[(c.dataset == k) & (c.method == m)].set_index("k")
            gl = lp[(lp.dataset == k) & (lp.method == m)].set_index("k")
            rows2.append(f"{DLAB[k]} & {NAME[m]} & {f3(gl.rmse.get(0))} & {f3(gc.rmse.get(0))} & "
                         f"{f3(gl.rmse.get(3))} & {f3(gc.rmse.get(3))} \\\\")
        rows.append("\\midrule")
        rows2.append("\\midrule")
        t = TAG[k]
        r = sp.loc[k]
        mac(f"exthr{t}", f"{r.threshold_mm:.0f}")
        mac(f"extprog{t}", f"{r.train_programmes:.0f}")
        mac(f"extgt{t}", f"{r.targets:.0f}")
        mac(f"extdmax{t}", f"{r.train_d_max:.0f}")
        mac(f"extbeyond{t}", f"{100 * r.beyond_train_max:.0f}")
        for m, mt in (("PAKA-S", "paka"), ("GBM", "gbm"), ("GBM-RES", "gbmres"), ("GBM-ME", "gbmme")):
            g = e[(e.dataset == k) & (e.method == m) & (e.k == 0)].iloc[0]
            mac(f"ex{mt}{t}", f3(g.rmse))
            mac(f"exunc{mt}{t}", f"{100 * g.unconservative:.0f}")
        cr = cs.loc[k]
        mac(f"circprog{t}", f"{cr.programmes:.0f}")
        mac(f"circcut{t}", f"{cr.dropped_up_to:.0f}")
        for m, mt in (("PAKA-S", "paka"), ("KA-FIXED", "fix"), ("DATA-RE", "data")):
            g = c[(c.dataset == k) & (c.method == m) & (c.k == 0)].iloc[0]
            mac(f"circ{mt}{t}", f3(g.rmse))
    write("tab_extrap.tex", "\n".join(rows[:-1]) + "\n")
    write("tab_circular.tex", "\n".join(rows2[:-1]) + "\n")


if __name__ == "__main__":
    domains()
    audit()
    lopo()
    scarcity()
    theory()
    unit()
    extrap_circ()
    conditional_icc()
    labs()
    circular_scarcity()
    write("numbers.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
    print(f"{len(M)} macros -> {GEN}")
