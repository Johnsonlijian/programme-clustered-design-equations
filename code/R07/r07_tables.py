"""R07 WP5 -- LaTeX fragments for the merged manuscript.

tab_audit_detect.tex : detection by corruption type (audit vs support gate)
tab_dossiers.tex     : per-anchor validity profile alongside the price
tab_downstream.tex   : clean vs corrupted deployment for the three cases

Rows only (booktabs), consumed through \\tabinput in main_r07.tex.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R07 = HERE.parent
OUT = R07 / "outputs"
GEN = HERE.parents[2] / "paper" / "manuscript" / "generated"
GEN.mkdir(parents=True, exist_ok=True)


def sstar(v):
    v = float(v)
    if not np.isfinite(v) or v > 1e6:
        return r"$\infty$"
    return f"{v:.1f}"


def signed(v):
    v = float(v)
    s = f"{v:+.3f}"
    return s.replace("-", r"\ensuremath{-}") if s.startswith("-") else s


def tab_audit_detect():
    d = pd.read_csv(OUT / "stress_results.csv")
    groups = [
        ("sign", "Sign flip (CaO; $\\lambda_s$)", "in-support", "sign/exponent"),
        ("formula", "Formula swap (K1$\\leftrightarrow$K3)", "in-support", "exponent"),
        ("unit", "Unit slip (mm$\\to$cm; \\%$\\to$fraction)", "out-of-support", "unit"),
        ("range", "Range drift (no CaO; $a/d>2.5$)", "out-of-support", "range"),
        ("clean", "Valid rule (control)", "---", "---"),
    ]
    rows = []
    for key, lab, grp, checks in groups:
        g = d[d.corrupt == key]
        n = len(g)
        na = int(g.audit_rejected.sum())
        ns = int(g.support_flagged.sum())
        rows.append(f"{lab} & {grp} & {n} & {na} & {ns} & {checks} \\\\")
    (GEN / "tab_audit_detect.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  -> tab_audit_detect.tex ({len(rows)} rows)")


def tab_dossiers():
    d = pd.read_csv(OUT / "anchor_dossiers.csv")
    rows = []
    for r in d.itertuples():
        terms = f"{r.n_consistent}/{r.n_inconsistent}/{r.n_contradicted}"
        worst = f"{r.worst_term.replace('ln_', '').replace('ln1p_', '').replace('x_', '')} ({r.worst_z:+.0f})"
        worst = worst.replace("-", r"\ensuremath{-}")
        rows.append(f"{r.anchor.replace('-', ' ')} & {terms} & {worst} & "
                    f"{sstar(r.S_star)} & {signed(r.paka_regret_max)} \\\\")
    (GEN / "tab_dossiers.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  -> tab_dossiers.tex ({len(rows)} rows)")


def tab_downstream():
    d = pd.read_csv(OUT / "downstream_summary.csv")
    rows = []
    for dom, rung in (("CARB", "K2"), ("FRP", "K2"), ("SFRC", "K2")):
        g = d[(d.domain == dom) & (d.rung == rung)]
        def m(case, meth):
            return g[(g.case == case) & (g.method == meth)]["mean"].iloc[0]
        label = {"CARB": "Carbonation K2", "FRP": "FRP-RC shear K2",
                 "SFRC": "SFRC shear K2"}[dom]
        rows.append(f"{label} & {m('clean','PAKA-S'):.3f} & {m('corrupt','KA-FIXED'):.3f} & "
                    f"{m('corrupt','PAKA-S'):.3f} & {m('clean','DATA-RE'):.3f} \\\\")
    (GEN / "tab_downstream.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  -> tab_downstream.tex ({len(rows)} rows)")


if __name__ == "__main__":
    print("writing R07 table fragments:")
    tab_audit_detect()
    tab_dossiers()
    tab_downstream()
    print("done")
