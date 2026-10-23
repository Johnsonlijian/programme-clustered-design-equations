"""R06 supplement appendix tables (gate-report condition C8).

tab_anchordefs.tex : the eleven imposed anchor forms -- the equation as
    implemented in the shared data layer, its source, and the verification
    status recorded in 07_source_registry.md / logs/human_todo.md. No new
    claims: every equation mirrors kag_data_r05.py / r06_rungs.py, every
    status mirrors the registry.
tab_progmeta.tex   : programme-level metadata of the four databases -- the
    label as recorded in the source database, the publication year where it
    could be parsed from the label (FRP, SFRC; the carbonation and deep-beam
    databases do not record one), the number of tests, the min--max range of
    the two variables that carry the anchor's coded exponents, and the
    calibration-overlap flag used by the circularity analysis (FRP <= 2002,
    SFRC <= 1987). DEEP-WB reanalyses the DEEP rows and is not repeated.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts")]
import r06_rungs as R  # noqa: E402

R06 = HERE.parent
GEN = R06.parents[1] / "paper" / "manuscript" / "generated"
GEN.mkdir(parents=True, exist_ok=True)


def esc(s) -> str:
    """Escape LaTeX specials in free-text database labels; collapse blanks."""
    out = str(s)
    for ch, rep in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                    ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}")]:
        out = out.replace(ch, rep)
    return re.sub(r"\s+", " ", out).strip()


def rng(v: pd.Series) -> str:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if v.empty:
        return "--"
    return f"{v.min():.4g}--{v.max():.4g}"


# --------------------------------------------------------------------------
# anchor definitions (equations as implemented; statuses from the registry)
# --------------------------------------------------------------------------
ANCHORS = [
    ("FRP K1", "ACI 440.1R-15",
     r"$V_c=0.4\sqrt{f'_c}\,b_w\,kd$ with $E_c=4700\sqrt{f'_c}$, "
     r"$n\rho=\rho E_f/E_c$, $k=\sqrt{2n\rho+(n\rho)^2}-n\rho$ "
     r"(form of Tureyen et al.\ 2003)",
     "guide; clause/version check pending"),
    ("FRP K2", r"ACI 440.1R-15 $\times\lambda_s$",
     r"FRP K1 multiplied by $\lambda_s=\sqrt{2/(1+0.004d)}\le1$, $d$ in mm "
     r"(ACI 318-19 size factor for members without shear reinforcement)",
     "clause check pending"),
    ("SFRC K1", "Narayanan--Darwish (1987)",
     r"$v=e\,(0.24 f_{sp}+80\rho\,(a/d)^{-1})+0.41\times4.15\,F$ with "
     r"$f_{sp}=f_{cu}/(20-\sqrt{F})+0.7+\sqrt{F}$, $f_{cu}=f'_c/0.8$, "
     r"$e=1$ for $a/d>2.8$ else $2.8\,d/a$; $V=v\,b\,d$; $F$ = fibre factor",
     "ref.\\ verified; eq.\\ text pending"),
    ("SFRC K2", r"Narayanan--Darwish $\times\lambda_s$",
     r"SFRC K1 multiplied by $\lambda_s$ (as FRP K2)",
     "as SFRC K1"),
    ("CARB K1", "Fick diffusion",
     r"$k\propto C^{1/2}$ (sharp diffusion front); CaO$_r$ and $w/b$ free",
     "---"),
    ("CARB K2", "Papadakis (1991)",
     r"$k\propto(C/a)^{1/2}$ with binding capacity $a\propto$ CaO$_r$; "
     r"$w/b$ free",
     "ref.\\ verified"),
    ("CARB K3", "Leemann--Moro (2015, 2017)",
     r"$k\propto C^{1/2}\,(w/b)/\mathrm{CaO}_r$ (CO$_2$ buffer capacity "
     r"per paste volume); no free exponents",
     "refs.\\ verified"),
    ("DEEP BEAM", "ACI 318-19 one-way shear",
     r"$V_c=0.66\,\lambda_s\,\rho_w^{1/3}\sqrt{f'_c}\,b_w d$ (Table 22.5.5.1; "
     r"imposed outside its nominal scope at $a/d\le2.5$)",
     "clause check pending"),
    ("DEEP ARCH", "Zsutty (1971)",
     r"$v=2.2\,(f'_c\rho\,d/a)^{1/3}(2.5\,d/a)$ MPa for $a/d<2.5$",
     "ref.\\ verified; eq.\\ text pending"),
    ("DEEP ARCHS", r"Zsutty $\times\lambda_s$",
     r"DEEP ARCH multiplied by $\lambda_s$ (as FRP K2)",
     "as DEEP ARCH"),
    ("DEEP ARCH99", "ACI 318-99 Eq.~(11-29)",
     r"$V_c=(3.5-2.5\,M_u/(V_ud))$ $(1.9\sqrt{f'_c}+2500\rho_w V_ud/M_u)\,b_wd$ "
     r"$\le6\sqrt{f'_c}\,b_wd$, first factor $\le2.5$, "
     r"$M_u/(V_ud)=\min(0.5\,a/d,\,1)$; constants in psi units "
     r"(SI: 0.158, 17.24, 0.5)",
     "original text to verify"),
]


def tab_anchordefs():
    rows = [f"{rung} & {basis} & {eq} & {status} \\\\"
            for rung, basis, eq, status in ANCHORS]
    (GEN / "tab_anchordefs.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  -> {GEN / 'tab_anchordefs.tex'} ({len(ANCHORS)} anchors)")


# --------------------------------------------------------------------------
# programme-level metadata
# --------------------------------------------------------------------------
META = {
    "FRP": dict(rung="K1", var1=("d", "d"), var2=("fc", "f'_c"), cut=2002,
                head="FRP-RC shear",
                note=r"ranges: effective depth $d$ (mm), cylinder strength $f'_c$ (MPa)"),
    "SFRC": dict(rung="K1", var1=("d", "d"), var2=("fc", "f'_c"), cut=1987,
                 head="SFRC shear",
                 note=r"ranges: $d$ (mm), $f'_c$ (MPa)"),
    "CARB": dict(rung="K1", var1=("co2", "C"), var2=("wb", "w/b"), cut=None,
                 head="Carbonation",
                 note=r"ranges: CO$_2$ concentration $C$ (\%), water--binder ratio $w/b$; label = reference code of the RILEM-derived dataset"),
    "DEEP": dict(rung="ARCH", var1=("d", "d"), var2=("a_d", "a/d"), cut=None,
                 head="RC deep beams",
                 note=r"ranges: $d$ (mm), shear-span ratio $a/d$; label = author field of the source database"),
}


def tab_progmeta():
    rows = []
    for name, spec in META.items():
        df, _, _ = R.load(name, spec["rung"])
        rows.append(r"\multicolumn{4}{l}{\textbf{" + spec["head"] + "} --- "
                    + spec["note"] + r"}\\")
        info = []
        for src, gg in df.groupby("source"):
            yrs = gg["year"].dropna() if "year" in gg.columns else pd.Series(dtype=float)
            yr = float(yrs.iloc[0]) if len(yrs) else np.nan
            info.append((yr if np.isfinite(yr) else np.inf, str(src), len(gg),
                         rng(gg[spec["var1"][0]]), rng(gg[spec["var2"][0]]), yr))
        for _, src, n, r1, r2, yr in sorted(info, key=lambda t: (t[0], t[1].lower())):
            ytxt = f"{yr:.0f}" if np.isfinite(yr) else "--"
            flag = (r"~$\dagger$" if spec["cut"] is not None and np.isfinite(yr)
                    and yr <= spec["cut"] else "")
            ranges = f"${spec['var1'][1]}$ {r1}; ${spec['var2'][1]}$ {r2}"
            rows.append(f"{esc(src)} & {ytxt} & {n} & {ranges}{flag} \\\\")
        rows.append(r"\midrule")
    rows.pop()  # drop the trailing \midrule
    (GEN / "tab_progmeta.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    n = sum(1 for r_ in rows if not r_.startswith("\\multicolumn") and r_ != "\\midrule")
    print(f"  -> {GEN / 'tab_progmeta.tex'} ({n} programmes)")


if __name__ == "__main__":
    print("writing R06 appendix tables:")
    tab_anchordefs()
    tab_progmeta()
    print("appendix tables done")
