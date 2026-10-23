"""R02 tables: complete LaTeX tabulars + numbers.tex, generated from outputs.

Each fragment is a whole tabular environment with its own header row, so it is
`\\input` in vertical mode.

Run after r02_domain_audit.py, r02_scarcity2.py and r02_theory.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROUND = Path(__file__).resolve().parents[1]
OUT = ROUND / "outputs"
GEN = ROUND / "note" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

DOMAIN_TEX = {"FRP": "FRP", "SFRC": "SFRC", "DEEP": "DEEP"}


def f3(x):
    return "---" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def f2(x):
    return "---" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def f1(x):
    return "---" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1f}"


def tabular(spec, rows, header):
    body = "\n".join([header + r" \\", r"\midrule"] + rows)
    return ("\\begin{tabular}{" + spec + "}\\toprule\n" + body +
            "\n\\bottomrule\\end{tabular}\n")


def main():
    audit = pd.read_csv(OUT / "domain_audit.csv")
    theory = pd.read_csv(OUT / "theory_prediction.csv")
    summ = pd.read_csv(OUT / "scarcity2_summary.csv")

    # ---------------- table: domains -------------------------------------------
    rows = []
    for _, r in audit.iterrows():
        rows.append(f"{DOMAIN_TEX[r.domain]} & {int(r.n)} & {int(r.programmes)} & "
                    f"{int(r.programmes_ge6)} & {f2(r.anchor_ratio_mean)} & {f2(r.anchor_ratio_cov)} & "
                    f"{f2(r.icc)} & {f2(r.eta2_d)} & "
                    f"{f2(r.d_within)} ({f2(r.d_within_se)}) & {f2(r.d_between)} ({f2(r.d_between_se)}) & "
                    f"{f1(r.d_z)} & {r.scope} \\\\")
    head = (r"域 & $n$ & 项目 & $\ge6$ & 锚比值 & CoV & ICC & $\eta^2_d$ & "
            r"项目内 & 项目间 & $z$ & 锚范围")
    (GEN / "tab_domains.tex").write_text(tabular("lccccccccccl", rows, head), encoding="utf-8")

    # ---------------- table: theory --------------------------------------------
    rows = []
    for ds in ["FRP", "SFRC", "DEEP"]:
        blk = theory[(theory.dataset == ds) & (theory.population == "all")]
        if blk.empty:
            continue
        r = blk.iloc[0]
        g = theory[(theory.dataset == ds) & (theory.population == "ge6")]
        g = g.iloc[0] if not g.empty else None
        ci = (f"[{f1(r.S_star_pred_lo)}, {f1(r.S_star_pred_hi)}]"
              if np.isfinite(r.S_star_pred_lo) else "---")
        rows.append(f"{DOMAIN_TEX[ds]} & {f2(r.A)} & {f3(r.B)} & {f1(r.S_star_pred)} & "
                    f"{f1(r.S_star_meas)} & {f2(r.ratio)} & {f2(r.gain_r2)} & "
                    f"{f1(g.S_star_pred) if g is not None else '---'} & {ci} \\\\")
    head = (r"域 & $A$ & $B$ & $S^\ast$预测 & $S^\ast$实测 & 比值 & $R^2$ & "
            r"$S^\ast$（ge6） & 95\% 区间")
    (GEN / "tab_theory.tex").write_text(tabular("lcccccccc", rows, head), encoding="utf-8")

    # ---------------- table: arms over S ---------------------------------------
    arms = [a for a in ["ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "EAT-T"]
            if a in set(summ.method)]
    rows = []
    for ds in ["FRP", "SFRC", "DEEP"]:
        blk = summ[(summ.dataset == ds) & (summ.k == 0)]
        if blk.empty:
            continue
        piv = blk.pivot_table(index="S_order", columns="method", values="rmse")
        lbl = blk.drop_duplicates("S_order").set_index("S_order").S
        for so in sorted(piv.index):
            cells = [DOMAIN_TEX[ds], "0", str(lbl[so])]
            best = piv.loc[so].min()
            for a in arms:
                v = piv.loc[so].get(a, np.nan)
                s = f3(v)
                if np.isfinite(v) and abs(v - best) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows = rows[:-1]
    head = r"域 & $k$ & $S$ & 规范原式 & 规范固定 & 锚定层次 & 纯数据层次 & 理论自适应"
    (GEN / "tab_arms.tex").write_text(tabular("llcccccc", rows, head), encoding="utf-8")

    # ---------------- macros ----------------------------------------------------
    M = {}
    for _, r in audit.iterrows():
        d = r.domain
        M[f"{d}N"] = str(int(r.n))
        M[f"{d}P"] = str(int(r.programmes))
        M[f"{d}Pge"] = str(int(r.programmes_ge6))
        M[f"{d}ICC"] = f2(r.icc)
        M[f"{d}ETA"] = f2(r.eta2_d)
        M[f"{d}DW"] = f2(r.d_within)
        M[f"{d}DWse"] = f2(r.d_within_se)
        M[f"{d}DB"] = f2(r.d_between)
        M[f"{d}DBse"] = f2(r.d_between_se)
        M[f"{d}DZ"] = f1(r.d_z)
        M[f"{d}AR"] = f2(r.anchor_ratio_mean)
        M[f"{d}ARcov"] = f2(r.anchor_ratio_cov)
    for _, r in theory[theory.population == "all"].iterrows():
        d = r.dataset
        M[f"{d}SSp"] = f1(r.S_star_pred)
        M[f"{d}SSm"] = f1(r.S_star_meas)
        M[f"{d}SSlo"] = f1(r.S_star_pred_lo)
        M[f"{d}SShi"] = f1(r.S_star_pred_hi)
        M[f"{d}Ratio"] = f2(r.ratio)
        M[f"{d}Rsq"] = f2(r.gain_r2)

    for ds in ["FRP", "SFRC", "DEEP"]:
        blk = summ[(summ.dataset == ds) & (summ.k == 0)]
        if blk.empty:
            continue
        for tag, so in (("GainThree", 0), ("GainAll", 4)):
            row = blk[(blk.S_order == so) & (blk.method == "KA-RE")]
            base = blk[(blk.S_order == so) & (blk.method == "DATA-RE")]
            if row.empty or base.empty:
                continue
            g = float(base.rmse.iloc[0] ** 2 - row.rmse.iloc[0] ** 2)
            M[f"{ds}{tag}"] = f"{g:+.3f}"

    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(M)} macros and 3 tabulars to {GEN}")


if __name__ == "__main__":
    main()
