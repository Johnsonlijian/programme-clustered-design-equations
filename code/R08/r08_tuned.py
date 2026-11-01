"""R08 post hoc: tuned ML baselines vs the best hierarchical predictor.

Builds outputs/tuned_vs_hier.csv and generated/tab_tuned.tex once
outputs/tuned_summary.csv exists (written by r08_tuned_baselines.py at the end
of the background job).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROOT = R08.parent.parent
OUT = R08 / "outputs"
R06 = ROOT / "rounds" / "R06_mechanism_pivot_2026-10-22"
GEN = ROOT / "paper" / "manuscript" / "generated"

DOMS = ["FRP", "SFRC", "CARB", "DEEP"]
S_LEV = ["3", "6", "12", "24"]
DOMLAB = {"FRP": "FRP-RC shear", "SFRC": "SFRC shear", "CARB": "Carbonation",
          "DEEP": "RC deep beams"}
ARMLAB = {"KA-FIXED": "ci", "KA-RE": "re", "DATA-RE": "dat", "PAKA-S": "paka"}


def f(x, d=3):
    return f"{float(x):.{d}f}"


def main():
    ts = pd.read_csv(OUT / "tuned_summary.csv", dtype={"S": str})
    ls = pd.read_csv(R06 / "outputs" / "ladder_summary.csv", dtype={"S": str})
    tpd = pd.read_csv(OUT / "tuned_per_draw.csv.gz", dtype={"S": str})
    lpd = pd.read_csv(R06 / "outputs" / "ladder_per_draw.csv.gz", dtype={"S": str})

    rows = []
    for dom in DOMS:
        for S in S_LEV:
            h = ls[(ls.dataset == dom) & (ls.S == S) & (ls.k == 0)
                   & ls.method.isin(["KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S"])]
            i = h.rmse.idxmin()
            g = ts[(ts.dataset == dom) & (ts.S == S) & (ts.k == 0)]
            gbm = g[g.method == "GBM"].rmse.iloc[0]
            gme = g[g.method == "GBM-ME"].rmse.iloc[0]
            bm = min(gbm, gme)
            rows.append(dict(domain=dom, S=S,
                             best_hier=ls.loc[i, "rmse"],
                             hier_rung=ls.loc[i, "rung"],
                             hier_method=ls.loc[i, "method"],
                             gbm=gbm, gbm_me=gme, best_ml=bm,
                             ml_minus_hier=bm - ls.loc[i, "rmse"]))
    comp = pd.DataFrame(rows)

    # paired per-target difference: GBM-ME minus best hierarchical
    rng = np.random.default_rng(0)
    diffs = []
    for r in comp.itertuples():
        a = (tpd[(tpd.dataset == r.domain) & (tpd.S == r.S) & (tpd.k == 0)
                 & (tpd.method == "GBM-ME")]
             .groupby("source").rmse.mean())
        b = (lpd[(lpd.dataset == r.domain) & (lpd.S == r.S) & (lpd.k == 0)
                 & (lpd.rung == r.hier_rung) & (lpd.method == r.hier_method)]
             .groupby("source").rmse.mean())
        j = a.index.intersection(b.index)
        d = (a.loc[j] - b.loc[j]).to_numpy()
        bs = rng.integers(0, len(d), size=(2000, len(d)))
        lo, hi = np.percentile(d[bs].mean(1), [2.5, 97.5])
        diffs.append((d.mean(), lo, hi, len(d)))
    comp["diff"] = [x[0] for x in diffs]
    comp["lo"] = [x[1] for x in diffs]
    comp["hi"] = [x[2] for x in diffs]
    comp["n_targets"] = [x[3] for x in diffs]
    comp.to_csv(OUT / "tuned_vs_hier.csv", index=False)
    print(comp.round(4).to_string(index=False))

    # ---- table body ----
    ts_i = ts.set_index(["dataset", "S", "k", "method"])
    lines, last = [], None
    for dom in DOMS:
        if last is not None:
            lines.append(r"\midrule")
        last = dom
        for meth, lab in (("GBM", "GBM (tuned)"), ("GBM-ME", "GBM-ME (tuned)")):
            cells = []
            for S in S_LEV:
                g = ts_i.loc[(dom, S, 0, meth)]
                cells.append(f"{f(g.rmse)} [{f(g.lo, 2)}, {f(g.hi, 2)}]")
            lines.append(f"{DOMLAB[dom]} & {lab} & " + " & ".join(cells) + r" \\")
        blk = comp[comp.domain == dom]
        cells = [f"{f(r.best_hier)} ({r.hier_rung}\\,{ARMLAB[r.hier_method]})"
                 for r in blk.itertuples()]
        lines.append(f"{DOMLAB[dom]} & best hier. (rung,arm) & "
                     + " & ".join(cells) + r" \\")
    (GEN / "tab_tuned.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT / 'tuned_vs_hier.csv'} and {GEN / 'tab_tuned.tex'}")


if __name__ == "__main__":
    main()
