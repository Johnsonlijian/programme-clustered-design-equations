"""R08 post hoc: does applying the certify verdict (free the contradicted SFRC fibre term)
change the k=0 comparison with boosting? See ../PRE_ANALYSIS_certified_anchor.md.

Rungs on IDENTICAL rows and targets (programmes with >= 6 tests), full training library:
  K1   Narayanan-Darwish 1987 as in R06 (sanity reference)
  K1C  N-D evaluated at F = 0 (plain-concrete part); coded d, b_w, f'c, rho, a/d; ln(1+F) free
  K2C  K1C x lambda_s
Protocol = r06_ladder.sc_task with S=None (ks=(0,3), n_draws=10, seed 0). GBM and GBM-ME do
not depend on the anchor and are run once, with the K1 frame.
Outputs: ../outputs/certified_per_draw.csv, certified_summary.csv, certified_paired.csv,
         certified_budget.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROUNDS = HERE.parents[1]
sys.path[:0] = [str(HERE),
                str(ROUNDS / "R06_mechanism_pivot_2026-10-22" / "scripts"),
                str(ROUNDS / "R05_consolidation_2026-10-21" / "scripts"),
                str(ROUNDS / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import r06_rungs as R  # noqa: E402
import r06_ladder as L  # noqa: E402
from kag_bench_r05 import run_programme  # noqa: E402
from kag_data import nd1987_v_kN  # noqa: E402
import r08_budget_check as BC  # noqa: E402

OUT = R08 / "outputs"
LMM = ("KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S")
ML = ("GBM", "GBM-ME")


def frame(rung):
    df, coded, free = R.load("SFRC", "K1")
    df = df.copy()
    if rung == "K1":
        pass
    elif rung in ("K1C", "K2C"):
        df["m"] = np.log(nd1987_v_kN(df.fc, df.b, df.d, df.rho, df.a_d, 0.0 * df.F))
        if rung == "K2C":
            df["m"] = df["m"] + np.log(R.lambda_s(df["d"]))
        coded = [c for c in coded if c != "ln1p_F"]
        free = list(free) + ["ln1p_F"]
    else:
        raise KeyError(rung)
    coded, free = L._canon_feats("SFRC", coded, free)
    return df.reset_index(drop=True), coded, free


def task(rung, target):
    df, coded, free = frame(rung)
    arms = LMM + ML if rung == "K1" else LMM
    rows, info = run_programme(df, coded, free, target, arms, ks=(0, 3), n_draws=10, seed=0)
    rows["rung"] = rung
    w = info.get("PAKA-S", {})
    rows["w_fixed"], rows["w_re"], rows["w_data"] = (w.get("w_KA-FIXED", np.nan), w.get("w_KA-RE", np.nan),
                                                      w.get("w_DATA-RE", np.nan))
    return rows


def boot_mean(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(v), size=(n_boot, len(v)))
    lo, hi = np.percentile(v[b].mean(1), [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def main():
    df, _, _ = frame("K1")
    n_s = df.groupby("source").size()
    targets = sorted(n_s[n_s >= 6].index)
    print(f"SFRC targets: {len(targets)}", flush=True)
    jobs = [delayed(task)(r, t) for r in ("K1", "K1C", "K2C") for t in targets]
    res = Parallel(n_jobs=12, backend="loky", batch_size=1)(jobs)
    pd_ = pd.concat(res, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd_.to_csv(OUT / "certified_per_draw.csv", index=False)

    per = pd_.groupby(["rung", "method", "k", "source"], as_index=False).agg(
        rmse=("rmse", "mean"), w_fixed=("w_fixed", "first"), w_re=("w_re", "first"), w_data=("w_data", "first"))
    # anchor-free arms are rung-independent: label them once
    ml = per[per.method.isin(ML)].copy()
    ml["rung"] = "-"
    lmm = per[per.method.isin(LMM)]
    per = pd.concat([lmm, ml], ignore_index=True)

    summ = []
    for (rung, meth, k), blk in per.groupby(["rung", "method", "k"]):
        m, lo, hi = boot_mean(blk.rmse.values)
        summ.append(dict(rung=rung, method=meth, k=k, programmes=len(blk), rmse=m, lo=lo, hi=hi,
                         w_fixed=blk.w_fixed.mean(), w_re=blk.w_re.mean(), w_data=blk.w_data.mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "certified_summary.csv", index=False)

    def vec(rung, meth, k):
        b = per[(per.rung == rung) & (per.method == meth) & (per.k == k)].set_index("source").rmse
        return b.reindex(targets)

    pairs = []
    for k in (0, 3):
        for a, b in [(("K1C", "PAKA-S"), ("-", "GBM")), (("K1C", "PAKA-S"), ("-", "GBM-ME")),
                     (("K2C", "PAKA-S"), ("-", "GBM")), (("K1", "PAKA-S"), ("-", "GBM")),
                     (("K1C", "PAKA-S"), ("K1", "PAKA-S")), (("K2C", "PAKA-S"), ("K1", "PAKA-S")),
                     (("K1C", "PAKA-S"), ("K1C", "DATA-RE")), (("K1C", "KA-FIXED"), ("K1", "KA-FIXED")),
                     (("K1C", "KA-RE"), ("K1", "KA-RE"))]:
            d = (vec(*a, k) - vec(*b, k)).dropna()
            m, lo, hi = boot_mean(d.values)
            pairs.append(dict(k=k, a=f"{a[1]}[{a[0]}]", b=f"{b[1]}[{b[0]}]", programmes=len(d),
                              diff=m, lo=lo, hi=hi))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "certified_paired.csv", index=False)

    # sanity gates (pre-registered in PRE_ANALYSIS_certified_anchor.md)
    s = summ.set_index(["rung", "method", "k"]).rmse
    gate_paka = abs(s[("K1", "PAKA-S", 0)] - 0.2602531050792629)
    gate_gbm = abs(s[("-", "GBM", 0)] - 0.241)
    dr = [s[(r, "DATA-RE", 0)] for r in ("K1", "K1C", "K2C")]
    print(f"gate PAKA-S K1 k=0 |diff| = {gate_paka:.2e}; GBM |diff to 0.241| = {gate_gbm:.4f}; "
          f"DATA-RE across rungs = {dr}", flush=True)

    budget = []
    for rung in ("K1", "K1C", "K2C"):
        d2, c2, f2 = frame(rung)
        b = BC.split_inputs(d2, c2, f2)
        budget.append(dict(rung=rung, p=b["p"], q=b["q"], B=b["B"], c=b["c"], c_F=b["c_F"], c_D=b["c_D"],
                           S_prespec=b["S_star"], S_split=b["S_split"]))
    pd.DataFrame(budget).to_csv(OUT / "certified_budget.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False))
    print(pairs.round(4).to_string(index=False))
    print(pd.DataFrame(budget).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
