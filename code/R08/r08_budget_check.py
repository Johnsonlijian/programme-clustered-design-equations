"""R08 post hoc diagnostics of the variance budget (written 2026-10-25, AFTER the R06
crossovers were observed; everything here is exploratory, not pre-specified).

1. Reproduce the archived pre-specified S* = p c / B for every rung (sanity gate).
2. Split the learning-cost term by arm. The pre-specified budget uses ONE c, computed from
   the anchor residual r = y - m with nothing fitted. The first-order argument behind it
   (r05_theory docstring) actually has two different residual variances:
       KA-FIXED : learns level + q free exponents   -> cost (1+q) c_F / S
       DATA-RE  : learns level + p coded + q free   -> cost (1+p+q) c_D / S
   with c_F from the residual after the fixed arm's own fit (level + free exponents) and
   c_D from the residual after the data-only fit (level + coded + free exponents). Equating
   B + (1+q) c_F / S = (1+p+q) c_D / S gives
       S*_split = [ (1+p+q) c_D - (1+q) c_F ] / B .
   The pre-specified formula is the special case c_F = c_D. When the anchor's misfit leaks
   into its own programme variance (c_F > c_D) the pre-specified S* is too large.
3. Programme-bootstrap intervals for S*_split (same resampling as r05_theory.boot_pred).

Outputs: ../outputs/budget_split.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROUNDS = HERE.parents[1]
sys.path[:0] = [str(ROUNDS / "R06_mechanism_pivot_2026-10-22" / "scripts"),
                str(ROUNDS / "R05_consolidation_2026-10-21" / "scripts"),
                str(ROUNDS / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import r06_rungs as R  # noqa: E402
import r05_theory as T  # noqa: E402
from kag_models_r05 import fast_lmm_gls, oneway_reml  # noqa: E402

OUT = R08 / "outputs"
THEORY = ROUNDS / "R06_mechanism_pivot_2026-10-22" / "outputs" / "ladder_theory.csv"
ALL = dict(R.RUNGS, **R.SENSITIVITY)


def contrast_var(r, X, g):
    n_j = pd.Series(g).value_counts().values
    if X.shape[1] == 0:
        _, tau, sig = oneway_reml(r, g)
    else:
        _, _, tau, sig = fast_lmm_gls(r, np.column_stack([np.ones(len(r)), X]), g)
    return tau ** 2 + sig ** 2 * float(np.mean(1.0 / n_j)), tau, sig


def split_inputs(df, coded, free):
    th = T.inputs(df, coded, free)
    g = df.source.values
    r = df.y.values - df.m.values
    Xf = df[free].values if free else np.zeros((len(df), 0))
    Xd = df[list(coded) + list(free)].values
    cF, tauF, sigF = contrast_var(r, Xf, g)
    cD, tauD, sigD = contrast_var(r, Xd, g)
    p, q, B = th["p"], len(free), th["B"]
    num = (1 + p + q) * cD - (1 + q) * cF
    if B <= 1e-8:
        s_split = np.inf
    else:
        s_split = max(num / B, 0.0)
    return dict(th, q=q, c_F=cF, tau_F=tauF, sig_F=sigF, c_D=cD, tau_D=tauD, sig_D=sigD,
                S_split=s_split)


def boot_split(df, coded, free, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    progs = df.source.unique()
    out = []
    for _ in range(n_boot):
        draw = rng.choice(progs, size=len(progs), replace=True)
        sub = pd.concat([df[df.source == s].assign(source=f"{s}__{i}") for i, s in enumerate(draw)],
                        ignore_index=True)
        try:
            out.append(split_inputs(sub, coded, free)["S_split"])
        except Exception:
            continue
    return np.array(out)


def main(n_boot=200):
    ref = pd.read_csv(THEORY)
    rows = []
    for name, rungs in ALL.items():
        for rung in rungs:
            df, coded, free = R.load(name, rung)
            s = split_inputs(df, coded, free)
            a = ref[(ref.domain == name) & (ref.rung == rung)].iloc[0]
            rel = abs(s["S_star"] - a.S_star) / a.S_star
            assert rel < 1e-6, f"{name} {rung}: S* {s['S_star']} != archived {a.S_star}"
            lo = hi = np.nan
            if n_boot and np.isfinite(s["S_split"]):
                bs = boot_split(df, coded, free, n_boot)
                fin = bs[np.isfinite(bs)]
                lo, hi = np.percentile(fin, [2.5, 97.5])
            rows.append(dict(domain=name, rung=rung, p=s["p"], q=s["q"], B=s["B"],
                             c_prespec=s["c"], c_F=s["c_F"], c_D=s["c_D"],
                             tau_F=s["tau_F"], tau_D=s["tau_D"], sig_F=s["sig_F"], sig_D=s["sig_D"],
                             S_prespec=s["S_star"], S_prespec_lo=a.S_star_lo, S_prespec_hi=a.S_star_hi,
                             S_split=s["S_split"], S_split_lo=lo, S_split_hi=hi,
                             S_obs=a.S_obs, S_obs_lo=a.S_obs_lo, S_obs_hi=a.S_obs_hi,
                             no_cross=a.obs_boot_no_crossing))
            print(f"{name:7s} {rung:6s} S*pre={s['S_star']:.2f} S*split={s['S_split']:.2f} "
                  f"[{lo:.1f},{hi:.1f}] obs={a.S_obs}", flush=True)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "budget_split.csv", index=False)
    return t


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
