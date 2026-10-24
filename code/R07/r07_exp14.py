"""R07 hardening experiments 1 and 4 (no model refits).

1. Tolerance curve: verdict distribution of the TOST rule as the equivalence
   margin epsilon varies over {0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50}, with
   epsilon expressed as a fraction of |theta0| (and as absolute log units for
   the silence assertions).
4. S* calibration: the 11 computable rungs; calibration slope and intercept of
   observed on predicted crossover (log scale), and CRPS of the budget's
   predictive distribution (lognormal fitted from the 95% interval) against two
   baselines: a fixed factor-two band and a constant predictor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R06 = Path(__file__).resolve().parents[1].parent / "R06_mechanism_pivot_2026-10-22"
R07 = Path(__file__).resolve().parent.parent
OUT = R07 / "outputs"
Z_ONE = 1.6448536269514722
SIG95 = 1.959963984540054


# ---------------------------------------------------------------- experiment 1
def tolerance_curve() -> pd.DataFrame:
    df = pd.read_csv(OUT / "empirical_terms.csv")
    rows = []
    for frac in (0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50):
        verdicts = []
        for r in df.itertuples():
            theta0, d, se = r.anchor_implied, r.dW, r.dW_se
            eps = abs(theta0) * frac if abs(theta0) > 1e-9 else frac * 0.2
            data = theta0 + d
            lo, hi = d - Z_ONE * se, d + Z_ONE * se
            if abs(theta0) > 1e-9:
                if (theta0 > 0 and data / se < -Z_ONE) or (theta0 < 0 and data / se > Z_ONE):
                    v = "CONTRADICTED"
                elif lo > -eps and hi < eps:
                    v = "CONFIRMED"
                elif abs(d) / se > 1.96:
                    v = "ATTENUATED"
                else:
                    v = "NOT-EVALUABLE"
            else:
                if abs(d) - Z_ONE * se > eps:
                    v = "CONTRADICTED"
                elif abs(d) + Z_ONE * se < eps:
                    v = "CONFIRMED"
                elif abs(d) / se > 1.96:
                    v = "ATTENUATED"
                else:
                    v = "NOT-EVALUABLE"
            verdicts.append(v)
        counts = pd.Series(verdicts).value_counts()
        rows.append(dict(eps_fraction=frac, **{k: int(counts.get(k, 0)) for k in
                                              ("CONFIRMED", "ATTENUATED", "CONTRADICTED", "NOT-EVALUABLE")}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment 4
def crps_lognormal(y: float, mu: float, sigma: float) -> float:
    """CRPS of a lognormal predictive distribution at observation y (all > 0)."""
    from math import erf, log, sqrt

    def Phi(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def phi(x: float) -> float:
        return np.exp(-0.5 * x * x) / sqrt(2 * np.pi)

    if y <= 0 or sigma <= 0:
        return float("nan")
    ly = log(y)
    z = (ly - mu) / sigma
    # CRPS(lognormal) = y (2 Phi(z) - 1) - 2 exp(mu + s^2/2) (Phi(z - s) + Phi(s/sqrt2) - 1)
    a = 2 * Phi(z) - 1
    b = np.exp(mu + 0.5 * sigma ** 2)
    c = Phi(z - sigma) + Phi(sigma / np.sqrt(2)) - 1
    return float(y * a - 2 * b * c)


def calibration() -> tuple[pd.DataFrame, pd.DataFrame]:
    th = pd.read_csv(R06 / "outputs" / "ladder_theory.csv")
    # the 11 computable rungs: finite predicted and observed crossover
    ok = th[np.isfinite(th.S_star) & np.isfinite(th.S_obs) & (th.S_star < 1e6)].copy()
    ok = ok[np.isfinite(ok.S_star_lo) & np.isfinite(ok.S_star_hi) &
            (ok.S_star_lo > 0) & (ok.S_star_hi > ok.S_star_lo)]
    ok["mu"] = np.log(ok.S_star)
    ok["sigma"] = (np.log(ok.S_star_hi) - np.log(ok.S_star_lo)) / (2 * SIG95)
    ok["crps_budget"] = [crps_lognormal(y, m, s) for y, m, s in zip(ok.S_obs, ok.mu, ok.sigma)]
    # baseline A: fixed factor-two band around the prediction (lognormal with sigma=ln2/1.96)
    sig_band = np.log(2.0) / SIG95
    ok["crps_factor2"] = [crps_lognormal(y, m, sig_band) for y, m in zip(ok.S_obs, ok.mu)]
    # baseline B: constant predictor at the geometric mean of observed crossovers
    mu0 = float(np.log(ok.S_obs).mean())
    sig0 = float(np.log(ok.S_obs).std())
    ok["crps_constant"] = [crps_lognormal(y, mu0, sig0) for y in ok.S_obs]

    x = np.log(ok.S_star.values)
    y = np.log(ok.S_obs.values)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # bootstrap over rungs
    rng = np.random.default_rng(0)
    slopes = []
    for _ in range(2000):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) < 1e-9:
            continue
        slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
    ci = np.percentile(slopes, [2.5, 97.5])
    summary = pd.DataFrame([dict(
        n=len(ok), slope=slope, intercept=intercept, r2=1 - ss_res / ss_tot,
        slope_lo=float(ci[0]), slope_hi=float(ci[1]),
        crps_budget=float(np.nanmean(ok.crps_budget)),
        crps_factor2=float(np.nanmean(ok.crps_factor2)),
        crps_constant=float(np.nanmean(ok.crps_constant)),
        mape_pct=float(np.mean(np.abs(ok.S_star / ok.S_obs - 1)) * 100),
    )])
    return summary, ok


def main() -> None:
    tc = tolerance_curve()
    tc.to_csv(OUT / "tolerance_curve.csv", index=False)
    print("== tolerance curve ==")
    print(tc.to_string(index=False))

    summary, detail = calibration()
    detail.to_csv(OUT / "calibration_rungs.csv", index=False)
    summary.to_csv(OUT / "calibration_summary.csv", index=False)
    print("\n== S* calibration (11 computable rungs) ==")
    print(summary.T.to_string(header=False))


if __name__ == "__main__":
    main()
