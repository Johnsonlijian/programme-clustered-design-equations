"""R08 post hoc diagnostics of the calibration slope (ln S_obs on ln S*), 2026-10-25/26.

Inputs: the archived R07 calibration table (pre-specified rungs) and the R08 prospective
repair tables. Nothing is re-estimated except the regressions below.
  A. OLS slope on the 11 rungs (manuscript) and on the 8 primary rungs (the three DEEP-WB rungs
     reuse the DEEP rows; they are a sensitivity variant, not independent points). Intervals:
     2000-resample bootstrap over rungs, as in R07 (r07_exp14.py).
  B. Regression-dilution correction: ln S* is an estimate; its sampling variance is taken from
     the archived 95% programme-bootstrap interval, se = (ln hi - ln lo) / (2 * 1.96).
     Reliability lambda = 1 - mean(se^2) / var(ln S*); corrected slope = OLS slope / lambda
     (classical errors-in-variables; Fuller 1987). lambda is recomputed in every resample.
  C. Prospective repaired rungs: predicted/observed ratio where the observed crossover is finite.
Outputs: ../outputs/calibration_diag.csv, ../outputs/prospective_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parents[1]
CAL = ROUNDS / "R07_merged_protocol_2026-10-22" / "outputs" / "calibration_rungs.csv"
OUT = HERE.parent / "outputs"
Z = 1.959964


def fit(x, y, se):
    b = np.polyfit(x, y, 1)[0]
    vx = np.var(x, ddof=1)
    lam = 1 - np.mean(se ** 2) / vx if vx > 0 else np.nan
    yhat = np.polyval(np.polyfit(x, y, 1), x)
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return b, lam, r2


def block(df, label, n_boot=2000):
    x = np.log(df.S_star.values)
    y = np.log(df.S_obs.values)
    se = (np.log(df.S_star_hi.values) - np.log(df.S_star_lo.values)) / (2 * Z)
    b, lam, r2 = fit(x, y, se)
    rng = np.random.default_rng(0)
    bs, be = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) < 1e-9:
            continue
        bb, ll, _ = fit(x[i], y[i], se[i])
        bs.append(bb)
        if ll > 0.1:
            be.append(bb / ll)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    elo, ehi = np.percentile(be, [2.5, 97.5])
    return dict(set=label, n=len(df), slope=b, slope_lo=lo, slope_hi=hi, r2=r2, reliability=lam,
                slope_eiv=b / lam, slope_eiv_lo=elo, slope_eiv_hi=ehi, eiv_resamples=len(be),
                ratio_min=float((df.S_star / df.S_obs).min()), ratio_max=float((df.S_star / df.S_obs).max()))


def prospective():
    rows = []
    sf = pd.read_csv(OUT / "certified_crossover.csv")
    for r in sf.itertuples():
        if r.rung in ("K1C", "K2C"):
            parent = "K1" if r.rung == "K1C" else "K2"
            p = sf[sf.rung == parent].iloc[0]
            rows.append(dict(domain="SFRC", rung=r.rung, parent=parent, S_pred=r.S_prespec,
                             S_pred_lo=r.S_prespec_lo, S_pred_hi=r.S_prespec_hi, S_split=r.S_split,
                             S_obs=r.S_obs, S_obs_lo=r.S_obs_lo, S_obs_hi=r.S_obs_hi,
                             no_cross=r.obs_boot_no_crossing, parent_obs=p.S_obs))
    rp = pd.read_csv(OUT / "repair_crossover.csv")
    for r in rp[rp.role == "repair"].itertuples():
        p = rp[(rp.domain == r.domain) & (rp.role == "parent")
               & (rp.rung == {"K1C": "K1", "K2C": "K2", "BEAMC": "BEAM"}[r.rung])].iloc[0]
        rows.append(dict(domain=r.domain, rung=r.rung, parent=p.rung, S_pred=r.S_prespec,
                         S_pred_lo=r.S_prespec_lo, S_pred_hi=r.S_prespec_hi, S_split=r.S_split,
                         S_obs=r.S_obs, S_obs_lo=r.S_obs_lo, S_obs_hi=r.S_obs_hi,
                         no_cross=r.obs_boot_no_crossing, parent_obs=p.S_obs))
    t = pd.DataFrame(rows)
    th = pd.read_csv(ROUNDS / "R06_mechanism_pivot_2026-10-22" / "outputs" / "ladder_theory.csv")
    t["parent_pred"] = [th[(th.domain == d) & (th.rung == p)].S_star.iloc[0] for d, p in zip(t.domain, t.parent)]
    t["ratio"] = np.where(np.isfinite(t.S_obs), t.S_pred / t.S_obs, np.nan)
    # observed "no crossing within S<=24" is compatible when the prediction is >= 12 (= 24/2)
    t["magnitude_ok"] = np.where(np.isfinite(t.S_obs), (t.ratio >= 0.5) & (t.ratio <= 2.0), t.S_pred >= 12)
    t["interval_overlap"] = np.where(np.isfinite(t.S_obs),
                                     (t.S_pred_lo <= t.S_obs_hi) & (t.S_obs_lo <= t.S_pred_hi), t.S_pred_hi >= 24)
    pred_dir = np.sign(t.S_pred - t.parent_pred)
    obs_rep = np.where(np.isfinite(t.S_obs), t.S_obs, 1e9)
    obs_par = np.where(np.isfinite(t.parent_obs), t.parent_obs, 0.0)   # parent 'none' = below S=3
    t["direction_ok"] = pred_dir == np.sign(obs_rep - obs_par)
    t.to_csv(OUT / "prospective_check.csv", index=False)
    return t


def main():
    cal = pd.read_csv(CAL)
    t = pd.DataFrame([block(cal, "all 11 (manuscript)"), block(cal[cal.domain != "DEEP-WB"], "8 primary")])
    t.to_csv(OUT / "calibration_diag.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).T.to_string())
    print(prospective().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
