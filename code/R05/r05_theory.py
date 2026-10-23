"""R05-E5: first-order prediction of where codified knowledge stops paying.

Compared arms (k = 0, new programme):
  KA-FIXED  code exponents imposed      -> MSE ~ s2 + B + c/S
  DATA-RE   exponents learned from S programmes
                                         -> MSE ~ s2 + (p + 1) c / S
with c = tau^2 + sigma^2 * mean(1/n_j)   (variance of a programme-mean contrast),
p the number of coded exponents (free exponents are learned by both arms and cancel),
and B the code's systematic misfit at a new programme,
  B = d_B' S_B d_B + d_W' S_W d_W - tr(S_B Cov d_B) - tr(S_W Cov d_W),
where d_W, d_B are the within- and between-programme exponent deviations from the code
(joint Mundlak fit of the code residual, free features as nuisance), S_B the covariance of
programme-mean features and S_W the pooled within-programme covariance; the trace terms
remove the estimation noise from the quadratic forms. Setting the two MSEs equal gives
  S* = p c / B   (programmes).
Every input is estimated on the database itself, without any held-out programme.
The observed crossover is where the KA-FIXED and DATA-RE curves of the scarcity
experiment cross (log-linear interpolation in S); its interval is a target-programme
bootstrap of the curves.

Outputs: outputs/theory_crossover.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import kag_data_r05 as D  # noqa: E402
from kag_models import within_between  # noqa: E402
from kag_models_r05 import fast_lmm_gls as lmm_gls  # noqa: E402
from kag_models_r05 import oneway_reml  # noqa: E402

OUT = HERE.parent / "outputs"
S_NUM = {"3": 3, "6": 6, "12": 12, "24": 24}


def inputs(df, coded, free):
    g = df.source.values
    r = df.y.values - df.m.values
    c0, tau, sig = oneway_reml(r, g)
    n_j = df.groupby("source").size().values
    c = tau ** 2 + sig ** 2 * float(np.mean(1.0 / n_j))
    Wc, Bc, _ = within_between(df[coded].values, g)
    blocks = [np.ones((len(df), 1)), Wc, Bc]
    if free:
        Wf, Bf, _ = within_between(df[free].values, g)
        blocks += [Wf, Bf]
    beta, cov, _, _ = lmm_gls(r, np.column_stack(blocks), g)
    p = len(coded)
    iw, ib = slice(1, 1 + p), slice(1 + p, 1 + 2 * p)
    dW, dB = beta[iw], beta[ib]
    SW = np.atleast_2d(np.cov(Wc, rowvar=False, ddof=1))
    means = pd.DataFrame(df[coded].values).groupby(g).mean().values
    SB = np.atleast_2d(np.cov(means, rowvar=False, ddof=1))
    Bq = float(dB @ SB @ dB + dW @ SW @ dW)
    noise = float(np.trace(SB @ cov[ib, ib]) + np.trace(SW @ cov[iw, iw]))
    B = max(Bq - noise, 1e-9)
    out = dict(p=p, c=c, tau=tau, sigma=sig, B_raw=Bq, B_noise=noise, B=B, S_star=p * c / B,
               programmes=int(len(n_j)), n=len(df))
    for col, tag in (("ln_d", "d"), ("x_cao", "cao")):
        if col in coded:
            j = list(coded).index(col)
            out[f"dW_{tag}"], out[f"dB_{tag}"] = float(dW[j]), float(dB[j])
    return out


def crossing(S_vals, diff):
    """First S where diff = RMSE_data - RMSE_fixed changes sign from + to - (log-linear)."""
    for i in range(len(S_vals) - 1):
        a, b = diff[i], diff[i + 1]
        if a > 0 >= b:
            x0, x1 = np.log(S_vals[i]), np.log(S_vals[i + 1])
            return float(np.exp(x0 + a / (a - b) * (x1 - x0)))
    return np.nan if diff[0] <= 0 else np.inf


def observed(name, n_boot=2000, seed=0):
    pp = pd.read_csv(OUT / "scarcity_per_draw.csv.gz")
    pp = pp[(pp.dataset == name) & (pp.k == 0) & (pp.method.isin(["KA-FIXED", "DATA-RE"])) & (pp.S != "all")]
    per = pp.groupby(["S", "method", "source"]).rmse.mean().unstack("method")
    per["d"] = per["DATA-RE"] - per["KA-FIXED"]
    tab = per.d.unstack("S")[["3", "6", "12", "24"]]
    S_vals = np.array([3, 6, 12, 24], float)
    arr = tab.to_numpy()
    point = crossing(S_vals, arr.mean(axis=0))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        bs.append(crossing(S_vals, arr[idx].mean(axis=0)))
    bs = np.array(bs)
    fin = bs[np.isfinite(bs)]
    lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size > 50 else (np.nan, np.nan))
    return point, lo, hi, float(np.mean(~np.isfinite(bs)))


def boot_pred(df, coded, free, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    progs = df.source.unique()
    out = []
    for _ in range(n_boot):
        draw = rng.choice(progs, size=len(progs), replace=True)
        sub = pd.concat([df[df.source == s].assign(source=f"{s}__{i}") for i, s in enumerate(draw)],
                        ignore_index=True)
        try:
            out.append(inputs(sub, coded, free)["S_star"])
        except Exception:
            continue
    return np.array(out)


def main():
    rows = []
    for name in D.DOMAINS:
        s = D.spec(name)
        df = D.load(name)
        th = inputs(df, s["coded"], s["free"])
        bs = boot_pred(df, s["coded"], s["free"])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        obs, olo, ohi, frac_none = observed(name)
        rows.append(dict(domain=name, **th, S_star_lo=lo, S_star_hi=hi, S_obs=obs, S_obs_lo=olo, S_obs_hi=ohi,
                         obs_boot_no_crossing=frac_none, ratio=th["S_star"] / obs if obs else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "theory_crossover.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
