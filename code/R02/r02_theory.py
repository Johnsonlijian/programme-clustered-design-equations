"""R02 diagnostic 1: closed-form crossover S* for the value of codified knowledge.

Model behind the pilot
----------------------
Specimens i in test programme j:   y_ij = m_ij + beta' phi_ij + u_j + e_ij,
u_j ~ N(0, tau^2), e_ij ~ N(0, sigma^2), phi = centred log-physical variables.
The design code asserts beta = 0; the truth is beta = delta (the code exponent
misfit, identified without confounding by the within-programme contrast).

For a new programme with k = 0 shots the unpredictable u_j and e_ij are common to
both estimators and cancel in the comparison, so the difference in squared error is
driven by how well each estimator recovers the systematic part at a programme mean:

  DATA-RE : beta estimated from the S programme means; with c = tau^2 + sigma^2/n_bar
            Var(beta_hat) = c (Sigma_B)^-1 / S  and
            E[(phi*'beta_hat - phi*'delta)^2] = tr(Sigma_B Var(beta_hat)) = p c / S.
  KA      : beta shrunk to the code; with lambda -> 0 the error is phi*'delta and
            E[(phi*'delta)^2] = delta' Sigma_B delta = B, independent of S.

Hence  G(S) = MSE_data(S) - MSE_ka(S) = A / S - B  with
  A = p (tau^2 + sigma^2 / n_bar),     B = delta' Sigma_B delta,
and the crossover, i.e. how many programmes the code is worth,

  S* = A / B = p (tau^2 + sigma^2 / n_bar) / (delta' Sigma_B delta).

Every input on the right is estimable from a training library alone: p feature
count, tau/sigma REML variance components, n_bar specimens per programme, delta the
within-programme exponent misfit, Sigma_B the between-programme covariance of the
centred features. Two populations are reported, because programme means of
singleton programmes are not design points: `all` = every programme in the
database, `ge6` = programmes with at least 6 specimens.

The measured counterpart is the R02 scarcity experiment
(outputs/scarcity2_summary.csv): G_meas(S) = RMSE_data^2 - RMSE_ka^2 at k = 0,
fitted with the same two-parameter law. The fit uses all library sizes; the variant
that drops S < p is reported as a sensitivity check.

Outputs: outputs/theory_inputs.csv, outputs/theory_prediction.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R01 = Path(__file__).resolve().parents[2] / "R01_direction_and_pilot_2026-10-20" / "scripts"
sys.path.insert(0, str(R01))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kag_data_r02 as K                                       # noqa: E402
from kag_models import KnowledgeAnchoredLMM, within_between    # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

N_BOOT = 30
SEED = 0


def eligible(df, min_n=6):
    """The population a k-shot benchmark draws targets from."""
    n_s = df.groupby("source").size()
    return df[df.source.isin(n_s[n_s >= min_n].index)].reset_index(drop=True)


def programme_covariance(Phi, groups):
    """Unweighted covariance of programme mean features (each programme one draw)."""
    means = pd.DataFrame(Phi).groupby(np.asarray(groups)).mean().values
    return np.cov(means, rowvar=False, ddof=1)


def theory_inputs(df, feats):
    """Estimate p, n_bar, tau, sigma, delta and Sigma_B on a (sub)library."""
    p = len(feats)
    Phi = df[feats].values
    r = df.y.values - df.m.values
    g = df.source.values

    m0 = KnowledgeAnchoredLMM(np.zeros(0, int), n_starts=4, seed=SEED).fit(
        r, np.zeros((len(df), 0)), g)
    tau, sig = m0.tau_, m0.sig_

    W, Bm, _ = within_between(Phi, g)
    wb = KnowledgeAnchoredLMM([0] * p + [1] * p, n_starts=4, seed=SEED).fit(
        r, np.column_stack([W, Bm]), g)
    delta = wb.b_[:p]
    delta_sd = np.sqrt(np.diag(wb.b_cov_))[:p]

    SB = programme_covariance(Phi, g)
    nbar = float(pd.Series(g).value_counts().values.mean())

    A = p * (tau ** 2 + sig ** 2 / nbar)
    Bv = float(delta @ SB @ delta)
    return dict(p=p, n_bar=nbar, tau=float(tau), sig=float(sig), A=float(A), B=Bv,
                S_star=float(A / Bv) if Bv > 0 else np.inf,
                delta=delta, delta_sd=delta_sd, Sigma_B=SB,
                n=int(len(df)), programmes=int(len(np.unique(g))))


def measured_curve(domain):
    """Measured knowledge gain on the MSE scale against S, k = 0."""
    s = pd.read_csv(OUT / "scarcity2_summary.csv")
    s = s[(s.dataset == domain) & (s.k == 0) & (s.method.isin(["KA-RE", "DATA-RE"]))]
    piv = s.pivot_table(index="S", columns="method", values="rmse").dropna()
    gain = piv["DATA-RE"].values ** 2 - piv["KA-RE"].values ** 2
    S = np.array([{"3": 3, "6": 6, "12": 12, "18": 18, "24": 24}.get(str(v), np.nan)
                  for v in piv.index])
    ok = np.isfinite(S)
    return S[ok], gain[ok]


def fit_gain(S, gain):
    """Least-squares fit of G(S) = A/S - B; return A, B, S* = A/B and R^2."""
    if len(S) < 3:
        return dict(A_meas=np.nan, B_meas=np.nan, S_star_meas=np.nan, r2=np.nan,
                    n_points=int(len(S)))
    X = np.column_stack([1.0 / S, -np.ones_like(S)])
    A, B = np.linalg.lstsq(X, gain, rcond=None)[0]
    pred = A / S - B
    ss_res = float(np.sum((gain - pred) ** 2))
    ss_tot = float(np.sum((gain - gain.mean()) ** 2))
    return dict(A_meas=float(A), B_meas=float(B),
                S_star_meas=float(A / B) if B > 0 else np.inf,
                r2=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                n_points=int(len(S)))


def bootstrap(df, feats, n_boot=N_BOOT, seed=SEED):
    """Programme-level bootstrap CI of the predicted S*."""
    rng = np.random.default_rng(seed)
    progs = df.source.unique()
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(progs, size=len(progs), replace=True)
        sub = pd.concat([df[df.source == g].assign(source=f"{g}__{i}") for i, g in enumerate(draw)],
                        ignore_index=True)
        try:
            vals.append(theory_inputs(sub, feats)["S_star"])
        except Exception:
            continue
    return np.array([v for v in vals if np.isfinite(v)])


def main():
    rows, feat_rows, summary = [], [], {}
    for name in K.LOADERS:
        df_all = K.load(name)
        feats = K.FEATURES[name]
        p = len(feats)
        S, gain = measured_curve(name)
        meas = fit_gain(S, gain)
        meas_strict = fit_gain(S[S >= p], gain[S >= p])

        for pop, df in (("all", df_all), ("ge6", eligible(df_all))):
            th = theory_inputs(df, feats)
            bs = bootstrap(df, feats)
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs.size else (np.nan, np.nan))
            rows.append(dict(
                dataset=name, population=pop, n=th["n"], programmes=th["programmes"], p=p,
                n_bar=round(th["n_bar"], 3), tau=th["tau"], sigma=th["sig"],
                A=th["A"], B=th["B"], S_star_pred=th["S_star"],
                S_star_pred_lo=lo, S_star_pred_hi=hi,
                S_star_meas=meas["S_star_meas"], S_star_meas_strict=meas_strict["S_star_meas"],
                ratio=th["S_star"] / meas["S_star_meas"] if meas["S_star_meas"] else np.nan,
                gain_r2=meas["r2"], gain_r2_strict=meas_strict["r2"],
                A_meas=meas["A_meas"], B_meas=meas["B_meas"],
                delta_norm=float(np.linalg.norm(th["delta"]))))
            if pop == "ge6":
                for j, f in enumerate(feats):
                    vb = th["Sigma_B"][j, j]
                    feat_rows.append(dict(dataset=name, feature=f, label=K.FEATURE_LABELS[f],
                                          delta=th["delta"][j], delta_sd=th["delta_sd"][j],
                                          var_between=vb,
                                          contribution=float(th["delta"][j] ** 2 * vb),
                                          share_of_B=float(th["delta"][j] ** 2 * vb / th["B"])))
                summary[name] = dict(p=p, n_bar=th["n_bar"], tau=th["tau"], sigma=th["sig"],
                                     A=th["A"], B=th["B"], S_star_pred=th["S_star"],
                                     S_star_lo=float(lo), S_star_hi=float(hi),
                                     S_star_meas=meas["S_star_meas"],
                                     gain_r2=meas["r2"], fit_points=meas["n_points"])
            print(f"{name:5s} pop={pop:3s}: S*_pred={th['S_star']:7.2f} "
                  f"[{lo:7.2f}, {hi:7.2f}]  S*_meas={meas['S_star_meas']:7.2f}  "
                  f"ratio={th['S_star'] / meas['S_star_meas']:5.2f}", flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "theory_prediction.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(OUT / "theory_inputs.csv", index=False)
    (OUT / "theory_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 220)
    print()
    print(tab[["dataset", "population", "p", "n_bar", "tau", "sigma", "A", "B",
               "S_star_pred", "S_star_pred_lo", "S_star_pred_hi", "S_star_meas",
               "ratio", "gain_r2"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
