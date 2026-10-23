"""R04: run the R02 pipeline on the concrete-carbonation (CARB) domain.

Produces, with the same definitions as R02 so the numbers are comparable:
  outputs/domain_audit_r04.csv   n, programmes, ICC, eta^2, within/between exponents
  outputs/scarcity_r04_*.csv     knowledge-value curve and policy arms
  outputs/theory_r04.csv         S* predicted from variance components vs measured

The CARB domain has two features: the codified sqrt(CO2) term x_co2 and the free
w/b term x_wb. The anchor fixes the CO2 exponent at 0.5; the model estimates the
deviation on both features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R02 = HERE.parents[1] / "R02_crossdomain_theory_2026-10-20" / "scripts"
R01 = HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts"
sys.path[:0] = [str(HERE), str(R02), str(R01)]

import kag_data_r04 as D                                        # noqa: E402
from kag_models import KnowledgeAnchoredLMM, lmm_gls, within_between  # noqa: E402
from kag_models_r02 import ka_fit                               # noqa: E402
from r02_scarcity2 import task                                  # noqa: E402
import r02_theory as T                                          # noqa: E402

OUT = HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)
FEATS = D.FEATURES


def between_share(x, g):
    s = pd.DataFrame({"x": x, "g": g})
    gm = s.groupby("g").x.transform("mean")
    tot = float(np.sum((s.x - s.x.mean()) ** 2))
    return float(np.sum((gm - s.x.mean()) ** 2) / tot) if tot > 0 else np.nan


def mundlak(df, feat):
    W, B, _ = within_between(df[[feat]].values, df.source.values)
    X = np.column_stack([np.ones(len(df)), W[:, 0], B[:, 0]])
    beta, cov, tau, sig = lmm_gls(df.y.values - df.m.values, X, df.source.values)
    gap = beta[2] - beta[1]
    se = np.sqrt(cov[2, 2] + cov[1, 1] - 2 * cov[1, 2])
    return dict(within=float(beta[1]), within_se=float(np.sqrt(cov[1, 1])),
                between=float(beta[2]), between_se=float(np.sqrt(cov[2, 2])),
                gap=float(gap), z=float(gap / se))


def audit(df):
    g = df.source.values
    n_s = pd.Series(g).value_counts()
    r = df.y.values - df.m.values
    m0 = ka_fit(r, np.zeros((len(df), 0)), g, lam=0.0, ml=False, n_starts=4)
    rec = dict(domain="CARB", n=len(df), programmes=int(n_s.size),
               programmes_ge6=int((n_s >= 6).sum()), specimens_ge6=int(n_s[n_s >= 6].sum()),
               tau=m0["tau"], sigma=m0["sig"],
               icc=m0["tau"] ** 2 / (m0["tau"] ** 2 + m0["sig"] ** 2))
    for f in FEATS:
        mk = mundlak(df, f)
        rec[f"eta2_{f}"] = between_share(df[f].values, g)
        rec[f"{f}_within"] = mk["within"]
        rec[f"{f}_within_se"] = mk["within_se"]
        rec[f"{f}_between"] = mk["between"]
        rec[f"{f}_between_se"] = mk["between_se"]
        rec[f"{f}_z"] = mk["z"]
    return rec


def main():
    df = D.load_carb()
    a = pd.DataFrame([audit(df)])
    a.to_csv(OUT / "domain_audit_r04.csv", index=False)
    print(a.round(3).to_string(index=False))

    # ---- knowledge-value curve ------------------------------------------------
    # The LMM is O(n^3); with 1580 observations a full sweep is impractical, so
    # each programme contributes at most CAP rows (deterministic, seed 0). The
    # audit above uses the full table; the benchmark uses the capped one.
    CAP = 10
    rng_cap = np.random.default_rng(0)
    parts = []
    for src, blk in df.groupby("source"):
        if len(blk) > CAP:
            idx = rng_cap.choice(len(blk), CAP, replace=False)
            parts.append(blk.iloc[np.sort(idx)])
        else:
            parts.append(blk)
    dfb = pd.concat(parts, ignore_index=True)
    print(f"benchmark table: {len(dfb)} rows after capping at {CAP}/programme")

    n_s = dfb.groupby("source").size()
    targets = sorted(n_s[n_s >= 6].index)
    S_LEVELS = (3, 6, 12, 24, None)
    B = 4
    frames = []
    for S in S_LEVELS:
        reps = range(B) if S is not None else [0]
        for tgt in targets:
            for b in reps:
                frames.append(task(dfb, FEATS, tgt, S, b, 4))
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.insert(0, "dataset", "CARB")
    per_draw.to_csv(OUT / "scarcity_r04_per_draw.csv.gz", index=False)

    per_prog = (per_draw.groupby(["S_label", "method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean"), cover90=("cover90", "mean")))
    rng = np.random.default_rng(0)
    rows = []
    for (S, k), blk in per_prog.groupby(["S_label", "k"]):
        progs = np.sort(blk.source.unique())
        piv = blk.pivot(index="source", columns="method", values="rmse").reindex(progs)
        boots = rng.integers(0, len(progs), size=(2000, len(progs)))
        for m in piv.columns:
            v = piv[m].values
            rec = dict(S=S, k=k, method=m, programmes=len(progs), rmse=float(np.nanmean(v)))
            rec["lo"], rec["hi"] = np.percentile(np.nanmean(v[boots], 1), [2.5, 97.5])
            if m != "DATA-RE" and "DATA-RE" in piv:
                d = piv["DATA-RE"].values - v
                rec["gain_vs_DATA"] = float(np.nanmean(d))
                rec["gain_lo"], rec["gain_hi"] = np.percentile(np.nanmean(d[boots], 1), [2.5, 97.5])
            rows.append(rec)
    summ = pd.DataFrame(rows)
    order = {"3": 0, "6": 1, "12": 2, "24": 3, "all": 4}
    summ["S_order"] = summ.S.map(order)
    summ = summ.sort_values(["k", "S_order", "method"])
    summ.to_csv(OUT / "scarcity_r04_summary.csv", index=False)

    # ---- theory --------------------------------------------------------------
    th = T.theory_inputs(df, FEATS)
    bs = T.bootstrap(df, FEATS, n_boot=30)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs.size else (np.nan, np.nan))
    gain = []
    for so in sorted(summ.S_order.unique()):
        blk = summ[(summ.S_order == so) & (summ.k == 0)]
        kd = blk[blk.method == "KA-RE"].rmse
        dd = blk[blk.method == "DATA-RE"].rmse
        if len(kd) and len(dd):
            gain.append((so, float(dd.iloc[0] ** 2 - kd.iloc[0] ** 2)))
    S_map = {0: 3, 1: 6, 2: 12, 3: 24}
    Sg = np.array([S_map[s] for s, _ in gain], float)
    G = np.array([v for _, v in gain])
    meas = T.fit_gain(Sg, G)
    rec = dict(domain="CARB", p=th["p"], n_bar=th["n_bar"], tau=th["tau"], sigma=th["sig"],
               A=th["A"], B=th["B"], S_star_pred=th["S_star"], S_star_lo=lo, S_star_hi=hi,
               S_star_meas=meas["S_star_meas"], ratio=th["S_star"] / meas["S_star_meas"],
               gain_r2=meas["r2"], A_meas=meas["A_meas"], B_meas=meas["B_meas"])
    pd.DataFrame([rec]).to_csv(OUT / "theory_r04.csv", index=False)
    print(pd.DataFrame([rec]).round(3).to_string(index=False))
    print(summ.pivot_table(index=["k", "method"], columns="S_order", values="rmse").round(3).to_string())


if __name__ == "__main__":
    main()
