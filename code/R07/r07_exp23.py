"""R07 hardening experiments 2 and 3.

2. PGLO: for the signature-selected rung of each domain and each of the three
   arms, compare the random 5-fold RMSE with the leave-one-programme-out RMSE.
   Random splits let the model condition on the remaining tests of the same
   programme (the leakage channel); LOPO does not.
3. Pointwise stacking vs programme-level stacking (PAKA): weights fitted on
   per-test marginal log scores instead of whole-programme joint scores, on the
   same LOPO fits.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R07 = HERE.parent
OUT = R07 / "outputs"
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R06_mechanism_pivot_2026-10-22" / "scripts"),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts")]
import r06_rungs as R  # noqa: E402
import r06_ladder as L  # noqa: E402
import kag_bench_r05 as B  # noqa: E402

ARMS = ("KA-FIXED", "KA-RE", "DATA-RE")
PICK = {"FRP": "K2", "SFRC": "K2", "CARB": "K3", "DEEP": "ARCHS"}
SEED = 0


def load(dom):
    df, coded, free = R.load(dom, PICK[dom])
    coded, free = L._canon_feats(dom, coded, free)
    return df.reset_index(drop=True), coded, free


def fit_arms(tr, coded, free):
    des = B.Design(coded, free).fit(tr.reset_index(drop=True))
    fitted = {a: B.fit_lmm_arm(a, tr.reset_index(drop=True), des, SEED) for a in ARMS}
    return des, fitted


def rmse(y, mu):
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def paka_weights(tr, coded, free):
    L_st = B.paka_stack_logdens(tr, coded, free, SEED)
    if L_st is None:
        return np.array([1.0, 0.0, 0.0])
    return B.stacking_weights(L_st, prior=[3.0, 0.0, 0.0])


def logdens(y, mu, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)


def point_weights(L, a0=3.0, iters=300):
    """MAP mixture weights for pointwise marginal densities, Dirichlet prior on arm 0."""
    A, n = L.shape
    w = np.ones(A) / A
    for _ in range(iters):
        resp = w[:, None] * np.exp(L)
        s = resp.sum(axis=0, keepdims=True)
        resp = resp / np.maximum(s, 1e-300)
        w_new = resp.sum(axis=1)
        w_new[0] += a0
        w = w_new / w_new.sum()
    return w


def experiment_pglo(folds=5, domains=("FRP", "SFRC", "CARB", "DEEP")):
    rows = []
    for dom in domains:
        df, coded, free = load(dom)
        n = len(df)
        rng = np.random.default_rng([SEED, sum(map(ord, dom))])
        idx = rng.permutation(n)
        parts = np.array_split(idx, folds)
        per_arm = {a: [] for a in ARMS}
        t0 = time.time()
        for f in range(folds):
            hold = np.sort(parts[f])
            tr = df.drop(index=hold).reset_index(drop=True)
            des, fitted = fit_arms(tr, coded, free)
            mask_hold = np.zeros(len(df), dtype=bool)
            mask_hold[hold] = True
            preds = {a: [] for a in ARMS}
            ys = []
            for src in df.loc[mask_hold, "source"].unique():
                te_j = df[df.source == src].reset_index(drop=True)
                pos = {orig: k for k, orig in enumerate(df.index[df.source == src])}
                kk, tt = [], []
                for k, orig in enumerate(df.index[df.source == src]):
                    (kk if not mask_hold[orig] else tt).append(k)
                if not tt or not kk:
                    continue
                for a in ARMS:
                    mu, _ = B.lmm_predict(a, fitted[a], des, te_j, np.array(kk), np.array(tt))
                    preds[a].append(mu)
                ys.append(te_j.y.values[np.array(tt)])
            y = np.concatenate(ys)
            for a in ARMS:
                p = np.concatenate(preds[a])
                per_arm[a].append(rmse(y, p))
        for a in ARMS:
            rows.append(dict(domain=dom, arm=a, rmse_random=float(np.mean(per_arm[a])),
                             folds=folds, seconds=round(time.time() - t0, 1)))
        print(f"{dom}: random-fold RMSE " + ", ".join(f"{a}={np.mean(per_arm[a]):.4f}" for a in ARMS),
              flush=True)
    rnd = pd.DataFrame(rows)

    summary = pd.read_csv(HERE.parents[1] / "R06_mechanism_pivot_2026-10-22" / "outputs" / "ladder_summary.csv")
    out = []
    for r in rnd.itertuples():
        ref = summary[(summary.dataset == r.domain) & (summary.rung == PICK[r.domain])
                      & (summary.S.astype(str) == "all") & (summary.k == 0) & (summary.method == r.arm)]
        if not len(ref):
            continue
        lopo = float(ref.rmse.iloc[0])
        out.append(dict(domain=r.domain, arm=r.arm, rmse_random=r.rmse_random, rmse_lopo=lopo,
                        pglo=float(np.log(r.rmse_random) - np.log(lopo)),
                        optimism_pct=float((r.rmse_random / lopo - 1) * 100)))
    return pd.DataFrame(out)


def experiment_pointwise(domains=("FRP", "SFRC", "CARB", "DEEP"), n_prog=10):
    rows = []
    for dom in domains:
        df, coded, free = load(dom)
        srcs = list(pd.unique(df.source))
        rng = np.random.default_rng([SEED, 7, sum(map(ord, dom))])
        pick = list(rng.choice(srcs, size=min(n_prog, len(srcs)), replace=False))
        errs = {"PAKA": [], "POINT": [], **{a: [] for a in ARMS}}
        for src in pick:
            tr = df[df.source != src].reset_index(drop=True)
            te = df[df.source == src].reset_index(drop=True)
            des, fitted = fit_arms(tr, coded, free)
            mus, vs = {}, {}
            for a in ARMS:
                mu, var = B.lmm_predict(a, fitted[a], des, te, np.array([], dtype=int),
                                        np.arange(len(te)))
                mus[a], vs[a] = mu, var
            w_p = paka_weights(tr, coded, free)
            mu_p = B.paka_stack_predict(w_p, fitted, des, te, np.array([], dtype=int),
                                        np.arange(len(te)))[0]
            L = np.vstack([logdens(te.y.values, mus[a], vs[a]) for a in ARMS])
            w_pt = point_weights(L)
            mu_pt = sum(w_pt[k] * mus[a] for k, a in enumerate(ARMS))
            y = te.y.values
            errs["PAKA"].append(rmse(y, mu_p))
            errs["POINT"].append(rmse(y, mu_pt))
            for a in ARMS:
                errs[a].append(rmse(y, mus[a]))
        rows.append(dict(domain=dom, n_programmes=len(pick),
                         **{k: float(np.mean(v)) for k, v in errs.items()}))
        print(f"{dom}: " + "  ".join(f"{k}={np.mean(v):.4f}" for k, v in errs.items()), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("pglo", "both"):
        pg = experiment_pglo()
        pg.to_csv(OUT / "pglo.csv", index=False)
        print("\n== PGLO (random - LOPO, log RMSE) ==")
        print(pg.to_string(index=False))
    if which in ("point", "both"):
        pt = experiment_pointwise()
        pt.to_csv(OUT / "stacking_pointwise.csv", index=False)
        print("\n== pointwise vs programme-level stacking (LOPO, 10 programmes/domain) ==")
        print(pt.to_string(index=False))
