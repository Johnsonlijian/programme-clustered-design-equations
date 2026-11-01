"""R10 hard gate: fully nested sensitivity for the PAKA prior strength a0.

The R09/R10 text claims a0 is immaterial over {0,3,6} but a0=3 was chosen on a
subset of the same evaluation targets (disclosed limitation). This script closes
that loop:

  1. Re-runs the stacking evaluation on the R06 scarcity grid with a0 in
     {0, 3, 6} as fixed values. The a0=3 arm must reproduce the frozen
     ladder_per_draw PAKA-S rows exactly (identity check), because subsets and
     draws reuse the R06 seed construction.
  2. Adds two *nested* selection rules that use only the training library:
       NEST-SCORE: a0* = argmax_a sum_j log sum_arm w_arm(a) exp(L[j,arm])
                   over the training library's own leave-programme-out joint
                   log densities L (the stacked predictive score).
       NEST-RMSE : a0* = argmin_a mean_j rmse_j(a) over out-of-fold point
                   predictions of the same inner folds (programme-level RMSE,
                   equal weight per held-out training programme).
     If the nested choice tracks a0=3, the "chosen on the same targets"
     critique is moot; if all fixed a0 give nearly identical predictions, the
     choice itself is inconsequential.

Grid: all 15 rungs, all targets (programmes with >= 6 tests), S in
{3,6,12,24,all}, b in 0..NB-1 subsets for S != all (NB=10; the frozen file used
20), b=0 for S=all; ks=(0,3), n_draws=10, same RNG construction as
r06_ladder.sc_task so a0=3 rows must match PAKA-S row-for-row on shared cells.

  python -X utf8 paka_a0_nested_r10.py run      # compute per-draw + info CSVs
  python -X utf8 paka_a0_nested_r10.py check    # identity vs ladder_per_draw + summary
  python -X utf8 paka_a0_nested_r10.py smoke    # 1 rung / 1 target / S=3 / b=0..1

Outputs (R10 outputs dir): paka_a0_per_draw.csv.gz, paka_a0_info.csv,
log_paka_a0.txt.
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent
sys.path[:0] = [
    str(HERE),
    str(ROUNDS / "R06_mechanism_pivot_2026-10-22" / "scripts"),
    str(ROUNDS / "R05_consolidation_2026-10-21" / "scripts"),
    str(ROUNDS / "R01_direction_and_pilot_2026-10-20" / "scripts"),
]
import r06_ladder as LAD  # noqa: E402
import r06_rungs as R  # noqa: E402
from kag_models_r05 import joint_predictive, mvn_logpdf, stacking_weights  # noqa: E402
from kag_bench_r05 import (Design, LMM_ARMS, fit_lmm_arm, lmm_predict,  # noqa: E402
                           paka_stack_predict, score)

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
A0_GRID = (0.0, 3.0, 6.0)
ARMS_OUT = ("PAKA-S0", "PAKA-S3", "PAKA-S6", "PAKA-NESTS", "PAKA-NESTR")
NB = 10                    # subsets per S level (frozen grid used 20)
S_LEVELS = (3, 6, 12, 24, None)
N_DRAWS = 10
KS = (0, 3)
NJ = 24


def stack_logdens_pointwise(tr, coded, free, seed=0, max_folds=6):
    """Extension of kag_bench_r05.paka_stack_logdens that also returns the
    out-of-fold per-point predictive mean/variance of each held-out training
    programme (same folds, same inner fits). Needed for the NEST-RMSE rule."""
    from sklearn.model_selection import GroupKFold
    g = tr.source.values
    ug = np.unique(g)
    if len(ug) < 3:
        return None, None
    rows, pts = [], []
    for itr, iva in GroupKFold(n_splits=min(max_folds, len(ug))).split(tr, groups=g):
        a_tr, a_va = tr.iloc[itr], tr.iloc[iva].reset_index(drop=True)
        des = Design(coded, free).fit(a_tr)
        mdls = {arm: fit_lmm_arm(arm, a_tr, des, seed, n_starts=2) for arm in LMM_ARMS}
        for pg in np.unique(a_va.source.values):
            blk = a_va[a_va.source == pg]
            row = []
            pt = dict(prog=pg, y=blk.y.values)
            for arm in LMM_ARMS:
                X, _ = des.arm_matrix(arm, blk)
                off = blk.m.values if arm != "DATA-RE" else None
                mu, C = joint_predictive(mdls[arm], X, off)
                row.append(mvn_logpdf(blk.y.values, mu, C))
                pt[f"mu_{arm}"], pt[f"var_{arm}"] = mu, np.diag(C)
            rows.append(row)
            pts.append(pt)
    return np.array(rows), pts


def nested_choices(L, pts):
    """Choose a0 in A0_GRID on the training library only. Returns
    (a0_score, a0_rmse, {a0: w})."""
    W = {a0: stacking_weights(L, prior=[a0, 0.0, 0.0]) for a0 in A0_GRID}
    # NEST-SCORE: stacked predictive score of the training programmes
    Ls = L - L.max(1, keepdims=True)
    E = np.exp(Ls)
    sc = {a0: float(np.sum(np.log(E @ w + 1e-300))) for a0, w in W.items()}
    a0_score = max(A0_GRID, key=lambda a: sc[a])
    # NEST-RMSE: programme-level RMSE of the mixture on out-of-fold points
    rm = {}
    for a0, w in W.items():
        per_prog = []
        for pt in pts:
            mus = np.array([pt[f"mu_{a}"] for a in LMM_ARMS])
            m = w @ mus
            per_prog.append(np.mean((pt["y"] - m) ** 2))
        rm[a0] = float(np.mean(per_prog))
    a0_rmse = min(A0_GRID, key=lambda a: rm[a])
    return a0_score, a0_rmse, W, sc, rm


def a0_task(name, rung, target, S, b):
    """r06_ladder.sc_task, restricted to the stack arms, evaluated under every
    a0 variant. Returns (per_draw rows, info row)."""
    df, coded, free = R.load(name, rung)
    coded, free = LAD._canon_feats(name, coded, free)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    subset = others
    if S is not None:
        rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    seed = b
    with threadpool_limits(1):
        keep = (df.source != target)
        tr = df[df.source.isin(subset) & keep].reset_index(drop=True)
        te = df[df.source == target].reset_index(drop=True)
        des = Design(coded, free).fit(tr)
        fitted = {arm: fit_lmm_arm(arm, tr, des, seed) for arm in LMM_ARMS}
        L_st, pts = stack_logdens_pointwise(tr, coded, free, seed)
        info = dict(dataset=name, rung=rung, S="all" if S is None else str(S),
                    subset=b, source=target, n_train=len(tr),
                    n_programmes=int(len(np.unique(tr.source.values))))
        if L_st is None:
            W = {a0: np.array([1.0, 0.0, 0.0]) for a0 in A0_GRID}
            a0_s = a0_r = np.nan
            info.update(a0_score=np.nan, a0_rmse=np.nan, L_rows=0)
        else:
            a0_s, a0_r, W, sc, rm = nested_choices(L_st, pts)
            info.update(a0_score=a0_s, a0_rmse=a0_r, L_rows=L_st.shape[0],
                        **{f"score_{a:g}": v for a, v in sc.items()},
                        **{f"rm_{a:g}": v for a, v in rm.items()})
        W["NESTS"] = W[a0_s] if not np.isnan(a0_s) else np.array([1.0, 0.0, 0.0])
        W["NESTR"] = W[a0_r] if not np.isnan(a0_r) else np.array([1.0, 0.0, 0.0])
        for a0 in A0_GRID:
            w = W[a0]
            info.update({f"w{a0:g}_{a}": float(v) for a, v in zip(LMM_ARMS, w)})
        variants = {f"PAKA-S{int(a0)}": W[a0] for a0 in A0_GRID}
        variants["PAKA-NESTS"] = W["NESTS"]
        variants["PAKA-NESTR"] = W["NESTR"]

        rng = np.random.default_rng([seed, zlib.crc32(str(target).encode("utf-8"))])
        n, rows = len(te), []
        for k in KS:
            if n - k < 3:
                continue
            perms = [np.arange(n)] if k == 0 else [rng.permutation(n) for _ in range(N_DRAWS)]
            for di, perm in enumerate(perms):
                kk, tt = perm[:k], perm[k:]
                for name_v, w in variants.items():
                    mu, var = paka_stack_predict(w, fitted, des, te, kk, tt)
                    rows.append(dict(source=target, n_source=n, k=k, draw=di,
                                     n_pred=len(tt), method=name_v,
                                     **score(te.y.values[tt], mu, var)))
        out = pd.DataFrame(rows)
        out["S"] = "all" if S is None else str(S)
        out["dataset"], out["rung"], out["subset"], out["n_train"] = name, rung, b, len(tr)
        return out, info


def tasks_for(name, rung, smoke=False):
    df, _, _ = R.load(name, rung)
    n_s = df.groupby("source").size()
    targets = sorted(n_s[n_s >= 6].index)
    if smoke:
        targets = targets[:1]
    for t in targets:
        for S in ((3,) if smoke else S_LEVELS):
            for b in (range(2) if smoke else (range(NB) if S is not None else [0])):
                yield name, rung, t, S, b


def mode_run(smoke=False):
    t0 = time.time()
    domains = {"FRP": ("K1",)} if smoke else LAD.ALL_DOMAINS
    frames, infos = [], []
    for name in domains:
        for rung in domains[name]:
            jobs = [delayed(a0_task)(*a) for a in tasks_for(name, rung, smoke)]
            res = Parallel(n_jobs=NJ, backend="loky", batch_size=2)(jobs)
            frames += [r[0] for r in res]
            infos += [r[1] for r in res]
            print(f"a0 {name} {rung}: {len(jobs)} tasks at {time.time() - t0:.0f}s",
                  flush=True)
    per = pd.concat(frames, ignore_index=True)
    tag = "_smoke" if smoke else ""
    per.to_csv(OUT / f"paka_a0_per_draw{tag}.csv.gz", index=False)
    pd.DataFrame(infos).to_csv(OUT / f"paka_a0_info{tag}.csv", index=False)
    print(f"total {time.time() - t0:.0f}s, {len(per)} rows")


def mode_check():
    """Identity: PAKA-S3 rows must equal the frozen PAKA-S rows on shared cells;
    then a0 sensitivity summary."""
    new = pd.read_csv(OUT / "paka_a0_per_draw.csv.gz")
    new["S"] = new.S.astype(str)
    old = pd.read_csv(LAD.OUT / "ladder_per_draw.csv.gz",
                      usecols=["dataset", "rung", "S", "subset", "source", "k",
                               "draw", "method", "rmse", "nlpd"])
    old["S"] = old.S.astype(str)
    key = ["dataset", "rung", "S", "subset", "source", "k", "draw"]
    mg = (new[new.method == "PAKA-S3"]
          .merge(old[old.method == "PAKA-S"], on=key, suffixes=("", "_old")))
    md = float(np.abs(mg.rmse - mg.rmse_old).max()) if len(mg) else np.nan
    print(f"identity PAKA-S3 vs PAKA-S: {len(mg)} shared rows, max|diff|={md:.2e}")
    info = pd.read_csv(OUT / "paka_a0_info.csv")
    print("\nnested a0 choices (share of tasks):")
    print(info[["a0_score", "a0_rmse"]].apply(lambda c: c.value_counts(normalize=True)
                                            .reindex([0.0, 3.0, 6.0]).round(3)))

    # worst shortfall (regret) per a0 variant: per (dataset,rung,S), RMSE of the
    # variant minus the best single LMM arm in the frozen file at k=0
    k0n = new[new.k == 0]
    per_prog_n = (k0n.groupby(["dataset", "rung", "S", "subset", "source", "method"])
                  .rmse.mean().reset_index())
    k0o = old[(old.k == 0) & old.method.isin(LMM_ARMS)]
    per_prog_o = (k0o.groupby(["dataset", "rung", "S", "subset", "source", "method"])
                  .rmse.mean().reset_index())
    best = (per_prog_o.groupby(["dataset", "rung", "S", "subset", "source"])
            .rmse.min().rename("best"))
    m = per_prog_n.merge(best.reset_index(),
                         on=["dataset", "rung", "S", "subset", "source"])
    m["shortfall"] = m.rmse - m.best
    reg = (m.groupby(["dataset", "rung", "S", "method"]).shortfall.mean()
           .groupby("method").max().rename("worst_shortfall"))
    reg_mean = (m.groupby(["method"]).shortfall.mean().rename("mean_shortfall"))
    summ = pd.concat([reg, reg_mean], axis=1)
    print("\nworst/mean shortfall per a0 variant (log RMSE, k=0):")
    print(summ.round(4).to_string())
    print(f"\nmax |worst_shortfall - PAKA-S3| over variants: "
          f"{float((summ.worst_shortfall - summ.worst_shortfall['PAKA-S3']).abs().max()):.4f}")

    # k=0 -> k=3 reduction at S=all, per domain
    alln = new[new.S == "all"]
    pp = (alln.groupby(["dataset", "method", "k", "source"]).rmse.mean()
          .groupby(["dataset", "method", "k"]).mean().unstack("k"))
    pp["red_%"] = 100 * (pp[0] - pp[3]) / pp[0]
    print("\nk=0 -> k=3 RMSE reduction at S=all (per domain):")
    print(pp.round(4).to_string())
    summ.to_csv(OUT / "paka_a0_regret_summary.csv")
    pp.to_csv(OUT / "paka_a0_kshot_summary.csv")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "smoke":
        mode_run(smoke=True)
    elif mode == "run":
        mode_run()
    elif mode == "check":
        mode_check()
    else:
        raise SystemExit(f"unknown mode {mode}")
