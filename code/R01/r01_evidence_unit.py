"""R01 diagnostic 6: is the test programme or the specimen the unit of evidence?

For a fixed specimen budget N, two training databases are built from the programmes
other than the held-out target:
  CONCENTRATED - whole programmes drawn at random until N specimens are reached;
  DISPERSED    - specimens drawn round-robin in pairs (up to two per programme per pass)
                 from as many programmes as possible until N specimens are reached.
If new-programme error depends on the number of programmes rather than on N, the
programme is the effective unit of evidence.

Outputs: outputs/evidence_unit_per_draw.csv.gz, outputs/evidence_unit_summary.csv
"""
from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import kag_data as K
from kag_bench import GBM_GRID, run_programme

OUT = Path(__file__).resolve().parents[1] / "outputs"
METHODS = ("KA-FIXED", "KA-RE", "DATA-RE", "GBM-RES")
BUDGETS = (24, 48, 96)
B = 15


def build(df, target, N, design, b):
    rng = np.random.default_rng([b, N, 1 if design == "DISPERSED" else 0, zlib.crc32(target.encode("utf-8"))])
    pool = df[df.source != target]
    progs = rng.permutation(np.sort(pool.source.unique()))
    take = []
    if design == "CONCENTRATED":
        for s in progs:
            rows = pool.index[pool.source == s].tolist()
            take += rows[: N - len(take)]
            if len(take) >= N:
                break
    else:   # round-robin in pairs: up to two specimens per programme per pass, so the
        # number of programmes is maximised while within-programme scatter stays estimable
        queues = [list(rng.permutation(pool.index[pool.source == s].to_numpy())) for s in progs]
        while len(take) < N and any(queues):
            for q in queues:
                for _ in range(2):
                    if q and len(take) < N:
                        take.append(q.pop())
    sub = pd.concat([pool.loc[take], df[df.source == target]])
    return sub.reset_index(drop=True), pool.loc[take].source.nunique(), len(take)


def task(df, feats, target, N, design, b):
    sub, n_prog, n = build(df, target, N, design, b)
    rows, _ = run_programme(sub, feats, target, METHODS, n_draws=10, seed=b, threads=1, ks=(0, 3),
                            gbm_grid=(GBM_GRID[1],))
    rows["N"], rows["design"], rows["subset"], rows["train_programmes"], rows["n_train"] = N, design, b, n_prog, n
    return rows


def main():
    t0 = time.time()
    frames = []
    for name in K.LOADERS:
        df = K.load(name)
        feats = K.FEATURES[name]
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        jobs = [(t, N, dz, b) for t in targets for N in BUDGETS for dz in ("CONCENTRATED", "DISPERSED")
                for b in range(B)]
        res = Parallel(n_jobs=12, backend="loky", batch_size=4)(
            delayed(task)(df, feats, t, N, dz, b) for t, N, dz, b in jobs)
        blk = pd.concat(res, ignore_index=True)
        blk.insert(0, "dataset", name)
        frames.append(blk)
        print(f"{name}: {len(jobs)} fits done at {time.time() - t0:.0f}s", flush=True)
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(OUT / "evidence_unit_per_draw.csv.gz", index=False)
    per_prog = (per_draw.groupby(["dataset", "N", "design", "method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean"), train_programmes=("train_programmes", "mean")))
    rng = np.random.default_rng(0)
    out = []
    for (ds, N, mth, k), blk in per_prog.groupby(["dataset", "N", "method", "k"]):
        piv = blk.pivot(index="source", columns="design", values="rmse")
        progs = piv.index.to_numpy()
        boots = rng.integers(0, len(progs), size=(2000, len(progs)))
        dlt = (piv["CONCENTRATED"] - piv["DISPERSED"]).to_numpy()   # > 0: dispersed better
        tp = blk.groupby("design").train_programmes.mean()
        lo, hi = np.percentile(np.mean(dlt[boots], 1), [2.5, 97.5])
        out.append(dict(dataset=ds, N=N, method=mth, k=k, rmse_concentrated=piv["CONCENTRATED"].mean(),
                        rmse_dispersed=piv["DISPERSED"].mean(), dispersed_gain=float(np.mean(dlt)),
                        gain_lo=lo, gain_hi=hi, programmes_concentrated=tp.get("CONCENTRATED"),
                        programmes_dispersed=tp.get("DISPERSED")))
    summ = pd.DataFrame(out)
    summ.to_csv(OUT / "evidence_unit_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
