"""R01 diagnostic 3: value of the codified anchor as the number of training
programmes shrinks (the small-data regime).

For every held-out programme with >= 6 specimens and every training size S, B random
subsets of S *other* programmes form the training database. Methods are refitted on
each subset and scored on the held-out programme at k = 0 and k = 3 shots.

Outputs: outputs/scarcity_per_draw.csv.gz, outputs/scarcity_summary.csv
"""
from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import kag_data as K
from kag_bench import GBM_GRID, run_programme

OUT = Path(__file__).resolve().parents[1] / "outputs"
S_LEVELS = (3, 6, 12, 24, None)          # None = all other programmes
METHODS = ("ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "GBM", "GBM-RES")


def task(df, feats, target, S, b, draws):
    others = np.sort(df.loc[df.source != target, "source"].unique())
    if S is None:
        subset = others
    else:
        rng = np.random.default_rng([b, S, zlib.crc32(target.encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    rows, info = run_programme(df, feats, target, METHODS, n_draws=draws, seed=b, threads=1,
                               ks=(0, 3), train_sources=subset, gbm_grid=(GBM_GRID[1],))
    rows["S"] = len(subset) if S is not None else -1
    rows["S_label"] = str(S) if S is not None else "all"
    rows["subset"] = b
    rows["n_train"] = info["_train"]["n"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    frames = []
    for name in K.LOADERS:
        df = K.load(name)
        feats = K.FEATURES[name]
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        jobs = [(t, S, b) for t in targets for S in S_LEVELS for b in (range(a.B) if S is not None else [0])]
        res = Parallel(n_jobs=a.jobs, backend="loky", batch_size=4)(
            delayed(task)(df, feats, t, S, b, a.draws) for t, S, b in jobs)
        blk = pd.concat(res, ignore_index=True)
        blk.insert(0, "dataset", name)
        frames.append(blk)
        print(f"{name}: {len(jobs)} fits done at {time.time() - t0:.0f}s", flush=True)
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(OUT / "scarcity_per_draw.csv.gz", index=False)

    per_prog = (per_draw.groupby(["dataset", "S_label", "method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean"), n_train=("n_train", "mean"), cover90=("cover90", "mean")))
    rng = np.random.default_rng(0)
    out = []
    for (ds, S, k), blk in per_prog.groupby(["dataset", "S_label", "k"]):
        progs = np.sort(blk.source.unique())
        piv = blk.pivot(index="source", columns="method", values="rmse").reindex(progs)
        boots = rng.integers(0, len(progs), size=(2000, len(progs)))
        for mname in piv.columns:
            v = piv[mname].values
            rec = dict(dataset=ds, S=S, k=k, method=mname, programmes=len(progs), rmse=float(np.mean(v)),
                       n_train=float(blk[blk.method == mname].n_train.mean()),
                       cover90=float(blk[blk.method == mname].cover90.mean()))
            rec["rmse_lo"], rec["rmse_hi"] = np.percentile(np.mean(v[boots], 1), [2.5, 97.5])
            for ref in ("DATA-RE", "GBM-RES", "GBM"):
                if ref in piv and mname == "KA-RE":
                    dlt = piv[ref].values - v       # > 0: knowledge-anchored model better
                    rec[f"gain_vs_{ref}"] = float(np.mean(dlt))
                    rec[f"gain_vs_{ref}_lo"], rec[f"gain_vs_{ref}_hi"] = np.percentile(np.mean(dlt[boots], 1), [2.5, 97.5])
                    rec[f"win_share_vs_{ref}"] = float(np.mean(dlt > 0))
            out.append(rec)
    summ = pd.DataFrame(out)
    order = {"3": 0, "6": 1, "12": 2, "24": 3, "all": 4}
    summ["S_order"] = summ.S.map(order)
    summ = summ.sort_values(["dataset", "k", "S_order", "method"])
    summ.to_csv(OUT / "scarcity_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.pivot_table(index=["dataset", "k", "method"], columns="S_order", values="rmse").round(3).to_string())
    ka = summ[summ.method == "KA-RE"][["dataset", "k", "S", "n_train", "gain_vs_DATA-RE", "gain_vs_DATA-RE_lo",
                                        "gain_vs_DATA-RE_hi", "win_share_vs_DATA-RE", "gain_vs_GBM-RES",
                                        "gain_vs_GBM-RES_lo", "gain_vs_GBM-RES_hi"]]
    print(ka.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
