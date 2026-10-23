"""R01 diagnostic 4: programme-level size extrapolation.

Training database = programmes whose mean effective depth is below the 75th
percentile of programme means; targets = programmes at or above it (>= 4
specimens). This mimics a new test programme that is larger than everything in the
database. Scored at k = 0 and k = 1 (and k = 3 where the programme allows it).

Outputs: outputs/extrapolation_per_draw.csv.gz, outputs/extrapolation_summary.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import kag_data as K
from kag_bench import run_programme

OUT = Path(__file__).resolve().parents[1] / "outputs"
METHODS = ("ANCHOR", "KA-FIXED", "KA-RE", "KA-WB", "DATA-RE", "DATA-WB", "GBM", "GBM-RES", "GP-RES")


def main():
    t0 = time.time()
    frames, splits = [], []
    for name in K.LOADERS:
        df = K.load(name)
        feats = K.FEATURES[name]
        prog = df.groupby("source").agg(n=("d", "size"), d_mean=("d", "mean"), d_max=("d", "max"))
        thr = prog.d_mean.quantile(0.75)
        train = prog.index[prog.d_mean < thr]
        targets = sorted(prog.index[(prog.d_mean >= thr) & (prog.n >= 4)])
        d_train_max = df[df.source.isin(train)].d.max()
        splits.append(dict(dataset=name, threshold_mm=thr, train_programmes=len(train),
                           train_n=int(df.source.isin(train).sum()), train_d_max=d_train_max,
                           targets=len(targets), target_n=int(df.source.isin(targets).sum()),
                           target_d_min=df[df.source.isin(targets)].d.min(),
                           target_d_max=df[df.source.isin(targets)].d.max(),
                           target_beyond_train_max=float((df[df.source.isin(targets)].d > d_train_max).mean())))
        res = Parallel(n_jobs=min(10, len(targets)), backend="loky")(
            delayed(run_programme)(df, feats, t, METHODS, 40, 0, 2, "m", (0, 1, 3), train) for t in targets)
        for t, (rows, _) in zip(targets, res):
            rows.insert(0, "dataset", name)
            frames.append(rows)
        print(f"{name}: {len(targets)} large-size target programmes done at {time.time() - t0:.0f}s", flush=True)
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(OUT / "extrapolation_per_draw.csv.gz", index=False)
    per_prog = (per_draw.groupby(["dataset", "method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean"), bias=("bias", "mean"), unconservative=("unconservative", "mean"),
                     cover90=("cover90", "mean")))
    summ = (per_prog.groupby(["dataset", "method", "k"], as_index=False)
            .agg(programmes=("source", "nunique"), rmse=("rmse", "mean"), bias=("bias", "mean"),
                 unconservative=("unconservative", "mean"), cover90=("cover90", "mean")))
    summ.to_csv(OUT / "extrapolation_summary.csv", index=False)
    pd.DataFrame(splits).to_csv(OUT / "extrapolation_split.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(splits).round(2).to_string(index=False))
    print(summ.pivot_table(index=["dataset", "method"], columns="k",
                           values=["rmse", "bias", "unconservative"]).round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
