"""R01 diagnostic 5: wrong knowledge. The anchor is corrupted everywhere (training and
target programmes) and the knowledge-using methods are refitted.

Corruptions
  size+0.3 / size-0.3 : exponent on d shifted by +/-0.3 around the median depth
  missing-term        : FRP - ACI 440 neutral-axis factor k replaced by its median
                        (reinforcement stiffness no longer enters);
                        SFRC - Narayanan-Darwish fibre contribution removed (F = 0)
Training size S = 6 random programmes (B subsets) and S = all.

Outputs: outputs/stress_per_draw.csv.gz, outputs/stress_summary.csv, outputs/stress_fit_info.csv
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
METHODS = ("KA-FIXED", "KA-RE", "KA-WB", "GBM-RES")
B_SMALL = 10


def corruptions(name, df):
    ln_d_med = np.median(df.ln_d)
    out = {"none": df.m.values,
           "size+0.3": df.m.values + 0.3 * (df.ln_d.values - ln_d_med),
           "size-0.3": df.m.values - 0.3 * (df.ln_d.values - ln_d_med)}
    if name == "FRP":
        rn = df.rho * df.Ef / (4700.0 * np.sqrt(df.fc))
        k = np.sqrt(2 * rn + rn ** 2) - rn
        out["missing-term"] = df.m.values - np.log(k.values) + np.log(np.median(k))
    else:
        out["missing-term"] = np.log(K.nd1987_v_kN(df.fc, df.b, df.d, df.rho, df.a_d, 0.0 * df.F).values)
    return out


def task(df, feats, target, variant, S, b):
    others = np.sort(df.loc[df.source != target, "source"].unique())
    subset = None
    if S is not None:
        rng = np.random.default_rng([b, S, zlib.crc32(target.encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    rows, info = run_programme(df, feats, target, METHODS, n_draws=10, seed=b, threads=1,
                               m_col=f"m_{variant}", ks=(0, 3), train_sources=subset, gbm_grid=(GBM_GRID[1],))
    rows["variant"], rows["S_label"], rows["subset"] = variant, ("all" if S is None else str(S)), b
    fit = [dict(variant=variant, S_label=("all" if S is None else str(S)), subset=b, source=target,
                method=mm, lam=v.get("lam"), tau=v.get("tau"), sig=v.get("sig"))
           for mm, v in info.items() if mm in ("KA-RE", "KA-WB")]
    return rows, fit


def main():
    t0 = time.time()
    frames, fits = [], []
    for name in K.LOADERS:
        df = K.load(name)
        feats = K.FEATURES[name]
        variants = corruptions(name, df)
        for v, mv in variants.items():
            df[f"m_{v}"] = mv
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        jobs = [(t, v, S, b) for t in targets for v in variants for S in (6, None)
                for b in (range(B_SMALL) if S is not None else [0])]
        res = Parallel(n_jobs=12, backend="loky", batch_size=4)(
            delayed(task)(df, feats, t, v, S, b) for t, v, S, b in jobs)
        for rows, fit in res:
            rows.insert(0, "dataset", name)
            frames.append(rows)
            fits += [dict(dataset=name, **f) for f in fit]
        print(f"{name}: {len(jobs)} fits done at {time.time() - t0:.0f}s", flush=True)
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(OUT / "stress_per_draw.csv.gz", index=False)
    pd.DataFrame(fits).to_csv(OUT / "stress_fit_info.csv", index=False)
    per_prog = (per_draw.groupby(["dataset", "variant", "S_label", "method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean")))
    summ = (per_prog.groupby(["dataset", "variant", "S_label", "method", "k"], as_index=False)
            .agg(rmse=("rmse", "mean"), programmes=("source", "nunique")))
    summ.to_csv(OUT / "stress_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.pivot_table(index=["dataset", "S_label", "k", "method"], columns="variant", values="rmse")
          .round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
