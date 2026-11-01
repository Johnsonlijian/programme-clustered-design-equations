"""R08 fairness re-run: machine-learning baselines in the library-size experiment with the
SAME tuning as the full-library comparison.

R05's library-size experiment (r05_scarcity.sc_task) ran mixed-effects boosting with one fixed
configuration (gbm_grid=(GBM_GRID[1],)) and no plain boosting; the full-library comparison
(r05_lopo) chose among four configurations by grouped inner cross-validation. Here GBM and
GBM-ME are re-run on the identical libraries (same rng formula, same targets, same seeds) with the
full four-configuration grid, in all four databases (DEEP features: d, b_w, f'c, rho, a/d; the
learners ignore the anchor).

Outputs: ../outputs/tuned_per_draw.csv.gz, tuned_summary.csv
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROUNDS = HERE.parents[1]
sys.path[:0] = [str(HERE),
                str(ROUNDS / "R06_mechanism_pivot_2026-10-22" / "scripts"),
                str(ROUNDS / "R05_consolidation_2026-10-21" / "scripts"),
                str(ROUNDS / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import kag_data_r05 as D  # noqa: E402
import r06_rungs as R  # noqa: E402
from kag_bench_r05 import GBM_GRID, run_programme  # noqa: E402

OUT = R08 / "outputs"
ARMS = ("GBM", "GBM-ME")
S_LEVELS = (3, 6, 12, 24)
NB = 20


def frame(name):
    if name == "DEEP":
        return R.load("DEEP", "ARCHS")
    s = D.spec(name)
    return D.load(name), list(s["coded"]), list(s["free"])


def task(name, target, S, b):
    df, coded, free = frame(name)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
    subset = rng.choice(others, size=min(S, len(others)), replace=False)
    rows, info = run_programme(df, coded, free, target, ARMS, ks=(0, 3), n_draws=10, seed=b,
                               train_sources=subset, gbm_grid=GBM_GRID)
    rows["S"], rows["dataset"], rows["subset"] = str(S), name, b
    return rows


def main(domains=("FRP", "SFRC", "CARB", "DEEP")):
    t0 = time.time()
    frames = []
    for name in domains:
        df, _, _ = frame(name)
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        jobs = [delayed(task)(name, t, S, b) for t in targets for S in S_LEVELS for b in range(NB)]
        res = Parallel(n_jobs=12, backend="loky", batch_size=2)(jobs)
        frames.append(pd.concat(res, ignore_index=True))
        print(f"tuned {name}: {len(jobs)} tasks at {time.time() - t0:.0f}s", flush=True)
        pd.concat(frames, ignore_index=True).to_csv(OUT / "tuned_per_draw.csv.gz", index=False)
    pp = pd.concat(frames, ignore_index=True)
    per = pp.groupby(["dataset", "S", "k", "method", "source"]).rmse.mean().reset_index()
    rng = np.random.default_rng(0)
    rows = []
    for (ds, S, k, m), blk in per.groupby(["dataset", "S", "k", "method"]):
        v = blk.rmse.to_numpy()
        bs = rng.integers(0, len(v), size=(2000, len(v)))
        lo, hi = np.percentile(v[bs].mean(1), [2.5, 97.5])
        rows.append(dict(dataset=ds, S=S, k=k, method=m, programmes=len(v), rmse=float(v.mean()), lo=lo, hi=hi))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "tuned_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["dataset", "k", "method"], columns="S", values="rmse").round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("FRP", "SFRC", "CARB", "DEEP"))
