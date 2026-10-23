"""Quick probe (not a paper result): does stacking over the trust ladder remove the
large-library regret of the one-standard-error selector? Subset of targets and libraries."""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kag_data_r05 as D  # noqa: E402
from kag_bench_r05 import run_programme  # noqa: E402

ARMS = ("KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S0", "PAKA-S", "PAKA-S6")


def task(name, target, S, b):
    s = D.spec(name)
    df = D.load(name)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    subset = others
    if S is not None:
        rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    rows, info = run_programme(df, s["coded"], s["free"], target, ARMS, ks=(0, 3), n_draws=10, seed=b,
                               train_sources=subset)
    rows["S"] = "all" if S is None else str(S)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    out = []
    for name in D.DOMAINS:
        df = D.load(name)
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)[:8]
        jobs = [delayed(task)(name, t, S, b) for t in targets for S in (3, 6, 12, None)
                for b in (range(6) if S is not None else [0])]
        r = pd.concat(Parallel(n_jobs=12, backend="loky", batch_size=2)(jobs), ignore_index=True)
        r.insert(0, "dataset", name)
        out.append(r)
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    r = pd.concat(out)
    pp = r.groupby(["dataset", "S", "k", "method", "source"]).rmse.mean().reset_index()
    print(pp.pivot_table(index=["dataset", "k", "method"], columns="S", values="rmse").round(3).to_string())
