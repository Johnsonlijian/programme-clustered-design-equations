"""R05-E7b: is the small-library value of the code an artefact of the code having been
calibrated on some of the same tests? Repeat the knowledge-value curve after removing
every programme published up to the anchor's calibration year (FRP <= 2002 for ACI
440.1R-15 via Tureyen & Frosch 2003; SFRC <= 1987 for Narayanan & Darwish 1987) from
both the libraries and the targets.

Outputs: outputs/circular_scarcity_per_draw.csv.gz, outputs/circular_scarcity_summary.csv
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
sys.path.insert(0, str(HERE))
import kag_data_r05 as D  # noqa: E402
from kag_bench_r05 import run_programme  # noqa: E402

OUT = HERE.parent / "outputs"
ARMS = ("KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S")
S_LEVELS = (3, 6, 12, None)
B = 10


def load_post(name):
    df = D.load(name)
    return df[~(df.year <= D.spec(name)["anchor_cut"])].reset_index(drop=True)


def task(name, target, S, b):
    s = D.spec(name)
    df = load_post(name)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    subset = others
    if S is not None:
        rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    rows, _ = run_programme(df, s["coded"], s["free"], target, ARMS, ks=(0, 3), n_draws=10, seed=b,
                            train_sources=subset)
    rows["S"] = "all" if S is None else str(S)
    return rows


def main():
    t0 = time.time()
    frames = []
    for name in ("FRP", "SFRC"):
        df = load_post(name)
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        jobs = [delayed(task)(name, t, S, b) for t in targets for S in S_LEVELS
                for b in (range(B) if S is not None else [0])]
        r = pd.concat(Parallel(n_jobs=12, backend="loky", batch_size=2)(jobs), ignore_index=True)
        r.insert(0, "dataset", name)
        frames.append(r)
        print(f"{name}: {len(jobs)} tasks at {time.time() - t0:.0f}s", flush=True)
    pdraw = pd.concat(frames, ignore_index=True)
    pdraw.to_csv(OUT / "circular_scarcity_per_draw.csv.gz", index=False)
    pp = pdraw.groupby(["dataset", "S", "k", "method", "source"], as_index=False).rmse.mean()
    rng = np.random.default_rng(0)
    rows = []
    for (ds, S, k), blk in pp.groupby(["dataset", "S", "k"]):
        piv = blk.pivot(index="source", columns="method", values="rmse")
        boots = rng.integers(0, len(piv), size=(2000, len(piv)))
        for m in piv.columns:
            v = piv[m].to_numpy()
            d = piv["DATA-RE"].to_numpy() - v
            lo, hi = np.percentile(np.mean(d[boots], 1), [2.5, 97.5])
            rows.append(dict(dataset=ds, S=S, k=k, method=m, programmes=len(piv), rmse=float(v.mean()),
                             gain_vs_data=float(d.mean()), gain_lo=lo, gain_hi=hi))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "circular_scarcity_summary.csv", index=False)
    print(summ[summ.k == 0].pivot_table(index=["dataset", "method"], columns="S", values="rmse").round(3).to_string())
    print(summ[(summ.k == 0) & (summ.method == "KA-FIXED")].round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
