"""R05-E3/E4: knowledge value against library size, and programmes versus specimens.

  python -X utf8 r05_scarcity.py scarcity   # S in {3,6,12,24,all}, B random libraries per S
  python -X utf8 r05_scarcity.py unit       # fixed specimen budget N: concentrated vs dispersed

Targets: every programme with >= 6 specimens; libraries are drawn from the *other*
programmes only. Outputs: outputs/scarcity_per_draw.csv.gz, outputs/scarcity_summary.csv,
outputs/unit_per_draw.csv.gz, outputs/unit_summary.csv
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
from kag_bench_r05 import GBM_GRID, run_programme  # noqa: E402

OUT = HERE.parent / "outputs"
S_LEVELS = (3, 6, 12, 24, None)
B = 20
SC_ARMS = ("ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "PAKA", "PAKA-S", "PAKA-S0", "PAKA-S6", "GBM-RES", "GBM-ME")
UNIT_ARMS = ("KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S")
BUDGETS = {"FRP": (24, 48, 96), "SFRC": (24, 48, 96), "CARB": (48, 96, 192)}


def sc_task(name, target, S, b):
    s = D.spec(name)
    df = D.load(name)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    subset = others
    if S is not None:
        rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    arms = SC_ARMS if name != "CARB" else tuple(a for a in SC_ARMS if a != "ANCHOR")
    rows, info = run_programme(df, s["coded"], s["free"], target, arms, ks=(0, 3), n_draws=10, seed=b,
                               train_sources=subset, gbm_grid=(GBM_GRID[1],))
    rows["S"] = "all" if S is None else str(S)
    rows["subset"], rows["n_train"] = b, info["_train"]["n"]
    rows["paka_choice"] = info.get("PAKA", {}).get("chosen")
    return rows


def unit_build(df, target, N, design, b):
    rng = np.random.default_rng([b, N, 1 if design == "DISPERSED" else 0, zlib.crc32(str(target).encode("utf-8"))])
    pool = df[df.source != target]
    progs = rng.permutation(np.sort(pool.source.unique()))
    take = []
    if design == "CONCENTRATED":
        for s in progs:
            take += pool.index[pool.source == s].tolist()[: N - len(take)]
            if len(take) >= N:
                break
    else:
        queues = [list(rng.permutation(pool.index[pool.source == s].to_numpy())) for s in progs]
        while len(take) < N and any(queues):
            for q in queues:
                for _ in range(2):
                    if q and len(take) < N:
                        take.append(q.pop())
    sub = pd.concat([pool.loc[take], df[df.source == target]]).reset_index(drop=True)
    return sub, pool.loc[take].source.nunique()


def unit_task(name, target, N, design, b):
    s = D.spec(name)
    sub, n_prog = unit_build(D.load(name), target, N, design, b)
    rows, _ = run_programme(sub, s["coded"], s["free"], target, UNIT_ARMS, ks=(0, 3), n_draws=10, seed=b)
    rows["N"], rows["design"], rows["subset"], rows["train_programmes"] = N, design, b, n_prog
    return rows


def boot_summary(per_prog, keys, ref_pairs, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for kv, blk in per_prog.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        piv = blk.pivot_table(index="source", columns="method", values="rmse")
        progs = piv.index.to_numpy()
        boots = rng.integers(0, len(progs), size=(2000, len(progs)))
        for m in piv.columns:
            v = piv[m].to_numpy()
            rec = dict(zip(keys, kv), method=m, programmes=len(progs), rmse=float(np.mean(v)))
            rec["rmse_lo"], rec["rmse_hi"] = np.percentile(np.mean(v[boots], 1), [2.5, 97.5])
            for a, bref in ref_pairs:
                if m == a and bref in piv:
                    d = piv[bref].to_numpy() - v          # > 0: arm a better than bref
                    rec[f"gain_vs_{bref}"] = float(np.mean(d))
                    rec[f"gain_vs_{bref}_lo"], rec[f"gain_vs_{bref}_hi"] = np.percentile(np.mean(d[boots], 1), [2.5, 97.5])
            out.append(rec)
    return pd.DataFrame(out)


def main(mode):
    t0 = time.time()
    frames = []
    for name in D.DOMAINS:
        df = D.load(name)
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        if mode == "scarcity":
            jobs = [delayed(sc_task)(name, t, S, b) for t in targets for S in S_LEVELS
                    for b in (range(B) if S is not None else [0])]
        else:
            jobs = [delayed(unit_task)(name, t, N, dz, b) for t in targets for N in BUDGETS[name]
                    for dz in ("CONCENTRATED", "DISPERSED") for b in range(15)]
        res = Parallel(n_jobs=12, backend="loky", batch_size=2)(jobs)
        blk = pd.concat(res, ignore_index=True)
        blk.insert(0, "dataset", name)
        frames.append(blk)
        print(f"{mode} {name}: {len(jobs)} tasks at {time.time() - t0:.0f}s", flush=True)
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(OUT / f"{mode}_per_draw.csv.gz", index=False)
    if mode == "scarcity":
        per_prog = (per_draw.groupby(["dataset", "S", "k", "method", "source"], as_index=False)
                    .agg(rmse=("rmse", "mean"), n_train=("n_train", "mean"), cover90=("cover90", "mean")))
        summ = boot_summary(per_prog, ["dataset", "S", "k"],
                            [("KA-RE", "DATA-RE"), ("KA-FIXED", "DATA-RE"), ("PAKA-S", "DATA-RE"),
                             ("PAKA-S", "KA-FIXED"), ("PAKA-S", "KA-RE"), ("PAKA", "DATA-RE")])
        ntr = per_prog.groupby(["dataset", "S"]).n_train.mean().rename("n_train").reset_index()
        summ = summ.merge(ntr, on=["dataset", "S"], how="left")
        ch = (per_draw[per_draw.method == "PAKA"].groupby(["dataset", "S"]).paka_choice
              .value_counts(normalize=True).rename("share").reset_index())
        ch.to_csv(OUT / "scarcity_paka_choices.csv", index=False)
    else:
        per_prog = (per_draw.groupby(["dataset", "N", "design", "k", "method", "source"], as_index=False)
                    .agg(rmse=("rmse", "mean"), train_programmes=("train_programmes", "mean")))
        per_prog["method_design"] = per_prog.method + "|" + per_prog.design
        pp = per_prog.drop(columns="method").rename(columns={"method_design": "method"})
        summ = boot_summary(pp, ["dataset", "N", "k"],
                            [(f"{m}|DISPERSED", f"{m}|CONCENTRATED") for m in UNIT_ARMS])
        tp = per_prog.groupby(["dataset", "N", "design"]).train_programmes.mean().unstack("design").reset_index()
        summ = summ.merge(tp, on=["dataset", "N"], how="left")
    order = {"3": 0, "6": 1, "12": 2, "24": 3, "all": 4}
    if mode == "scarcity":
        summ["S_order"] = summ.S.map(order)
        summ = summ.sort_values(["dataset", "k", "S_order", "method"])
    summ.to_csv(OUT / f"{mode}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    if mode == "scarcity":
        print(summ.pivot_table(index=["dataset", "k", "method"], columns="S_order", values="rmse").round(3).to_string())
    else:
        print(summ.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scarcity")
