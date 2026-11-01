"""R08 blind test of the variance budget on the certified SFRC anchors (K1C, K2C).

  python -X utf8 r08_certified_ladder.py predict    # record S* (pre-specified form and
                                                    # arm-split form) with programme-bootstrap
                                                    # intervals; refuses to overwrite
  python -X utf8 r08_certified_ladder.py scarcity   # library-size experiment, identical to
                                                    # r06_ladder.sc_task (same library draws,
                                                    # seeds, targets); then observed crossover

The predictions are written BEFORE the scarcity run (timestamp in the CSV); the rungs were
defined in ../PRE_ANALYSIS_certified_anchor.md. Outputs in ../outputs/.
"""
from __future__ import annotations

import sys
import time
import zlib
from datetime import datetime, timezone
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
import r06_ladder as L  # noqa: E402
import r05_theory as T  # noqa: E402
from kag_bench_r05 import run_programme  # noqa: E402
import r08_budget_check as BC  # noqa: E402
from r08_certified_sfrc import frame  # noqa: E402

OUT = R08 / "outputs"
RUNGS = ("K1C", "K2C")
ARMS = ("KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S")
S_LEVELS = (3, 6, 12, 24)
NB = 20
R06_PER_DRAW = ROUNDS / "R06_mechanism_pivot_2026-10-22" / "outputs" / "ladder_per_draw.csv.gz"


def mode_predict(n_boot=200):
    f = OUT / "certified_prediction.csv"
    if f.exists():
        raise SystemExit(f"{f} exists; predictions are write-once")
    rows = []
    for rung in RUNGS:
        df, coded, free = frame(rung)
        s = BC.split_inputs(df, coded, free)
        bs_pre = T.boot_pred(df, coded, free, n_boot=n_boot)
        bs_split = BC.boot_split(df, coded, free, n_boot)
        fin = bs_split[np.isfinite(bs_split)]
        rows.append(dict(domain="SFRC", rung=rung, p=s["p"], q=s["q"], B=s["B"], c=s["c"],
                         c_F=s["c_F"], c_D=s["c_D"],
                         S_prespec=s["S_star"], S_prespec_lo=np.percentile(bs_pre, 2.5),
                         S_prespec_hi=np.percentile(bs_pre, 97.5),
                         S_split=s["S_split"], S_split_lo=np.percentile(fin, 2.5),
                         S_split_hi=np.percentile(fin, 97.5),
                         written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)


def sc_task(rung, target, S, b):
    df, coded, free = frame(rung)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
    subset = rng.choice(others, size=min(S, len(others)), replace=False)
    rows, info = run_programme(df, coded, free, target, ARMS, ks=(0, 3), n_draws=10, seed=b,
                               train_sources=subset)
    rows["S"], rows["dataset"], rows["rung"] = str(S), "SFRC", rung
    rows["subset"], rows["n_train"] = b, info["_train"]["n"]
    return rows


def mode_scarcity():
    if not (OUT / "certified_prediction.csv").exists():
        raise SystemExit("run `predict` first")
    t0 = time.time()
    df, _, _ = frame("K1C")
    n_s = df.groupby("source").size()
    targets = sorted(n_s[n_s >= 6].index)
    frames = []
    for rung in RUNGS:
        jobs = [delayed(sc_task)(rung, t, S, b) for t in targets for S in S_LEVELS for b in range(NB)]
        res = Parallel(n_jobs=12, backend="loky", batch_size=2)(jobs)
        frames.append(pd.concat(res, ignore_index=True))
        print(f"scarcity SFRC {rung}: {len(jobs)} tasks at {time.time() - t0:.0f}s", flush=True)
    pp = pd.concat(frames, ignore_index=True)
    pp.to_csv(OUT / "certified_ladder_per_draw.csv.gz", index=False)

    # DATA-RE must equal R06's SFRC DATA-RE draw for draw (same rows, libraries, seeds)
    k1 = pd.read_csv(D_long(R06_PER_DRAW))
    k1["S"] = k1.S.astype(str)
    k1 = k1[(k1.dataset == "SFRC") & (k1.rung.isin(["K1", "K2"])) & (k1.S != "all")]
    old = k1[(k1.rung == "K1") & (k1.method == "DATA-RE")]
    key = ["S", "subset", "source", "k", "draw"]
    new = pp[(pp.method == "DATA-RE") & (pp.rung == "K1C")]
    mg = new.merge(old, on=key, suffixes=("", "_r06"))
    md = float(np.abs(mg.rmse - mg.rmse_r06).max())
    print(f"DATA-RE identity vs R06 SFRC K1: {len(mg)} matched rows, max |diff| = {md:.2e}", flush=True)
    assert len(mg) == len(new) and md < 1e-9, "DATA-RE differs from R06; stop and report"
    summarise(pp, k1)
    print(f"total {time.time() - t0:.0f}s")


def mode_summary():
    pp = pd.read_csv(OUT / "certified_ladder_per_draw.csv.gz")
    pp["S"] = pp.S.astype(str)
    k1 = pd.read_csv(D_long(R06_PER_DRAW))
    k1["S"] = k1.S.astype(str)
    k1 = k1[(k1.dataset == "SFRC") & (k1.rung.isin(["K1", "K2"])) & (k1.S != "all")]
    summarise(pp, k1)


def summarise(pp, k1):
    allp = pd.concat([k1, pp], ignore_index=True)
    pred = pd.read_csv(OUT / "certified_prediction.csv")
    rows = []
    for rung in ("K1", "K2") + RUNGS:
        obs, lo, hi, none, _ = L.observed_ladder(allp, "SFRC", rung)
        r = dict(rung=rung, S_obs=obs, S_obs_lo=lo, S_obs_hi=hi, obs_boot_no_crossing=none)
        if rung in RUNGS:
            p = pred[pred.rung == rung].iloc[0]
            r.update(S_prespec=p.S_prespec, S_prespec_lo=p.S_prespec_lo, S_prespec_hi=p.S_prespec_hi,
                     S_split=p.S_split, S_split_lo=p.S_split_lo, S_split_hi=p.S_split_hi)
        rows.append(r)
    t = pd.DataFrame(rows)
    per = (allp[allp.k == 0].groupby(["rung", "S", "method", "source"]).rmse.mean()
           .groupby(["rung", "S", "method"]).mean().unstack("S")[["3", "6", "12", "24"]])
    t.to_csv(OUT / "certified_crossover.csv", index=False)
    per.to_csv(OUT / "certified_scarcity_curves.csv")
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(per.round(4).to_string())


def D_long(p):
    return L.wpath(p)


if __name__ == "__main__":
    {"predict": mode_predict, "scarcity": mode_scarcity, "summary": mode_summary}[sys.argv[1]]()
