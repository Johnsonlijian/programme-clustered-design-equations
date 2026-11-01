"""R08 prospective repair test: free every CONTRADICTED term (see ../PRE_ANALYSIS_repair_all.md).

  python -X utf8 r08_repair.py predict    # write-once S* predictions with bootstrap intervals
  python -X utf8 r08_repair.py scarcity   # library-size experiment (+ S=all) as r06_ladder
  python -X utf8 r08_repair.py summary    # observed crossovers, paired comparisons, budget check

Repaired rungs (parent -> repair):
  FRP  K1 -> K1C, K2 -> K2C   : ln_ad moved from coded to free (ACI 440 asserts no a/d term)
  DEEP BEAM -> BEAMC          : ln_ad moved from coded to free (ACI 318-19 one-way shear)
The SFRC repairs (K1C, K2C) were run by r08_certified_sfrc.py / r08_certified_ladder.py.
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
import r06_rungs as R  # noqa: E402
import r06_ladder as L  # noqa: E402
import r05_theory as T  # noqa: E402
from kag_bench_r05 import run_programme  # noqa: E402
import r08_budget_check as BC  # noqa: E402

OUT = R08 / "outputs"
REPAIRS = {("FRP", "K1C"): "K1", ("FRP", "K2C"): "K2", ("DEEP", "BEAMC"): "BEAM"}
LMM = ("KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S")
ML = ("GBM", "GBM-ME")
S_LEVELS = (3, 6, 12, 24)
NB = 20
R06_PER_DRAW = ROUNDS / "R06_mechanism_pivot_2026-10-22" / "outputs" / "ladder_per_draw.csv.gz"
FREED = "ln_ad"


def frame(domain, rung):
    parent = REPAIRS[(domain, rung)]
    df, coded, free = R.load(domain, parent)
    coded = [c for c in coded if c != FREED]
    free = list(free) + [FREED]
    coded, free = L._canon_feats(domain, coded, free)
    return df.reset_index(drop=True), coded, free


def mode_predict(n_boot=200):
    f = OUT / "repair_prediction.csv"
    if f.exists():
        raise SystemExit(f"{f} exists; predictions are write-once")
    rows = []
    for (domain, rung), parent in REPAIRS.items():
        df, coded, free = frame(domain, rung)
        s = BC.split_inputs(df, coded, free)
        bs_pre = T.boot_pred(df, coded, free, n_boot=n_boot)
        bs_split = BC.boot_split(df, coded, free, n_boot)
        fin = bs_split[np.isfinite(bs_split)]
        rows.append(dict(domain=domain, rung=rung, parent=parent, p=s["p"], q=s["q"], B=s["B"], c=s["c"],
                         c_F=s["c_F"], c_D=s["c_D"], S_prespec=s["S_star"],
                         S_prespec_lo=np.percentile(bs_pre, 2.5), S_prespec_hi=np.percentile(bs_pre, 97.5),
                         S_split=s["S_split"], S_split_lo=np.percentile(fin, 2.5),
                         S_split_hi=np.percentile(fin, 97.5),
                         written_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)


def sc_task(domain, rung, target, S, b):
    df, coded, free = frame(domain, rung)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    subset = others
    arms = LMM
    if S is not None:
        rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    elif domain == "DEEP":
        arms = LMM + ML
    rows, info = run_programme(df, coded, free, target, arms, ks=(0, 3), n_draws=10, seed=b,
                               train_sources=subset)
    rows["S"] = "all" if S is None else str(S)
    rows["dataset"], rows["rung"] = domain, rung
    rows["subset"], rows["n_train"] = b, info["_train"]["n"]
    return rows


def mode_scarcity():
    if not (OUT / "repair_prediction.csv").exists():
        raise SystemExit("run `predict` first")
    t0 = time.time()
    frames = []
    for (domain, rung) in REPAIRS:
        df, _, _ = frame(domain, rung)
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        jobs = [delayed(sc_task)(domain, rung, t, S, b) for t in targets for S in S_LEVELS + (None,)
                for b in (range(NB) if S is not None else [0])]
        res = Parallel(n_jobs=12, backend="loky", batch_size=2)(jobs)
        frames.append(pd.concat(res, ignore_index=True))
        print(f"scarcity {domain} {rung}: {len(jobs)} tasks at {time.time() - t0:.0f}s", flush=True)
    pp = pd.concat(frames, ignore_index=True)
    pp.to_csv(OUT / "repair_per_draw.csv.gz", index=False)
    mode_summary()
    print(f"total {time.time() - t0:.0f}s")


def boot_mean(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(v), size=(n_boot, len(v)))
    lo, hi = np.percentile(v[b].mean(1), [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def mode_summary():
    pp = pd.read_csv(OUT / "repair_per_draw.csv.gz")
    pp["S"] = pp.S.astype(str)
    old = pd.read_csv(L.wpath(R06_PER_DRAW))
    old["S"] = old.S.astype(str)
    key = ["S", "subset", "source", "k", "draw"]
    for (domain, rung), parent in REPAIRS.items():
        o = old[(old.dataset == domain) & (old.rung == parent) & (old.method == "DATA-RE")]
        n = pp[(pp.dataset == domain) & (pp.rung == rung) & (pp.method == "DATA-RE")]
        mg = n.merge(o, on=key, suffixes=("", "_r06"))
        md = float(np.abs(mg.rmse - mg.rmse_r06).max())
        print(f"DATA-RE identity {domain} {rung} vs R06 {parent}: {len(mg)}/{len(n)} rows, max |diff| = {md:.2e}",
              flush=True)
        assert len(mg) == len(n) and md < 1e-9, "DATA-RE differs from R06; stop and report"

    want = pd.MultiIndex.from_tuples(sorted({(d, p) for (d, _), p in REPAIRS.items()}))
    parents = old[pd.MultiIndex.from_arrays([old.dataset, old.rung]).isin(want)]
    allp = pd.concat([parents, pp], ignore_index=True)
    pred = pd.read_csv(OUT / "repair_prediction.csv")
    th = pd.read_csv(ROUNDS / "R06_mechanism_pivot_2026-10-22" / "outputs" / "ladder_theory.csv")
    rows = []
    for (domain, rung), parent in REPAIRS.items():
        for rr in (parent, rung):
            obs, lo, hi, none, _ = L.observed_ladder(allp, domain, rr)
            r = dict(domain=domain, rung=rr, role="parent" if rr == parent else "repair",
                     S_obs=obs, S_obs_lo=lo, S_obs_hi=hi, obs_boot_no_crossing=none)
            if rr == rung:
                p = pred[(pred.domain == domain) & (pred.rung == rung)].iloc[0]
                r.update(B=p.B, S_prespec=p.S_prespec, S_prespec_lo=p.S_prespec_lo, S_prespec_hi=p.S_prespec_hi,
                         S_split=p.S_split, S_split_lo=p.S_split_lo, S_split_hi=p.S_split_hi)
            else:
                t = th[(th.domain == domain) & (th.rung == rr)].iloc[0]
                r.update(B=t.B, S_prespec=t.S_star, S_prespec_lo=t.S_star_lo, S_prespec_hi=t.S_star_hi)
            rows.append(r)
    cross = pd.DataFrame(rows)
    cross.to_csv(OUT / "repair_crossover.csv", index=False)

    per = (allp.groupby(["dataset", "rung", "S", "k", "method", "source"]).rmse.mean().reset_index())
    curves = per[per.k == 0].groupby(["dataset", "rung", "method", "S"]).rmse.mean().unstack("S")
    curves = curves[[c for c in ["3", "6", "12", "24", "all"] if c in curves.columns]]
    curves.to_csv(OUT / "repair_scarcity_curves.csv")

    pairs = []
    for (domain, rung), parent in REPAIRS.items():
        for S in ["3", "6", "12", "24", "all"]:
            for k in (0, 3):
                for meth in ("KA-FIXED", "PAKA-S"):
                    a = per[(per.dataset == domain) & (per.rung == rung) & (per.S == S) & (per.k == k)
                            & (per.method == meth)].set_index("source").rmse
                    b = per[(per.dataset == domain) & (per.rung == parent) & (per.S == S) & (per.k == k)
                            & (per.method == meth)].set_index("source").rmse
                    d = (a - b).dropna()
                    if len(d) == 0:
                        continue
                    m, lo, hi = boot_mean(d.values)
                    pairs.append(dict(domain=domain, repair=rung, parent=parent, S=S, k=k, method=meth,
                                      programmes=len(d), diff=m, lo=lo, hi=hi))
        if domain == "DEEP":
            for k in (0, 3):
                for ml in ML:
                    a = per[(per.dataset == domain) & (per.rung == rung) & (per.S == "all") & (per.k == k)
                            & (per.method == "PAKA-S")].set_index("source").rmse
                    b = per[(per.dataset == domain) & (per.rung == rung) & (per.S == "all") & (per.k == k)
                            & (per.method == ml)].set_index("source").rmse
                    d = (a - b).dropna()
                    m, lo, hi = boot_mean(d.values)
                    pairs.append(dict(domain=domain, repair=rung, parent=ml, S="all", k=k, method="PAKA-S vs " + ml,
                                      programmes=len(d), diff=m, lo=lo, hi=hi))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "repair_paired.csv", index=False)

    full = (per[per.S == "all"].groupby(["dataset", "rung", "k", "method"]).rmse
            .apply(lambda v: pd.Series(boot_mean(v.values), index=["rmse", "lo", "hi"])).unstack())
    full.to_csv(OUT / "repair_fulllib.csv")
    pd.set_option("display.width", 220)
    print(cross.round(3).to_string(index=False))
    print(curves.round(4).to_string())
    print(pairs.round(4).to_string(index=False))
    print(full.round(4).to_string())


if __name__ == "__main__":
    {"predict": mode_predict, "scarcity": mode_scarcity, "summary": mode_summary}[sys.argv[1]]()
