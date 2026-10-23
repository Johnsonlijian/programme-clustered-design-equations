"""R05-E2/E6/E7: new-programme prediction with the full library, size extrapolation and
calibration-circularity sensitivity.

  python -X utf8 r05_lopo.py lopo         # every programme with >= 6 specimens held out in turn
  python -X utf8 r05_lopo.py extrap       # shear domains: programmes ranked by mean depth; train on the
                                          # smaller 75 % of programmes, predict the larger ones (>= 4 specimens)
  python -X utf8 r05_lopo.py circular     # shear domains: drop every programme published up to the anchor's
                                          # calibration year (FRP <= 2002, SFRC <= 1987) from training AND targets

Outputs: outputs/{mode}_per_draw.csv.gz, outputs/{mode}_summary.csv, outputs/{mode}_fit_info.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kag_data_r05 as D  # noqa: E402
from kag_bench_r05 import ALL_ARMS, run_programme, summarise  # noqa: E402

OUT = HERE.parent / "outputs"


def arms_for(name):
    a = ALL_ARMS
    if name == "CARB":
        a = tuple(x for x in a if x not in ("ANCHOR", "GP-RES"))
    return a


def task(name, target, train_sources, ks, drop_before=None):
    s = D.spec(name)
    df = D.load(name)
    if drop_before is not None:
        df = df[~(df.year <= drop_before)].reset_index(drop=True)
    return run_programme(df, s["coded"], s["free"], target, arms_for(name), ks=ks, n_draws=40, seed=0,
                         train_sources=train_sources)


def main(mode):
    t0 = time.time()
    frames, infos, splits = [], {}, []
    names = list(D.DOMAINS) if mode == "lopo" else ["FRP", "SFRC"]
    for name in names:
        s = D.spec(name)
        df = D.load(name)
        drop = s["anchor_cut"] if mode == "circular" else None
        if drop is not None:
            df = df[~(df.year <= drop)].reset_index(drop=True)
        n_s = df.groupby("source").size()
        if mode == "extrap":
            prog = df.groupby("source").agg(n=("d", "size"), d_mean=("d", "mean"))
            thr = prog.d_mean.quantile(0.75)
            train = list(prog.index[prog.d_mean < thr])
            targets = sorted(prog.index[(prog.d_mean >= thr) & (prog.n >= 4)])
            ks = (0, 1, 3)
            dmax = df[df.source.isin(train)].d.max()
            splits.append(dict(dataset=name, threshold_mm=thr, train_programmes=len(train),
                               train_n=int(df.source.isin(train).sum()), train_d_max=dmax, targets=len(targets),
                               target_n=int(df.source.isin(targets).sum()),
                               target_d_max=df[df.source.isin(targets)].d.max(),
                               beyond_train_max=float((df[df.source.isin(targets)].d > dmax).mean())))
        else:
            train, targets, ks = None, sorted(n_s[n_s >= 6].index), (0, 1, 2, 3, 5)
            if mode == "circular":
                splits.append(dict(dataset=name, dropped_up_to=drop, programmes=int(n_s.size), n=len(df),
                                   targets=len(targets)))
        res = Parallel(n_jobs=min(12, len(targets)), backend="loky")(
            delayed(task)(name, t, train, ks, drop) for t in targets)
        for t, (rows, info) in zip(targets, res):
            rows.insert(0, "dataset", name)
            frames.append(rows)
            infos[f"{name}::{t}"] = info
        print(f"{mode} {name}: {len(targets)} targets at {time.time() - t0:.0f}s", flush=True)
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(OUT / f"{mode}_per_draw.csv.gz", index=False)
    per_prog, summ = summarise(per_draw)
    per_prog.to_csv(OUT / f"{mode}_per_programme.csv", index=False)
    summ.to_csv(OUT / f"{mode}_summary.csv", index=False)
    (OUT / f"{mode}_fit_info.json").write_text(json.dumps(infos, indent=1, default=str))
    if splits:
        pd.DataFrame(splits).to_csv(OUT / f"{mode}_split.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.pivot_table(index=["dataset", "method"], columns="k", values="rmse").round(3).to_string())
    k0 = summ[summ.k == 0][["dataset", "method", "rmse", "rmse_lo", "rmse_hi", "bias", "unconservative", "cover90",
                            "minus_ref", "minus_ref_lo", "minus_ref_hi"]]
    print(k0.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lopo")
