"""R02 experiment 2: dense scarcity curve + evidence-adaptive trust.

Repeats the R01 scarcity experiment on a denser grid of training-library sizes and
adds the R02 method arm:

  ANCHOR    raw design-code prediction, programme variance only
  KA-FIXED  code exponents fixed, global calibration + programme effect
  KA-RE     code exponents with the REML trust scale (R01 default)
  DATA-RE   no code, same random-intercept structure
  EAT       arm chosen by grouped cross-validation *inside the training library*
            (evidence-adaptive trust), then refitted on the whole library

Every arm is scored on a held-out programme at k = 0 and k = 3 shots. The k shots
enter all arms through the same Gaussian conditioning, so the comparison isolates
the base predictor and the trust it places in the code.

Outputs: outputs/scarcity2_per_draw.csv.gz, outputs/scarcity2_summary.csv,
         outputs/scarcity2_eat_choices.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

R01 = Path(__file__).resolve().parents[2] / "R01_direction_and_pilot_2026-10-20" / "scripts"
sys.path.insert(0, str(R01))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kag_data_r02 as K                                          # noqa: E402
from kag_models import oneway_components                          # noqa: E402
from kag_models_r02 import (ka_fit, predict_new_group, select_arm,  # noqa: E402
                            s_star_policy)

OUT = Path(__file__).resolve().parents[1] / "outputs"
S_LEVELS = (3, 6, 12, 24, None)
METHODS = ("ANCHOR", "KA-FIXED", "KA-RE", "DATA-RE", "EAT-T")
Z90 = 1.6448536269514722


def score(y, mu, var):
    res = y - mu
    sd = np.sqrt(var)
    return dict(rmse=float(np.sqrt(np.mean(res ** 2))), bias=float(np.mean(res)),
                unconservative=float(np.mean(res < 0)),
                cover90=float(np.mean(np.abs(res) <= Z90 * sd)),
                nlpd=float(np.mean(0.5 * np.log(2 * np.pi * var) + res ** 2 / (2 * var))))


def task(df, feats, target, S, b, draws, seed=0, include_cv=False):
    others = np.sort(df.loc[df.source != target, "source"].unique())
    if S is None:
        subset = others
    else:
        rng = np.random.default_rng([b, S, zlib.crc32(target.encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)

    tr = df[df.source.isin(subset)].reset_index(drop=True)
    te = df[df.source == target].reset_index(drop=True)
    g = tr.source.values
    cen = tr[feats].values.mean(0)
    Xtr, Xte = tr[feats].values - cen, te[feats].values - cen
    ytr, yte = tr.y.values, te.y.values
    mtr, mte = tr.m.values, te.m.values

    preds = {}          # name -> ("anchor", mu, var) or ("model", fitted)
    models = {}
    choice, choice_t, sstar = "", "", np.nan

    # ANCHOR: raw code, programme-level variance from moment components
    tau2, sig2, bias0 = oneway_components(ytr - mtr, g)
    preds["ANCHOR"] = ("anchor", mte, np.full(len(te), tau2 + sig2))

    # fixed / free / flat trust
    for name, lam in (("KA-FIXED", 0.0), ("KA-RE", None), ("DATA-RE", 10.0)):
        mdl = ka_fit(ytr, Xtr, g, offset=mtr, lam=lam, ml=False, n_starts=2,
                     seed=seed + zlib.crc32(name.encode("utf-8")) % 1000)
        preds[name] = ("model", mdl)
        models[name] = mdl

    # evidence-adaptive trust, cross-validated inside the training library
    if include_cv:
        try:
            best, _ = select_arm(ytr, Xtr, g, offset=mtr, seed=seed)
            preds["EAT"] = preds[best]
            choice = best
        except Exception:
            preds["EAT"] = preds["KA-RE"]
            choice = "KA-RE(fallback)"

    # theory-driven trust: cross over at the estimated S*
    try:
        pol, sstar = s_star_policy(Xtr, g, models["KA-FIXED"], models["KA-RE"])
        preds["EAT-T"] = preds[pol]
        choice_t = pol
    except Exception:
        preds["EAT-T"] = preds["KA-RE"]
        choice_t = "KA-RE(fallback)"

    rng = np.random.default_rng([seed, zlib.crc32(target.encode("utf-8")), S or 0, b])
    n = len(te)
    rows = []
    for k in (0, 3):
        if n - k < 3:
            continue
        drew = [np.arange(n)] if k == 0 else [rng.permutation(n) for _ in range(draws)]
        for di, perm in enumerate(drew):
            kk, tt = perm[:k], perm[k:]
            for name, spec in preds.items():
                if spec[0] == "model":
                    mu, var = predict_new_group(spec[1], Xte[tt], offset_t=mte[tt],
                                                X_k=Xte[kk], y_k=yte[kk], offset_k=mte[kk])
                else:
                    mu, var = spec[1][tt], spec[2][tt]
                rows.append(dict(source=target, n_source=n, k=k, draw=di, method=name,
                                 eat_choice=choice, eat_t_choice=choice_t, s_star_est=sstar,
                                 **score(yte[tt], mu, var)))
    blk = pd.DataFrame(rows)
    blk["S"] = len(subset) if S is not None else -1
    blk["S_label"] = str(S) if S is not None else "all"
    blk["subset"] = b
    blk["n_train"] = len(tr)
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cv", action="store_true", help="also run the cross-validated selector arm")
    a = ap.parse_args()
    t0 = time.time()
    frames = []
    for name in K.LOADERS:
        df = K.load(name)
        feats = K.FEATURES[name]
        n_s = df.groupby("source").size()
        targets = sorted(n_s[n_s >= 6].index)
        jobs = [(t, S, b) for t in targets for S in S_LEVELS
                for b in (range(a.B) if S is not None else [0])]
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(lambda job: task(df, feats, job[0], job[1], job[2],
                                               a.draws, include_cv=a.cv), jobs))
        blk = pd.concat(res, ignore_index=True)
        blk.insert(0, "dataset", name)
        frames.append(blk)
        print(f"{name}: {len(jobs)} fits at {time.time() - t0:.0f}s", flush=True)

    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(OUT / "scarcity2_per_draw.csv.gz", index=False)

    per_prog = (per_draw.groupby(["dataset", "S_label", "method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean"), unconservative=("unconservative", "mean"),
                     cover90=("cover90", "mean"), nlpd=("nlpd", "mean"), n_train=("n_train", "mean")))
    rng = np.random.default_rng(0)
    out = []
    for (ds, S, k), blk in per_prog.groupby(["dataset", "S_label", "k"]):
        progs = np.sort(blk.source.unique())
        piv = blk.pivot(index="source", columns="method", values="rmse").reindex(progs)
        boots = rng.integers(0, len(progs), size=(2000, len(progs)))
        for mname in piv.columns:
            v = piv[mname].values
            rec = dict(dataset=ds, S=S, k=k, method=mname, programmes=len(progs),
                       rmse=float(np.nanmean(v)))
            rec["rmse_lo"], rec["rmse_hi"] = np.percentile(
                np.nanmean(v[boots], 1), [2.5, 97.5])
            for ref in ("DATA-RE", "KA-RE"):
                if ref in piv and mname != ref:
                    d = piv[ref].values - v          # > 0: this arm better than ref
                    rec[f"gain_vs_{ref}"] = float(np.nanmean(d))
                    rec[f"gain_vs_{ref}_lo"], rec[f"gain_vs_{ref}_hi"] = np.percentile(
                        np.nanmean(d[boots], 1), [2.5, 97.5])
                    rec[f"win_share_vs_{ref}"] = float(np.nanmean(d > 0))
            out.append(rec)
    summ = pd.DataFrame(out)
    order = {"3": 0, "4": 1, "6": 2, "8": 3, "12": 4, "16": 5, "24": 6, "all": 7}
    summ["S_order"] = summ.S.map(order)
    summ = summ.sort_values(["dataset", "k", "S_order", "method"])
    summ.to_csv(OUT / "scarcity2_summary.csv", index=False)

    ch = (per_draw[per_draw.method == "EAT"]
          .groupby(["dataset", "S_label", "eat_choice"]).size().rename("n").reset_index())
    ch.to_csv(OUT / "scarcity2_eat_choices.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.pivot_table(index=["dataset", "k", "method"], columns="S_order",
                           values="rmse").round(3).to_string())
    print("\nEAT choices:")
    print(ch.pivot_table(index=["dataset", "S_label"], columns="eat_choice", values="n",
                         fill_value=0).to_string())
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
