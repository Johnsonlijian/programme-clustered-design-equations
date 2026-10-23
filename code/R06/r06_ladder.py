"""R06 mechanism-fidelity ladder: the value of a design equation as a function of its
mechanistic content.

  python -X utf8 r06_ladder.py theory      # pre-registration S* per (domain, rung); appends
                                          # only rungs missing from ladder_theory_prereg.csv
  python -X utf8 r06_ladder.py scarcity    # sub-library experiment per (domain, rung)
  python -X utf8 r06_ladder.py crossover   # observed vs predicted crossover S*
  python -X utf8 r06_ladder.py extrap      # between-programme size extrapolation (FRP, SFRC)
  python -X utf8 r06_ladder.py within      # within-programme size extrapolation
  python -X utf8 r06_ladder.py all         # theory -> scarcity -> crossover -> extrap -> within
  python -X utf8 r06_ladder.py scarcity_deep   # DEEP-only pass, appended to the shared files
  python -X utf8 r06_ladder.py within_deep     # DEEP within pass (ARCH, ARCHS), appended
  python -X utf8 r06_ladder.py <mode> --smoke  # 1 target / 1 S / 2 subsets into outputs/smoke/

Every rung of a domain is built on identical rows (r06_rungs.load), so the anchor-free
DATA-RE arm must be identical across rungs; the runner asserts this.

Outputs (outputs/): ladder_theory_prereg.csv, ladder_per_draw.csv.gz, ladder_summary.csv,
ladder_paired.csv, ladder_theory.csv, ladder_extrap_per_draw.csv.gz,
ladder_extrap_summary.csv, ladder_extrap_paired.csv, ladder_within_per_draw.csv.gz,
ladder_within_summary.csv, ladder_within_paired.csv, ladder_within_targets.csv,
log_<mode>.txt
"""
from __future__ import annotations

import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),
                str(HERE.parents[1] / "R05_consolidation_2026-10-21" / "scripts"),
                str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import r06_rungs as R  # noqa: E402
import r05_theory as T  # noqa: E402
import kag_data_r05 as D  # noqa: E402
from kag_bench_r05 import run_programme, summarise  # noqa: E402
from r05_scarcity import boot_summary  # noqa: E402

OUT = HERE.parent / "outputs"
SMOKE = OUT / "smoke"
ARMS = ("KA-FIXED", "KA-RE", "DATA-RE", "PAKA-S")
PAIR_ARMS = ("KA-FIXED", "KA-RE", "PAKA-S")
S_LEVELS = (3, 6, 12, 24, None)
B = 20
# all datasets iterated by theory/scarcity/crossover: primary domains + sensitivity
ALL_DOMAINS = dict(R.RUNGS, **R.SENSITIVITY)
# mechanism pivots per domain: (lower-fidelity rung, higher-fidelity rung) pairs
RUNG_PAIRS = {"FRP": [("K1", "K2")], "SFRC": [("K1", "K2")],
              "CARB": [("K1", "K2"), ("K1", "K3"), ("K2", "K3")],
              "DEEP": [("BEAM", "ARCH")], "DEEP-WB": [("BEAM", "ARCH")]}
WITHIN = {"FRP": R.RUNGS["FRP"], "SFRC": R.RUNGS["SFRC"], "DEEP": ("ARCH", "ARCHS")}
WITHIN_PAIRS = dict(RUNG_PAIRS, DEEP=[("ARCH", "ARCHS")],
                    **{"DEEP-WB": [("ARCH", "ARCHS")]}, CARB=[])


def wpath(p: Path) -> str:
    return D.long_path(p)


def _canon_feats(name, coded, free):
    """Canonical feature order across a domain's rungs. DATA-RE's design is
    column_stack(coded, free); the FastAnchoredLMM solution is column-order sensitive at
    ~1e-6 (observed up to 3e-4 for CARB K3), so ordering every rung's features by the
    first rung's coded+free order keeps DATA-RE column-identical across rungs. The
    coded/free split per rung is preserved."""
    if name == "CARB":
        canon = ["x_co2", "x_cao", "x_wb"]
    elif name.startswith("DEEP"):
        canon = list(R.DEEP_FEATURES) + (["ln_wb"] if name == "DEEP-WB" else [])
    else:
        s = D.spec(name)
        canon = list(s["coded"]) + list(s["free"])
    canon = [f for f in canon if f in set(coded) | set(free)]
    canon += [f for f in list(coded) + list(free) if f not in canon]
    return [f for f in canon if f in coded], [f for f in canon if f in free]


def sc_task(name, rung, target, S, b):
    """r05_scarcity.sc_task restricted to the LMM arms, on a rung's frame."""
    df, coded, free = R.load(name, rung)
    coded, free = _canon_feats(name, coded, free)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    subset = others
    if S is not None:
        rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
        subset = rng.choice(others, size=min(S, len(others)), replace=False)
    rows, info = run_programme(df, coded, free, target, ARMS, ks=(0, 3), n_draws=10, seed=b,
                               train_sources=subset)
    rows["S"] = "all" if S is None else str(S)
    rows["dataset"], rows["rung"] = name, rung
    rows["subset"], rows["n_train"] = b, info["_train"]["n"]
    return rows


def check_datare(per_draw, pair_cols):
    """DATA-RE ignores the anchor, so across rungs it must be identical row for row."""
    d = per_draw[per_draw.method == "DATA-RE"]
    key = [c for c in pair_cols if c != "dataset"]
    md_all = 0.0
    for ds, blk in d.groupby("dataset"):
        tab = blk.pivot_table(index=key, columns="rung", values="rmse")
        assert len(tab.columns) == blk.rung.nunique(), f"{ds}: rung coverage uneven"
        assert tab.notna().all().all(), f"{ds}: unpaired DATA-RE rows across rungs"
        diff = tab.sub(tab.iloc[:, 0], axis=0)
        md = float(np.abs(diff.to_numpy()).max())
        md_all = max(md_all, md)
        print(f"DATA-RE identity {ds}: max |rung_i - rung_0| = {md:.3e} "
              f"over {len(tab)}x{tab.shape[1]} cells", flush=True)
        assert md < 1e-9, f"{ds}: DATA-RE differs across rungs; stop and report"
    return md_all


def rung_paired(per_draw, group_cols, rung_pair, methods=PAIR_ARMS, values=("rmse",),
                pair_extra=(), n_boot=2000, seed=0):
    """Rung A - rung B paired on identical (sub-library, draw): per-programme mean
    difference, then a 2000-programme-bootstrap interval."""
    ra, rb = rung_pair
    tag = f"{ra}_minus_{rb}"
    pair = ["dataset", "k", "source", "draw", "method"] + list(pair_extra)
    d = per_draw[per_draw.method.isin(methods)]
    a = d[d.rung == ra].set_index(pair)
    b = d[d.rung == rb].set_index(pair)
    diff = (a[list(values)] - b[list(values)]).dropna()
    per_prog = diff.groupby(group_cols + ["method", "source"]).mean().reset_index()
    rng = np.random.default_rng(seed)
    out = []
    for kv, blk in per_prog.groupby(group_cols + ["method"]):
        kv = kv if isinstance(kv, tuple) else (kv,)
        rec = dict(zip(group_cols + ["method"], kv), programmes=blk.source.nunique())
        for val in values:
            v = blk[val].to_numpy()
            boots = rng.integers(0, len(v), size=(n_boot, len(v)))
            lo, hi = np.percentile(np.mean(v[boots], 1), [2.5, 97.5])
            if val == "rmse":
                rec[f"diff_{tag}"], rec["lo"], rec["hi"] = float(np.mean(v)), lo, hi
            else:
                rec[f"{val}_{tag}"] = float(np.mean(v))
                rec[f"{val}_{tag}_lo"], rec[f"{val}_{tag}_hi"] = lo, hi
        out.append(rec)
    return pd.DataFrame(out)


def paired_all(per_draw, group_cols, pairs, values=("rmse",), pair_extra=()):
    """Rung-pair differences for every (domain, rung-pair) present in per_draw."""
    out = []
    for ds, pp in pairs.items():
        sub = per_draw[per_draw.dataset == ds]
        if len(sub) == 0:
            continue
        for ra, rb in pp:
            if ra not in set(sub.rung) or rb not in set(sub.rung):
                continue
            out.append(rung_paired(sub, group_cols, (ra, rb), values=values,
                                   pair_extra=pair_extra))
    return pd.concat(out, ignore_index=True)


def prog_boot(df, keys, col, n_boot=2000, seed=0):
    """Mean of col over `source` rows per group with a programme-bootstrap interval."""
    rng = np.random.default_rng(seed)
    recs = []
    for kv, blk in df.groupby(keys):
        kv = kv if isinstance(kv, tuple) else (kv,)
        v = blk[col].to_numpy(float)
        boots = rng.integers(0, len(v), size=(n_boot, len(v)))
        lo, hi = np.percentile(np.nanmean(v[boots], 1), [2.5, 97.5])
        recs.append(dict(zip(keys, kv), programmes=len(v),
                         **{col: float(np.nanmean(v)), col + "_lo": lo, col + "_hi": hi}))
    return pd.DataFrame(recs)


def mode_theory(smoke=False):
    """Pre-registration table. Rows already in ladder_theory_prereg.csv are kept
    verbatim; only missing (domain, rung) rows are computed and appended."""
    out_dir = SMOKE if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "ladder_theory_prereg.csv"
    existing = pd.read_csv(wpath(f)) if (f.exists() and not smoke) else None
    have = set(zip(existing.domain, existing.rung)) if existing is not None else set()
    rungs = {"FRP": ("K1",)} if smoke else ALL_DOMAINS
    n_boot = 20 if smoke else 200
    rows = []
    for name, rr in rungs.items():
        for rung in rr:
            if (name, rung) in have:
                continue
            df, coded, free = R.load(name, rung)
            th = T.inputs(df, coded, free)
            bs = T.boot_pred(df, coded, free, n_boot=n_boot)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(domain=name, rung=rung,
                             **{k: th[k] for k in ("p", "c", "tau", "sigma", "B_raw",
                                                   "B_noise", "B", "S_star")},
                             S_star_lo=lo, S_star_hi=hi,
                             dW_d=th.get("dW_d"), dB_d=th.get("dB_d"),
                             dW_cao=th.get("dW_cao"), dB_cao=th.get("dB_cao"),
                             written_utc=datetime.now(timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ")))
            print(f"theory {name} {rung}: S*={th['S_star']:.2f} [{lo:.1f},{hi:.1f}]", flush=True)
    if not rows:
        print(f"theory: nothing to add ({len(have)} rows already registered)")
        if existing is not None:
            pd.set_option("display.width", 250)
            print(existing.round(4).to_string(index=False))
        return
    t = pd.DataFrame(rows)
    if existing is not None:
        t = pd.concat([existing, t], ignore_index=True)
    t.to_csv(wpath(f), index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))


def mode_scarcity(smoke=False, domains=None, append=False):
    out_dir = SMOKE if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = out_dir / "ladder_per_draw.csv.gz"
    have = set()
    if append and f.exists() and not smoke:
        _have = pd.read_csv(wpath(f), usecols=["dataset", "rung"])
        have = set(zip(_have.dataset, _have.rung))
    frames = []
    for name in (domains or ALL_DOMAINS):
        for rung in ALL_DOMAINS[name]:
            if (name, rung) in have:
                print(f"scarcity {name} {rung}: already in per_draw, skipped", flush=True)
                continue
            df, coded, free = R.load(name, rung)
            n_s = df.groupby("source").size()
            targets = sorted(n_s[n_s >= 6].index)
            s_levels = (3,) if smoke else S_LEVELS
            n_b = 2 if smoke else B
            if smoke:
                targets = targets[:1]
            jobs = [delayed(sc_task)(name, rung, t, S, b) for t in targets for S in s_levels
                    for b in (range(n_b) if S is not None else [0])]
            res = Parallel(n_jobs=12, backend="loky", batch_size=2)(jobs)
            frames.append(pd.concat(res, ignore_index=True))
            print(f"scarcity {name} {rung}: {len(jobs)} tasks at {time.time() - t0:.0f}s",
                  flush=True)
    if append and f.exists() and not smoke:
        old = pd.read_csv(wpath(f))
        frames.insert(0, old)
    if not frames:
        raise RuntimeError("scarcity: nothing computed and no existing per_draw")
    per_draw = pd.concat(frames, ignore_index=True)
    if append:
        key = ["dataset", "rung", "S", "subset", "source", "k", "draw", "method"]
        per_draw = per_draw.drop_duplicates(key)
    per_draw.to_csv(wpath(f), index=False)
    check_datare(per_draw, ["dataset", "S", "k", "source", "subset", "draw"])
    per_prog = (per_draw.groupby(["dataset", "rung", "S", "k", "method", "source"],
                                 as_index=False)
                .agg(rmse=("rmse", "mean"), n_train=("n_train", "mean")))
    summ = boot_summary(per_prog, ["dataset", "rung", "S", "k"],
                        [("KA-FIXED", "DATA-RE"), ("KA-RE", "DATA-RE"), ("PAKA-S", "DATA-RE")])
    ntr = per_prog.groupby(["dataset", "rung", "S"]).n_train.mean().rename("n_train").reset_index()
    summ = summ.merge(ntr, on=["dataset", "rung", "S"], how="left")
    order = {"3": 0, "6": 1, "12": 2, "24": 3, "all": 4}
    summ["S_order"] = summ.S.map(order)
    summ = summ.sort_values(["dataset", "rung", "k", "S_order", "method"])
    summ.to_csv(wpath(out_dir / "ladder_summary.csv"), index=False)
    paired = paired_all(per_draw, ["dataset", "S", "k"], RUNG_PAIRS, pair_extra=("S", "subset"))
    paired.to_csv(wpath(out_dir / "ladder_paired.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.pivot_table(index=["dataset", "rung", "k", "method"], columns="S",
                           values="rmse").round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


def observed_ladder(pp, name, rung, n_boot=2000, seed=0):
    """r05_theory.observed's logic restricted to one rung of the ladder table."""
    pp = pp[(pp.dataset == name) & (pp.rung == rung) & (pp.k == 0)
            & (pp.method.isin(["KA-FIXED", "DATA-RE"])) & (pp.S != "all")]
    per = pp.groupby(["S", "method", "source"]).rmse.mean().unstack("method")
    per["d"] = per["DATA-RE"] - per["KA-FIXED"]
    tab = per.d.unstack("S")
    s_cols = sorted(tab.columns, key=float)
    tab = tab[s_cols]
    S_vals = np.array([float(c) for c in s_cols])
    arr = tab.to_numpy()
    point = T.crossing(S_vals, arr.mean(axis=0))
    rng = np.random.default_rng(seed)
    bs = np.array([T.crossing(S_vals, arr[rng.integers(0, len(arr), len(arr))].mean(axis=0))
                   for _ in range(n_boot)])
    fin = bs[np.isfinite(bs)]
    lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size > 50 else (np.nan, np.nan))
    return point, lo, hi, float(np.mean(~np.isfinite(bs))), per


def mode_crossover(smoke=False):
    out_dir = SMOKE if smoke else OUT
    n_boot = 200 if smoke else 2000
    pp = pd.read_csv(wpath(out_dir / "ladder_per_draw.csv.gz"))
    th = pd.read_csv(wpath(out_dir / "ladder_theory_prereg.csv"))
    rows, diffs = [], {}
    for name, rr in ALL_DOMAINS.items():
        for rung in rr:
            if not ((pp.dataset == name) & (pp.rung == rung)).any():
                continue
            obs, olo, ohi, frac_none, per = observed_ladder(pp, name, rung, n_boot=n_boot)
            rows.append(dict(domain=name, rung=rung, S_obs=obs, S_obs_lo=olo,
                             S_obs_hi=ohi, obs_boot_no_crossing=frac_none))
            diffs[(name, rung)] = -(per["d"].unstack("S")).mean()
    t = th.merge(pd.DataFrame(rows), on=["domain", "rung"], how="left")
    t["ratio"] = t.S_star / t.S_obs
    t.to_csv(wpath(out_dir / "ladder_theory.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print("mean KA-FIXED minus DATA-RE RMSE by S (negative = anchor better):")
    print(pd.DataFrame(diffs).T.round(3).to_string())


def ex_task(name, rung, target, train):
    df, coded, free = R.load(name, rung)
    coded, free = _canon_feats(name, coded, free)
    rows, info = run_programme(df, coded, free, target, ARMS, ks=(0, 1, 3), n_draws=40,
                               seed=0, train_sources=train)
    rows["rung"] = rung
    return rows, info


def mode_extrap(smoke=False):
    out_dir = SMOKE if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = []
    for name in ("FRP", "SFRC"):
        for rung in R.RUNGS[name]:
            df, coded, free = R.load(name, rung)
            prog = df.groupby("source").agg(n=("d", "size"), d_mean=("d", "mean"))
            thr = prog.d_mean.quantile(0.75)
            train = list(prog.index[prog.d_mean < thr])
            targets = sorted(prog.index[(prog.d_mean >= thr) & (prog.n >= 4)])
            if smoke:
                targets = targets[:1]
            res = Parallel(n_jobs=min(12, len(targets)), backend="loky")(
                delayed(ex_task)(name, rung, t, train) for t in targets)
            for rows, _info in res:
                rows.insert(0, "dataset", name)
                frames.append(rows)
            print(f"extrap {name} {rung}: {len(targets)} targets, thr={thr:.0f} mm, "
                  f"{len(train)} train programmes at {time.time() - t0:.0f}s", flush=True)
    per_draw = pd.concat(frames, ignore_index=True)
    per_draw.to_csv(wpath(out_dir / "ladder_extrap_per_draw.csv.gz"), index=False)
    check_datare(per_draw, ["dataset", "k", "source", "draw"])
    per_prog, summ = summarise(per_draw, keys=("dataset", "rung"))
    summ.to_csv(wpath(out_dir / "ladder_extrap_summary.csv"), index=False)
    paired = paired_all(per_draw, ["dataset", "k"], RUNG_PAIRS)
    paired.to_csv(wpath(out_dir / "ladder_extrap_paired.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.pivot_table(index=["dataset", "rung", "method"], columns="k",
                           values="rmse").round(3).to_string())
    print(paired.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


def wi_task(name, rung, target):
    df, coded, free = R.load(name, rung)
    coded, free = _canon_feats(name, coded, free)
    rows, info = run_programme(df, coded, free, target, ARMS, ks=(1, 2, 3), seed=0,
                             order_by="d")
    rows["rung"] = rung
    return rows, info


def mode_within(smoke=False, domains=None, append=False):
    out_dir = SMOKE if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = out_dir / "ladder_within_per_draw.csv.gz"
    ftab = out_dir / "ladder_within_targets.csv"
    names = domains or WITHIN
    have = set()
    if append and f.exists() and not smoke:
        old_pd = pd.read_csv(wpath(f), usecols=["dataset", "rung"])
        have = set(zip(old_pd.dataset, old_pd.rung))
    frames, tabs = [], []
    for name in names:
        rungs = WITHIN[name]
        df0, _, _ = R.load(name, rungs[0])
        pg = (df0.groupby("source")
              .agg(n=("d", "size"), n_d=("d", "nunique"),
                   ln_d_min=("ln_d", "min"), ln_d_max=("ln_d", "max")))
        targets = sorted(pg.index[(pg.n >= 4) & (pg.n_d >= 3)])
        tabs.append(pg.loc[targets].reset_index().assign(dataset=name))
        if smoke:
            targets = targets[:1]
        for rung in rungs:
            if (name, rung) in have:
                print(f"within {name} {rung}: already in per_draw, skipped", flush=True)
                continue
            res = Parallel(n_jobs=min(12, len(targets)), backend="loky")(
                delayed(wi_task)(name, rung, t) for t in targets)
            for rows, _info in res:
                rows.insert(0, "dataset", name)
                frames.append(rows)
            print(f"within {name} {rung}: {len(targets)} targets at {time.time() - t0:.0f}s",
                  flush=True)
    if append and f.exists() and not smoke:
        frames.insert(0, pd.read_csv(wpath(f)))
    if not frames:
        raise RuntimeError("within: nothing computed and no existing per_draw")
    per_draw = pd.concat(frames, ignore_index=True)
    if append:
        per_draw = per_draw.drop_duplicates(["dataset", "rung", "k", "source", "draw",
                                             "method"])
    per_draw.to_csv(wpath(f), index=False)
    new_tabs = pd.concat(tabs, ignore_index=True)
    if append and ftab.exists() and not smoke:
        old_tabs = pd.read_csv(wpath(ftab))
        new_tabs = (pd.concat([old_tabs, new_tabs], ignore_index=True)
                    .drop_duplicates(["dataset", "source"]))
    new_tabs.to_csv(wpath(ftab), index=False)
    check_datare(per_draw, ["dataset", "k", "source", "draw"])
    per_prog, summ = summarise(per_draw, keys=("dataset", "rung"))
    npd = (per_draw.groupby(["dataset", "rung", "k", "method", "source"], as_index=False)
           .agg(n_pred=("n_pred", "mean")))
    summ = summ.merge(prog_boot(npd, ["dataset", "rung", "k", "method"], "n_pred")
                      .drop(columns="programmes"),
                      on=["dataset", "rung", "k", "method"], how="left")
    summ.to_csv(wpath(out_dir / "ladder_within_summary.csv"), index=False)
    paired = paired_all(per_draw, ["dataset", "k"], WITHIN_PAIRS, values=("rmse", "bias"))
    paired.to_csv(wpath(out_dir / "ladder_within_paired.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["dataset", "rung", "k", "method", "programmes", "rmse", "rmse_lo", "rmse_hi",
            "bias", "bias_lo", "bias_hi", "unconservative", "n_pred"]
    print(summ[cols].round(3).to_string(index=False))
    print(paired.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


def ch_task(name, rung, T, target, library):
    """One prequential task: library = programmes with year <= T; evaluate target."""
    df, coded, free = R.load(name, rung)
    coded, free = _canon_feats(name, coded, free)
    rows, info = run_programme(df, coded, free, target, ARMS, ks=(0,), n_draws=1, seed=0,
                               train_sources=library)
    rows["dataset"], rows["rung"] = name, rung
    rows["T"], rows["S"], rows["n_train"] = T, len(library), info["_train"]["n"]
    return rows


def mode_chrono(smoke=False):
    """Prequential libraries (FRP, SFRC, K1/K2, k=0): for each cutoff year T, train on
    programmes published by T and evaluate every later programme with >= 6 tests."""
    out_dir = SMOKE if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs, meta = [], []
    for name in ("FRP", "SFRC"):
        for rung in R.RUNGS[name]:
            df, coded, free = R.load(name, rung)
            yr = df.groupby("source").year.first()
            n_s = df.groupby("source").size()
            for cutoff in sorted(yr.dropna().unique()):
                library = sorted(yr.index[yr <= cutoff])
                targets = sorted(yr.index[(yr > cutoff) & (n_s >= 6)])
                if len(library) < 3 or not targets:
                    continue
                for t in targets:
                    jobs.append(delayed(ch_task)(name, rung, cutoff, t, library))
                meta.append(dict(dataset=name, rung=rung, T=cutoff, S=len(library),
                                 n_targets=len(targets)))
                if smoke and len(jobs) >= 2:
                    break
            if smoke and jobs:
                break
        if smoke and jobs:
            break
    if smoke:
        jobs = jobs[:2]
    res = Parallel(n_jobs=min(12, max(len(jobs), 1)), backend="loky")(jobs)
    per = pd.concat(res, ignore_index=True)
    per.to_csv(wpath(out_dir / "ladder_chrono_per_target.csv"), index=False)
    print(f"chrono: {len(jobs)} tasks, {len(meta)} cutoffs at {time.time() - t0:.0f}s",
          flush=True)
    # diff curve per target, pooled into S bins; programme bootstrap over targets
    pv = (per[per.method.isin(["KA-FIXED", "DATA-RE"])]
          .pivot_table(index=["dataset", "rung", "T", "S", "source"], columns="method",
                       values="rmse"))
    pv["diff"] = pv["KA-FIXED"] - pv["DATA-RE"]
    pv = pv.reset_index()
    bins = [(3, 5), (6, 11), (12, 23), (24, np.inf)]
    lab = {i: (f"{a}-{b}" if np.isfinite(b) else ">=24") for i, (a, b) in enumerate(bins)}
    pv["s_bin"] = np.nan
    for i, (a, b) in enumerate(bins):
        pv.loc[(pv.S >= a) & (pv.S <= b), "s_bin"] = i
    rng = np.random.default_rng(0)
    rows = []
    for (ds, rung, sb), blk in pv.dropna(subset=["s_bin"]).groupby(["dataset", "rung", "s_bin"]):
        v = blk["diff"].to_numpy()
        boots = rng.integers(0, len(v), size=(2000, len(v)))
        lo, hi = np.percentile(np.mean(v[boots], 1), [2.5, 97.5])
        rows.append(dict(dataset=ds, rung=rung, s_bin=lab[int(sb)], bin_i=int(sb),
                         s_lo=int(blk.S.min()), s_hi=int(blk.S.max()),
                         s_centre=float(np.exp(np.log(blk.S).mean())),
                         targets=len(blk), rmse_ka_fixed=blk["KA-FIXED"].mean(),
                         rmse_data_re=blk["DATA-RE"].mean(),
                         diff_ka_fixed_minus_data_re=float(np.mean(v)), lo=lo, hi=hi))
    summ = pd.DataFrame(rows)
    crow = []
    for (ds, rung), blk in summ.groupby(["dataset", "rung"]):
        blk = blk.sort_values("s_centre")
        S_v = blk.s_centre.to_numpy()
        # crossing() detects the scarcity-lab convention: DATA-RE - KA-FIXED going + -> -
        d_v = -blk.diff_ka_fixed_minus_data_re.to_numpy()
        point = T.crossing(S_v, d_v)
        # bootstrap: resample target programmes within each bin, recompute bin means
        pools = [pv[(pv.dataset == ds) & (pv.rung == rung) & (pv.s_bin == i)]["diff"]
                 .to_numpy() for i in blk.bin_i]
        bs = []
        if all(len(v) for v in pools):
            for _ in range(2000):
                bs.append(T.crossing(S_v, -np.array(
                    [v[rng.integers(0, len(v), len(v))].mean() for v in pools])))
        bs = np.array(bs)
        fin = bs[np.isfinite(bs)]
        lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size > 50 else (np.nan, np.nan))
        crow.append(dict(dataset=ds, rung=rung, S_obs=point, S_obs_lo=lo, S_obs_hi=hi,
                         obs_boot_no_crossing=float(np.mean(~np.isfinite(bs))) if bs.size
                         else np.nan))
    cross = pd.DataFrame(crow)
    summ.to_csv(wpath(out_dir / "ladder_chrono_summary.csv"), index=False)
    cross.to_csv(wpath(out_dir / "ladder_chrono_crossover.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(cross.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


def mode_clusters(smoke=False):
    """Theory inputs with laboratory grouping (r05_labs.lab_of) instead of publications."""
    from r05_labs import lab_of
    out_dir = SMOKE if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    rungs = {"FRP": ("K1",)} if smoke else {d: R.RUNGS[d] for d in ("FRP", "SFRC", "CARB")}
    n_boot = 20 if smoke else 200
    rows = []
    for name, rr in rungs.items():
        for rung in rr:
            df, coded, free = R.load(name, rung)
            d2 = df.copy()
            d2["source"] = df.source.map(lambda s: lab_of(s, name))
            th = T.inputs(d2, coded, free)
            bs = T.boot_pred(d2, coded, free, n_boot=n_boot)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(domain=name, rung=rung,
                             **{k: th[k] for k in ("p", "c", "tau", "sigma", "B_raw",
                                                   "B_noise", "B", "S_star")},
                             S_star_lo=lo, S_star_hi=hi,
                             dW_d=th.get("dW_d"), dB_d=th.get("dB_d"),
                             dW_cao=th.get("dW_cao"), dB_cao=th.get("dB_cao"),
                             grouping="lab", groups=th["programmes"],
                             written_utc=datetime.now(timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ")))
            print(f"clusters {name} {rung}: {th['programmes']} labs, "
                  f"S*={th['S_star']:.2f} [{lo:.1f},{hi:.1f}]", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(wpath(out_dir / "ladder_theory_labs.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))


SMOKE_MODES = dict(theory=mode_theory, scarcity=mode_scarcity, crossover=mode_crossover,
                   extrap=mode_extrap, within=mode_within, chrono=mode_chrono,
                   clusters=mode_clusters)


def mode_all(smoke=False):
    for mode in ("theory", "scarcity", "crossover", "extrap", "within", "chrono",
                 "clusters"):
        t0 = time.time()
        if smoke:
            SMOKE_MODES[mode](smoke=True)
            print(f"smoke {mode} done in {time.time() - t0:.0f}s", flush=True)
            continue
        log = OUT / f"log_{mode}.txt"
        with open(wpath(log), "w", encoding="utf-8") as fh:
            subprocess.run([sys.executable, "-X", "utf8", str(HERE / "r06_ladder.py"), mode],
                           stdout=fh, stderr=subprocess.STDOUT, check=True)
        print(f"{mode} done in {time.time() - t0:.0f}s -> {log}", flush=True)


MODES = dict(theory=mode_theory, crossover=mode_crossover, extrap=mode_extrap,
             chrono=mode_chrono, clusters=mode_clusters)


def main(mode):
    OUT.mkdir(exist_ok=True)
    if mode == "all" or mode == "smoke":
        mode_all(smoke=(mode == "smoke" or "--smoke" in sys.argv))
    elif "--smoke" in sys.argv:
        SMOKE_MODES[mode](smoke=True)
    elif mode == "scarcity":
        mode_scarcity(append=True)
    elif mode == "within":
        mode_within(append=True)
    elif mode == "scarcity_deep":
        mode_scarcity(domains=["DEEP"], append=True)
    elif mode == "within_deep":
        mode_within(domains=["DEEP"], append=True)
    elif mode == "deep":
        for m in ("theory", "scarcity_deep", "crossover", "within_deep"):
            t0 = time.time()
            log = OUT / f"log_{m}.txt"
            with open(wpath(log), "w", encoding="utf-8") as fh:
                subprocess.run([sys.executable, "-X", "utf8", str(HERE / "r06_ladder.py"), m],
                               stdout=fh, stderr=subprocess.STDOUT, check=True)
            print(f"{m} done in {time.time() - t0:.0f}s -> {log}", flush=True)
    else:
        MODES[mode]()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
