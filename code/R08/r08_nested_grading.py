"""R08 nested grading check (post hoc robustness; written 2026-10-26 before running).

Question: the repair rule releases the terms that the within-programme equivalence test grades
CONTRADICTED on the WHOLE database. Would the same terms be released if the grading were
repeated inside each training library, i.e. without the held-out programme?

A. Whole database: one-sided p-value of every contradiction test and its Holm adjustment within
   each anchor (family = all asserted terms of the anchor). The R07 grading (r07_tost.py) reports
   the unadjusted decision; its docstring mentions Holm but the adjustment is not applied.
B. Leave-one-programme-out libraries: for every target programme (>= 6 tests), drop it, refit the
   joint Mundlak signature (r06_numbers.signature, identical to the archived fits) and re-grade
   every asserted term with the same decision rule (r07_tost.verdict). Report, per term, the share
   of libraries in which it is graded CONTRADICTED (unadjusted and Holm-adjusted).
C. Sub-libraries of S = 6, 12, 24 programmes, drawn exactly as in the library-size experiment
   (rng = default_rng([b, S, crc32(target)]), b = 0..19, target excluded): same shares.
The coded exponent theta0 of each term is held at its whole-database value (it is the anchor's
assertion, not an outcome). A failed or singular fit counts as NOT-EVALUABLE.

Outputs: ../outputs/nested_wholedb.csv, nested_lopo.csv, nested_sub.csv
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
R08 = HERE.parent
ROUNDS = HERE.parents[1]
sys.path[:0] = [str(HERE),
                str(ROUNDS / "R07_merged_protocol_2026-10-22" / "scripts"),
                str(ROUNDS / "R06_mechanism_pivot_2026-10-22" / "scripts"),
                str(ROUNDS / "R05_consolidation_2026-10-21" / "scripts"),
                str(ROUNDS / "R01_direction_and_pilot_2026-10-20" / "scripts")]
import r06_rungs as R  # noqa: E402
import r06_numbers as N  # noqa: E402
import r07_tost as V  # noqa: E402

OUT = R08 / "outputs"
TOST = ROUNDS / "R07_merged_protocol_2026-10-22" / "outputs" / "tost_verdicts.csv"
ANCHORS = [("FRP", "K1"), ("FRP", "K2"), ("SFRC", "K1"), ("SFRC", "K2"),
           ("CARB", "K1"), ("CARB", "K2"), ("CARB", "K3"),
           ("DEEP", "BEAM"), ("DEEP", "ARCH"), ("DEEP", "ARCHS"), ("DEEP", "ARCH99")]
S_LEVELS = (6, 12, 24)
NB = 20


def contra_p(theta0, d, se):
    """One-sided p-value of the contradiction test used by r07_tost.verdict."""
    if not (np.isfinite(d) and np.isfinite(se) and se > 0):
        return 1.0
    if abs(theta0) > 1e-9:
        z = (theta0 + d) / se
        return float(norm.cdf(z)) if theta0 > 0 else float(norm.sf(z))
    return float(norm.sf((abs(d) - 0.10) / se))


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def grade(df, coded, free, theta):
    """Verdicts (unadjusted), contradiction p and Holm p for every coded term of one library."""
    try:
        sg = N.signature(df, coded, free)
        dW, se = sg["dW"], sg["dW_se"]
    except Exception:  # noqa: BLE001
        dW = pd.Series(np.nan, index=coded)
        se = pd.Series(np.nan, index=coded)
    rows = []
    for f in coded:
        d, s = float(dW[f]), float(se[f])
        if np.isfinite(d) and np.isfinite(s) and s > 0:
            v = V.verdict(SimpleNamespace(anchor_implied=theta[f], dW=d, dW_se=s))
        else:
            v = "NOT-EVALUABLE"
        rows.append(dict(feature=f, dW=d, dW_se=s, verdict=v, p=contra_p(theta[f], d, s)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    t["contra"] = t.verdict == "CONTRADICTED"
    t["contra_holm"] = t.contra & (t.p_holm <= 0.05)
    return t


def theta_of(tost, dom, rung):
    g = tost[(tost.domain == dom) & (tost.rung == rung)]
    return dict(zip(g.feature, g.anchor_implied))


def job_sub(dom, rung, target, theta):
    df, coded, free = R.load(dom, rung)
    others = np.sort(df.loc[df.source != target, "source"].unique())
    out = []
    for S in S_LEVELS:
        for b in range(NB):
            rng = np.random.default_rng([b, S, zlib.crc32(str(target).encode("utf-8"))])
            sub = rng.choice(others, size=min(S, len(others)), replace=False)
            t = grade(df[df.source.isin(sub)].reset_index(drop=True), coded, free, theta)
            t["S"], t["subset"], t["target"] = S, b, target
            out.append(t)
    r = pd.concat(out, ignore_index=True)
    r["domain"], r["rung"] = dom, rung
    return r


def main():
    tost = pd.read_csv(TOST)
    whole, lopo, jobs = [], [], []
    for dom, rung in ANCHORS:
        df, coded, free = R.load(dom, rung)
        theta = theta_of(tost, dom, rung)
        t = grade(df, coded, free, theta)
        ref = tost[(tost.domain == dom) & (tost.rung == rung)].set_index("feature")
        assert np.allclose(t.set_index("feature").dW, ref.dW.reindex(t.feature), atol=1e-6), (dom, rung)
        assert (t.set_index("feature").verdict == ref.verdict_tost.reindex(t.feature)).all(), (dom, rung)
        t["domain"], t["rung"] = dom, rung
        whole.append(t)
        n_s = df.groupby("source").size()
        for tg in sorted(n_s[n_s >= 6].index):
            u = grade(df[df.source != tg].reset_index(drop=True), coded, free, theta)
            u["domain"], u["rung"], u["target"] = dom, rung, tg
            lopo.append(u)
            jobs.append(delayed(job_sub)(dom, rung, tg, theta))
        print(f"{dom} {rung}: whole-database grading reproduced; LOPO done", flush=True)
    W = pd.concat(whole, ignore_index=True)
    W.to_csv(OUT / "nested_wholedb.csv", index=False)
    Lr = pd.concat(lopo, ignore_index=True)
    sub = pd.concat(Parallel(n_jobs=12, backend="loky", batch_size=1)(jobs), ignore_index=True)

    key = ["domain", "rung", "feature"]
    L = Lr.groupby(key).agg(libraries=("contra", "size"), share_contra=("contra", "mean"),
                            share_contra_holm=("contra_holm", "mean"),
                            share_not_evaluable=("verdict", lambda v: float((v == "NOT-EVALUABLE").mean()))
                            ).reset_index()
    L = L.merge(W[key + ["verdict", "p", "p_holm"]].rename(columns={"verdict": "verdict_wholedb",
                                                                    "p": "p_wholedb",
                                                                    "p_holm": "p_holm_wholedb"}), on=key)
    L.to_csv(OUT / "nested_lopo.csv", index=False)
    Sm = sub.groupby(key + ["S"]).agg(libraries=("contra", "size"), share_contra=("contra", "mean"),
                                       share_contra_holm=("contra_holm", "mean"),
                                       share_not_evaluable=("verdict",
                                                            lambda v: float((v == "NOT-EVALUABLE").mean()))
                                       ).reset_index()
    Sm = Sm.merge(W[key + ["verdict"]].rename(columns={"verdict": "verdict_wholedb"}), on=key)
    Sm.to_csv(OUT / "nested_sub.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== A. whole database: contradicted terms, raw and Holm-adjusted p ==")
    print(W[W.contra][["domain", "rung", "feature", "dW", "dW_se", "p", "p_holm", "contra_holm"]]
          .round(4).to_string(index=False))
    print("\n== B. LOPO libraries: terms ever graded contradicted ==")
    print(L[(L.share_contra > 0) | (L.verdict_wholedb == "CONTRADICTED")].round(3).to_string(index=False))
    print("\n== C. sub-libraries: contradicted-term trigger rates ==")
    piv = Sm[(Sm.verdict_wholedb == "CONTRADICTED")].pivot_table(
        index=["domain", "rung", "feature"], columns="S", values=["share_contra", "share_contra_holm"])
    print(piv.round(3).to_string())
    other = Sm[Sm.verdict_wholedb != "CONTRADICTED"]
    print("\nfalse-trigger rate (terms not contradicted on the whole database), mean over terms by S:")
    print(other.groupby("S")[["share_contra", "share_contra_holm"]].mean().round(4).to_string())
    print("max over terms by S:")
    print(other.groupby("S")[["share_contra", "share_contra_holm"]].max().round(4).to_string())


if __name__ == "__main__":
    main()
