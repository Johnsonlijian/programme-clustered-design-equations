"""R05 benchmark worker: leave-one-programme-out, k-shot, all arms.

Arms (all receive the k shots of the held-out programme through Gaussian
conditioning on a programme effect; nothing else about the target is used):
  ANCHOR    code as published (no fitting)
  KA-FIXED  code exponents imposed; global level, free exponents, programme effects learned
  KA-RE     code exponents as a prior; deviations shrunk with a REML trust scale (coded and
            free columns carry separate scales)
  DATA-RE   no code; same hierarchical power law with one shrinkage scale
  PAKA      programme-aware knowledge anchoring: chooses KA-FIXED / KA-RE / DATA-RE by
            leave-programme-out error on the *training* programmes with a one-standard-
            error rule toward more knowledge
  GBM       gradient boosting on the raw features (grouped inner tuning)
  GBM-RES   gradient boosting on the code residual
  GBM-ME    mixed-effects gradient boosting (programme intercept, MERF-type EM)
  GP-RES    Gaussian process on the code residual (Kennedy-O'Hagan-type discrepancy)
"""
from __future__ import annotations

import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GroupKFold, KFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "R01_direction_and_pilot_2026-10-20" / "scripts")]
from kag_models import ResidualVarianceEB, within_between  # noqa: E402
from kag_models_r05 import (GBMME, FastAnchoredLMM, joint_predictive, mvn_logpdf,  # noqa: E402
                            oneway_reml, stacking_weights)

warnings.filterwarnings("ignore")
Z90 = 1.6448536269514722
GBM_GRID = (
    dict(max_leaf_nodes=4, max_iter=300, learning_rate=0.05, min_samples_leaf=10),
    dict(max_leaf_nodes=8, max_iter=300, learning_rate=0.05, min_samples_leaf=15),
    dict(max_leaf_nodes=16, max_iter=200, learning_rate=0.05, min_samples_leaf=5),
    dict(max_leaf_nodes=8, max_iter=600, learning_rate=0.03, min_samples_leaf=10),
)
LMM_ARMS = ("KA-FIXED", "KA-RE", "DATA-RE")
ALL_ARMS = ("ANCHOR",) + LMM_ARMS + ("PAKA", "PAKA-S", "PAKA-S0", "PAKA-S6", "GBM", "GBM-RES", "GBM-ME", "GP-RES")


def folds(groups, n=5):
    ng = len(np.unique(groups))
    if ng >= 2:
        return GroupKFold(n_splits=min(n, ng))
    return KFold(n_splits=min(n, len(groups)), shuffle=True, random_state=0)


class Design:
    """Centred design blocks for one training library (centre from training only)."""

    def __init__(self, coded, free):
        self.coded, self.free = list(coded), list(free)

    def fit(self, df):
        self.c_cod = df[self.coded].values.mean(0) if self.coded else np.zeros(0)
        self.c_free = df[self.free].values.mean(0) if self.free else np.zeros(0)
        return self

    def blocks(self, df):
        n = len(df)
        Xc = df[self.coded].values - self.c_cod if self.coded else np.zeros((n, 0))
        Xf = df[self.free].values - self.c_free if self.free else np.zeros((n, 0))
        return Xc, Xf

    def arm_matrix(self, arm, df):
        Xc, Xf = self.blocks(df)
        if arm == "KA-FIXED":
            return Xf, [0] * Xf.shape[1]
        if arm == "KA-RE":
            return np.column_stack([Xc, Xf]), [0] * Xc.shape[1] + [1] * Xf.shape[1]
        if arm == "DATA-RE":
            X = np.column_stack([Xc, Xf])
            return X, [0] * X.shape[1]
        raise KeyError(arm)


def fit_lmm_arm(arm, tr, des, seed=0, n_starts=3):
    X, cg = des.arm_matrix(arm, tr)
    off = tr.m.values if arm != "DATA-RE" else None
    return FastAnchoredLMM(cg, n_starts=n_starts, seed=seed).fit(tr.y.values, X, tr.source.values, offset=off)


def lmm_predict(arm, mdl, des, te, kk, tt):
    X, _ = des.arm_matrix(arm, te)
    off = te.m.values if arm != "DATA-RE" else np.zeros(len(te))
    if len(kk) == 0:
        return mdl.predict_new_group(X[tt], offset_t=off[tt])
    return mdl.predict_new_group(X[tt], offset_t=off[tt], X_k=X[kk], y_k=te.y.values[kk], offset_k=off[kk])


def paka_select(tr, coded, free, seed=0, n_folds=4):
    """Leave-programme-out trust selection on the training library only."""
    g = tr.source.values
    ug = np.unique(g)
    if len(ug) < 3:
        return "KA-FIXED", {}
    err = {a: {} for a in LMM_ARMS}
    for itr, iva in GroupKFold(n_splits=min(n_folds, len(ug))).split(tr, groups=g):
        a_tr, a_va = tr.iloc[itr], tr.iloc[iva]
        des = Design(coded, free).fit(a_tr)
        for arm in LMM_ARMS:
            try:
                mdl = fit_lmm_arm(arm, a_tr, des, seed, n_starts=2)
                mu, _ = lmm_predict(arm, mdl, des, a_va, np.array([], int), np.arange(len(a_va)))
                sq = (a_va.y.values - mu) ** 2
                for pg in np.unique(a_va.source.values):
                    err[arm][pg] = float(np.mean(sq[a_va.source.values == pg]))
            except Exception:
                for pg in np.unique(a_va.source.values):
                    err[arm][pg] = np.inf
    common = [p for p in ug if all(np.isfinite(err[a].get(p, np.inf)) for a in LMM_ARMS)]
    if len(common) < 3:
        return "KA-FIXED", {}
    mse = {a: float(np.mean([err[a][p] for p in common])) for a in LMM_ARMS}
    best = min(mse, key=mse.get)
    chosen = best
    ladder = list(LMM_ARMS)
    for a in ladder[:ladder.index(best)][::-1]:          # walk toward more knowledge
        d = np.array([err[a][p] - err[best][p] for p in common])
        if d.mean() <= d.std(ddof=1) / np.sqrt(d.size):
            chosen = a
        else:
            break
    return chosen, mse


STACK_PRIOR = 3.0      # Dirichlet pseudo-count on KA-FIXED, in held-out-programme units


def paka_stack_weights(tr, coded, free, seed=0, max_folds=6, prior_strength=STACK_PRIOR):
    """Stacking weights over the trust ladder from leave-programme-out *joint* log
    densities of whole held-out training programmes (training library only). The code
    enters with a prior weight worth `prior_strength` held-out programmes, so a library
    of three programmes cannot overturn it on noise alone, while a library of twenty can."""
    L = paka_stack_logdens(tr, coded, free, seed, max_folds)
    if L is None:
        return np.array([1.0, 0.0, 0.0])
    return stacking_weights(L, prior=[prior_strength, 0.0, 0.0])


def paka_stack_logdens(tr, coded, free, seed=0, max_folds=6):
    """Held-out-programme joint log densities (programmes x ladder arms); None if < 3 programmes."""
    g = tr.source.values
    ug = np.unique(g)
    if len(ug) < 3:
        return None
    rows = []
    for itr, iva in GroupKFold(n_splits=min(max_folds, len(ug))).split(tr, groups=g):
        a_tr, a_va = tr.iloc[itr], tr.iloc[iva].reset_index(drop=True)
        des = Design(coded, free).fit(a_tr)
        mdls = {arm: fit_lmm_arm(arm, a_tr, des, seed, n_starts=2) for arm in LMM_ARMS}
        for pg in np.unique(a_va.source.values):
            blk = a_va[a_va.source == pg]
            row = []
            for arm in LMM_ARMS:
                X, _ = des.arm_matrix(arm, blk)
                off = blk.m.values if arm != "DATA-RE" else None
                mu, C = joint_predictive(mdls[arm], X, off)
                row.append(mvn_logpdf(blk.y.values, mu, C))
            rows.append(row)
    return np.array(rows)


def paka_stack_predict(w, fitted, des, te, kk, tt):
    """Mixture over the ladder; the k shots update the weights by their marginal
    likelihood under each arm (programme-specific trust) before conditioning."""
    w = np.asarray(w, float).copy()
    mus, vars_ = [], []
    if len(kk):
        lk = []
        for arm in LMM_ARMS:
            X, _ = des.arm_matrix(arm, te)
            off = te.m.values if arm != "DATA-RE" else None
            mu, C = joint_predictive(fitted[arm], X[kk], None if off is None else off[kk])
            lk.append(mvn_logpdf(te.y.values[kk], mu, C))
        lk = np.array(lk)
        w = w * np.exp(lk - lk.max())
        w = w / w.sum()
    for arm in LMM_ARMS:
        mu, var = lmm_predict(arm, fitted[arm], des, te, kk, tt)
        mus.append(mu)
        vars_.append(var)
    mus, vars_ = np.array(mus), np.array(vars_)
    mean = w @ mus
    var = w @ (vars_ + mus ** 2) - mean ** 2
    return mean, np.maximum(var, 1e-10)


def _gbm_nested(X, t, g, seed, grid):
    best = None
    for cfg in grid:
        oof = np.zeros_like(t)
        for itr, iva in folds(g).split(X, t, g):
            oof[iva] = HistGradientBoostingRegressor(random_state=seed, **cfg).fit(X[itr], t[itr]).predict(X[iva])
        mse = float(np.mean((t - oof) ** 2))
        if best is None or mse < best[0]:
            best = (mse, cfg, t - oof)
    return HistGradientBoostingRegressor(random_state=seed, **best[1]).fit(X, t), best[2], best[1]


def _gbmme_oof(X, y, g, cfg, seed, n_folds=3):
    oof = np.zeros_like(y)
    for itr, iva in folds(g, n_folds).split(X, y, g):
        oof[iva] = GBMME(cfg, n_iter=4, seed=seed).fit(X[itr], y[itr], g[itr]).predict(X[iva])
    return y - oof


def _gp(X, t, seed):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    k = ConstantKernel(0.05, (1e-4, 10.0)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(0.03, (1e-4, 1.0))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=seed).fit((X - mu) / sd, t)
    return lambda Q: gp.predict((Q - mu) / sd)


def score(y, mu, var):
    res = y - mu
    sd = np.sqrt(var)
    return dict(rmse=float(np.sqrt(np.mean(res ** 2))), bias=float(np.mean(res)),
                unconservative=float(np.mean(res < 0)), cover90=float(np.mean(np.abs(res) <= Z90 * sd)),
                width90=float(np.mean(2 * Z90 * sd)),
                nlpd=float(np.mean(0.5 * np.log(2 * np.pi * var) + res ** 2 / (2 * var))))


def run_programme(df, coded, free, target, arms=ALL_ARMS, ks=(0, 1, 2, 3, 5), n_draws=40, seed=0,
                  train_sources=None, gbm_grid=GBM_GRID, threads=1, order_by=None):
    """k-shot evaluation of one held-out programme. With order_by=None (default) the k
    conditioning tests are random draws; with order_by=<column> the k conditioning tests are
    the k smallest values of that column and the targets are the tests strictly above them
    (within-programme extrapolation, one deterministic draw per k)."""
    with threadpool_limits(threads):
        keep = (df.source != target) if train_sources is None else df.source.isin(train_sources)
        tr = df[keep & (df.source != target)].reset_index(drop=True)
        te = df[df.source == target].reset_index(drop=True)
        g = tr.source.values
        feats = list(coded) + list(free)
        Phi_tr, Phi_te = tr[feats].values, te[feats].values
        des = Design(coded, free).fit(tr)
        pred, info = {}, {"_train": dict(n=len(tr), programmes=int(len(np.unique(g))))}

        def eb(name, mu_te, oof_res):
            c, tau, sig = oneway_reml(oof_res, g)
            e = ResidualVarianceEB(tau ** 2, sig ** 2)
            mu_adj = mu_te + c
            info[name] = dict(tau=tau, sig=sig, bias=c)
            pred[name] = lambda kk, tt: e.condition(mu_adj[tt], mu_adj[kk], te.y.values[kk])

        if "ANCHOR" in arms:
            c, tau, sig = oneway_reml(tr.y.values - tr.m.values, g)
            v = tau ** 2 + sig ** 2
            pred["ANCHOR"] = lambda kk, tt: (te.m.values[tt], np.full(len(tt), v))
        fitted = {}
        for arm in LMM_ARMS:
            if arm in arms or "PAKA" in arms or "PAKA-S" in arms:
                fitted[arm] = fit_lmm_arm(arm, tr, des, seed)
                info[arm] = dict(lam=";".join(f"{x:.4f}" for x in fitted[arm].lam_), tau=fitted[arm].tau_,
                                 sig=fitted[arm].sig_)
        for arm in LMM_ARMS:
            if arm in arms:
                pred[arm] = (lambda a: (lambda kk, tt: lmm_predict(a, fitted[a], des, te, kk, tt)))(arm)
        if "PAKA" in arms:
            chosen, mse = paka_select(tr, coded, free, seed)
            info["PAKA"] = dict(chosen=chosen, **{f"cv_{k}": v for k, v in mse.items()})
            pred["PAKA"] = lambda kk, tt: lmm_predict(chosen, fitted[chosen], des, te, kk, tt)
        stack_arms = [(a, a0) for a, a0 in (("PAKA-S", STACK_PRIOR), ("PAKA-S0", 0.0), ("PAKA-S6", 6.0))
                      if a in arms]
        if stack_arms:
            L_st = paka_stack_logdens(tr, coded, free, seed)
            for st_arm, a0 in stack_arms:
                w_st = (np.array([1.0, 0.0, 0.0]) if L_st is None
                        else stacking_weights(L_st, prior=[a0, 0.0, 0.0]))
                info[st_arm] = dict(zip([f"w_{a}" for a in LMM_ARMS], map(float, w_st)))
                pred[st_arm] = (lambda w: (lambda kk, tt: paka_stack_predict(w, fitted, des, te, kk, tt)))(w_st)
        best_cfg = gbm_grid[0]
        if "GBM" in arms or "GBM-ME" in arms:
            mdl, oof, best_cfg = _gbm_nested(Phi_tr, tr.y.values, g, seed, gbm_grid)
            if "GBM" in arms:
                eb("GBM", mdl.predict(Phi_te), oof)
        if "GBM-RES" in arms:
            mdl, oof, _ = _gbm_nested(Phi_tr, tr.y.values - tr.m.values, g, seed, gbm_grid)
            eb("GBM-RES", te.m.values + mdl.predict(Phi_te), oof)
        if "GBM-ME" in arms:
            me = GBMME(best_cfg, n_iter=6, seed=seed).fit(Phi_tr, tr.y.values, g)
            eb("GBM-ME", me.predict(Phi_te), _gbmme_oof(Phi_tr, tr.y.values, g, best_cfg, seed))
        if "GP-RES" in arms:
            oof = np.zeros(len(tr))
            r_tr = tr.y.values - tr.m.values
            for itr, iva in folds(g).split(Phi_tr, r_tr, g):
                oof[iva] = r_tr[iva] - _gp(Phi_tr[itr], r_tr[itr], seed)(Phi_tr[iva])
            eb("GP-RES", te.m.values + _gp(Phi_tr, r_tr, seed)(Phi_te), oof)

        rng = np.random.default_rng([seed, zlib.crc32(str(target).encode("utf-8"))])
        n, rows = len(te), []
        for k in ks:
            if order_by is not None:
                v = te[order_by].values
                o = np.argsort(v, kind="stable")
                kk = o[:k]
                thr = v[kk].max() if k else -np.inf
                tt = o[k:][v[o[k:]] > thr]
                if len(tt) < 2:
                    continue
                draws = [(kk, tt)]
            else:
                if n - k < 3:
                    continue
                perms = [np.arange(n)] if k == 0 else [rng.permutation(n) for _ in range(n_draws)]
                draws = [(perm[:k], perm[k:]) for perm in perms]
            for di, (kk, tt) in enumerate(draws):
                for name, f in pred.items():
                    mu, var = f(kk, tt)
                    rows.append(dict(source=target, n_source=n, k=k, draw=di, n_pred=len(tt), method=name,
                                     **score(te.y.values[tt], mu, var)))
        return pd.DataFrame(rows), info


def summarise(per_draw, keys=("dataset",), n_boot=2000, seed=0, ref="PAKA-S"):
    keys = list(keys)
    per_prog = (per_draw.groupby(keys + ["method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean"), bias=("bias", "mean"), unconservative=("unconservative", "mean"),
                     cover90=("cover90", "mean"), width90=("width90", "mean"), nlpd=("nlpd", "mean")))
    rng = np.random.default_rng(seed)
    out = []
    for kv, blk in per_prog.groupby(keys + ["k"]):
        kv = kv if isinstance(kv, tuple) else (kv,)
        progs = np.sort(blk.source.unique())
        piv = {m: b.set_index("source").reindex(progs) for m, b in blk.groupby("method")}
        boots = rng.integers(0, len(progs), size=(n_boot, len(progs)))
        for m, b in piv.items():
            rec = dict(zip(keys + ["k"], kv), method=m, programmes=len(progs))
            for col in ("rmse", "bias", "unconservative", "cover90", "width90", "nlpd"):
                v = b[col].values
                rec[col] = float(np.nanmean(v))
                rec[col + "_lo"], rec[col + "_hi"] = np.percentile(np.nanmean(v[boots], 1), [2.5, 97.5])
            if ref in piv and m != ref:
                dlt = b["rmse"].values - piv[ref]["rmse"].values
                rec["minus_ref"] = float(np.nanmean(dlt))
                rec["minus_ref_lo"], rec["minus_ref_hi"] = np.percentile(np.nanmean(dlt[boots], 1), [2.5, 97.5])
                rec["ref_better_share"] = float(np.mean(dlt > 0))
            out.append(rec)
    return per_prog, pd.DataFrame(out)
