"""R01 pilot: leave-one-programme-out k-shot benchmark (worker functions).

For a held-out test programme s, every method is fitted on the other programmes
only. The k shots are k randomly drawn specimens of s with observed strength; the
remaining specimens of s are scored. All methods receive the k shots through the
same Gaussian conditioning on a programme-level effect, so differences between
methods come from the base predictor and its programme-level variance model.
"""
from __future__ import annotations

import warnings
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import GroupKFold, KFold
from threadpoolctl import threadpool_limits

from kag_models import KnowledgeAnchoredLMM, ResidualVarianceEB, oneway_components, within_between

warnings.filterwarnings("ignore")
Z90 = 1.6448536269514722
KS = (0, 1, 2, 3, 5)
GBM_GRID = (
    dict(max_leaf_nodes=4, max_iter=300, learning_rate=0.05, min_samples_leaf=10),
    dict(max_leaf_nodes=8, max_iter=300, learning_rate=0.05, min_samples_leaf=15),
    dict(max_leaf_nodes=16, max_iter=200, learning_rate=0.05, min_samples_leaf=5),
    dict(max_leaf_nodes=8, max_iter=600, learning_rate=0.03, min_samples_leaf=10),
)
ALL_METHODS = ("ANCHOR", "KA-FIXED", "KA-RE", "KA-WB", "DATA-RE", "DATA-WB",
               "GBM", "GBM-FEAT", "GBM-RES", "GP-RES")
KNOWLEDGE_METHODS = ("ANCHOR", "KA-FIXED", "KA-RE", "KA-WB", "GBM-FEAT", "GBM-RES", "GP-RES")


def _gkf(groups, n=5):
    """Grouped folds; with a single training programme only specimen-level folds exist."""
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        return GroupKFold(n_splits=min(n, n_groups))
    return KFold(n_splits=min(n, len(groups)), shuffle=True, random_state=0)


def _gbm_nested(X, t, g, seed, grid=GBM_GRID):
    """Grouped inner CV over a small grid; returns fitted model and grouped OOF residuals."""
    best = None
    for cfg in grid:
        oof = np.zeros_like(t)
        for tr, va in _gkf(g).split(X, t, g):
            mdl = HistGradientBoostingRegressor(random_state=seed, **cfg).fit(X[tr], t[tr])
            oof[va] = mdl.predict(X[va])
        mse = np.mean((t - oof) ** 2)
        if best is None or mse < best[0]:
            best = (mse, cfg, t - oof)
    mdl = HistGradientBoostingRegressor(random_state=seed, **best[1]).fit(X, t)
    return mdl, best[2], best[1]


def _gp_fit(X, t, seed):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    k = ConstantKernel(0.05, (1e-4, 10.0)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(0.03, (1e-4, 1.0))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    gp.fit((X - mu) / sd, t)
    return lambda Q: gp.predict((Q - mu) / sd)


def _gp_nested(X, t, g, seed):
    oof = np.zeros_like(t)
    for tr, va in _gkf(g).split(X, t, g):
        oof[va] = _gp_fit(X[tr], t[tr], seed)(X[va])
    return _gp_fit(X, t, seed), t - oof


def _score(y, mu, var):
    res = y - mu
    sd = np.sqrt(var)
    return dict(rmse=float(np.sqrt(np.mean(res ** 2))), bias=float(np.mean(res)),
                unconservative=float(np.mean(res < 0)),
                cover90=float(np.mean(np.abs(res) <= Z90 * sd)),
                width90=float(np.mean(2 * Z90 * sd)),
                nlpd=float(np.mean(0.5 * np.log(2 * np.pi * var) + res ** 2 / (2 * var))))


def run_programme(df, feats, target, methods=ALL_METHODS, n_draws=40, seed=0, threads=2,
                  m_col="m", ks=KS, train_sources=None, gbm_grid=GBM_GRID):
    """Fit all requested methods without programme `target` (optionally only on the
    programmes in `train_sources`), then score k-shot draws for k in `ks`."""
    with threadpool_limits(threads):
        keep = (df.source != target) if train_sources is None else df.source.isin(train_sources)
        tr = df[keep & (df.source != target)].reset_index(drop=True)
        te = df[df.source == target].reset_index(drop=True)
        g = tr.source.values
        Phi_tr, Phi_te = tr[feats].values, te[feats].values
        y_tr, y_te = tr.y.values, te.y.values
        m_tr, m_te = tr[m_col].values, te[m_col].values
        cen = Phi_tr.mean(0)
        W_tr, B_tr, cen_wb = within_between(Phi_tr, g)
        W_te, B_te, _ = within_between(Phi_te, te.source.values, center=cen_wb)
        p = len(feats)
        pred = {}   # method -> function(idx_shots, idx_test) -> (mu, var)
        info = {}

        def lmm_method(name, y_off, X_tr, X_te, groups_of_cols, off_te):
            mdl = KnowledgeAnchoredLMM(groups_of_cols, n_starts=3, seed=seed).fit(y_tr, X_tr, g, offset=y_off)
            info[name] = dict(lam=";".join(f"{v:.4f}" for v in mdl.lam_), tau=mdl.tau_, sig=mdl.sig_)

            def f(kk, tt):
                if len(kk) == 0:
                    return mdl.predict_new_group(X_te[tt], offset_t=off_te[tt])
                return mdl.predict_new_group(X_te[tt], offset_t=off_te[tt], X_k=X_te[kk], y_k=y_te[kk],
                                             offset_k=off_te[kk])
            pred[name] = f

        def eb_method(name, mu_te, oof_res):
            tau2, sig2, bias = oneway_components(oof_res, g)
            eb = ResidualVarianceEB(tau2, sig2)
            mu_adj = mu_te + bias
            info[name] = dict(tau=np.sqrt(tau2), sig=np.sqrt(sig2), bias=bias)

            def f(kk, tt):
                return eb.condition(mu_adj[tt], mu_adj[kk], y_te[kk])
            pred[name] = f

        r_tr = y_tr - m_tr
        if "ANCHOR" in methods:
            tau2, sig2, _ = oneway_components(r_tr, g)
            pred["ANCHOR"] = lambda kk, tt: (m_te[tt], np.full(len(tt), tau2 + sig2))
        if "KA-FIXED" in methods:
            eb_method("KA-FIXED", m_te.copy(), r_tr)
        if "KA-RE" in methods:
            lmm_method("KA-RE", m_tr, Phi_tr - cen, Phi_te - cen, [0] * p, m_te)
        if "KA-WB" in methods:
            lmm_method("KA-WB", m_tr, np.column_stack([W_tr, B_tr]), np.column_stack([W_te, B_te]),
                       [0] * p + [1] * p, m_te)
        zeros = np.zeros(len(te))
        if "DATA-RE" in methods:
            lmm_method("DATA-RE", None, Phi_tr - cen, Phi_te - cen, [0] * p, zeros)
        if "DATA-WB" in methods:
            lmm_method("DATA-WB", None, np.column_stack([W_tr, B_tr]), np.column_stack([W_te, B_te]),
                       [0] * p + [1] * p, zeros)
        if "GBM" in methods:
            mdl, oof, cfg = _gbm_nested(Phi_tr, y_tr, g, seed, gbm_grid)
            eb_method("GBM", mdl.predict(Phi_te), oof)
            info["GBM"]["cfg"] = str(cfg)
        if "GBM-FEAT" in methods:
            Xa_tr, Xa_te = np.column_stack([Phi_tr, m_tr]), np.column_stack([Phi_te, m_te])
            mdl, oof, cfg = _gbm_nested(Xa_tr, y_tr, g, seed, gbm_grid)
            eb_method("GBM-FEAT", mdl.predict(Xa_te), oof)
            info["GBM-FEAT"]["cfg"] = str(cfg)
        if "GBM-RES" in methods:
            mdl, oof, cfg = _gbm_nested(Phi_tr, r_tr, g, seed, gbm_grid)
            eb_method("GBM-RES", m_te + mdl.predict(Phi_te), oof)
            info["GBM-RES"]["cfg"] = str(cfg)
        if "GP-RES" in methods:
            f_gp, oof = _gp_nested(Phi_tr, r_tr, g, seed)
            eb_method("GP-RES", m_te + f_gp(Phi_te), oof)
        info["_train"] = dict(n=len(tr), programmes=int(len(np.unique(g))))

        rng = np.random.default_rng([seed, zlib.crc32(target.encode("utf-8"))])
        n = len(te)
        rows = []
        for k in ks:
            if n - k < 3:
                continue
            draws = [np.arange(n)] if k == 0 else [rng.permutation(n) for _ in range(n_draws)]
            for di, perm in enumerate(draws):
                kk, tt = perm[:k], perm[k:]
                for name, f in pred.items():
                    mu, var = f(kk, tt)
                    rows.append(dict(source=target, n_source=n, k=k, draw=di, method=name,
                                     **_score(y_te[tt], mu, var)))
        return pd.DataFrame(rows), info


def summarise(per_draw: pd.DataFrame, n_boot=2000, seed=0, ref="KA-WB"):
    """Programme-weighted means with cluster-bootstrap CIs and paired gaps to `ref`."""
    per_prog = (per_draw.groupby(["dataset", "method", "k", "source"], as_index=False)
                .agg(rmse=("rmse", "mean"), bias=("bias", "mean"), abs_bias=("bias", lambda v: np.mean(np.abs(v))),
                     unconservative=("unconservative", "mean"), cover90=("cover90", "mean"),
                     width90=("width90", "mean"), nlpd=("nlpd", "mean")))
    rng = np.random.default_rng(seed)
    out = []
    for (ds, k), blk in per_prog.groupby(["dataset", "k"]):
        progs = np.sort(blk.source.unique())
        piv = {mname: b.set_index("source").reindex(progs) for mname, b in blk.groupby("method")}
        boots = rng.integers(0, len(progs), size=(n_boot, len(progs)))
        for mname, b in piv.items():
            rec = dict(dataset=ds, k=k, method=mname, programmes=len(progs))
            for col in ("rmse", "abs_bias", "unconservative", "cover90", "width90", "nlpd"):
                v = b[col].values
                bs = np.nanmean(v[boots], axis=1)
                rec[col] = float(np.nanmean(v))
                rec[col + "_lo"], rec[col + "_hi"] = np.percentile(bs, [2.5, 97.5])
            if ref in piv and mname != ref:
                dlt = b["rmse"].values - piv[ref]["rmse"].values
                bs = np.nanmean(dlt[boots], axis=1)
                rec["rmse_minus_ref"] = float(np.nanmean(dlt))
                rec["rmse_minus_ref_lo"], rec["rmse_minus_ref_hi"] = np.percentile(bs, [2.5, 97.5])
                rec["ref_better_share"] = float(np.mean(dlt > 0))
            out.append(rec)
    return per_prog, pd.DataFrame(out)
