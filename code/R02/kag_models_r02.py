"""R02 models: knowledge-anchored random-intercept model with the trust scale
lambda either fixed (for evidence-based model selection) or estimated (REML),
and the evidence-adaptive trust (EAT) selector.

Same likelihood as R01's KnowledgeAnchoredLMM:

    y_i - offset_i = c + sum_g X[i, g] b_g + u_{s(i)} + e_i
    b_g ~ N(0, lam^2),  u_s ~ N(0, tau^2),  e_i ~ N(0, sig^2)

`ka_fit` estimates (c, tau, sig) for a given lam; with `lam=None` it also estimates
lam by REML, which reproduces R01's KA-RE arm. `ml=True` returns the marginal
log-likelihood with the fixed effect integrated at its GLS estimate, which is what
makes *nested priors of different width comparable on the same data*: the evidence
of the code being right (lam -> 0) against the evidence of the code being untrusted
(lam large). `eat_select` returns the lam with the largest evidence, plus the
Gaussian-approximation posterior weights over the grid.

Everything the selector uses is computed on training programmes only; the k shots
of the target programme enter only through the Gaussian conditioning of the
programme effect, exactly as in R01.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp

LOG2PI = np.log(2.0 * np.pi)


def group_index(groups):
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    return inv


def _core(y, X, ZZt, lam, tau, sig):
    """Cholesky factor of V and GLS intercept for one parameter point."""
    n = y.size
    V = tau ** 2 * ZZt + sig ** 2 * np.eye(n)
    if X.shape[1] and lam > 0:
        V = V + (lam ** 2) * (X @ X.T)
    cf = cho_factor(V, lower=True, check_finite=False)
    one = np.ones_like(y)
    Vi1 = cho_solve(cf, one, check_finite=False)
    Viy = cho_solve(cf, y, check_finite=False)
    a = float(one @ Vi1)
    c = float(one @ Viy) / a
    r = y - c
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return cf, Vi1, c, r, logdet, a


def _negprof(theta, y, X, ZZt, lam, ml):
    tau, sig = np.exp(theta)
    try:
        cf, Vi1, c, r, logdet, a = _core(y, X, ZZt, lam, tau, sig)
    except np.linalg.LinAlgError:
        return 1e12
    Vir = cho_solve(cf, r, check_finite=False)
    q = float(r @ Vir)
    if ml:                      # -2 log marginal likelihood (c at GLS)
        return 0.5 * (logdet + q + y.size * LOG2PI)
    return 0.5 * (logdet + np.log(a) + q)      # REML


def ka_fit(y, X, groups, offset=None, lam=0.1, ml=True, n_starts=4, seed=0,
           bounds=dict(tau=(1e-4, 3.0), sig=(1e-3, 3.0))):
    """Fit the random-intercept knowledge model. `lam=None` estimates lam by REML.

    Returns a dict with c, lam, tau, sig, loglik (marginal, ML scale), and the
    posterior pieces needed for k-shot conditioning of a new programme.
    """
    y = np.asarray(y, float) - (0.0 if offset is None else np.asarray(offset, float))
    X = np.asarray(X, float).reshape(len(y), -1)
    inv = group_index(groups)
    ZZt = np.eye(inv.max() + 1)[inv] @ np.eye(inv.max() + 1)[inv].T

    lam_hi = 10.0

    def neg(th):
        if lam is None:
            tau, sig, l = np.exp(th[0]), np.exp(th[1]), np.exp(th[2])
            return _negprof([th[0], th[1]], y, X, ZZt, l, ml)
        tau, sig = np.exp(th[0]), np.exp(th[1])
        return _negprof(th, y, X, ZZt, lam, ml)

    rng = np.random.default_rng(seed)
    lo = [np.log(bounds["tau"][0]), np.log(bounds["sig"][0])]
    hi = [np.log(bounds["tau"][1]), np.log(bounds["sig"][1])]
    starts = [np.log([0.1, 0.15])] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]
    if lam is None:
        starts = [np.r_[s, np.log(0.1)] for s in starts]
        lo = lo + [np.log(1e-4)]
        hi = hi + [np.log(lam_hi)]
    best = None
    for s0 in starts:
        res = minimize(neg, s0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or res.fun < best.fun:
            best = res
    th = best.x
    tau, sig = float(np.exp(th[0])), float(np.exp(th[1]))
    l_hat = float(np.exp(th[2])) if lam is None else float(lam)

    cf, Vi1, c, r, logdet, a = _core(y, X, ZZt, l_hat, tau, sig)
    Vir = cho_solve(cf, r, check_finite=False)
    loglik = float(-0.5 * (logdet + float(r @ Vir) + y.size * LOG2PI))   # ML evidence

    if X.shape[1] and l_hat > 0:
        w = l_hat ** 2
        b = w * (X.T @ Vir)
        ViX = cho_solve(cf, X, check_finite=False)
        b_cov = w * np.eye(X.shape[1]) - (w ** 2) * (X.T @ ViX)
    else:
        b = np.zeros(X.shape[1])
        b_cov = np.zeros((X.shape[1], X.shape[1]))
    return dict(c=c, c_var=1.0 / a, lam=l_hat, tau=tau, sig=sig, loglik=loglik,
                b=b, b_cov=b_cov, X=X, groups=inv)


def predict_new_group(m, X_t, offset_t=None, X_k=None, y_k=None, offset_k=None):
    """Gaussian conditioning of a new programme's k observed specimens."""
    X_t = np.asarray(X_t, float).reshape(-1, m["b"].size)
    mu = m["c"] + X_t @ m["b"] + (0.0 if offset_t is None else np.asarray(offset_t, float))
    Cb = m["b_cov"]
    shared = m["tau"] ** 2 + m["c_var"]
    if X_k is None or len(X_k) == 0:
        var = m["sig"] ** 2 + shared + np.einsum("ij,jk,ik->i", X_t, Cb, X_t)
        return mu, np.maximum(var, 1e-10)
    X_k = np.asarray(X_k, float).reshape(-1, m["b"].size)
    mu_k = m["c"] + X_k @ m["b"] + (0.0 if offset_k is None else np.asarray(offset_k, float))
    e_k = np.asarray(y_k, float) - mu_k
    C_kk = X_k @ Cb @ X_k.T + shared + m["sig"] ** 2 * np.eye(len(X_k))
    C_tk = X_t @ Cb @ X_k.T + shared
    cf = cho_factor(C_kk, lower=True)
    mu = mu + C_tk @ cho_solve(cf, e_k)
    var = (m["sig"] ** 2 + shared + np.einsum("ij,jk,ik->i", X_t, Cb, X_t)
           - np.einsum("ij,ji->i", C_tk, cho_solve(cf, C_tk.T)))
    return mu, np.maximum(var, 1e-10)


def eat_select(y, X, groups, offset=None, grid=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0, 10.0),
               seed=0, n_starts=3):
    """Evidence-adaptive trust by marginal likelihood (see note in r02_note).

    Kept for the record, but the marginal-likelihood selector is *not* the method
    used in R02: it scores the fit to the training programmes, not the ability to
    predict an unseen programme, and on a 3-programme library it still selected
    lambda = 0.2. `select_arm` below is the operational selector.
    """
    fits = [ka_fit(y, X, groups, offset=offset, lam=g, ml=True, n_starts=n_starts, seed=seed)
            for g in grid]
    ll = np.array([f["loglik"] for f in fits])
    wts = np.exp(ll - logsumexp(ll))
    j = int(np.argmax(ll))
    return dict(lam=grid[j], weights=wts, grid=grid, model=fits[j], fits=fits,
                R=float(ll[j] - logsumexp(ll)))


ARM_LAM = {"KA-FIXED": 0.0, "KA-RE": None, "DATA-RE": 10.0}


def select_arm(y, X, groups, offset=None, arms=None, n_folds=4, seed=0, n_starts=2):
    """Evidence-adaptive trust: score every arm on held-out *training programmes*
    at k = 0 and apply a one-standard-error rule along the trust ladder.

    Errors are paired by programme, so the standard error of an arm difference
    shrinks with the number of programmes: with three programmes the comparison
    is too noisy to license abandoning the code and the rule stays with
    KA-FIXED; with twenty-odd programmes it follows the evidence. That is the
    intended, stated behaviour, not a tuning artefact.

    The selector sees only training programme labels; the target programme
    contributes nothing.
    """
    arms = dict(ARM_LAM if arms is None else arms)
    ladder = [a for a in ("KA-FIXED", "KA-RE", "DATA-RE") if a in arms]
    from sklearn.model_selection import GroupKFold

    g = np.asarray(groups)
    ug = np.unique(g)
    if len(ug) < 3:
        return "KA-FIXED", {a: np.nan for a in arms}
    n_folds = int(min(n_folds, len(ug)))
    # per-programme mean squared error, aligned on programme id
    err = {a: {} for a in arms}
    for tr, va in GroupKFold(n_splits=n_folds).split(X, y, g):
        for name, lam in arms.items():
            try:
                mdl = ka_fit(y[tr], X[tr], g[tr],
                             offset=None if offset is None else offset[tr],
                             lam=lam, ml=False, n_starts=n_starts, seed=seed)
                mu, _ = predict_new_group(mdl, X[va],
                                          offset_t=None if offset is None else offset[va])
                sq = (y[va] - mu) ** 2
                for gid in np.unique(g[va]):
                    err[name][gid] = float(np.mean(sq[g[va] == gid]))
            except Exception:
                for gid in np.unique(g[va]):
                    err[name][gid] = np.inf

    common = [p for p in ug if all(p in err[a] for a in arms) and
              np.all(np.isfinite([err[a][p] for a in arms]))]
    mse = {a: float(np.mean([err[a][p] for p in common])) if common else np.inf for a in arms}
    best = min(mse, key=mse.get)
    if len(common) < 3:
        return "KA-FIXED", mse
    chosen = best
    j = ladder.index(best) if best in ladder else len(ladder)
    for a in ladder[:j][::-1]:                      # walk toward more knowledge
        d = np.array([err[a][p] - err[best][p] for p in common])
        se = float(np.std(d, ddof=1) / np.sqrt(d.size))
        if d.mean() <= se:
            chosen = a
        else:
            break
    return chosen, mse



def s_star_policy(X, groups, model_fixed, model_dev, p=None):
    """Theory-driven trust policy: estimate the crossover S* inside the training
    library and trust the code while the library is smaller than it.

    A = p (tau^2 + sigma^2 / n_bar) from the code-fixed fit;
    B = delta' Sigma_B delta with delta the *shrunk* exponent deviation of the
        KA-RE fit, not the flat-prior fit: on a 3-6 programme library the flat
        estimate is dominated by noise and biases S* down, while the shrinkage
        estimate is stable and is the model's own estimate of the misfit.
    S* = A / B.  Returns (policy, S_star).
    """
    X = np.asarray(X, float)
    g = np.asarray(groups)
    ug, inv = np.unique(g, return_inverse=True)
    S = len(ug)
    nbar = len(X) / S
    p = X.shape[1] if p is None else p
    A = p * (model_fixed["tau"] ** 2 + model_fixed["sig"] ** 2 / nbar)
    means = np.zeros((S, X.shape[1]))
    np.add.at(means, inv, X)
    means /= np.bincount(inv)[:, None]
    SB = np.cov(means, rowvar=False, ddof=1)
    delta = np.asarray(model_dev["b"], float)
    B = float(delta @ SB @ delta)
    if B <= 1e-8:
        return "KA-FIXED", float("inf")
    S_star = A / B
    return ("KA-FIXED" if S < S_star else "KA-RE"), float(S_star)




