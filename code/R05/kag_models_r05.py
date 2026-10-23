"""R05 models.

FastAnchoredLMM
    The R01 knowledge-anchored random-intercept model,
        y_i - offset_i = c + X_i b + u_{j(i)} + e_i,
        b_g ~ N(0, lam_g^2 I) per column group g,  u_j ~ N(0, tau^2),  e_i ~ N(0, sig^2),
    with REML hyper-parameters, evaluated in O(n p^2) per likelihood call instead of
    O(n^3): V = D + U U' with D block-diagonal (one compound-symmetry block per
    programme, inverse and log-determinant in closed form) and U = X Lam^(1/2), so the
    Woodbury identity gives V^-1 and log|V| through a p x p system. Groups can be
    frozen at a fixed lam (lam = 0 removes the columns' freedom: the anchor's exponent
    is imposed; a large lam makes the exponent data-determined).
    Numerically identical to kag_models.KnowledgeAnchoredLMM (tests/test_r05.py).

GBMME
    Mixed-effects gradient boosting (MERF-type EM): boosting is refitted on the
    programme-effect-corrected response, programme effects are re-estimated as BLUPs
    from the boosting residuals; a strong flexible baseline that, unlike plain
    boosting, does not let programme effects masquerade as feature effects.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor


def group_index(groups):
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    return inv


class FastAnchoredLMM:
    def __init__(self, col_groups, fixed_lam=None, bounds=None, n_starts=3, seed=0):
        self.col_group = np.asarray(col_groups, int)
        self.n_lam = int(self.col_group.max()) + 1 if self.col_group.size else 0
        self.fixed = dict(fixed_lam or {})
        self.bounds = bounds or dict(lam=(1e-4, 5.0), tau=(1e-4, 3.0), sig=(1e-3, 3.0))
        self.n_starts = n_starts
        self.seed = seed

    # ---- D^-1 (block compound symmetry) applied to columns -------------------------
    def _prep(self, groups):
        inv = group_index(groups)
        self._inv = inv
        self._ng = np.bincount(inv).astype(float)

    def _Dinv(self, A, tau2, sig2):
        """Apply D^-1 to the columns of A (n x q)."""
        A2 = A if A.ndim == 2 else A[:, None]
        sums = np.zeros((self._ng.size, A2.shape[1]))
        np.add.at(sums, self._inv, A2)
        f = tau2 / (sig2 + self._ng * tau2)
        out = (A2 - (f[:, None] * sums)[self._inv]) / sig2
        return out if A.ndim == 2 else out[:, 0]

    def _logdetD(self, tau2, sig2):
        return float(np.sum((self._ng - 1.0) * np.log(sig2) + np.log(sig2 + self._ng * tau2)))

    def _unpack(self, th):
        lam = np.empty(self.n_lam)
        k = 0
        for gi in range(self.n_lam):
            if gi in self.fixed:
                lam[gi] = self.fixed[gi]
            else:
                lam[gi] = np.exp(th[k])
                k += 1
        return lam, float(np.exp(th[k])), float(np.exp(th[k + 1]))

    def _pieces(self, th, y, X):
        lam, tau, sig = self._unpack(th)
        tau2, sig2 = tau * tau, sig * sig
        one = np.ones_like(y)
        if X.shape[1]:
            w = lam[self.col_group]
            keep = w > 0
            U = X[:, keep] * w[keep]
        else:
            U = np.zeros((len(y), 0))
        R = np.column_stack([one, y, U])
        DR = self._Dinv(R, tau2, sig2)
        q = U.shape[1]
        if q:
            M = np.eye(q) + U.T @ DR[:, 2:]
            cM = np.linalg.cholesky(M)
            logdetM = 2.0 * np.sum(np.log(np.diag(cM)))
            UtD = DR[:, 2:].T @ R[:, :2]                           # q x 2
            sol = np.linalg.solve(M, UtD)
            G = R[:, :2].T @ DR[:, :2] - UtD.T @ sol               # [1 y]' V^-1 [1 y]
        else:
            logdetM = 0.0
            G = R[:, :2].T @ DR[:, :2]
        return lam, tau2, sig2, G, logdetM

    def _nreml(self, th, y, X):
        try:
            lam, tau2, sig2, G, logdetM = self._pieces(th, y, X)
        except np.linalg.LinAlgError:
            return 1e12
        a, b, yy = G[0, 0], G[0, 1], G[1, 1]
        if a <= 0:
            return 1e12
        q = yy - b * b / a
        return 0.5 * (self._logdetD(tau2, sig2) + logdetM + np.log(a) + q)

    def fit(self, y, X, groups, offset=None):
        y = np.asarray(y, float) - (0.0 if offset is None else np.asarray(offset, float))
        X = np.asarray(X, float).reshape(len(y), -1)
        self._prep(groups)
        free = [g for g in range(self.n_lam) if g not in self.fixed]
        lo = np.log([self.bounds["lam"][0]] * len(free) + [self.bounds["tau"][0], self.bounds["sig"][0]])
        hi = np.log([self.bounds["lam"][1]] * len(free) + [self.bounds["tau"][1], self.bounds["sig"][1]])
        rng = np.random.default_rng(self.seed)
        starts = [np.log([0.1] * len(free) + [0.1, 0.15])] + [rng.uniform(lo, hi) for _ in range(self.n_starts - 1)]
        best = None
        for s0 in starts:
            r = minimize(self._nreml, s0, args=(y, X), method="L-BFGS-B", bounds=list(zip(lo, hi)))
            if best is None or r.fun < best.fun:
                best = r
        self.theta_, self.reml_ = best.x, float(best.fun)
        self.lam_, tau, sig = self._unpack(best.x)
        self.tau_, self.sig_ = tau, sig
        tau2, sig2 = tau * tau, sig * sig
        # posterior of c and b
        one = np.ones_like(y)
        w = self.lam_[self.col_group] if X.shape[1] else np.zeros(0)
        U = X * w if X.shape[1] else np.zeros((len(y), 0))
        R = np.column_stack([one, y, X]) if X.shape[1] else np.column_stack([one, y])
        DR = self._Dinv(R, tau2, sig2)
        if X.shape[1]:
            DU = DR[:, 2:] * w
            M = np.eye(X.shape[1]) + U.T @ DU
            UtDR = U.T @ DR
            VinvR_quad = R.T @ DR - UtDR.T @ np.linalg.solve(M, UtDR)   # R' V^-1 R
        else:
            VinvR_quad = R.T @ DR
        a = VinvR_quad[0, 0]
        self.c_ = float(VinvR_quad[0, 1] / a)
        self.c_var_ = float(1.0 / a)
        if X.shape[1]:
            XtVr = VinvR_quad[2:, 1] - self.c_ * VinvR_quad[2:, 0]       # X' V^-1 (y - c)
            XtVX = VinvR_quad[2:, 2:]
            W2 = w * w
            self.b_ = W2 * XtVr
            self.b_cov_ = np.diag(W2) - (W2[:, None] * XtVX) * W2[None, :]
        else:
            self.b_, self.b_cov_ = np.zeros(0), np.zeros((0, 0))
        return self

    def reliance(self):
        w2 = self.lam_[self.col_group] ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w2 > 0, np.diag(self.b_cov_) / w2, 1.0)

    def predict_new_group(self, X_t, offset_t=None, X_k=None, y_k=None, offset_k=None):
        p = self.b_.size
        X_t = np.asarray(X_t, float)
        X_t = X_t if X_t.ndim == 2 else X_t.reshape(-1, p)
        mu_t = self.c_ + X_t @ self.b_ + (0.0 if offset_t is None else np.asarray(offset_t, float))
        Cb = self.b_cov_
        tt = self.tau_ ** 2 + self.c_var_
        base = self.sig_ ** 2 + tt + np.einsum("ij,jk,ik->i", X_t, Cb, X_t)
        if X_k is None or len(X_k) == 0:
            return mu_t, np.maximum(base, 1e-10)
        X_k = np.asarray(X_k, float)
        X_k = X_k if X_k.ndim == 2 else X_k.reshape(-1, p)
        mu_k = self.c_ + X_k @ self.b_ + (0.0 if offset_k is None else np.asarray(offset_k, float))
        e_k = np.asarray(y_k, float) - mu_k
        C_kk = X_k @ Cb @ X_k.T + tt + self.sig_ ** 2 * np.eye(len(X_k))
        C_tk = X_t @ Cb @ X_k.T + tt
        sol = np.linalg.solve(C_kk, np.column_stack([e_k, C_tk.T]))
        mu = mu_t + C_tk @ sol[:, 0]
        var = base - np.einsum("ij,ji->i", C_tk, sol[:, 1:])
        return mu, np.maximum(var, 1e-10)


def joint_predictive(mdl, X_t, offset_t=None):
    """Mean and full covariance of a new programme's responses before any shot:
    X Cb X' + (tau^2 + var c) 11' + sig^2 I."""
    p = mdl.b_.size
    X_t = np.asarray(X_t, float)
    X_t = X_t if X_t.ndim == 2 else X_t.reshape(-1, p)
    mu = mdl.c_ + X_t @ mdl.b_ + (0.0 if offset_t is None else np.asarray(offset_t, float))
    C = X_t @ mdl.b_cov_ @ X_t.T + (mdl.tau_ ** 2 + mdl.c_var_) + mdl.sig_ ** 2 * np.eye(len(mu))
    return mu, C


def mvn_logpdf(y, mu, C):
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, y - mu)
    return float(-0.5 * (z @ z) - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * np.log(2 * np.pi))


def stacking_weights(logdens, prior=None):
    """Maximise sum_p log sum_a w_a exp(L[p, a]) + sum_a prior_a log w_a over the simplex
    (stacking of predictive distributions, Yao et al., 2018, with an optional Dirichlet
    pseudo-count `prior` expressed in held-out-programme units)."""
    L = np.asarray(logdens, float)
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    a = np.zeros(L.shape[1]) if prior is None else np.asarray(prior, float)

    def f(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        return -np.sum(np.log(E @ w + 1e-300)) - np.sum(a * np.log(w + 1e-300))

    r = minimize(f, np.zeros(L.shape[1]), method="L-BFGS-B")
    w = np.exp(r.x - r.x.max())
    return w / w.sum()


def fast_lmm_gls(y, X, groups, bounds=((1e-4, 3.0), (1e-3, 3.0)), n_starts=3, seed=0):
    """Random-intercept LMM with unpenalised fixed effects X (constant column included):
    REML (tau, sigma) and GLS beta with Cov(beta). V is block compound symmetry, so V^-1
    and log|V| are closed form per programme -- O(n p^2) instead of O(n^3).
    Same estimator as kag_models.lmm_gls (tests/test_r05.py)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    helper = FastAnchoredLMM(np.zeros(0, int))
    helper._prep(groups)
    R = np.column_stack([X, y])

    def parts(th):
        tau2, sig2 = np.exp(2 * th[0]), np.exp(2 * th[1])
        DR = helper._Dinv(R, tau2, sig2)
        Q = R.T @ DR
        A = Q[:-1, :-1]
        beta = np.linalg.solve(A, Q[:-1, -1])
        q = Q[-1, -1] - Q[:-1, -1] @ beta
        return tau2, sig2, A, beta, q

    def nll(th):
        try:
            tau2, sig2, A, beta, q = parts(th)
            sign, logdetA = np.linalg.slogdet(A)
        except np.linalg.LinAlgError:
            return 1e12
        return 0.5 * (helper._logdetD(tau2, sig2) + logdetA + q)

    lo, hi = np.log([bounds[0][0], bounds[1][0]]), np.log([bounds[0][1], bounds[1][1]])
    rng = np.random.default_rng(seed)
    best = None
    for s0 in [np.log([0.1, 0.15])] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]:
        r = minimize(nll, s0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun:
            best = r
    tau2, sig2, A, beta, _ = parts(best.x)
    return beta, np.linalg.inv(A), float(np.sqrt(tau2)), float(np.sqrt(sig2))


def oneway_reml(res, groups):
    """REML random-intercept fit of a residual vector: (mean, tau, sig)."""
    m = FastAnchoredLMM(np.zeros(0, int), n_starts=3).fit(res, np.zeros((len(res), 0)), groups)
    return m.c_, m.tau_, m.sig_


class GBMME:
    """MERF-type mixed-effects boosting with a random programme intercept."""

    def __init__(self, gbm_kw, n_iter=8, seed=0):
        self.kw, self.n_iter, self.seed = dict(gbm_kw), n_iter, seed

    def fit(self, X, y, groups):
        inv = group_index(groups)
        ng = np.bincount(inv).astype(float)
        u = np.zeros(ng.size)
        tau2, sig2 = 0.01, 0.01
        for _ in range(self.n_iter):
            self.model_ = HistGradientBoostingRegressor(random_state=self.seed, **self.kw).fit(X, y - u[inv])
            r = y - self.model_.predict(X)
            c, tau, sig = oneway_reml(r, inv)
            tau2, sig2 = tau ** 2, sig ** 2
            sums = np.bincount(inv, weights=r - c, minlength=ng.size)
            u_new = c + tau2 * sums / (ng * tau2 + sig2)
            if np.max(np.abs(u_new - u)) < 1e-3:
                u = u_new
                break
            u = u_new
        self.tau2_, self.sig2_ = tau2, sig2
        self.offset_ = float(np.mean(u))      # population-level shift (programme effects not centred)
        return self

    def predict(self, X):
        return self.model_.predict(X) + self.offset_
