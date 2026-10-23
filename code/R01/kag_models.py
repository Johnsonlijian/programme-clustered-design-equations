"""KAG-SmallData R01 pilot: models.

KnowledgeAnchoredLMM
    y_i - offset_i = c + sum_g X_g[i] b_g + u_{s(i)} + e_i
    b_g ~ N(0, lam_g^2 I)   (deviation from the anchor's exponents, in exponent units)
    u_s ~ N(0, tau^2)       (test-programme / laboratory effect)
    e_i ~ N(0, sig^2)       (within-programme scatter)
  offset = ln V_anchor  -> knowledge-anchored (KA) model; offset = 0 -> data-only model.
  Column groups let the within-programme and between-programme parts of each
  feature carry separate trust parameters (Mundlak / within-between form).
  Hyper-parameters (lam_g, tau, sig) by REML with a flat prior on c.

Few-shot prediction for an unseen programme is exact Gaussian conditioning of the
programme's test points on its k observed shots (uncertainty of b included).

ResidualVarianceEB
    the same few-shot conditioning for any point predictor whose new-programme
    residual variance components (tau^2, sig^2) were estimated from grouped
    out-of-fold residuals on the training programmes.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

LOG2PI = np.log(2.0 * np.pi)


def group_index(groups):
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    return inv


def within_between(Phi, groups, center=None):
    """Split features into within-programme deviations and centred programme means."""
    Phi = np.asarray(Phi, float)
    inv = group_index(groups)
    G = inv.max() + 1
    sums = np.zeros((G, Phi.shape[1]))
    np.add.at(sums, inv, Phi)
    means = sums / np.bincount(inv, minlength=G)[:, None]
    gm = means[inv]
    c = Phi.mean(0) if center is None else center
    return Phi - gm, gm - c, c


class KnowledgeAnchoredLMM:
    def __init__(self, n_groups_of_cols, bounds=None, n_starts=6, seed=0):
        self.col_group = np.asarray(n_groups_of_cols, int)
        self.n_lam = int(self.col_group.max()) + 1 if self.col_group.size else 0
        self.bounds = bounds or dict(lam=(1e-4, 5.0), tau=(1e-4, 3.0), sig=(1e-3, 3.0))
        self.n_starts = n_starts
        self.seed = seed

    # ---- marginal covariance pieces -------------------------------------------------
    def _V(self, theta, X, ZZt):
        lam = np.exp(theta[: self.n_lam])
        tau, sig = np.exp(theta[self.n_lam]), np.exp(theta[self.n_lam + 1])
        n = X.shape[0]
        V = tau ** 2 * ZZt + sig ** 2 * np.eye(n)
        if X.shape[1]:
            w = lam[self.col_group] ** 2
            V += (X * w) @ X.T
        return V

    def _reml(self, theta, y, X, ZZt):
        V = self._V(theta, X, ZZt)
        try:
            cf = cho_factor(V, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e12
        one = np.ones_like(y)
        Vi1 = cho_solve(cf, one, check_finite=False)
        Viy = cho_solve(cf, y, check_finite=False)
        a = one @ Vi1
        c = (one @ Viy) / a
        r = y - c
        Vir = Viy - c * Vi1
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
        return 0.5 * (logdet + np.log(a) + r @ Vir)

    def fit(self, y, X, groups, offset=None):
        y = np.asarray(y, float) - (0.0 if offset is None else np.asarray(offset, float))
        X = np.asarray(X, float).reshape(len(y), -1)
        inv = group_index(groups)
        Z = np.eye(inv.max() + 1)[inv]
        ZZt = Z @ Z.T
        lo = np.log([self.bounds["lam"][0]] * self.n_lam + [self.bounds["tau"][0], self.bounds["sig"][0]])
        hi = np.log([self.bounds["lam"][1]] * self.n_lam + [self.bounds["tau"][1], self.bounds["sig"][1]])
        rng = np.random.default_rng(self.seed)
        best = None
        starts = [np.log([0.1] * self.n_lam + [0.1, 0.15])]
        starts += [rng.uniform(lo, hi) for _ in range(self.n_starts - 1)]
        for s0 in starts:
            res = minimize(self._reml, s0, args=(y, X, ZZt), method="L-BFGS-B",
                           bounds=list(zip(lo, hi)))
            if best is None or res.fun < best.fun:
                best = res
        self.theta_ = best.x
        self.reml_ = float(best.fun)
        self.lam_ = np.exp(best.x[: self.n_lam])
        self.tau_ = float(np.exp(best.x[self.n_lam]))
        self.sig_ = float(np.exp(best.x[self.n_lam + 1]))
        V = self._V(best.x, X, ZZt)
        cf = cho_factor(V, lower=True, check_finite=False)
        one = np.ones_like(y)
        Vi1 = cho_solve(cf, one, check_finite=False)
        Viy = cho_solve(cf, y, check_finite=False)
        self.c_ = float((one @ Viy) / (one @ Vi1))
        self.c_var_ = float(1.0 / (one @ Vi1))
        if X.shape[1]:
            w = self.lam_[self.col_group] ** 2
            Vir = cho_solve(cf, y - self.c_, check_finite=False)
            self.b_ = w * (X.T @ Vir)
            ViX = cho_solve(cf, X, check_finite=False)
            self.b_cov_ = np.diag(w) - (w[:, None] * (X.T @ ViX)) * w[None, :]
        else:
            self.b_ = np.zeros(0)
            self.b_cov_ = np.zeros((0, 0))
        return self

    def reliance(self):
        """Posterior/prior variance ratio of each deviation coefficient: 1 = the data
        left the anchor's exponent untouched (knowledge-reliant); 0 = data-determined."""
        w = self.lam_[self.col_group] ** 2
        return np.diag(self.b_cov_) / w

    def predict_new_group(self, X_t, offset_t=None, X_k=None, y_k=None, offset_k=None):
        X_t = np.asarray(X_t, float).reshape(-1, self.b_.size)
        mu_t = self.c_ + X_t @ self.b_ + (0.0 if offset_t is None else np.asarray(offset_t, float))
        Cb = self.b_cov_
        if X_k is None or len(X_k) == 0:
            var = self.sig_ ** 2 + self.tau_ ** 2 + self.c_var_ + np.einsum("ij,jk,ik->i", X_t, Cb, X_t)
            return mu_t, var
        X_k = np.asarray(X_k, float).reshape(-1, self.b_.size)
        mu_k = self.c_ + X_k @ self.b_ + (0.0 if offset_k is None else np.asarray(offset_k, float))
        e_k = np.asarray(y_k, float) - mu_k
        tt = self.tau_ ** 2 + self.c_var_
        C_kk = X_k @ Cb @ X_k.T + tt + self.sig_ ** 2 * np.eye(len(X_k))
        C_tk = X_t @ Cb @ X_k.T + tt
        cf = cho_factor(C_kk, lower=True)
        mu = mu_t + C_tk @ cho_solve(cf, e_k)
        var = (self.sig_ ** 2 + tt + np.einsum("ij,jk,ik->i", X_t, Cb, X_t)
               - np.einsum("ij,ji->i", C_tk, cho_solve(cf, C_tk.T)))
        return mu, np.maximum(var, 1e-10)


def lmm_gls(y, X, groups, bounds=((1e-4, 3.0), (1e-3, 3.0)), n_starts=4, seed=0):
    """Random-intercept LMM with unpenalised fixed effects X (include a constant
    column), REML variance components; returns beta, Cov(beta), tau, sig."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    inv = group_index(groups)
    Z = np.eye(inv.max() + 1)[inv]
    ZZt = Z @ Z.T
    n = len(y)

    def parts(th):
        tau, sig = np.exp(th)
        cf = cho_factor(tau ** 2 * ZZt + sig ** 2 * np.eye(n), lower=True, check_finite=False)
        ViX = cho_solve(cf, X, check_finite=False)
        Viy = cho_solve(cf, y, check_finite=False)
        A = X.T @ ViX
        beta = np.linalg.solve(A, X.T @ Viy)
        r = y - X @ beta
        return cf, A, beta, r

    def nll(th):
        try:
            cf, A, beta, r = parts(th)
        except np.linalg.LinAlgError:
            return 1e12
        Vir = cho_solve(cf, r, check_finite=False)
        sign, logdetA = np.linalg.slogdet(A)
        return 0.5 * (2.0 * np.sum(np.log(np.diag(cf[0]))) + logdetA + r @ Vir)

    lo, hi = np.log([bounds[0][0], bounds[1][0]]), np.log([bounds[0][1], bounds[1][1]])
    rng = np.random.default_rng(seed)
    best = None
    for s0 in [np.log([0.1, 0.15])] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]:
        res = minimize(nll, s0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or res.fun < best.fun:
            best = res
    cf, A, beta, _ = parts(best.x)
    tau, sig = np.exp(best.x)
    return beta, np.linalg.inv(A), float(tau), float(sig)


def oneway_components(res, groups):
    """Method-of-moments one-way random-effects estimates (tau^2, sig^2) and mean."""
    res = np.asarray(res, float)
    inv = group_index(groups)
    G = inv.max() + 1
    n_g = np.bincount(inv, minlength=G).astype(float)
    m_g = np.bincount(inv, weights=res, minlength=G) / n_g
    N = res.size
    grand = res.mean()
    ssw = np.sum((res - m_g[inv]) ** 2)
    ssb = np.sum(n_g * (m_g - grand) ** 2)
    msw = ssw / max(N - G, 1)
    msb = ssb / max(G - 1, 1)
    n0 = (N - np.sum(n_g ** 2) / N) / max(G - 1, 1)
    tau2 = max(0.0, (msb - msw) / n0) if G > 1 else 0.0
    return float(tau2), float(msw), float(grand)


class ResidualVarianceEB:
    """Few-shot conditioning for a point predictor with programme-level residual
    variance tau2 and within-programme residual variance sig2."""

    def __init__(self, tau2, sig2):
        self.tau2, self.sig2 = float(tau2), float(sig2)

    def condition(self, mu_t, mu_k=None, y_k=None):
        mu_t = np.asarray(mu_t, float)
        if mu_k is None or len(mu_k) == 0:
            return mu_t, np.full(mu_t.shape, self.tau2 + self.sig2)
        e = np.asarray(y_k, float) - np.asarray(mu_k, float)
        k = e.size
        shrink = self.tau2 / (k * self.tau2 + self.sig2)
        u = shrink * e.sum()
        var_u = self.tau2 * self.sig2 / (k * self.tau2 + self.sig2)
        return mu_t + u, np.full(mu_t.shape, self.sig2 + var_u)
