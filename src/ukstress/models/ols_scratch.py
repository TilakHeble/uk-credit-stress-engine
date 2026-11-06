"""Ordinary least squares from first principles.

Written to show what ``statsmodels.OLS`` does under the hood, and tested against it to machine
precision (``tests/test_ols_scratch.py``). The production models use statsmodels; this module
exists so every number in the notebooks can be traced back to linear algebra.

    beta_hat = (X'X)^-1 X'y                     normal equation (solved via QR, not an inverse)
    sigma2   = e'e / (n - k)                    unbiased residual variance
    Var(b)   = sigma2 (X'X)^-1                  classical (homoskedastic) covariance
    Var_HC0  = (X'X)^-1 X' diag(e^2) X (X'X)^-1 White's heteroskedasticity-robust "sandwich"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class OLSResult:
    params: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    sigma2: float
    rsquared: float
    rsquared_adj: float
    fvalue: float
    f_pvalue: float
    nobs: int
    df_resid: int
    cov: np.ndarray = field(repr=False)
    names: list[str] = field(default_factory=list)

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        q = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return np.column_stack([self.params - q * self.bse, self.params + q * self.bse])


def add_constant(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    return np.column_stack([np.ones(len(X)), X])


def fit_ols(X: np.ndarray, y: np.ndarray, cov_type: str = "nonrobust", names=None,
            maxlags: int = 4) -> OLSResult:
    """Fit OLS. ``X`` must already include a constant column if one is wanted.

    The coefficients come from a QR decomposition (X = QR, so R beta = Q'y). This is numerically
    stabler than inverting X'X, which squares the condition number.
    """
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    n, k = X.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")
    q, r = np.linalg.qr(X)
    beta = np.linalg.solve(r, q.T @ y)
    fitted = X @ beta
    resid = y - fitted
    df_resid = n - k
    sigma2 = resid @ resid / df_resid
    xtx_inv = np.linalg.inv(r.T @ r)  # (X'X)^-1 = (R'R)^-1

    if cov_type == "nonrobust":
        cov = sigma2 * xtx_inv
    elif cov_type == "HAC":
        cov = newey_west(X, resid, maxlags)  # statsmodels default: no small-sample correction
    elif cov_type in ("HC0", "HC1"):
        meat = X.T @ (X * resid[:, None] ** 2)
        cov = xtx_inv @ meat @ xtx_inv
        if cov_type == "HC1":
            cov *= n / df_resid
    else:
        raise ValueError(f"unsupported cov_type {cov_type!r}")

    bse = np.sqrt(np.diag(cov))
    tvalues = beta / bse
    pvalues = 2 * stats.t.sf(np.abs(tvalues), df_resid)

    has_const = np.any(np.all(X == 1.0, axis=0))
    centre = y.mean() if has_const else 0.0
    tss = np.sum((y - centre) ** 2)
    rss = resid @ resid
    r2 = 1 - rss / tss
    df_model = k - 1 if has_const else k
    r2_adj = 1 - (1 - r2) * (n - (1 if has_const else 0)) / df_resid
    fvalue = ((tss - rss) / df_model) / (rss / df_resid)
    f_pvalue = stats.f.sf(fvalue, df_model, df_resid)
    return OLSResult(beta, bse, tvalues, pvalues, resid, fitted, sigma2, r2, r2_adj, fvalue,
                     f_pvalue, n, df_resid, cov, list(names or []))


def newey_west(X: np.ndarray, resid: np.ndarray, maxlags: int) -> np.ndarray:
    """Newey-West (HAC) covariance with Bartlett weights w_l = 1 - l / (L + 1).

    The "meat" adds weighted autocovariances of the score x_t * e_t up to ``maxlags`` quarters,
    so it stays valid when residuals are autocorrelated (as overlapping trailing rates are).
    """
    X = np.asarray(X, float)
    u = X * np.asarray(resid, float)[:, None]
    meat = u.T @ u
    for lag in range(1, maxlags + 1):
        w = 1 - lag / (maxlags + 1)
        g = u[lag:].T @ u[:-lag]
        meat += w * (g + g.T)
    bread = np.linalg.inv(X.T @ X)
    return bread @ meat @ bread


def f_test_nested(rss_restricted: float, rss_full: float, q: int, df_resid_full: int):
    """F = ((RSS_r - RSS_f) / q) / (RSS_f / df_f) for q linear restrictions."""
    f = ((rss_restricted - rss_full) / q) / (rss_full / df_resid_full)
    return f, stats.f.sf(f, q, df_resid_full)


def information_criteria(rss: float, n: int, k: int) -> tuple[float, float]:
    """Gaussian log-likelihood based AIC and BIC (statsmodels' definitions)."""
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    return -2 * llf + 2 * k, -2 * llf + np.log(n) * k


def fit_gradient_descent(X, y, lr: float = 0.1, n_iter: int = 20_000, tol: float = 1e-12):
    """Minimise mean squared error by batch gradient descent.

    Features should be standardised first, or the step size that suits one feature diverges on
    another. Returns the coefficients and the loss history (to plot convergence).
    """
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    n, k = X.shape
    beta = np.zeros(k)
    history = []
    for _ in range(n_iter):
        resid = X @ beta - y
        loss = resid @ resid / n
        history.append(loss)
        grad = 2 / n * X.T @ resid
        beta -= lr * grad
        if len(history) > 1 and abs(history[-2] - loss) < tol:
            break
    return beta, np.array(history)


def leverage(X: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix H = X (X'X)^-1 X' without forming the n x n matrix."""
    q, _ = np.linalg.qr(np.asarray(X, float))
    return np.sum(q**2, axis=1)


def cooks_distance(res: OLSResult, X: np.ndarray) -> np.ndarray:
    """Cook's D_i = e_i^2 / (k s^2) * h_i / (1 - h_i)^2.

    How far all fitted values move when observation i is dropped.
    """
    h = leverage(X)
    k = X.shape[1]
    return res.resid**2 / (k * res.sigma2) * h / (1 - h) ** 2
