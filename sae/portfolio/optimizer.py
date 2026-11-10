"""
Portfolio optimisation routines.

Implements four complementary methods covering different regimes:

  1. mean_variance_optimize   – Markowitz (1952) mean-variance optimisation
  2. risk_parity_optimize     – Equal Risk Contribution (Roncalli 2013)
  3. black_litterman           – Black-Litterman (1990) Bayesian view incorporation
  4. hierarchical_risk_parity  – Lopez de Prado (2016) HRP via Ward clustering

All functions accept a (N,) expected return vector mu and an (N, N) covariance
matrix sigma.  They return a (N,) weight vector summing to 1 (long-only by default).

References
----------
Markowitz, H. (1952).  "Portfolio Selection."  Journal of Finance, 7(1), pp. 77-91.
Black, F. and Litterman, R. (1990).  "Asset Allocation: Combining Investor Views
  with Market Equilibrium."  Goldman Sachs Fixed Income Research.
Ledoit, O. and Wolf, M. (2003).  "Honey, I Shrunk the Sample Covariance Matrix."
  Journal of Portfolio Management, 30(4), pp. 110-119.
Lopez de Prado, M. (2016).  "Building Diversified Portfolios that Outperform
  Out-of-Sample."  Journal of Portfolio Management, 42(4), pp. 59-69.
Roncalli, T. (2013).  "Introduction to Risk Parity and Budgeting."
  Chapman & Hall/CRC Financial Mathematics Series.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from typing import Optional, List


# ── 1. Mean-Variance Optimisation ────────────────────────────────────────────

def mean_variance_optimize(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    max_weight: float = 0.10,
    target_vol: Optional[float] = None,
) -> np.ndarray:
    """
    Markowitz mean-variance portfolio.

    Solves:
      min  w^T Sigma w  -  (1/risk_aversion) * mu^T w
      s.t. sum(w) = 1,  [w >= 0 if long_only],  [w_i <= max_weight]

    If target_vol is set, the solution is rescaled to match it.

    Parameters
    ----------
    mu            : (N,) expected returns.
    sigma         : (N, N) covariance matrix.
    risk_aversion : higher = more risk-averse (larger Sigma weight).
    long_only     : enforce w >= 0.
    max_weight    : per-asset cap.
    target_vol    : if set, rescale portfolio to this annualised vol.

    Returns
    -------
    (N,) weight vector.
    """
    N = len(mu)

    def objective(w):
        port_var = w @ sigma @ w
        port_ret = mu @ w
        return port_var - port_ret / risk_aversion

    def grad(w):
        return 2.0 * sigma @ w - mu / risk_aversion

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_weight)] * N if long_only else [(-max_weight, max_weight)] * N

    w0 = np.ones(N) / N
    result = minimize(
        objective, w0, jac=grad, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    w = np.clip(result.x, 0, None)
    w /= w.sum() + 1e-12

    if target_vol is not None:
        port_vol = np.sqrt(w @ sigma @ w)
        if port_vol > 1e-8:
            w = w * target_vol / port_vol

    return w


# ── 2. Risk Parity (Equal Risk Contribution) ─────────────────────────────────

def risk_parity_optimize(
    sigma: np.ndarray,
    risk_budget: Optional[np.ndarray] = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Equal Risk Contribution (risk parity) portfolio.

    Finds w such that each asset contributes equally to portfolio variance:
      w_i * (Sigma w)_i / (w^T Sigma w) = 1/N   for all i.

    Solves the equivalent convex formulation (Roncalli 2013):
      min  sum_i  b_i * ln(w_i)  - 0.5 * ln(w^T Sigma w)
      s.t. w > 0

    then normalises to unit sum.

    Parameters
    ----------
    sigma       : (N, N) covariance matrix.
    risk_budget : (N,) risk budget weights (default: equal 1/N).

    Returns
    -------
    (N,) weight vector.
    """
    N = sigma.shape[0]
    b = np.ones(N) / N if risk_budget is None else np.array(risk_budget) / sum(risk_budget)

    def objective(w):
        port_var = w @ sigma @ w
        return 0.5 * np.log(port_var) - np.sum(b * np.log(w + 1e-12))

    def grad(w):
        port_var = float(w @ sigma @ w)
        return (sigma @ w) / port_var - b / (w + 1e-12)

    # Initialise close to equal-weight
    w0 = np.ones(N) / N
    bounds = [(1e-6, None)] * N
    result = minimize(
        objective, w0, jac=grad, method="L-BFGS-B",
        bounds=bounds, options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = np.maximum(result.x, 1e-10)
    w /= w.sum()
    return w


# ── 3. Black-Litterman ────────────────────────────────────────────────────────

def black_litterman(
    sigma: np.ndarray,
    mkt_weights: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: Optional[np.ndarray] = None,
    tau: float = 0.025,
    delta: float = 2.5,
) -> np.ndarray:
    """
    Black-Litterman model: combines market equilibrium with investor views.

    Equilibrium returns (reverse optimisation):
      Pi = delta * Sigma * w_mkt

    View uncertainty (default: proportional-to-variance diagonal):
      Omega = diag(P (tau Sigma) P^T)

    Posterior mean:
      mu_BL = [(tau Sigma)^{-1} + P^T Omega^{-1} P]^{-1}
              [(tau Sigma)^{-1} Pi + P^T Omega^{-1} Q]

    Posterior covariance:
      Sigma_BL = [(tau Sigma)^{-1} + P^T Omega^{-1} P]^{-1}

    Final MVO weights using mu_BL and (Sigma + Sigma_BL):
      w* = (1/delta) (Sigma + Sigma_BL)^{-1} mu_BL

    Parameters
    ----------
    sigma       : (N, N) covariance matrix.
    mkt_weights : (N,) market-cap weights (benchmark).
    P           : (K, N) pick matrix — each row defines one view.
    Q           : (K,) view expected returns.
    omega       : (K, K) view uncertainty matrix.  Default: proportional.
    tau         : scaling of prior uncertainty (typically 0.01-0.05).
    delta       : market risk aversion coefficient.

    Returns
    -------
    (N,) weight vector (not constrained to unit sum; re-normalise if desired).
    """
    N = sigma.shape[0]
    tau_sigma = tau * sigma
    Pi = delta * sigma @ mkt_weights  # equilibrium expected returns

    if omega is None:
        # Proportional-to-variance view uncertainty (Black-Litterman default)
        omega = np.diag(np.diag(P @ tau_sigma @ P.T))

    omega_inv = np.linalg.inv(omega + np.eye(len(Q)) * 1e-10)
    tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(N) * 1e-10)

    # Posterior precision
    post_prec = tau_sigma_inv + P.T @ omega_inv @ P
    post_cov = np.linalg.inv(post_prec + np.eye(N) * 1e-10)

    # Posterior mean
    post_mean = post_cov @ (tau_sigma_inv @ Pi + P.T @ omega_inv @ Q)

    # MVO on posterior
    total_sigma = sigma + post_cov
    total_sigma_inv = np.linalg.inv(total_sigma + np.eye(N) * 1e-10)

    w_bl = (1.0 / delta) * total_sigma_inv @ post_mean
    # Rescale to unit sum
    w_bl = np.clip(w_bl, 0, None)
    w_sum = w_bl.sum()
    if w_sum > 1e-8:
        w_bl /= w_sum
    else:
        w_bl = mkt_weights.copy()

    return w_bl


# ── 4. Hierarchical Risk Parity ───────────────────────────────────────────────

def hierarchical_risk_parity(
    sigma: np.ndarray,
    assets: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).

    Algorithm:
      1. Compute distance matrix D_ij = sqrt(0.5 * (1 - rho_ij)).
      2. Ward hierarchical clustering on D.
      3. Quasi-diagonalise: reorder assets by tree leaf order.
      4. Recursive bisection: at each split, allocate proportionally
         to the inverse variance of the sub-cluster equal-weight portfolio.

    Parameters
    ----------
    sigma  : (N, N) covariance matrix.
    assets : asset names (optional, for output mapping).

    Returns
    -------
    (N,) weight array in original asset order.
    """
    N = sigma.shape[0]
    if assets is None:
        assets = list(range(N))

    # 1. Distance matrix from correlation
    std = np.sqrt(np.diag(sigma))
    std = np.where(std < 1e-12, 1.0, std)
    corr = sigma / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))

    # 2. Ward linkage clustering
    n_upper = N * (N - 1) // 2
    condensed = dist[np.triu_indices(N, k=1)]
    link = linkage(condensed, method="ward")

    # 3. Quasi-diagonalisation: get leaf order
    sorted_idx = leaves_list(link).tolist()

    # 4. Recursive bisection
    weights = np.ones(N)
    clusters = [sorted_idx]

    while clusters:
        clusters_next = []
        for cluster in clusters:
            if len(cluster) > 1:
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_L = _cluster_var(sigma, left)
                var_R = _cluster_var(sigma, right)

                # Weight allocated to left = inverse-variance share of right
                alpha = 1.0 - var_L / (var_L + var_R + 1e-12)
                weights[right] *= alpha
                weights[left] *= (1.0 - alpha)

                clusters_next.extend([left, right])
        clusters = clusters_next

    # Map back to original order
    w = np.zeros(N)
    for rank, orig in enumerate(sorted_idx):
        w[orig] = weights[orig]

    w /= w.sum() + 1e-12
    return w


def _cluster_var(sigma: np.ndarray, idx: list) -> float:
    """Equal-weight portfolio variance within a cluster."""
    if not idx:
        return 0.0
    sub = sigma[np.ix_(idx, idx)]
    n = len(idx)
    w = np.ones(n) / n
    return float(w @ sub @ w)
