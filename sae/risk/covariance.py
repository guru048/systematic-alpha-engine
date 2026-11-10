"""
Covariance estimation: Ledoit-Wolf shrinkage and Random Matrix Theory denoising.

References
----------
Ledoit, O. and Wolf, M. (2004).  "A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices."  Journal of Multivariate Analysis,
  88(2), pp. 365-411.

Marchenko, V.A. and Pastur, L.A. (1967).  "Distribution of Eigenvalues for
  Some Sets of Random Matrices."  Matematicheskii Sbornik, 72(4), pp. 507-536.

Bouchaud, J.-P. and Potters, M. (2009).  "Financial Applications of Random
  Matrix Theory: a Short Review."  arXiv:0910.1205.
"""

import numpy as np
import pandas as pd
from scipy import linalg


# ── Ledoit-Wolf shrinkage ─────────────────────────────────────────────────────

class LedoitWolfEstimator:
    """
    Analytical Ledoit-Wolf covariance shrinkage estimator.

    Shrinks the sample covariance toward a scaled identity target:
      Sigma_hat = (1 - alpha) * S + alpha * mu * I
    where alpha is chosen to minimise the Frobenius-norm distance to the oracle.

    Uses scikit-learn's LedoitWolf under the hood for the analytical solution.
    """

    def __init__(self, assume_centered: bool = False):
        self.assume_centered = assume_centered
        self.covariance_: np.ndarray = None
        self.precision_: np.ndarray = None
        self.shrinkage_: float = None

    def fit(self, X: np.ndarray) -> "LedoitWolfEstimator":
        """
        Fit to returns matrix X of shape (T, N).

        Parameters
        ----------
        X : (T, N) array of returns (rows = observations, cols = assets).
        """
        from sklearn.covariance import LedoitWolf as _LW
        lw = _LW(assume_centered=self.assume_centered)
        lw.fit(X)
        self.covariance_ = lw.covariance_
        self.precision_ = lw.precision_
        self.shrinkage_ = float(lw.shrinkage_)
        return self

    def fit_dataframe(self, returns: pd.DataFrame) -> "LedoitWolfEstimator":
        self.columns_ = returns.columns.tolist()
        return self.fit(returns.values)

    def as_dataframe(self) -> pd.DataFrame:
        """Return covariance as a labelled DataFrame."""
        cols = getattr(self, "columns_", None) or range(self.covariance_.shape[0])
        return pd.DataFrame(self.covariance_, index=cols, columns=cols)


# ── Random Matrix Theory denoising ───────────────────────────────────────────

def marchenko_pastur_pdf(
    sigma_sq: float,
    q: float,
    n_points: int = 1000,
) -> tuple:
    """
    Marchenko-Pastur probability density for a Wishart matrix.

    The limiting density of eigenvalues of (1/T) X^T X where X ~ N(0, sigma_sq)
    has support [lambda_min, lambda_max]:

      lambda_{min,max} = sigma_sq * (1 ± sqrt(1/q))^2

    Parameters
    ----------
    sigma_sq : variance of each entry in X.
    q        : ratio T/N (must be > 1 for the law to hold).

    Returns
    -------
    (lambda_grid, pdf_values)
    """
    q_inv = 1.0 / q
    lam_min = sigma_sq * (1.0 - np.sqrt(q_inv)) ** 2
    lam_max = sigma_sq * (1.0 + np.sqrt(q_inv)) ** 2

    lam = np.linspace(lam_min + 1e-10, lam_max, n_points)
    pdf = (q / (2.0 * np.pi * sigma_sq)) * np.sqrt((lam_max - lam) * (lam - lam_min)) / lam
    return lam, pdf


def rmt_denoise(
    cov: np.ndarray,
    n_obs: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Denoise a covariance matrix using Random Matrix Theory.

    Eigenvalues below the Marchenko-Pastur noise threshold are replaced by
    their mean (preserving trace), effectively filtering out noise dimensions.

    Parameters
    ----------
    cov   : (N, N) covariance matrix (sample or shrunk).
    n_obs : number of observations T used to estimate `cov`.
    alpha : scaling factor for the MP threshold (default 1.0 = exact threshold).
            Use alpha > 1 for more aggressive denoising.

    Returns
    -------
    denoised : (N, N) positive-semidefinite denoised covariance matrix.
    """
    N = cov.shape[0]
    q = n_obs / N  # T/N ratio

    # Standardise to correlation for eigen-analysis
    std = np.sqrt(np.diag(cov))
    std[std < 1e-12] = 1e-12
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    # Eigendecomposition of correlation matrix
    eigvals, eigvecs = np.linalg.eigh(corr)  # ascending order

    # Marchenko-Pastur noise upper bound
    # sigma_sq ≈ 1 (for a correlation matrix) minus the variance explained by signal
    # We estimate sigma_sq by fitting the MP bulk to the empirical distribution
    sigma_sq = np.mean(eigvals[eigvals <= (1.0 + np.sqrt(1.0 / q)) ** 2 * alpha])
    sigma_sq = max(sigma_sq, 1e-6)

    lambda_max_noise = sigma_sq * (1.0 + np.sqrt(1.0 / q)) ** 2 * alpha

    # Identify signal vs noise eigenvalues
    is_noise = eigvals < lambda_max_noise

    # Replace noise eigenvalues with their mean (preserves trace of correlation matrix)
    noise_eigvals = eigvals[is_noise]
    noise_mean = noise_eigvals.mean() if noise_eigvals.size > 0 else eigvals.min()

    denoised_eigvals = eigvals.copy()
    denoised_eigvals[is_noise] = noise_mean

    # Clip to non-negative (numerical safety)
    denoised_eigvals = np.maximum(denoised_eigvals, 1e-8)

    # Reconstruct correlation matrix
    corr_denoised = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T
    # Renormalise diagonal to 1 (small numerical errors)
    d = np.sqrt(np.diag(corr_denoised))
    d[d < 1e-12] = 1.0
    corr_denoised /= np.outer(d, d)
    np.fill_diagonal(corr_denoised, 1.0)

    # Rescale back to covariance
    cov_denoised = corr_denoised * np.outer(std, std)

    n_signal = int((~is_noise).sum())
    return cov_denoised, {"n_signal_factors": n_signal, "lambda_max_noise": lambda_max_noise}
