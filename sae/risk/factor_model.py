"""
PCA-based statistical factor risk model.

A statistical risk model decomposes the covariance matrix into:
  Sigma = B * F * B^T + Delta

where:
  B     (N x K) = factor exposures (loadings)
  F     (K x K) = factor covariance matrix
  Delta (N x N) = diagonal specific (idiosyncratic) covariance

We estimate B via PCA on the denoised correlation matrix, using the K
largest eigenvalues (outside the Marchenko-Pastur noise bulk).

The model is similar in spirit to BARRA's USE4 and Axioma fundamental
factor models, but uses statistical rather than fundamental factors.

Usage
-----
>>> model = PCAFactorModel(n_factors=10)
>>> model.fit(returns)
>>> sigma_hat = model.covariance()
>>> factor_ret = model.factor_returns(returns)
"""

import numpy as np
import pandas as pd
from typing import Optional
from .covariance import LedoitWolfEstimator, rmt_denoise


class PCAFactorModel:
    """
    PCA statistical factor risk model.

    Parameters
    ----------
    n_factors       : Number of statistical factors (default: auto from RMT).
    shrink          : Apply Ledoit-Wolf shrinkage before PCA (default: True).
    min_var_explained: Minimum cumulative variance to determine n_factors automatically.
    """

    def __init__(
        self,
        n_factors: Optional[int] = None,
        shrink: bool = True,
        min_var_explained: float = 0.6,
    ):
        self.n_factors = n_factors
        self.shrink = shrink
        self.min_var_explained = min_var_explained

        # Fitted attributes
        self.loadings_: np.ndarray = None      # (N, K)
        self.factor_cov_: np.ndarray = None    # (K, K)
        self.specific_var_: np.ndarray = None  # (N,)
        self.eigvals_: np.ndarray = None
        self.eigvecs_: np.ndarray = None
        self.columns_: list = None
        self.n_obs_: int = None

    # ── Fitting ──────────────────────────────────────────────────────────

    def fit(self, returns: pd.DataFrame) -> "PCAFactorModel":
        """
        Fit the factor model to a returns DataFrame.

        Parameters
        ----------
        returns : (T, N) DataFrame of daily returns.
        """
        self.columns_ = returns.columns.tolist()
        X = returns.values
        T, N = X.shape
        self.n_obs_ = T

        # 1. Sample covariance (optionally shrunk)
        if self.shrink:
            lw = LedoitWolfEstimator()
            lw.fit(X)
            cov_est = lw.covariance_
        else:
            cov_est = np.cov(X.T)

        # 2. RMT denoising
        cov_denoised, rmt_info = rmt_denoise(cov_est, T)
        self.rmt_info_ = rmt_info

        # 3. Standardise to correlation for PCA
        std = np.sqrt(np.diag(cov_denoised))
        std[std < 1e-12] = 1.0
        corr = cov_denoised / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)

        # 4. PCA via eigendecomposition (descending order)
        eigvals_all, eigvecs_all = np.linalg.eigh(corr)
        idx = np.argsort(eigvals_all)[::-1]
        eigvals_all = eigvals_all[idx]
        eigvecs_all = eigvecs_all[:, idx]

        # 5. Determine number of factors
        if self.n_factors is None:
            K = rmt_info["n_signal_factors"]
            K = max(K, 1)
            # Also cap by min_var_explained
            cum_var = np.cumsum(eigvals_all) / eigvals_all.sum()
            k_var = int(np.searchsorted(cum_var, self.min_var_explained)) + 1
            K = max(K, k_var)
            K = min(K, N // 2)  # sanity cap
        else:
            K = min(self.n_factors, N - 1)

        self.K_ = K
        self.eigvals_ = eigvals_all[:K]
        self.eigvecs_ = eigvecs_all[:, :K]

        # 6. Factor loadings: B = eigvecs * sqrt(eigvals)   (N x K)
        self.loadings_ = eigvecs_all[:, :K] * np.sqrt(eigvals_all[:K])

        # 7. Factor covariance = I_K (by construction from PCA)
        self.factor_cov_ = np.diag(eigvals_all[:K])

        # 8. Specific variance = diag(Sigma_full) - diag(B B^T)
        systematic_var = np.sum(self.loadings_ ** 2, axis=1)  # (N,)
        total_var = np.diag(cov_denoised)
        self.specific_var_ = np.maximum(total_var - systematic_var, 1e-8)

        return self

    # ── Covariance matrices ───────────────────────────────────────────────

    def covariance(self) -> pd.DataFrame:
        """
        Full (N x N) covariance matrix: B * F * B^T + diag(Delta).
        """
        systematic = self.loadings_ @ self.factor_cov_ @ self.loadings_.T
        specific = np.diag(self.specific_var_)
        cov = systematic + specific
        return pd.DataFrame(cov, index=self.columns_, columns=self.columns_)

    def factor_exposures(self) -> pd.DataFrame:
        """Return (N x K) factor loading DataFrame."""
        cols = [f"PC{k+1}" for k in range(self.K_)]
        return pd.DataFrame(self.loadings_, index=self.columns_, columns=cols)

    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total variance explained by each factor."""
        return self.eigvals_ / self.eigvals_.sum()

    # ── Factor returns ────────────────────────────────────────────────────

    def factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Project returns onto factor space: f_t = B^+ * r_t (least squares).

        Returns (T, K) DataFrame of factor returns.
        """
        X = returns.reindex(columns=self.columns_).values
        # Pseudo-inverse of loadings matrix
        B_pinv = np.linalg.pinv(self.loadings_)  # (K, N)
        f = (B_pinv @ X.T).T  # (T, K)
        cols = [f"PC{k+1}" for k in range(self.K_)]
        return pd.DataFrame(f, index=returns.index, columns=cols)
