"""
Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

References
----------
Bailey, D.H. and Lopez de Prado, M. (2012).  "The Sharpe Ratio Efficient Frontier."
  Journal of Risk, 15(2), pp. 3-44.

Bailey, D.H. and Lopez de Prado, M. (2014).  "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
  Journal of Portfolio Management, 40(5), pp. 94-107.

PSR
---
The PSR measures the probability that the true Sharpe ratio SR* exceeds
a benchmark SR_0, given that we observed SR_hat over T returns:

  PSR(SR_0) = Phi [ (SR_hat - SR_0) * sqrt(T-1)
                    / sqrt(1 - gamma_1 * SR_hat + (gamma_2 - 1)/4 * SR_hat^2) ]

where gamma_1 = skewness, gamma_2 = excess kurtosis.

DSR
---
The DSR adjusts for multiple testing by computing the expected maximum Sharpe
ratio across N_T independently evaluated strategies using order-statistics:

  E[SR_max(N_T)] = (1 - gamma_E) * Phi^{-1}(1 - 1/N_T)
                   + gamma_E    * Phi^{-1}(1 - 1/(N_T * e))

  DSR = PSR(E[SR_max(N_T)])

where gamma_E ≈ 0.5772 is the Euler-Mascheroni constant.
"""

import numpy as np
from scipy import stats
from typing import Union


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    frequency: int = 252,
) -> dict:
    """
    Compute the Probabilistic Sharpe Ratio.

    Parameters
    ----------
    returns      : array of returns (at the given frequency).
    sr_benchmark : annualised benchmark Sharpe ratio to test against.
    frequency    : trading days per year (252 for daily, 52 for weekly).

    Returns
    -------
    dict with keys: SR_hat, PSR, skew, kurtosis
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 10:
        return {"SR_hat": np.nan, "PSR": np.nan, "skew": np.nan, "kurtosis": np.nan}

    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma < 1e-12:
        return {"SR_hat": 0.0, "PSR": 0.0, "skew": 0.0, "kurtosis": 0.0}

    # Sample statistics
    sr_hat_per_period = mu / sigma  # per-period (not annualised yet)
    skew = float(stats.skew(r))
    kurt_excess = float(stats.kurtosis(r, fisher=True))  # excess kurtosis

    # PSR denominator: adjustment for non-normality
    denom_sq = 1.0 - skew * sr_hat_per_period + (kurt_excess + 2.0) / 4.0 * sr_hat_per_period ** 2
    denom_sq = max(denom_sq, 1e-10)

    # De-annualise the benchmark for period-level comparison
    sr_bench_period = sr_benchmark / np.sqrt(frequency)

    numer = (sr_hat_per_period - sr_bench_period) * np.sqrt(T - 1)
    psr = float(stats.norm.cdf(numer / np.sqrt(denom_sq)))

    return {
        "SR_hat": float(sr_hat_per_period * np.sqrt(frequency)),
        "PSR": psr,
        "skew": skew,
        "kurtosis": kurt_excess,
        "T": T,
    }


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 1,
    frequency: int = 252,
) -> dict:
    """
    Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    returns  : array of daily (or periodic) returns.
    n_trials : number of strategies/backtests evaluated (the "trial multiplicity").
               If n_trials == 1, DSR == PSR with benchmark 0.
    frequency: trading periods per year.

    Returns
    -------
    dict with keys: SR_hat, PSR, DSR, min_SR (expected maximum over n_trials),
                    skew, kurtosis
    """
    EULER_GAMMA = 0.5772156649015328

    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)

    base = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, frequency=frequency)
    if np.isnan(base["SR_hat"]):
        return {k: np.nan for k in ["SR_hat", "PSR", "DSR", "min_SR", "skew", "kurtosis"]}

    # PSR with 0 benchmark
    psr = base["PSR"]

    # Expected maximum Sharpe ratio across n_trials (order statistics)
    if n_trials > 1:
        # Bailey & Lopez de Prado (2014) Eq. 8
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials) if n_trials > 1 else 0.0
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)) if n_trials > 1 else 0.0
        min_sr_annual = (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2
    else:
        min_sr_annual = 0.0

    # DSR = PSR evaluated at the expected maximum SR
    dsr_result = probabilistic_sharpe_ratio(r, sr_benchmark=min_sr_annual, frequency=frequency)
    dsr = dsr_result["PSR"]

    return {
        "SR_hat": base["SR_hat"],
        "PSR": psr,
        "DSR": dsr,
        "min_SR": min_sr_annual,
        "skew": base["skew"],
        "kurtosis": base["kurtosis"],
        "n_trials": n_trials,
        "T": T,
    }


def min_track_record_length(
    sr_benchmark: float,
    sr_hat: float,
    alpha: float = 0.05,
    skew: float = 0.0,
    kurt_excess: float = 0.0,
) -> float:
    """
    Minimum Track Record Length (MinTRL) — how many observations are needed
    to establish SR_hat > sr_benchmark at significance level alpha.

    MinTRL = 1 + (1 - gamma_1 * SR + (gamma_2-1)/4 * SR^2) * (z_alpha / (SR - SR_0))^2

    where SR = annualised SR_hat, SR_0 = annualised sr_benchmark.
    """
    sr_diff = sr_hat - sr_benchmark
    if abs(sr_diff) < 1e-10:
        return np.inf
    z_alpha = stats.norm.ppf(1.0 - alpha)
    # Per-period ratio
    adj = 1.0 - skew * sr_hat + (kurt_excess - 1.0) / 4.0 * sr_hat ** 2
    adj = max(adj, 1e-6)
    return 1.0 + adj * (z_alpha / sr_diff) ** 2
