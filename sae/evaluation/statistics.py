"""
Statistical utilities for factor evaluation and hypothesis testing.

Functions
---------
newey_west_tstat     : Newey-West heteroskedasticity- and autocorrelation-robust t-stat
benjamini_hochberg   : BH step-up procedure for controlling false discovery rate
bonferroni_correction: Family-wise error rate correction
fama_macbeth         : Fama-MacBeth cross-sectional regression t-statistics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Sequence, Tuple


def newey_west_tstat(
    x: np.ndarray,
    max_lag: int = None,
    ddof: int = 0,
) -> float:
    """
    Newey-West (1987) heteroskedasticity- and autocorrelation-consistent t-statistic.

    t_NW = mean(x) / NW_std_error(x)

    The NW long-run variance estimator uses Bartlett (triangular) kernel:
      V_NW = gamma_0 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * gamma_j

    where gamma_j = (1/T) * sum_t (x_t - x_bar)(x_{t-j} - x_bar).

    Parameters
    ----------
    x       : 1-D array of observations (e.g. IC series).
    max_lag : Number of lags.  Default: 4*(T/100)^0.25 (Andrews 1991 rule).
    ddof    : Degrees of freedom correction (usually 0).

    Returns
    -------
    float : NW t-statistic.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return 0.0

    if max_lag is None:
        max_lag = max(1, int(np.floor(4.0 * (n / 100.0) ** 0.25)))

    x_demeaned = x - x.mean()
    # gamma_0
    gamma_0 = np.dot(x_demeaned, x_demeaned) / n
    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        weight = 1.0 - j / (max_lag + 1.0)  # Bartlett kernel
        gamma_j = np.dot(x_demeaned[j:], x_demeaned[:-j]) / n
        nw_var += 2.0 * weight * gamma_j

    # Variance of the sample mean
    nw_var_mean = max(nw_var / n, 1e-15)
    return float(x.mean() / np.sqrt(nw_var_mean))


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Benjamini-Hochberg (1995) step-up procedure for controlling FDR.

    Parameters
    ----------
    p_values : sequence of p-values (one per hypothesis).
    alpha    : desired FDR level (default 0.05).

    Returns
    -------
    rejected  : bool array — True if hypothesis i is rejected.
    adj_pvals : BH-adjusted p-values (use for ranking).
    threshold : the BH critical value used.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    # BH critical values: p_(k) <= k/m * alpha
    critical = ranks / m * alpha
    # Find largest k where p_(k) <= k/m*alpha
    # Because of step-up, we find the largest k and reject all <= k
    bh_condition = (p[order] <= (np.arange(1, m + 1) / m * alpha))
    if not bh_condition.any():
        threshold = 0.0
        rejected = np.zeros(m, dtype=bool)
    else:
        k_max = np.where(bh_condition)[0][-1]
        threshold = p[order[k_max]]
        rejected = p <= threshold

    # Adjusted p-values: p_adj_i = min_{j >= i} (m * p_(j) / j)
    adj_pvals = np.empty(m)
    min_adj = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        adj = min(1.0, m * p[idx] / (i + 1))
        min_adj = min(min_adj, adj)
        adj_pvals[idx] = min_adj

    return rejected, adj_pvals, threshold


def bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Bonferroni family-wise error rate correction.

    Rejects hypothesis i if p_i <= alpha / m.

    Returns
    -------
    rejected  : bool array.
    threshold : alpha / m.
    """
    p = np.asarray(p_values, dtype=float)
    threshold = alpha / len(p)
    return p <= threshold, threshold


def fama_macbeth(
    factors: pd.DataFrame,
    returns: pd.DataFrame,
    n_lags: int = None,
) -> pd.DataFrame:
    """
    Fama-MacBeth (1973) two-pass cross-sectional regression.

    Pass 1: at each time t, run cross-sectional OLS:
      return_{i,t} = lambda_0 + lambda_1 * f_{i,t-1} + ... + epsilon_{i,t}

    Pass 2: time-series mean and NW t-stat of the lambda series.

    Parameters
    ----------
    factors : DataFrame (T x N) of factor exposures (one column per stock).
    returns : DataFrame (T x N) of forward daily returns.
    n_lags  : NW lags.  Default: auto.

    Returns
    -------
    DataFrame with columns: lambda_mean, lambda_std, t_stat_NW
    """
    dates = factors.index.intersection(returns.index)
    lambdas = []

    for date in dates:
        f = factors.loc[date].dropna()
        r = returns.loc[date, f.index].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        X = np.column_stack([np.ones(len(common)), f[common].values])
        y = r[common].values
        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            lambdas.append(coef)
        except np.linalg.LinAlgError:
            continue

    if not lambdas:
        return pd.DataFrame()

    lam = np.array(lambdas)
    result = {}
    for j, label in enumerate(["intercept", "lambda"]):
        series = lam[:, j]
        result[label] = {
            "mean": series.mean(),
            "std": series.std(ddof=1),
            "t_stat_NW": newey_west_tstat(series, n_lags),
        }

    return pd.DataFrame(result).T
