"""
Walk-forward backtesting engine.

Implements a strict walk-forward (expanding or rolling window) backtest
with no look-ahead bias.  At each rebalance date:
  1. Compute signals and covariance using ONLY past data.
  2. Construct portfolio weights.
  3. Apply weights to the NEXT out-of-sample period.

The engine is agnostic to the signal/portfolio function: callers pass
function objects following the SignalFn / PortfolioFn protocols.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Tuple


SignalFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
"""f(train_prices, train_volume) → (N,) signal scores"""

PortfolioFn = Callable[[np.ndarray, np.ndarray, list], np.ndarray]
"""f(signals, cov, assets) → (N,) portfolio weights summing to 1"""


class WalkForwardBacktest:
    """
    Walk-forward backtest engine.

    Parameters
    ----------
    train_window   : Days of history used to compute signals & covariance.
    test_window    : Days of out-of-sample evaluation per step.
    step           : Days between rebalances (must be <= test_window).
    expanding      : If True, use expanding (not rolling) training window.
    transaction_cost_bps : Round-trip transaction cost in basis points.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 63,
        expanding: bool = False,
        transaction_cost_bps: float = 10.0,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.expanding = expanding
        self.tc_bps = transaction_cost_bps / 10_000  # convert to fraction

    def run(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        signal_fn: SignalFn,
        portfolio_fn: PortfolioFn,
        benchmark_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run the walk-forward backtest.

        Parameters
        ----------
        prices        : (T, N) closing prices.
        volume        : (T, N) daily volume.
        signal_fn     : computes alpha signals from training data.
        portfolio_fn  : constructs portfolio from signals + covariance.
        benchmark_weights : (N,) benchmark weights (default: equal-weight).

        Returns
        -------
        dict with keys:
            portfolio_returns : Series (T_oos)
            benchmark_returns : Series (T_oos)
            weights_history   : dict {date: weight array}
            rebalance_dates   : list of dates
        """
        dates = prices.index
        assets = prices.columns.tolist()
        N = len(assets)
        T = len(dates)

        if benchmark_weights is None:
            benchmark_weights = np.ones(N) / N

        port_returns = {}
        bench_returns = {}
        weights_hist = {}
        rebalance_dates = []
        prev_weights = None

        t = self.train_window
        while t + self.test_window <= T:
            # Training window
            train_start = 0 if self.expanding else t - self.train_window
            train_prices = prices.iloc[train_start:t]
            train_volume = volume.iloc[train_start:t]
            test_prices = prices.iloc[t: t + self.test_window]

            # 1. Compute signals (strictly from past data)
            try:
                signals = signal_fn(train_prices, train_volume)
            except Exception:
                t += self.step
                continue

            # 2. Estimate covariance from training returns
            train_rets = train_prices.pct_change().dropna()
            if len(train_rets) < 20:
                t += self.step
                continue
            cov = np.cov(train_rets.values.T)
            cov = (cov + cov.T) / 2  # symmetrise

            # 3. Construct portfolio weights
            try:
                weights = portfolio_fn(signals, cov, assets)
            except Exception:
                weights = benchmark_weights.copy()

            weights = np.array(weights, dtype=float)
            weights = np.clip(weights, 0, None)
            w_sum = weights.sum()
            if w_sum > 1e-8:
                weights /= w_sum
            else:
                weights = benchmark_weights.copy()

            rebalance_date = dates[t]
            rebalance_dates.append(rebalance_date)

            # 4. Apply transaction costs at rebalance
            if prev_weights is not None:
                turnover = np.abs(weights - prev_weights).sum()
                tc_today = turnover * self.tc_bps
            else:
                tc_today = 0.0
            prev_weights = weights.copy()

            # 5. Compute out-of-sample returns
            test_rets = test_prices.pct_change().dropna()
            port_ret_series = (test_rets.values @ weights)
            bench_ret_series = (test_rets.values @ benchmark_weights)

            # Apply TC to first day
            if len(port_ret_series) > 0:
                port_ret_series[0] -= tc_today

            for i, d in enumerate(test_rets.index):
                port_returns[d] = float(port_ret_series[i])
                bench_returns[d] = float(bench_ret_series[i])
                weights_hist[d] = weights.copy()

            t += self.step

        return {
            "portfolio_returns": pd.Series(port_returns).sort_index(),
            "benchmark_returns": pd.Series(bench_returns).sort_index(),
            "weights_history": weights_hist,
            "rebalance_dates": rebalance_dates,
            "assets": assets,
        }
