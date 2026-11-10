"""
Synthetic market data generator.

Generates realistic OHLCV + fundamental data for N stocks over T trading days
using an explicit linear factor model, so ground-truth factor signals are known.

Factor model:
  r_{i,t} = beta_mkt_i * f_mkt_t
            + beta_sec_i * f_sec_{s(i),t}
            + load_mom_i  * f_mom_t
            + load_val_i  * f_val_t
            + load_qual_i * f_qual_t
            + epsilon_{i,t}

where factor processes are AR(1) for persistence, and fundamental values
are constructed to correlate with the style loadings (so B/P ~ load_val,
ROE ~ load_qual), giving observable factors positive IC.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class MarketDataGenerator:
    """
    Generates synthetic OHLCV + fundamental data for a cross-section of stocks.

    Parameters
    ----------
    n_stocks : int
        Number of stocks to simulate (default 100).
    n_days : int
        Number of trading days (default 1260 ≈ 5 years).
    n_sectors : int
        Number of sectors (default 5).
    seed : int
        RNG seed for reproducibility.
    start_date : str
        First business day of the simulated universe.
    """

    def __init__(
        self,
        n_stocks: int = 100,
        n_days: int = 1260,
        n_sectors: int = 5,
        seed: int = 42,
        start_date: str = "2019-01-02",
    ):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.n_sectors = n_sectors
        self.seed = seed
        self.start_date = start_date

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Dict[str, Any]:
        """
        Generate and return a dict containing:
            returns, prices, volume, open, high, low  : DataFrames (T x N)
            fundamentals                               : dict of DataFrames
            metadata                                   : dict (sector_id, loadings…)
        """
        rng = np.random.RandomState(self.seed)
        N, T, S = self.n_stocks, self.n_days, self.n_sectors

        # ── Sector assignment ────────────────────────────────────────────
        stocks_per_sector = N // S
        sector_id = np.repeat(np.arange(S), stocks_per_sector)
        # pad if N % S != 0
        sector_id = np.concatenate(
            [sector_id, np.zeros(N - len(sector_id), dtype=int)]
        )

        # ── Factor loadings (stock-specific, fixed) ──────────────────────
        market_beta = rng.uniform(0.5, 1.5, N)
        sector_beta = rng.uniform(0.3, 0.8, N)

        # Style loadings: drawn from N(0, sigma_load)
        load_mom = rng.randn(N) * 0.35
        load_val = rng.randn(N) * 0.30
        load_qual = rng.randn(N) * 0.25

        # ── Factor return processes (AR(1) for autocorrelation/persistence)
        # Market: drift ~7% p.a., vol ~18% p.a.
        f_mkt = self._ar1(T, mu=0.07 / 252, sigma=0.18 / np.sqrt(252), phi=0.05, rng=rng)

        # Sector factors: 0 mean drift, 12% annualised vol
        f_sec = np.column_stack(
            [self._ar1(T, mu=0.0, sigma=0.12 / np.sqrt(252), phi=0.08, rng=rng) for _ in range(S)]
        )

        # Style factors (small positive drift = factor premium)
        f_mom = self._ar1(T, mu=0.04 / 252, sigma=0.08 / np.sqrt(252), phi=0.12, rng=rng)
        f_val = self._ar1(T, mu=0.025 / 252, sigma=0.065 / np.sqrt(252), phi=0.08, rng=rng)
        f_qual = self._ar1(T, mu=0.025 / 252, sigma=0.055 / np.sqrt(252), phi=0.06, rng=rng)

        # ── Idiosyncratic shocks ─────────────────────────────────────────
        idio_ann_vol = rng.uniform(0.18, 0.40, N)
        idio_daily_vol = idio_ann_vol / np.sqrt(252)
        epsilon = rng.randn(T, N) * idio_daily_vol

        # ── Stock returns ────────────────────────────────────────────────
        returns = np.zeros((T, N))
        for i in range(N):
            s = sector_id[i]
            returns[:, i] = (
                market_beta[i] * f_mkt
                + sector_beta[i] * f_sec[:, s]
                + load_mom[i] * f_mom
                + load_val[i] * f_val
                + load_qual[i] * f_qual
                + epsilon[:, i]
            )

        # ── Prices (geometric Brownian + log-returns) ────────────────────
        log_prices = np.log(100.0) + np.cumsum(returns, axis=0)
        prices = np.exp(log_prices)

        # ── OHLC (simple intraday spread model) ──────────────────────────
        abs_ret = np.abs(returns)
        intraday_range = abs_ret * prices * rng.uniform(1.0, 2.0, (T, N))
        high = prices + intraday_range * rng.uniform(0, 1, (T, N))
        low = np.maximum(
            prices - intraday_range * rng.uniform(0, 1, (T, N)),
            prices * 0.5,  # floor at 50% to avoid negatives
        )
        open_ = prices * np.exp(rng.randn(T, N) * 0.004)
        open_ = np.clip(open_, low, high)

        # ── Volume (log-normal, higher on big-move days) ─────────────────
        base_vol = rng.lognormal(np.log(5e5), 0.6, (T, N))
        vol_multiplier = 1.0 + 3.0 * abs_ret / (abs_ret.mean() + 1e-8)
        volume = base_vol * vol_multiplier

        # ── Fundamental data ─────────────────────────────────────────────
        # Book value: correlated with load_val so B/P is a valid factor
        book_value_base = 50.0 * np.exp(-load_val * 0.6 + rng.randn(N) * 0.3)
        book_value_base = np.clip(book_value_base, 5.0, 300.0)

        # ROE: correlated with load_qual so ROE is a valid factor
        roe_base = 0.12 + load_qual * 0.15 + rng.randn(N) * 0.04
        roe_base = np.clip(roe_base, 0.02, 0.35)

        # Debt/equity: inversely correlated with load_qual (high quality → low leverage)
        debt_to_equity = np.exp(0.5 - load_qual * 0.4 + rng.randn(N) * 0.3)
        debt_to_equity = np.clip(debt_to_equity, 0.05, 5.0)

        # Build time-varying book value (slowly evolving)
        book_df = self._evolve_fundamental(book_value_base, T, growth_rate=0.08, vol=0.12, rng=rng)
        roe_df = self._evolve_fundamental(roe_base, T, growth_rate=0.0, vol=0.06, rng=rng)
        de_df = self._evolve_fundamental(debt_to_equity, T, growth_rate=0.0, vol=0.08, rng=rng)

        # ── Date index & stock names ──────────────────────────────────────
        dates = pd.bdate_range(self.start_date, periods=T)
        stocks = [f"S{i:03d}" for i in range(N)]

        def df(arr):
            return pd.DataFrame(arr, index=dates, columns=stocks)

        return {
            "returns": df(returns),
            "prices": df(prices),
            "volume": df(volume),
            "open": df(open_),
            "high": df(high),
            "low": df(low),
            "fundamentals": {
                "book_value": df(book_df),
                "roe": df(roe_df),
                "debt_to_equity": df(de_df),
            },
            "metadata": {
                "stocks": stocks,
                "dates": dates,
                "sector_id": sector_id,
                "n_sectors": S,
                # True factor loadings (for validation)
                "load_mom": load_mom,
                "load_val": load_val,
                "load_qual": load_qual,
                "market_beta": market_beta,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ar1(
        T: int,
        mu: float,
        sigma: float,
        phi: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """AR(1) process: x_t = mu*(1-phi) + phi*x_{t-1} + sigma*eps_t."""
        x = np.zeros(T)
        x[0] = mu + sigma * rng.randn()
        for t in range(1, T):
            x[t] = mu * (1 - phi) + phi * x[t - 1] + sigma * np.sqrt(1 - phi ** 2) * rng.randn()
        return x

    @staticmethod
    def _evolve_fundamental(
        base: np.ndarray,
        T: int,
        growth_rate: float,
        vol: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Slowly-evolving fundamental: annual mean reversion to base with lognormal noise.
        Updates quarterly to match real reporting cadence.
        """
        N = len(base)
        out = np.zeros((T, N))
        current = base.copy()
        out[0] = current
        quarter = 63  # ~63 trading days per quarter

        for t in range(1, T):
            if t % quarter == 0:
                # Quarterly update: mean-revert toward base + drift
                drift = (1 + growth_rate / 4)
                shock = np.exp(rng.randn(N) * vol / np.sqrt(4) - (vol ** 2) / 8)
                # 30% mean reversion per quarter
                current = 0.30 * base * drift + 0.70 * current * drift * shock
            out[t] = current

        return out
