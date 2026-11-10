"""
Technical / price-based alpha factors.

All factors return DataFrames (T x N) with NaN for warm-up periods.
Factors are raw (unranked) values; ranking happens in FactorEvaluator.

Implemented factors
-------------------
Momentum
  mom_12_1   : 12-1 month past return (Jegadeesh & Titman 1993)
  mom_6_1    : 6-1 month past return
  mom_1m_rev : 1-month reversal (DeBondt & Thaler)
  rsi_14     : 14-day RSI

Volatility / Risk
  vol_21     : 21-day realised daily volatility (annualised)
  vol_63     : 63-day realised daily volatility (annualised)
  idio_vol   : idiosyncratic volatility (residual vs equal-weighted index)
  skew_60    : 60-day return skewness

Mean-Reversion
  z_score_52 : 52-week z-score of price (mean-reversion signal)
  corr_mkt   : 60-day correlation with equal-weight market (beta proxy)
"""

import numpy as np
import pandas as pd


class TechnicalFactors:
    """Compute technical / price-based alpha factors from OHLCV data."""

    def __init__(self, prices: pd.DataFrame, volume: pd.DataFrame):
        """
        Parameters
        ----------
        prices : DataFrame (T x N) of daily closing prices.
        volume : DataFrame (T x N) of daily share volume.
        """
        self.prices = prices
        self.volume = volume
        self.returns = prices.pct_change()

    # ── Momentum ────────────────────────────────────────────────────────

    def mom_12_1(self) -> pd.DataFrame:
        """12-1 month total return: price change from t-252 to t-21."""
        return self.prices.shift(21) / self.prices.shift(252) - 1.0

    def mom_6_1(self) -> pd.DataFrame:
        """6-1 month total return: price change from t-126 to t-21."""
        return self.prices.shift(21) / self.prices.shift(126) - 1.0

    def mom_1m_rev(self) -> pd.DataFrame:
        """1-month reversal: negative of last-month return (short-term mean reversion)."""
        return -(self.prices / self.prices.shift(21) - 1.0)

    def rsi_14(self) -> pd.DataFrame:
        """
        14-day RSI.  Values > 70 → overbought, < 30 → oversold.
        We negate so that low RSI (oversold) gives high factor value.
        """
        delta = self.returns.copy()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return -rsi  # negate: low RSI = oversold = buy signal

    # ── Volatility / Risk ────────────────────────────────────────────────

    def vol_21(self) -> pd.DataFrame:
        """21-day annualised realised volatility."""
        return self.returns.rolling(21, min_periods=15).std() * np.sqrt(252)

    def vol_63(self) -> pd.DataFrame:
        """63-day annualised realised volatility."""
        return self.returns.rolling(63, min_periods=40).std() * np.sqrt(252)

    def idio_vol(self) -> pd.DataFrame:
        """
        Idiosyncratic volatility: std of daily residuals over 63-day rolling window
        after removing equal-weight market return.
        """
        mkt = self.returns.mean(axis=1)
        resid = self.returns.subtract(mkt, axis=0)
        return resid.rolling(63, min_periods=40).std() * np.sqrt(252)

    def skew_60(self) -> pd.DataFrame:
        """60-day return skewness (negative skew = crash risk factor)."""
        return self.returns.rolling(60, min_periods=40).skew()

    # ── Mean-Reversion ───────────────────────────────────────────────────

    def z_score_52(self) -> pd.DataFrame:
        """
        52-week price z-score: (price - 252d mean) / 252d std.
        Negative z → mean-reversion buy signal.
        """
        mu = self.prices.rolling(252, min_periods=126).mean()
        sigma = self.prices.rolling(252, min_periods=126).std()
        return -(self.prices - mu) / (sigma + 1e-8)

    def corr_mkt_60(self) -> pd.DataFrame:
        """
        60-day rolling correlation with equal-weighted index return.
        Proxy for systematic beta; low correlation → more idiosyncratic.
        """
        mkt = self.returns.mean(axis=1)

        def _corr(col):
            return col.rolling(60, min_periods=40).corr(mkt)

        return self.returns.apply(_corr)

    # ── Convenience ─────────────────────────────────────────────────────

    def all_factors(self) -> dict:
        """Return dict of all technical factor DataFrames."""
        return {
            "mom_12_1": self.mom_12_1(),
            "mom_6_1": self.mom_6_1(),
            "mom_1m_rev": self.mom_1m_rev(),
            "rsi_14": self.rsi_14(),
            "vol_21": self.vol_21(),
            "vol_63": self.vol_63(),
            "idio_vol": self.idio_vol(),
            "skew_60": self.skew_60(),
            "z_score_52w": self.z_score_52(),
            "corr_mkt_60": self.corr_mkt_60(),
        }
