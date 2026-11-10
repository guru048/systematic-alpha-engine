"""
Fundamental / valuation alpha factors.

Factors are based on balance-sheet and income data aligned to price data.
Since fundamental data is updated quarterly, signals change slowly and IC
is typically computed at 1-month or 3-month forward horizons.

Implemented factors
-------------------
Value
  book_to_price     : B/P ratio (book value / market cap)
  earnings_to_price : estimated earnings yield (ROE * book / price)
  sales_to_price    : proxy sales yield (ROE proxy for asset efficiency)

Quality
  roe               : Return on Equity
  asset_turnover    : ROE / (D/E + 1) — simple profitability-per-unit-leverage
  low_leverage      : negative D/E ratio (high quality = low debt)
  earnings_quality  : stability of earnings (1 / rolling std of earnings yield)

Fama-French proxies
  ff_smb_proxy      : negative of log market cap (small-minus-big)
  ff_hml_proxy      : book-to-price (high-minus-low)
"""

import numpy as np
import pandas as pd


class FundamentalFactors:
    """Compute fundamental alpha factors from price + balance-sheet data."""

    def __init__(
        self,
        prices: pd.DataFrame,
        fundamentals: dict,
    ):
        """
        Parameters
        ----------
        prices       : DataFrame (T x N) of closing prices.
        fundamentals : dict with keys 'book_value', 'roe', 'debt_to_equity'.
                       Each is a DataFrame (T x N).
        """
        self.prices = prices
        self.book_value = fundamentals["book_value"]
        self.roe = fundamentals["roe"]
        self.debt_to_equity = fundamentals["debt_to_equity"]

    # ── Value ────────────────────────────────────────────────────────────

    def book_to_price(self) -> pd.DataFrame:
        """Book-to-price ratio = book value per share / closing price."""
        return self.book_value / (self.prices + 1e-8)

    def earnings_to_price(self) -> pd.DataFrame:
        """
        Earnings yield = (ROE * Book Value) / Price.
        Approximates E/P using reported ROE times book.
        """
        earnings = self.roe * self.book_value
        return earnings / (self.prices + 1e-8)

    def sales_to_price(self) -> pd.DataFrame:
        """
        Proxy for sales yield: asset turnover × book / price.
        (Asset turnover approximated as ROE / (D/E + 1) normalised.)
        """
        asset_turnover = self.roe / (self.debt_to_equity + 1.0)
        sales_proxy = asset_turnover * self.book_value
        return sales_proxy / (self.prices + 1e-8)

    # ── Quality ──────────────────────────────────────────────────────────

    def quality_roe(self) -> pd.DataFrame:
        """Return on equity — higher is higher quality."""
        return self.roe.copy()

    def low_leverage(self) -> pd.DataFrame:
        """Negative debt-to-equity: high value → low leverage → high quality."""
        return -self.debt_to_equity

    def earnings_stability(self) -> pd.DataFrame:
        """
        Earnings quality: 1 / rolling std of quarterly earnings yield.
        Stocks with stable earnings get a higher score.
        """
        ey = self.earnings_to_price()
        rolling_std = ey.rolling(252, min_periods=63).std()
        return 1.0 / (rolling_std + 1e-6)

    # ── Fama-French proxies ──────────────────────────────────────────────

    def ff_smb_proxy(self) -> pd.DataFrame:
        """Negative log market cap — small-cap premium factor."""
        return -np.log(self.prices + 1e-8)

    def ff_hml_proxy(self) -> pd.DataFrame:
        """Book-to-price — high-minus-low value factor."""
        return self.book_to_price()

    # ── Composite ────────────────────────────────────────────────────────

    def all_factors(self) -> dict:
        return {
            "book_to_price": self.book_to_price(),
            "earnings_to_price": self.earnings_to_price(),
            "sales_to_price": self.sales_to_price(),
            "roe": self.quality_roe(),
            "low_leverage": self.low_leverage(),
            "earnings_stability": self.earnings_stability(),
            "ff_smb_proxy": self.ff_smb_proxy(),
            "ff_hml_proxy": self.ff_hml_proxy(),
        }
