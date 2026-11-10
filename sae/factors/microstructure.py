"""
Market microstructure alpha factors.

Implements classical microstructure measures from the literature:

  Roll (1984)   : effective bid-ask spread from return autocovariance
  Amihud (2002) : illiquidity = |return| / dollar volume
  Kyle (1985)   : lambda — price impact coefficient (OLS slope of |r| on sqrt(vol))

Lower liquidity (higher Amihud, higher Roll spread) predicts higher future returns
as a risk premium (liquidity risk factor).  Stocks with lower lambda have better
market depth and can absorb larger trades without large price impact.
"""

import numpy as np
import pandas as pd


class MicrostructureFactors:
    """Compute market microstructure factors from OHLCV data."""

    def __init__(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
    ):
        self.returns = returns
        self.prices = prices
        self.volume = volume
        self.dollar_volume = prices * volume  # shares × price = dollar volume

    # ── Roll (1984) bid-ask spread ───────────────────────────────────────

    def roll_spread(self, window: int = 63) -> pd.DataFrame:
        """
        Roll (1984) effective spread estimator.

          c = 2 * sqrt(max(-Cov(r_t, r_{t-1}), 0))

        Uses rolling covariance over `window` days.
        """
        r = self.returns

        def _roll_cov(col: pd.Series) -> pd.Series:
            cov = col.rolling(window, min_periods=window // 2).cov(col.shift(1))
            spread = 2.0 * np.sqrt((-cov).clip(lower=0.0))
            return spread

        return r.apply(_roll_cov)

    # ── Amihud (2002) illiquidity ─────────────────────────────────────────

    def amihud_illiquidity(self, window: int = 21) -> pd.DataFrame:
        """
        Amihud (2002) daily illiquidity ratio.

          ILLIQ_t = (1/D) * sum_{d=1}^{D} |r_d| / DVOL_d   × 10^6

        Units: price impact per $1M traded.  Higher = more illiquid.
        We negate for use as a *factor* (positive score → more liquid → better).
        """
        abs_ret = self.returns.abs()
        dvol = self.dollar_volume

        daily_illiq = abs_ret / (dvol + 1.0)  # +1 avoids divide-by-zero
        illiq = daily_illiq.rolling(window, min_periods=window // 2).mean() * 1e6

        return -illiq  # negate: factor = liquidity (high = more liquid)

    # ── Kyle (1985) lambda ────────────────────────────────────────────────

    def kyle_lambda(self, window: int = 63) -> pd.DataFrame:
        """
        Kyle (1985) lambda: price impact coefficient.

          |r_t| ≈ lambda * sqrt(DVOL_t) + noise

        We estimate lambda by rolling OLS: lambda = Cov(|r|, sqrt(DVOL)) / Var(sqrt(DVOL)).
        Lower lambda → better liquidity.  Negate for factor convention.
        """
        abs_ret = self.returns.abs()
        sqrt_dvol = self.dollar_volume.apply(np.sqrt)

        def _lambda(col_name: str) -> pd.Series:
            y = abs_ret[col_name]
            x = sqrt_dvol[col_name]

            cov_yx = y.rolling(window, min_periods=window // 2).cov(x)
            var_x = x.rolling(window, min_periods=window // 2).var()

            lam = cov_yx / (var_x + 1e-10)
            return -lam  # negate: factor = market depth (high = more liquid)

        return pd.DataFrame(
            {c: _lambda(c) for c in self.returns.columns},
            index=self.returns.index,
        )

    # ── Volume-based factors ─────────────────────────────────────────────

    def volume_momentum(self, fast: int = 5, slow: int = 21) -> pd.DataFrame:
        """
        Volume momentum: ratio of short-term to long-term average volume.
        Rising volume alongside price → continuation signal.
        """
        v_fast = self.volume.rolling(fast, min_periods=fast).mean()
        v_slow = self.volume.rolling(slow, min_periods=slow // 2).mean()
        return np.log(v_fast / (v_slow + 1e-8))

    def turnover_ratio(self, window: int = 21) -> pd.DataFrame:
        """
        Share turnover: volume / average volume over `window` days.
        High turnover → high attention / momentum reinforcement.
        """
        avg_vol = self.volume.rolling(252, min_periods=63).mean()
        return self.volume.rolling(window).mean() / (avg_vol + 1e-8)

    # ── Convenience ──────────────────────────────────────────────────────

    def all_factors(self) -> dict:
        return {
            "roll_spread": self.roll_spread(),
            "amihud_illiq": self.amihud_illiquidity(),
            "kyle_lambda": self.kyle_lambda(),
            "volume_mom": self.volume_momentum(),
            "turnover": self.turnover_ratio(),
        }
