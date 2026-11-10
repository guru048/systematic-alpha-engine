"""
Factor evaluation metrics.

FactorEvaluator computes IC (Information Coefficient), ICIR, factor decay,
and turnover for a panel of alpha factors.

Key definitions
---------------
IC_t = Spearman rank correlation between factor[t] and forward_return[t+h]
ICIR = mean(IC) / std(IC)                 — analogous to Sharpe ratio
Factor decay: IC at different holding horizons h = 1, 5, 10, 20, 60 days
Factor turnover: 1 - Spearman rank autocorr(factor_t, factor_{t-1})
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional


class FactorEvaluator:
    """
    Evaluate one or more alpha factor DataFrames against forward returns.

    Parameters
    ----------
    factor_dict : dict of {name: DataFrame (T x N)}
    returns     : DataFrame (T x N) of daily returns
    """

    def __init__(self, factor_dict: dict, returns: pd.DataFrame):
        self.factors = factor_dict
        self.returns = returns

    # ── IC series ────────────────────────────────────────────────────────

    def ic_series(
        self,
        factor: pd.DataFrame,
        forward_days: int = 1,
        min_stocks: int = 20,
    ) -> pd.Series:
        """
        Compute daily cross-sectional IC series.

        IC_t = Spearman(factor[t], return[t + forward_days])

        NaN rows (insufficient data) are dropped.
        """
        f_dates = factor.index
        r_dates = self.returns.index

        ics = {}
        for t, date in enumerate(f_dates):
            fwd_t = t + forward_days
            if fwd_t >= len(r_dates):
                break
            fwd_date = r_dates[fwd_t]

            f_vals = factor.loc[date].dropna()
            r_vals = self.returns.loc[fwd_date, f_vals.index].dropna()

            common = f_vals.index.intersection(r_vals.index)
            if len(common) < min_stocks:
                continue

            ic, _ = stats.spearmanr(f_vals[common].values, r_vals[common].values)
            if not np.isnan(ic):
                ics[date] = ic

        return pd.Series(ics, name="IC")

    # ── Summary statistics for a single factor ───────────────────────────

    def factor_stats(
        self,
        name: str,
        forward_days: int = 1,
    ) -> dict:
        """
        Return a dict of summary statistics for one factor.

        Keys: IC_mean, IC_std, ICIR, t_stat_NW, frac_positive, annual_IC
        """
        from .statistics import newey_west_tstat

        factor = self.factors[name]
        ic = self.ic_series(factor, forward_days=forward_days)

        if len(ic) < 10:
            return {k: np.nan for k in ["IC_mean", "IC_std", "ICIR", "t_stat_NW", "frac_positive", "annual_IC"]}

        ic_mean = ic.mean()
        ic_std = ic.std(ddof=1) if len(ic) > 1 else np.nan
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        t_nw = newey_west_tstat(ic.values)
        frac_pos = (ic > 0).mean()
        annual_ic = ic_mean * np.sqrt(252 / forward_days)  # rough annualisation

        return {
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "ICIR": icir,
            "t_stat_NW": t_nw,
            "frac_positive": frac_pos,
            "annual_IC": annual_ic,
        }

    # ── Factor decay ─────────────────────────────────────────────────────

    def factor_decay(
        self,
        name: str,
        horizons: Optional[List[int]] = None,
    ) -> pd.Series:
        """
        IC at different holding horizons — measures how quickly signal decays.

        Parameters
        ----------
        horizons : list of forward-return horizons (days).  Default: [1,5,10,20,60]

        Returns
        -------
        Series indexed by horizon with mean IC values.
        """
        if horizons is None:
            horizons = [1, 5, 10, 20, 60]

        factor = self.factors[name]
        decay = {}
        for h in horizons:
            # h-day cumulative return as target
            cum_ret = self.returns.rolling(h).sum().shift(-h)
            ic = self.ic_series(cum_ret, forward_days=0)  # factor[t] vs cum_ret[t]

            # Use direct corr approach for cumulative returns
            cum_ret_aligned = cum_ret
            ics_h = {}
            for t, date in enumerate(factor.index):
                if date not in cum_ret_aligned.index:
                    continue
                f_vals = factor.loc[date].dropna()
                r_vals = cum_ret_aligned.loc[date, f_vals.index].dropna()
                common = f_vals.index.intersection(r_vals.index)
                if len(common) < 20:
                    continue
                ic_val, _ = stats.spearmanr(f_vals[common].values, r_vals[common].values)
                if not np.isnan(ic_val):
                    ics_h[date] = ic_val

            decay[h] = np.nanmean(list(ics_h.values())) if ics_h else np.nan

        return pd.Series(decay, name=name)

    # ── Factor turnover ──────────────────────────────────────────────────

    def factor_turnover(self, name: str) -> float:
        """
        Average daily factor turnover = 1 - Spearman rank autocorrelation.

        Turnover near 0 → very persistent ranks (low trading cost).
        Turnover near 1 → ranks flip daily (expensive to trade).
        """
        factor = self.factors[name].copy()
        # Daily cross-sectional rank
        ranked = factor.rank(axis=1, pct=True)
        # Lag-1 rank correlation
        autocorrs = []
        for t in range(1, len(ranked)):
            row_t = ranked.iloc[t].dropna()
            row_lag = ranked.iloc[t - 1, :].reindex(row_t.index).dropna()
            common = row_t.index.intersection(row_lag.index)
            if len(common) < 10:
                continue
            rho, _ = stats.spearmanr(row_t[common].values, row_lag[common].values)
            if not np.isnan(rho):
                autocorrs.append(rho)
        if not autocorrs:
            return np.nan
        return 1.0 - float(np.mean(autocorrs))

    # ── Full summary table ───────────────────────────────────────────────

    def summary_table(
        self, forward_days: int = 1, include_turnover: bool = True
    ) -> pd.DataFrame:
        """
        Return a DataFrame with one row per factor and columns:
          IC_mean, IC_std, ICIR, t_stat_NW, frac_positive, annual_IC, [turnover]
        """
        rows = {}
        for name in self.factors:
            row = self.factor_stats(name, forward_days=forward_days)
            if include_turnover:
                row["turnover"] = self.factor_turnover(name)
            rows[name] = row

        df = pd.DataFrame(rows).T
        return df.sort_values("t_stat_NW", ascending=False)
