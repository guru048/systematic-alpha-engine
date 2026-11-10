"""
Brinson-Hood-Beebower (BHB) performance attribution.

The BHB model decomposes active return (portfolio − benchmark) into
three effects at the sector level:

  Allocation effect  = (w_P - w_B) * (R_B_sector - R_B_total)
    How much did over/underweighting sectors contribute?

  Selection effect   = w_B * (R_P_sector - R_B_sector)
    Within each sector, how much did stock selection add?

  Interaction effect = (w_P - w_B) * (R_P_sector - R_B_sector)
    Joint effect of allocation and selection decisions.

  Active return = Allocation + Selection + Interaction

Reference
---------
Brinson, G., Hood, L. and Beebower, G. (1986).  "Determinants of Portfolio
  Performance."  Financial Analysts Journal, 42(4), pp. 39-44.
"""

import numpy as np
import pandas as pd
from typing import List


class BrinsonAttribution:
    """
    Brinson-Hood-Beebower sector attribution.

    Parameters
    ----------
    sector_map : list of sector labels (length N), one per asset.
    """

    def __init__(self, sector_map: List):
        self.sector_map = np.array(sector_map)
        self.sectors = sorted(set(sector_map))

    def attribute(
        self,
        port_weights: np.ndarray,
        bench_weights: np.ndarray,
        asset_returns: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute BHB attribution for a single period.

        Parameters
        ----------
        port_weights  : (N,) portfolio weights.
        bench_weights : (N,) benchmark weights.
        asset_returns : (N,) realised returns for the period.

        Returns
        -------
        DataFrame indexed by sector with columns:
          w_port, w_bench, r_port_sector, r_bench_sector,
          allocation, selection, interaction, total_active
        """
        rows = {}
        R_B_total = float(bench_weights @ asset_returns)

        for s in self.sectors:
            mask = (self.sector_map == s)
            w_P_s = port_weights[mask].sum()
            w_B_s = bench_weights[mask].sum()

            # Sector returns
            if w_P_s > 1e-8:
                r_P_s = float((port_weights[mask] * asset_returns[mask]).sum() / w_P_s)
            else:
                r_P_s = float(asset_returns[mask].mean()) if mask.any() else 0.0

            if w_B_s > 1e-8:
                r_B_s = float((bench_weights[mask] * asset_returns[mask]).sum() / w_B_s)
            else:
                r_B_s = float(asset_returns[mask].mean()) if mask.any() else 0.0

            alloc = (w_P_s - w_B_s) * (r_B_s - R_B_total)
            selec = w_B_s * (r_P_s - r_B_s)
            inter = (w_P_s - w_B_s) * (r_P_s - r_B_s)

            rows[s] = {
                "w_port": w_P_s,
                "w_bench": w_B_s,
                "r_port_sector": r_P_s,
                "r_bench_sector": r_B_s,
                "allocation": alloc,
                "selection": selec,
                "interaction": inter,
                "total_active": alloc + selec + inter,
            }

        result = pd.DataFrame(rows).T
        result.index.name = "sector"
        result.loc["Total"] = result.sum()
        return result

    def attribute_period(
        self,
        port_returns: pd.Series,
        bench_returns: pd.Series,
        weights_history: dict,
        bench_weights: np.ndarray,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run attribution over multiple periods and aggregate.

        Parameters
        ----------
        port_returns    : portfolio return series.
        bench_returns   : benchmark return series.
        weights_history : {date: weight_array} from backtest.
        bench_weights   : (N,) static benchmark weights.
        prices          : (T, N) price DataFrame for asset returns.

        Returns
        -------
        Aggregated attribution DataFrame.
        """
        asset_ret = prices.pct_change().dropna()
        total_alloc = {s: 0.0 for s in self.sectors}
        total_selec = {s: 0.0 for s in self.sectors}
        total_inter = {s: 0.0 for s in self.sectors}
        n_periods = 0

        for date in port_returns.index:
            if date not in weights_history or date not in asset_ret.index:
                continue
            w_P = weights_history[date]
            r_assets = asset_ret.loc[date].values
            attr = self.attribute(w_P, bench_weights, r_assets)
            for s in self.sectors:
                if s in attr.index:
                    total_alloc[s] += attr.loc[s, "allocation"]
                    total_selec[s] += attr.loc[s, "selection"]
                    total_inter[s] += attr.loc[s, "interaction"]
            n_periods += 1

        summary = pd.DataFrame({
            "allocation": total_alloc,
            "selection": total_selec,
            "interaction": total_inter,
        })
        summary["total_active"] = summary.sum(axis=1)
        summary.index.name = "sector"

        # Annualise (approximate)
        if n_periods > 0:
            ann = 252.0 / max(n_periods, 1)
            summary *= ann

        summary.loc["Total"] = summary.sum()
        return summary
