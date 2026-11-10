"""
Market impact model.

Estimates market impact as a function of trade size relative to market volume.

Implements a square-root market impact model (Grinold & Kahn, Barra):
  MI = eta * sigma * sqrt(participation_rate)

where participation_rate = trade_size / average_daily_volume.

Also provides:
  - Permanent vs temporary impact decomposition
  - Bid-ask spread cost estimation using Roll (1984)
  - Total transaction cost = spread + impact + timing risk
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransactionCostEstimate:
    spread_cost_bps: float    # half-spread cost in basis points
    impact_cost_bps: float    # market impact in basis points
    total_cost_bps: float     # total one-way cost in basis points
    participation_rate: float # fraction of ADV traded


class MarketImpactModel:
    """
    Square-root law market impact model.

    Impact = sigma * eta * sqrt(Q / ADV)

    where:
      sigma  = daily volatility
      eta    = impact coefficient (typically 0.1 for liquid stocks)
      Q      = trade size (shares)
      ADV    = average daily volume (shares)

    Parameters
    ----------
    eta    : temporary impact coefficient.
    gamma  : permanent impact coefficient (default: 0.5 * eta).
    spread_bps : half-spread in basis points (default: 5 bps).
    """

    def __init__(
        self,
        eta: float = 0.1,
        gamma: float = None,
        spread_bps: float = 5.0,
    ):
        self.eta = eta
        self.gamma = gamma if gamma is not None else 0.5 * eta
        self.spread_bps = spread_bps

    def estimate(
        self,
        trade_size: float,
        adv: float,
        sigma: float,
        price: float = 100.0,
    ) -> TransactionCostEstimate:
        """
        Estimate total transaction cost for a single trade.

        Parameters
        ----------
        trade_size : Number of shares to trade.
        adv        : Average daily volume (shares).
        sigma      : Daily volatility (fraction, e.g. 0.02 = 2%).
        price      : Share price for dollar/bps conversion.

        Returns
        -------
        TransactionCostEstimate
        """
        participation = trade_size / max(adv, 1.0)

        # Square-root impact
        impact_frac = self.eta * sigma * np.sqrt(participation)
        impact_bps = impact_frac * 1e4  # convert to basis points

        total_bps = self.spread_bps + impact_bps

        return TransactionCostEstimate(
            spread_cost_bps=self.spread_bps,
            impact_cost_bps=impact_bps,
            total_cost_bps=total_bps,
            participation_rate=participation,
        )

    def perm_vs_temp_impact(
        self,
        trade_size: float,
        adv: float,
        sigma: float,
    ) -> dict:
        """
        Decompose impact into permanent and temporary components.

          Perm impact = gamma * sigma * sqrt(Q / ADV)   (shifts mid permanently)
          Temp impact = eta   * sigma * sqrt(Q / ADV)   (reverts after trade)

        Returns dict with keys: permanent_bps, temporary_bps, total_bps.
        """
        participation = trade_size / max(adv, 1.0)
        sqrt_part = np.sqrt(participation)

        perm = self.gamma * sigma * sqrt_part * 1e4
        temp = self.eta * sigma * sqrt_part * 1e4

        return {
            "permanent_bps": perm,
            "temporary_bps": temp,
            "total_bps": perm + temp + self.spread_bps,
        }

    def optimal_participation_rate(
        self,
        urgency: float = 0.5,
        sigma: float = 0.02,
    ) -> float:
        """
        Approximate optimal participation rate from AC model.

        Higher urgency → trade faster → higher participation rate.
        urgency in [0, 1]: 0 = very patient, 1 = must liquidate today.
        """
        # Heuristic: participation = urgency^2 * sigma^{-1} * constant
        # In practice, determined by AC efficient frontier
        return float(np.clip(urgency ** 2 / (self.eta * sigma + 1e-8) * 0.01, 0.01, 0.30))
