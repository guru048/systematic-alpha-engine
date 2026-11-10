"""
Almgren-Chriss (2001) optimal execution model.

Finds the optimal trajectory for liquidating X shares over T periods
that minimises a weighted sum of expected execution cost and variance.

Model Setup
-----------
- X  : initial position (shares to sell, > 0)
- T  : time horizon (days)
- N  : number of trading periods (default = T)
- sigma : daily price volatility ($/day^0.5 or fraction)
- eta  : temporary impact coefficient ($ per share per share/day)
- gamma: permanent impact coefficient ($ per share per share/day)
- lam  : risk aversion ($ per unit of variance)

Dynamics
--------
S_j = S_{j-1} - gamma * (x_j / tau)         (permanent impact)
n_j = x_j / tau                              (trading rate, shares/day)
~S_j = S_j - eta * n_j                       (execution price with temporary impact)

where x_j = shares sold in period j, tau = T/N.

Optimal Trajectory (Almgren-Chriss 2001, Theorem 3.1)
------------------------------------------------------
kappa * sinh(kappa * tau) = sqrt(lambda * sigma^2 / (eta - gamma*tau/2))

x(t) = X * sinh(kappa * (T - t)) / sinh(kappa * T)

For typical params (eta >> gamma*tau/2), kappa ≈ sqrt(lambda * sigma^2 / eta).

Efficient Frontier
------------------
Parameterised by lambda: for each lambda, we get a (E[cost], Var[cost]) pair.
Higher lambda → faster liquidation, lower variance but higher cost.

References
----------
Almgren, R. and Chriss, N. (2001).  "Optimal Execution of Portfolio Transactions."
  Journal of Risk, 3(2), pp. 5-39.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of an Almgren-Chriss optimal execution."""
    times: np.ndarray           # (N+1,) — time grid t_0, t_1, ..., t_N
    holdings: np.ndarray        # (N+1,) — shares remaining after each period
    trades: np.ndarray          # (N,)   — shares sold in each period
    expected_cost: float        # E[implementation shortfall] in $
    cost_variance: float        # Var[implementation shortfall] in $^2
    cost_std: float             # sqrt(Var)
    kappa: float                # optimal decay parameter
    risk_aversion: float        # lambda used


class AlmgrenChriss:
    """
    Almgren-Chriss optimal execution trajectory.

    Parameters
    ----------
    sigma  : Daily price volatility (as a fraction of price, e.g. 0.02 = 2%).
    eta    : Temporary impact coefficient.  Typical: 0.1 * sigma for liquid stocks.
    gamma  : Permanent impact coefficient.  Typical: 0.1 * eta.
    price  : Current mid-price per share (for dollar cost estimates).
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 0.002,
        gamma: float = 0.0002,
        price: float = 100.0,
    ):
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.price = price

    def optimal_trajectory(
        self,
        X: float,
        T: float,
        N: int = None,
        risk_aversion: float = 1e-6,
    ) -> ExecutionResult:
        """
        Compute the optimal liquidation trajectory.

        Parameters
        ----------
        X             : Shares to liquidate (positive).
        T             : Liquidation horizon in days.
        N             : Number of trading periods (default: T, i.e. one per day).
        risk_aversion : Lambda — higher → faster liquidation.

        Returns
        -------
        ExecutionResult containing trajectory and cost statistics.
        """
        if N is None:
            N = max(1, int(T))
        tau = T / N

        # Effective temporary impact: eta_eff = eta - gamma*tau/2
        eta_eff = self.eta - 0.5 * self.gamma * tau
        eta_eff = max(eta_eff, self.eta * 0.01)  # floor to avoid division issues

        # kappa from the transcendental equation:
        # kappa * sinh(kappa*tau) = sqrt(lambda * sigma^2 / eta_eff)
        rhs = np.sqrt(risk_aversion * (self.sigma ** 2) / eta_eff)
        kappa = self._solve_kappa(rhs, tau)

        # Optimal holdings trajectory: x(t_j) = X * sinh(kappa*(T-t_j)) / sinh(kappa*T)
        times = np.linspace(0, T, N + 1)
        denom = np.sinh(kappa * T) if kappa * T > 1e-6 else T  # limit: linear schedule
        if abs(denom) < 1e-12:
            denom = 1.0

        holdings = np.array([X * np.sinh(kappa * (T - t)) / denom for t in times])
        holdings = np.clip(holdings, 0, X)
        holdings[-1] = 0.0  # fully liquidated at end

        # Shares traded each period
        trades = holdings[:-1] - holdings[1:]
        trades = np.clip(trades, 0, None)

        # Expected cost (Almgren-Chriss eq. 20)
        # E[shortfall] = 0.5*gamma*X^2*S + epsilon*X*S
        #               + eta_eff * X^2 * kappa*tau / (2*tanh(0.5*kappa*T*tau))
        # Simplified (using the trajectory directly):
        trade_rates = trades / tau  # shares per day
        perm_cost = 0.5 * self.gamma * (X ** 2) * self.price
        temp_cost = self.eta * np.sum(trade_rates ** 2) * tau * self.price

        # Variance: sigma^2 * sum_j x_j^2 * tau
        cost_variance = (self.sigma * self.price) ** 2 * np.sum(holdings[:-1] ** 2) * tau

        return ExecutionResult(
            times=times,
            holdings=holdings,
            trades=trades,
            expected_cost=float(perm_cost + temp_cost),
            cost_variance=float(cost_variance),
            cost_std=float(np.sqrt(cost_variance)),
            kappa=float(kappa),
            risk_aversion=risk_aversion,
        )

    def efficient_frontier(
        self,
        X: float,
        T: float,
        N: int = None,
        lambdas: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Compute the execution efficient frontier over a range of risk aversions.

        Returns
        -------
        (expected_costs, cost_stds, trajectories) arrays.
        """
        if lambdas is None:
            lambdas = np.logspace(-8, -4, 50)

        costs, stds = [], []
        for lam in lambdas:
            res = self.optimal_trajectory(X, T, N, lam)
            costs.append(res.expected_cost)
            stds.append(res.cost_std)

        return np.array(costs), np.array(stds), lambdas

    def vwap_schedule(self, X: float, N: int) -> np.ndarray:
        """VWAP benchmark: uniform liquidation (naive strategy)."""
        return np.full(N, X / N)

    def twap_schedule(self, X: float, N: int) -> np.ndarray:
        """TWAP: same as VWAP for uniform volume (alias)."""
        return self.vwap_schedule(X, N)

    def implementation_shortfall(
        self,
        trades: np.ndarray,
        price_path: np.ndarray,
        X: float,
    ) -> float:
        """
        Compute realised implementation shortfall given a price path.

        IS = arrival_price * X - sum_j (trade_j * exec_price_j)

        Parameters
        ----------
        trades     : (N,) shares sold each period.
        price_path : (N,) mid-prices at each period.
        X          : Total shares sold.

        Returns
        -------
        IS in dollars per unit of initial price.
        """
        arrival_price = price_path[0]
        execution_value = np.sum(trades * price_path)
        benchmark_value = arrival_price * X
        return float(benchmark_value - execution_value)

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _solve_kappa(rhs: float, tau: float, n_iter: int = 100) -> float:
        """
        Solve kappa * sinh(kappa * tau) = rhs via Newton-Raphson.

        For small kappa*tau: sinh(kappa*tau) ≈ kappa*tau,
        so kappa^2 * tau ≈ rhs → kappa ≈ sqrt(rhs / tau).
        """
        if rhs < 1e-12:
            return 1e-8
        # Initial guess from small-kappa approximation
        kappa = np.sqrt(rhs / max(tau, 1e-6))
        for _ in range(n_iter):
            f = kappa * np.sinh(kappa * tau) - rhs
            df = np.sinh(kappa * tau) + kappa * tau * np.cosh(kappa * tau)
            if abs(df) < 1e-12:
                break
            kappa_new = kappa - f / df
            if kappa_new <= 0:
                kappa_new = kappa * 0.5
            if abs(kappa_new - kappa) < 1e-10 * kappa:
                kappa = kappa_new
                break
            kappa = kappa_new
        return max(kappa, 1e-8)
