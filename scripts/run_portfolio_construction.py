"""
Portfolio Construction Demo
============================
Compares four portfolio construction methods on 100 synthetic stocks:
  1. Mean-Variance Optimisation (Markowitz 1952)
  2. Risk Parity / Equal Risk Contribution (Roncalli 2013)
  3. Black-Litterman (Black & Litterman 1990)
  4. Hierarchical Risk Parity (Lopez de Prado 2016)

Uses Ledoit-Wolf shrinkage + RMT denoising for the covariance estimate.
Also demos Almgren-Chriss optimal execution for the resulting rebalance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sae.data.generator import MarketDataGenerator
from sae.factors.technical import TechnicalFactors
from sae.risk.covariance import LedoitWolfEstimator, rmt_denoise
from sae.risk.factor_model import PCAFactorModel
from sae.portfolio.optimizer import (
    mean_variance_optimize,
    risk_parity_optimize,
    black_litterman,
    hierarchical_risk_parity,
)
from sae.execution.almgren_chriss import AlmgrenChriss
from sae.execution.impact import MarketImpactModel


def portfolio_metrics(w, mu, sigma, freq=252):
    """Annualised return, vol, Sharpe for a portfolio. mu must be annual."""
    ret = float(w @ mu)                                   # mu already annual
    vol = float(np.sqrt(w @ sigma @ w)) * np.sqrt(freq)  # sigma is daily
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe


def main():
    print("=" * 60)
    print("  Systematic Alpha Engine — Portfolio Construction")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────
    print("\n[1/5] Generating market data…")
    gen = MarketDataGenerator(n_stocks=100, n_days=1260, n_sectors=5, seed=42)
    data = gen.generate()
    prices  = data["prices"]
    returns = data["returns"]
    volume  = data["volume"]
    stocks  = data["metadata"]["stocks"]
    N = len(stocks)

    # Use last 252 days as estimation window
    train_ret = returns.iloc[-252:].dropna()

    # ── 2. Covariance estimation ──────────────────────────────────────
    print("[2/5] Estimating covariance (Ledoit-Wolf + RMT denoising)…")
    lw = LedoitWolfEstimator()
    lw.fit(train_ret.values)
    cov_lw = lw.covariance_
    print(f"      Ledoit-Wolf shrinkage α: {lw.shrinkage_:.4f}")

    cov_rmt, rmt_info = rmt_denoise(cov_lw, n_obs=len(train_ret))
    print(f"      RMT noise threshold:    λ_max = {rmt_info['lambda_max_noise']:.4f}")
    print(f"      Signal factors kept:    {rmt_info['n_signal_factors']}")

    # Factor model
    print("      Fitting PCA factor risk model…")
    pfm = PCAFactorModel(shrink=True)
    pfm.fit(train_ret)
    evr = pfm.explained_variance_ratio()
    cum_evr = np.cumsum(evr)
    print(f"      Factors: {pfm.K_}, cumulative variance: {cum_evr[-1] * 100:.1f}%")

    # ── 3. Expected returns (from momentum signal) ────────────────────
    print("[3/5] Computing expected returns (12-1m momentum signal)…")
    tech = TechnicalFactors(prices, volume)
    mom = tech.mom_12_1().iloc[-1].fillna(0).values
    # Cross-sectionally normalise
    mu_pred = (mom - mom.mean()) / (mom.std() + 1e-8)
    # Scale: 1-std stock earns ~3% annual excess return (realistic factor premium)
    mu_annual = mu_pred * 0.03  # 3% per unit of z-score
    mu_daily = mu_annual / 252

    # ── 4. Portfolio optimisation ─────────────────────────────────────
    print("[4/5] Optimising portfolios…")
    sigma = cov_rmt

    # Market-cap proxy: equal weight
    mkt_weights = np.ones(N) / N

    # 1. Mean-variance
    w_mv = mean_variance_optimize(mu_daily, sigma, risk_aversion=1.0, long_only=True, max_weight=0.05)

    # 2. Risk Parity
    w_rp = risk_parity_optimize(sigma)

    # 3. Black-Litterman — 3 views: momentum top-quintile outperform by 2% p.a.
    top20 = np.argsort(mom)[-20:]
    bot20 = np.argsort(mom)[:20]
    P = np.zeros((1, N))
    P[0, top20] = 1.0 / 20
    P[0, bot20] = -1.0 / 20
    Q = np.array([0.02 / 252])  # 2% annualised per day
    w_bl = black_litterman(sigma, mkt_weights, P, Q, tau=0.025, delta=2.5)

    # 4. HRP
    w_hrp = hierarchical_risk_parity(sigma)

    # ── 5. Compare ────────────────────────────────────────────────────
    print("[5/5] Comparing portfolio metrics…")

    portfolios = {
        "Mean-Variance":      w_mv,
        "Risk Parity":        w_rp,
        "Black-Litterman":    w_bl,
        "HRP":                w_hrp,
        "Equal Weight":       mkt_weights,
    }

    print(f"\n  {'Portfolio':<20} {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>7} "
          f"{'MaxW':>7} {'EffN':>7} {'Gini':>7}")
    print("  " + "-" * 65)

    for name, w in portfolios.items():
        ret, vol, sharpe = portfolio_metrics(w, mu_annual, sigma)
        max_w = w.max()
        # Effective N = 1 / sum(w^2) (portfolio concentration)
        eff_n = 1.0 / (np.sum(w ** 2) + 1e-12)
        # Gini coefficient of weights (0 = uniform, 1 = concentrated)
        sw = np.sort(w)
        n = len(sw)
        gini = (2 * np.arange(1, n + 1) - n - 1) @ sw / (n * sw.sum() + 1e-12)
        print(
            f"  {name:<20} {ret * 100:>7.2f}% {vol * 100:>7.2f}% "
            f"{sharpe:>7.3f} {max_w * 100:>6.1f}% {eff_n:>7.1f} {gini:>7.4f}"
        )

    # ── 6. Risk decomposition for Risk Parity ────────────────────────
    print("\n  Risk Parity — Asset Risk Contributions (top 10):")
    port_var = w_rp @ sigma @ w_rp
    mrc = sigma @ w_rp  # marginal risk contributions
    rc = w_rp * mrc / (port_var + 1e-12)
    rc_pct = rc / rc.sum()
    top_idx = np.argsort(rc_pct)[-10:][::-1]
    for i in top_idx:
        print(f"    {stocks[i]:>5}  weight={w_rp[i] * 100:.2f}%  risk_contrib={rc_pct[i] * 100:.2f}%")

    # ── 7. Almgren-Chriss execution ───────────────────────────────────
    print("\n  Almgren-Chriss Optimal Execution (rebalancing $10M portfolio):")
    sigma_stock = float(train_ret.std().mean())
    # Realistic impact params: $100 stock, trading 10,000 shares/day moves price ~1 bps
    ac = AlmgrenChriss(sigma=sigma_stock, eta=2e-7, gamma=5e-8, price=100.0)

    X_shares = 100_000  # 100k shares = $10M position
    T_days = 10         # 10-day horizon

    for lam_name, lam in [("Low risk aversion (λ=1e-7)", 1e-7),
                           ("Med risk aversion (λ=1e-6)", 1e-6),
                           ("High risk aversion (λ=1e-5)", 1e-5)]:
        res = ac.optimal_trajectory(X_shares, T_days, risk_aversion=lam)
        cost_bps = res.expected_cost / (X_shares * 100.0) * 1e4
        print(f"    {lam_name:35s}  "
              f"E[cost]=${res.expected_cost:,.0f} ({cost_bps:.1f}bps)  "
              f"σ[cost]=${res.cost_std:,.0f}  "
              f"κ={res.kappa:.4f}")

    # Efficient frontier
    costs, stds, lambdas = ac.efficient_frontier(X_shares, T_days)
    print(f"\n    Efficient frontier: {len(lambdas)} points, "
          f"cost range ${costs.min():,.0f}—${costs.max():,.0f}, "
          f"risk range ${stds.min():,.0f}—${stds.max():,.0f}")

    # ── 8. Market impact model ────────────────────────────────────────
    print("\n  Market Impact Estimates (square-root law):")
    impact_model = MarketImpactModel(eta=0.1, spread_bps=5.0)
    adv = float(volume.mean().mean())
    for size_pct, label in [(0.01, "1% ADV"), (0.05, "5% ADV"), (0.20, "20% ADV")]:
        trade = adv * size_pct
        est = impact_model.estimate(trade, adv, sigma_stock)
        print(f"    {label:12s}  spread={est.spread_cost_bps:.1f}bps  "
              f"impact={est.impact_cost_bps:.1f}bps  total={est.total_cost_bps:.1f}bps")

    print("\n✓  Portfolio construction demo complete.")


if __name__ == "__main__":
    main()
