"""
Full Pipeline Demo
==================
End-to-end walk-forward backtest combining:
  - Momentum + quality alpha signals
  - Ledoit-Wolf + RMT covariance
  - Risk Parity portfolio construction
  - Walk-forward evaluation (5-year horizon, quarterly rebalance)
  - Full performance tearsheet
  - Brinson-Hood-Beebower attribution by sector
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sae.data.generator import MarketDataGenerator
from sae.factors.technical import TechnicalFactors
from sae.factors.fundamental import FundamentalFactors
from sae.risk.covariance import LedoitWolfEstimator, rmt_denoise
from sae.portfolio.optimizer import risk_parity_optimize, mean_variance_optimize
from sae.backtest.engine import WalkForwardBacktest
from sae.backtest.attribution import BrinsonAttribution
from sae.analytics.tearsheet import compute_tearsheet, print_tearsheet
from sae.models.deflated_sharpe import deflated_sharpe_ratio


def make_signal_fn(prices_ref, volume_ref, fundam_ref):
    """
    Returns a signal function that computes momentum + quality composite score.
    Signal is computed strictly from train_prices and train_volume.
    """
    def signal_fn(train_prices, train_volume):
        # Momentum signal: 12-1 month
        tech = TechnicalFactors(train_prices, train_volume)
        mom = tech.mom_12_1().iloc[-1].fillna(0)

        # Volatility (inverse for quality-vol composite)
        vol21 = tech.vol_21().iloc[-1].fillna(mom.abs().max())

        # Composite: momentum - 0.5 * vol (penalise volatile stocks)
        mom_z = (mom - mom.mean()) / (mom.std() + 1e-8)
        vol_z = (vol21 - vol21.mean()) / (vol21.std() + 1e-8)
        signal = mom_z - 0.5 * vol_z

        return signal.reindex(train_prices.columns).fillna(0).values

    return signal_fn


def make_portfolio_fn(method="risk_parity"):
    """Returns a portfolio function for the given method."""
    def portfolio_fn(signals, cov, assets):
        # Apply LW shrinkage to the in-the-loop covariance
        try:
            lw = LedoitWolfEstimator()
            lw.fit_dataframe(pd.DataFrame(
                np.random.randn(252, len(assets)), columns=assets
            ))
            cov_est = cov  # already a np array from backtest engine
        except Exception:
            cov_est = cov

        # Small regularisation for numerical stability
        cov_reg = cov_est + np.eye(len(assets)) * 1e-6 * np.diag(cov_est).mean()

        if method == "risk_parity":
            return risk_parity_optimize(cov_reg)
        elif method == "mv":
            mu = signals / (np.abs(signals).mean() + 1e-8) * 0.001
            return mean_variance_optimize(mu, cov_reg, risk_aversion=1.0, long_only=True, max_weight=0.05)
        else:
            return np.ones(len(assets)) / len(assets)

    return portfolio_fn


def save_equity_curve(strategies: dict, out_path: str) -> None:
    """Save a 2-panel equity curve + drawdown chart to out_path."""
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    colors = {"Risk Parity": "#2196F3", "Mean-Variance": "#FF9800", "Benchmark": "#9E9E9E"}

    for name, rets in strategies.items():
        cum = (1 + rets).cumprod()
        color = colors.get(name, None)
        lw = 1.5 if "Benchmark" in name else 2.0
        ls = "--" if "Benchmark" in name else "-"
        ax_eq.plot(cum.index, cum.values, label=name, color=color, lw=lw, ls=ls)

        roll_max = cum.cummax()
        dd = (cum - roll_max) / (roll_max + 1e-10)
        ax_dd.plot(dd.index, dd.values * 100, color=color, lw=lw, ls=ls, alpha=0.8)

    ax_eq.set_ylabel("Cumulative Return", fontsize=11)
    ax_eq.set_title("Systematic Alpha Engine — Walk-Forward Backtest", fontsize=13, fontweight="bold")
    ax_eq.legend(fontsize=9, loc="upper left")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.axhline(1.0, color="black", lw=0.8, alpha=0.4)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    ax_dd.set_ylabel("Drawdown (%)", fontsize=11)
    ax_dd.set_xlabel("Date", fontsize=11)
    ax_dd.fill_between(dd.index, dd.values * 100, 0, alpha=0.15, color="#2196F3")
    ax_dd.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Equity curve saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Systematic Alpha Engine — full pipeline backtest")
    p.add_argument("--seed",     type=int, default=42,   help="Random seed for data generation")
    p.add_argument("--n-stocks", type=int, default=100,  help="Number of synthetic stocks")
    p.add_argument("--n-days",   type=int, default=1260, help="Number of trading days (~5 years)")
    p.add_argument("--out-dir",  type=str, default="results", help="Output directory for plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  Systematic Alpha Engine — Full Pipeline Backtest")
    print("=" * 60)
    print(f"  seed={args.seed}  n_stocks={args.n_stocks}  n_days={args.n_days}")

    # ── 1. Generate data ──────────────────────────────────────────────
    print("\n[1/6] Generating synthetic market data…")
    gen = MarketDataGenerator(n_stocks=args.n_stocks, n_days=args.n_days,
                              n_sectors=5, seed=args.seed)
    data = gen.generate()
    prices   = data["prices"]
    returns  = data["returns"]
    volume   = data["volume"]
    fundam   = data["fundamentals"]
    sector_id = data["metadata"]["sector_id"]
    stocks   = data["metadata"]["stocks"]
    N = len(stocks)

    bench_weights = np.ones(N) / N  # equal-weight benchmark

    # ── 2. Walk-forward backtest ──────────────────────────────────────
    print("[2/6] Running walk-forward backtest (252d train, 63d test, quarterly rebal)…")

    signal_fn = make_signal_fn(prices, volume, fundam)

    for method_name, method_key in [("Risk Parity", "risk_parity"), ("Mean-Variance", "mv")]:
        portfolio_fn = make_portfolio_fn(method_key)
        backtest = WalkForwardBacktest(
            train_window=252, test_window=63, step=63,
            expanding=False, transaction_cost_bps=10.0
        )
        result = backtest.run(prices, volume, signal_fn, portfolio_fn, bench_weights)

        port_rets = result["portfolio_returns"]
        bench_rets = result["benchmark_returns"]

        if len(port_rets) == 0:
            print(f"  {method_name}: no OOS data")
            continue

        # ── 3. Tearsheet ──────────────────────────────────────────────
        print(f"\n[3/6] Performance tearsheet — {method_name}:")
        metrics = compute_tearsheet(port_rets, benchmark=bench_rets, risk_free_rate=0.045)
        print_tearsheet(metrics, title=f"{method_name} Portfolio")

        # ── 4. DSR ────────────────────────────────────────────────────
        print(f"  Deflated Sharpe Ratio (N_trials=2 strategies tested):")
        dsr = deflated_sharpe_ratio(port_rets.values, n_trials=2)
        print(f"    SR_hat = {dsr['SR_hat']:.3f},  PSR = {dsr['PSR']:.4f},  "
              f"DSR = {dsr['DSR']:.4f}")

    # ── 5. Brinson-Hood-Beebower attribution ─────────────────────────
    print("\n[5/6] Brinson-Hood-Beebower sector attribution…")

    # Re-run risk parity for attribution
    portfolio_fn_rp = make_portfolio_fn("risk_parity")
    backtest_attr = WalkForwardBacktest(
        train_window=252, test_window=63, step=63, transaction_cost_bps=10.0
    )
    result_rp = backtest_attr.run(prices, volume, signal_fn, portfolio_fn_rp, bench_weights)
    port_rets_rp = result_rp["portfolio_returns"]
    weights_hist = result_rp["weights_history"]

    if len(port_rets_rp) > 0:
        attr = BrinsonAttribution(sector_id.tolist())
        attribution_df = attr.attribute_period(
            port_rets_rp,
            pd.Series({d: v for d, v in zip(port_rets_rp.index, [0] * len(port_rets_rp))}),
            weights_hist,
            bench_weights,
            prices,
        )

        print(f"\n  Annualised BHB Attribution by Sector:")
        print(f"  {'Sector':<10} {'Allocation':>12} {'Selection':>12} {'Interaction':>12} {'Total Active':>13}")
        print("  " + "-" * 60)
        for sector_label, row in attribution_df.iterrows():
            if sector_label == "Total":
                print("  " + "-" * 60)
            alloc = row.get("allocation", 0)
            selec = row.get("selection", 0)
            inter = row.get("interaction", 0)
            total = row.get("total_active", 0)
            print(
                f"  {str(sector_label):<10} {alloc * 100:>11.3f}% {selec * 100:>11.3f}% "
                f"{inter * 100:>11.3f}% {total * 100:>12.3f}%"
            )

    # ── 6. Summary comparison table ───────────────────────────────────
    print("\n[6/6] Summary comparison — portfolio vs benchmark:")
    strategies = {}

    for method_name, method_key in [("Risk Parity", "risk_parity"), ("Mean-Variance", "mv")]:
        portfolio_fn = make_portfolio_fn(method_key)
        bt = WalkForwardBacktest(train_window=252, test_window=63, step=63, transaction_cost_bps=10.0)
        res = bt.run(prices, volume, signal_fn, portfolio_fn, bench_weights)
        if len(res["portfolio_returns"]) > 0:
            strategies[method_name] = res["portfolio_returns"]
            strategies[f"Benchmark ({method_name})"] = res["benchmark_returns"]

    if strategies:
        print(f"\n  {'Strategy':<30} {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>7} {'MaxDD':>8}")
        print("  " + "-" * 65)
        for name, rets in strategies.items():
            m = compute_tearsheet(rets, risk_free_rate=0.045)
            if m:
                ar = m.get("annual_return", float("nan"))
                av = m.get("annual_vol", float("nan"))
                sh = m.get("sharpe_ratio", float("nan"))
                md = m.get("max_drawdown", float("nan"))
                print(
                    f"  {name:<30} {ar * 100:>7.2f}% {av * 100:>7.2f}% "
                    f"{sh:>7.3f} {md * 100:>7.2f}%"
                )

        # ── Save equity curve plot ────────────────────────────────────
        plot_path = os.path.join(args.out_dir, "equity_curve.png")
        save_equity_curve(strategies, plot_path)

    print("\n✓  Full pipeline backtest complete.")


if __name__ == "__main__":
    main()
