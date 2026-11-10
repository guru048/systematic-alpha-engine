"""
Factor Research Demo
====================
Generates synthetic market data, computes 15+ alpha factors, and evaluates
them using IC, ICIR, Newey-West t-statistics, factor decay, and BH FDR correction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sae.data.generator import MarketDataGenerator
from sae.factors.technical import TechnicalFactors
from sae.factors.fundamental import FundamentalFactors
from sae.factors.microstructure import MicrostructureFactors
from sae.evaluation.metrics import FactorEvaluator
from sae.evaluation.statistics import benjamini_hochberg
from scipy import stats


def main():
    print("=" * 60)
    print("  Systematic Alpha Engine — Factor Research")
    print("=" * 60)

    # ── 1. Generate synthetic data ────────────────────────────────────
    print("\n[1/4] Generating synthetic market data (100 stocks × 5 years)…")
    gen = MarketDataGenerator(n_stocks=100, n_days=1260, n_sectors=5, seed=42)
    data = gen.generate()

    prices  = data["prices"]
    returns = data["returns"]
    volume  = data["volume"]
    fundam  = data["fundamentals"]

    print(f"      {prices.shape[0]} trading days × {prices.shape[1]} stocks")
    print(f"      Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
    ann_ret = (1 + returns.mean()).pow(252) - 1
    print(f"      Avg annual return: {ann_ret.mean() * 100:.1f}%  "
          f"Avg annual vol: {(returns.std() * np.sqrt(252)).mean() * 100:.1f}%")

    # ── 2. Compute factors ────────────────────────────────────────────
    print("\n[2/4] Computing alpha factors…")
    tech = TechnicalFactors(prices, volume)
    fund = FundamentalFactors(prices, fundam)
    micro = MicrostructureFactors(returns, prices, volume)

    all_factors = {
        **tech.all_factors(),
        **fund.all_factors(),
        **micro.all_factors(),
    }
    print(f"      {len(all_factors)} factors computed")

    # ── 3. IC / ICIR evaluation ───────────────────────────────────────
    print("\n[3/4] Computing IC, ICIR, Newey-West t-stats (1-day forward return)…")
    evaluator = FactorEvaluator(all_factors, returns)
    summary = evaluator.summary_table(forward_days=1, include_turnover=True)

    print(f"\n  {'Factor':<24} {'IC_mean':>8} {'IC_std':>8} {'ICIR':>7} {'t_NW':>7} {'Hit%':>6} {'Turnover':>9}")
    print("  " + "-" * 72)
    for name, row in summary.head(20).iterrows():
        print(
            f"  {name:<24} "
            f"{row.get('IC_mean', float('nan')):>8.4f} "
            f"{row.get('IC_std', float('nan')):>8.4f} "
            f"{row.get('ICIR', float('nan')):>7.3f} "
            f"{row.get('t_stat_NW', float('nan')):>7.3f} "
            f"{row.get('frac_positive', float('nan')) * 100:>5.1f}% "
            f"{row.get('turnover', float('nan')):>9.4f}"
        )

    # ── 4. Multiple testing correction (BH) ──────────────────────────
    print("\n[4/4] Multiple testing correction (Benjamini-Hochberg FDR = 5%)…")
    t_stats = summary["t_stat_NW"].dropna()
    p_values = 2 * stats.norm.sf(np.abs(t_stats.values))
    rejected, adj_p, threshold = benjamini_hochberg(p_values)

    print(f"\n  Bonferroni threshold (α/m):  p < {0.05 / len(p_values):.4f}")
    print(f"  BH threshold (5% FDR):       p < {threshold:.4f}")
    n_sig = int(rejected.sum())
    print(f"  Significant factors: {n_sig}/{len(p_values)}")

    sig_factors = t_stats.index[rejected].tolist()
    if sig_factors:
        print(f"  Factors passing BH:  {', '.join(sig_factors[:10])}")

    # ── 5. Factor decay ───────────────────────────────────────────────
    print("\n  Factor Decay (IC at different holding horizons):")
    top3 = t_stats.head(3).index.tolist()
    decay_rows = {}
    for fname in top3:
        decay = evaluator.factor_decay(fname, horizons=[1, 5, 10, 20, 60])
        decay_rows[fname] = decay

    decay_df = pd.DataFrame(decay_rows).T
    decay_df.columns = [f"h={h}d" for h in [1, 5, 10, 20, 60]]

    print(f"\n  {'Factor':<24} " + "  ".join(f"{c:>7}" for c in decay_df.columns))
    print("  " + "-" * 60)
    for name, row in decay_df.iterrows():
        vals = "  ".join(f"{v:>7.4f}" if not np.isnan(v) else "     nan" for v in row)
        print(f"  {name:<24} {vals}")

    print("\n✓  Factor research complete.")
    return summary


if __name__ == "__main__":
    main()
