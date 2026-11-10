"""
ML Alpha Model Demo
===================
Trains a LightGBM model with Combinatorial Purged Cross-Validation,
evaluates OOS IC, computes Probabilistic and Deflated Sharpe Ratios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sae.data.generator import MarketDataGenerator
from sae.factors.technical import TechnicalFactors
from sae.factors.fundamental import FundamentalFactors
from sae.models.alpha_model import AlphaModel
from sae.models.deflated_sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from sae.evaluation.statistics import newey_west_tstat
from scipy import stats


def build_simple_factors(prices, volume, fundamentals):
    """Fast subset of factors for ML feature matrix."""
    tech = TechnicalFactors(prices, volume)
    fund = FundamentalFactors(prices, fundamentals)
    return {
        "mom_12_1":        tech.mom_12_1(),
        "mom_6_1":         tech.mom_6_1(),
        "mom_1m_rev":      tech.mom_1m_rev(),
        "vol_21":          tech.vol_21(),
        "vol_63":          tech.vol_63(),
        "idio_vol":        tech.idio_vol(),
        "z_score_52w":     tech.z_score_52(),
        "book_to_price":   fund.book_to_price(),
        "earnings_to_price": fund.earnings_to_price(),
        "roe":             fund.quality_roe(),
        "low_leverage":    fund.low_leverage(),
    }


def main():
    print("=" * 60)
    print("  Systematic Alpha Engine — ML Alpha Model")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────
    print("\n[1/5] Generating data…")
    gen = MarketDataGenerator(n_stocks=100, n_days=1260, n_sectors=5, seed=42)
    data = gen.generate()
    prices   = data["prices"]
    returns  = data["returns"]
    volume   = data["volume"]
    fundam   = data["fundamentals"]
    stocks   = data["metadata"]["stocks"]

    # ── 2. Build factors ─────────────────────────────────────────────
    print("[2/5] Computing features…")
    factor_dict = build_simple_factors(prices, volume, fundam)
    print(f"      {len(factor_dict)} factors × {prices.shape[0]} days × {prices.shape[1]} stocks")

    # ── 3. Build feature matrix ───────────────────────────────────────
    print("[3/5] Building feature matrix (cross-sectional z-score normalised)…")
    feature_df = AlphaModel.build_feature_matrix(
        factor_dict, returns, forward_days=1, min_stocks=20
    )
    if feature_df.empty:
        print("  ERROR: empty feature matrix")
        return

    feature_names = list(factor_dict.keys())
    X = feature_df[feature_names].values
    y = feature_df["target"].values
    print(f"      Feature matrix: {X.shape[0]:,} obs × {X.shape[1]} features")
    print(f"      NaN rate: {np.isnan(X).mean():.1%}")

    # ── 4. Train LightGBM with CPCV ──────────────────────────────────
    print("[4/5] Training LightGBM with Combinatorial Purged CV (6 folds, 1 test)…")
    model = AlphaModel(
        n_splits=6, test_splits=1, purge_days=5, embargo_days=5,
        lgbm_params={"n_estimators": 150, "num_leaves": 31, "verbose": -1},
    )
    model.fit(X, y, feature_names=feature_names)

    n_splits = model.cv.get_n_splits()
    print(f"      Total CV splits: {n_splits} (C({6},{1}) combinations)")
    print(f"      Models trained:  {len(model.models_)}")

    # OOS IC
    oos_mask = ~np.isnan(model.oos_predictions_)
    n_oos = oos_mask.sum()
    oos_ic = model.oos_ic()
    print(f"      OOS observations: {n_oos:,}")
    print(f"      OOS IC (Spearman): {oos_ic:.4f}")

    # ── 5. Sharpe / DSR ──────────────────────────────────────────────
    print("[5/5] Computing Sharpe, PSR, and Deflated Sharpe Ratio…")

    # Simulate returns of a simple long-short strategy from ML predictions
    # (quintile long top-20, short bottom-20 within each cross-section)
    dates_unique = feature_df["_date"].unique()
    strat_returns = []

    for date in dates_unique:
        mask = (feature_df["_date"] == date)
        pred = model.oos_predictions_[mask]
        tgt  = y[mask]
        valid = ~(np.isnan(pred) | np.isnan(tgt))
        if valid.sum() < 20:
            continue
        p = pred[valid]
        t = tgt[valid]
        # Long-short quintile return
        n = len(p)
        q = max(1, n // 5)
        ranked = np.argsort(p)
        long_ret  = t[ranked[-q:]].mean()
        short_ret = t[ranked[:q]].mean()
        strat_returns.append(long_ret - short_ret)

    strat_arr = np.array(strat_returns)
    strat_arr = strat_arr[~np.isnan(strat_arr)]

    psr_res = probabilistic_sharpe_ratio(strat_arr, sr_benchmark=0.0)
    dsr_res = deflated_sharpe_ratio(strat_arr, n_trials=len(factor_dict))

    n_trials = len(factor_dict)
    print(f"\n  Strategy: Long-short quintile (OOS L/S return z-scores)")
    print(f"  OOS Sharpe (annualised):  {psr_res['SR_hat']:.3f}")
    print(f"  Skewness:                 {psr_res['skew']:.3f}")
    print(f"  Excess kurtosis:          {psr_res['kurtosis']:.3f}")
    print(f"  PSR (benchmark SR=0):     {psr_res['PSR']:.4f}")
    print(f"  DSR (N_trials={n_trials:2d}):         {dsr_res['DSR']:.4f}")
    print(f"  Min SR (expected max):    {dsr_res['min_SR']:.3f}")

    # Feature importance
    imp = model.feature_importance()
    if not imp.empty:
        print(f"\n  Feature Importance (LightGBM):")
        for fname, val in imp.head(8).items():
            bar = "█" * int(val / imp.max() * 20)
            print(f"    {fname:<24} {bar:<20} {val:.0f}")

    print("\n✓  ML alpha model demo complete.")
    return model


if __name__ == "__main__":
    main()
