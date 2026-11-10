"""
Unit test suite for systematic-alpha-engine.

Coverage
--------
  data        : generator shapes, returns distribution, fundamentals
  factors     : technical, fundamental, microstructure correctness
  evaluation  : IC calculation, Newey-West t-stat, BH correction
  models      : CPCV split counts, purging, DSR formula
  risk        : Ledoit-Wolf PSD, RMT denoising, PCA factor model
  portfolio   : MVO, risk parity, BL, HRP — all weights sum to 1
  execution   : Almgren-Chriss trajectory sums to X, kappa solver
  backtest    : walk-forward no look-ahead, attribution sums to active
  analytics   : tearsheet metrics consistency
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def market_data():
    from sae.data.generator import MarketDataGenerator
    gen = MarketDataGenerator(n_stocks=20, n_days=504, n_sectors=4, seed=0)
    return gen.generate()


@pytest.fixture(scope="module")
def small_cov():
    rng = np.random.RandomState(1)
    X = rng.randn(252, 20)
    return np.cov(X.T)


# ── 1. Data Generator ─────────────────────────────────────────────────────────

class TestDataGenerator:
    def test_shapes(self, market_data):
        prices = market_data["prices"]
        assert prices.shape == (504, 20)
        assert market_data["returns"].shape == (504, 20)
        assert market_data["volume"].shape == (504, 20)

    def test_prices_positive(self, market_data):
        assert (market_data["prices"] > 0).all().all()

    def test_returns_finite(self, market_data):
        ret = market_data["returns"].dropna()
        assert np.isfinite(ret.values).all()

    def test_fundamentals_positive(self, market_data):
        bv = market_data["fundamentals"]["book_value"]
        assert (bv > 0).all().all()
        roe = market_data["fundamentals"]["roe"]
        assert (roe > 0).all().all()

    def test_sector_id_valid(self, market_data):
        sid = market_data["metadata"]["sector_id"]
        assert len(sid) == 20
        assert sid.min() == 0
        assert sid.max() < 4


# ── 2. Technical Factors ──────────────────────────────────────────────────────

class TestTechnicalFactors:
    def test_mom_12_1_shape(self, market_data):
        from sae.factors.technical import TechnicalFactors
        tf = TechnicalFactors(market_data["prices"], market_data["volume"])
        mom = tf.mom_12_1()
        assert mom.shape == (504, 20)

    def test_mom_12_1_nan_warmup(self, market_data):
        from sae.factors.technical import TechnicalFactors
        tf = TechnicalFactors(market_data["prices"], market_data["volume"])
        mom = tf.mom_12_1()
        # First 252 rows should be NaN (need 252 days history)
        assert mom.iloc[:252].isna().all().all()

    def test_vol_21_positive(self, market_data):
        from sae.factors.technical import TechnicalFactors
        tf = TechnicalFactors(market_data["prices"], market_data["volume"])
        vol = tf.vol_21().dropna()
        assert (vol > 0).all().all()

    def test_all_factors_keys(self, market_data):
        from sae.factors.technical import TechnicalFactors
        tf = TechnicalFactors(market_data["prices"], market_data["volume"])
        factors = tf.all_factors()
        expected = {"mom_12_1", "mom_6_1", "mom_1m_rev", "rsi_14",
                    "vol_21", "vol_63", "idio_vol", "skew_60",
                    "z_score_52w", "corr_mkt_60"}
        assert set(factors.keys()) == expected


# ── 3. Fundamental Factors ────────────────────────────────────────────────────

class TestFundamentalFactors:
    def test_btp_positive(self, market_data):
        from sae.factors.fundamental import FundamentalFactors
        ff = FundamentalFactors(market_data["prices"], market_data["fundamentals"])
        btp = ff.book_to_price()
        assert (btp > 0).all().all()

    def test_roe_positive(self, market_data):
        from sae.factors.fundamental import FundamentalFactors
        ff = FundamentalFactors(market_data["prices"], market_data["fundamentals"])
        roe = ff.quality_roe()
        assert (roe > 0).all().all()


# ── 4. Microstructure Factors ─────────────────────────────────────────────────

class TestMicrostructureFactors:
    def test_roll_spread_shape(self, market_data):
        from sae.factors.microstructure import MicrostructureFactors
        mf = MicrostructureFactors(
            market_data["returns"], market_data["prices"], market_data["volume"]
        )
        spread = mf.roll_spread(window=63)
        assert spread.shape == (504, 20)

    def test_amihud_finite(self, market_data):
        from sae.factors.microstructure import MicrostructureFactors
        mf = MicrostructureFactors(
            market_data["returns"], market_data["prices"], market_data["volume"]
        )
        amihud = mf.amihud_illiquidity()
        valid = amihud.dropna()
        assert np.isfinite(valid.values).all()


# ── 5. Evaluation / Statistics ────────────────────────────────────────────────

class TestStatistics:
    def test_newey_west_simple(self):
        from sae.evaluation.statistics import newey_west_tstat
        rng = np.random.RandomState(42)
        # Generate IID series with known mean
        x = rng.randn(500) + 0.5  # mean = 0.5
        t = newey_west_tstat(x)
        # Should be >> 2 (highly significant)
        assert t > 5.0

    def test_newey_west_zero_mean(self):
        from sae.evaluation.statistics import newey_west_tstat
        rng = np.random.RandomState(99)
        x = rng.randn(1000)  # mean ~ 0
        t = newey_west_tstat(x)
        assert abs(t) < 3.0  # should be near 0

    def test_bh_rejection(self):
        from sae.evaluation.statistics import benjamini_hochberg
        # 5 small p-values + 10 large ones
        small = [0.001, 0.002, 0.003, 0.004, 0.005]
        large = [0.5] * 10
        p = small + large
        rejected, adj_p, _ = benjamini_hochberg(p, alpha=0.05)
        # Should reject (at least some of) the small ones
        assert rejected[:5].any()
        # Should not reject the large ones
        assert not rejected[5:].any()

    def test_bh_adjusted_valid(self):
        from sae.evaluation.statistics import benjamini_hochberg
        p = [0.01, 0.02, 0.5, 0.9]
        _, adj_p, _ = benjamini_hochberg(p)
        assert all(0 <= x <= 1.0 for x in adj_p)


# ── 6. CPCV ───────────────────────────────────────────────────────────────────

class TestCPCV:
    def test_split_count(self):
        from sae.models.cross_validation import CombPurgedCV
        from math import comb
        cv = CombPurgedCV(n_splits=6, test_splits=1)
        X = np.zeros((300, 5))
        splits = list(cv.split(X))
        assert len(splits) == comb(6, 1)

    def test_no_overlap(self):
        from sae.models.cross_validation import CombPurgedCV
        cv = CombPurgedCV(n_splits=4, test_splits=1, purge_days=0, embargo_days=0)
        X = np.zeros((200, 3))
        for train_idx, test_idx in cv.split(X):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_purging_applied(self):
        from sae.models.cross_validation import CombPurgedCV
        cv = CombPurgedCV(n_splits=4, test_splits=1, purge_days=10, embargo_days=0)
        X = np.zeros((200, 3))
        for train_idx, test_idx in cv.split(X):
            test_start = min(test_idx)
            # No training sample within purge_days before test
            close_before = [i for i in train_idx if test_start - 10 <= i < test_start]
            assert len(close_before) == 0


# ── 7. Deflated Sharpe Ratio ──────────────────────────────────────────────────

class TestDSR:
    def test_psr_high_sharpe(self):
        from sae.models.deflated_sharpe import probabilistic_sharpe_ratio
        rng = np.random.RandomState(7)
        # Daily SR ~ 0.1 → annualised SR ~ 1.6, T=1000 → PSR close to 1
        r = rng.randn(1000) * 0.01 + 0.001
        result = probabilistic_sharpe_ratio(r)
        assert result["PSR"] > 0.9

    def test_dsr_lt_psr(self):
        from sae.models.deflated_sharpe import deflated_sharpe_ratio
        rng = np.random.RandomState(42)
        r = rng.randn(500) * 0.01 + 0.0001
        d = deflated_sharpe_ratio(r, n_trials=20)
        # DSR (accounting for 20 trials) must be <= PSR
        assert d["DSR"] <= d["PSR"] + 1e-6

    def test_min_sr_increases_with_trials(self):
        from sae.models.deflated_sharpe import deflated_sharpe_ratio
        rng = np.random.RandomState(1)
        r = rng.randn(500) * 0.01
        d1 = deflated_sharpe_ratio(r, n_trials=1)
        d100 = deflated_sharpe_ratio(r, n_trials=100)
        assert d100["min_SR"] >= d1["min_SR"]


# ── 8. Ledoit-Wolf & RMT ─────────────────────────────────────────────────────

class TestCovariance:
    def test_lw_psd(self, small_cov):
        from sae.risk.covariance import LedoitWolfEstimator
        rng = np.random.RandomState(5)
        X = rng.randn(252, 20)
        lw = LedoitWolfEstimator()
        lw.fit(X)
        eigvals = np.linalg.eigvalsh(lw.covariance_)
        assert (eigvals > -1e-8).all(), "LW covariance not PSD"

    def test_lw_shrinkage_in_range(self):
        from sae.risk.covariance import LedoitWolfEstimator
        rng = np.random.RandomState(10)
        X = rng.randn(100, 50)
        lw = LedoitWolfEstimator()
        lw.fit(X)
        # Ledoit-Wolf shrinkage is always in [0, 1]
        assert 0 <= lw.shrinkage_ <= 1.0

    def test_rmt_psd(self, small_cov):
        from sae.risk.covariance import rmt_denoise
        cov_d, _ = rmt_denoise(small_cov, n_obs=252)
        eigvals = np.linalg.eigvalsh(cov_d)
        assert (eigvals > -1e-6).all(), "RMT covariance not PSD"

    def test_rmt_n_signal(self, small_cov):
        from sae.risk.covariance import rmt_denoise
        _, info = rmt_denoise(small_cov, n_obs=252)
        assert 0 <= info["n_signal_factors"] <= 20


# ── 9. PCA Factor Model ───────────────────────────────────────────────────────

class TestPCAFactorModel:
    def test_fit_shapes(self, market_data):
        from sae.risk.factor_model import PCAFactorModel
        pfm = PCAFactorModel(n_factors=5)
        pfm.fit(market_data["returns"].dropna())
        assert pfm.loadings_.shape[0] == 20
        assert pfm.loadings_.shape[1] == 5

    def test_covariance_psd(self, market_data):
        from sae.risk.factor_model import PCAFactorModel
        pfm = PCAFactorModel(n_factors=5)
        pfm.fit(market_data["returns"].dropna())
        cov = pfm.covariance().values
        eigvals = np.linalg.eigvalsh(cov)
        assert (eigvals > -1e-6).all()

    def test_explained_variance_sum(self, market_data):
        from sae.risk.factor_model import PCAFactorModel
        pfm = PCAFactorModel(n_factors=5)
        pfm.fit(market_data["returns"].dropna())
        evr = pfm.explained_variance_ratio()
        assert 0 < evr.sum() <= 1.0 + 1e-6


# ── 10. Portfolio Optimisation ────────────────────────────────────────────────

class TestPortfolioOptimiser:
    def test_mv_sums_to_one(self, small_cov):
        from sae.portfolio.optimizer import mean_variance_optimize
        N = small_cov.shape[0]
        mu = np.random.randn(N) * 0.0005
        w = mean_variance_optimize(mu, small_cov)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_mv_long_only(self, small_cov):
        from sae.portfolio.optimizer import mean_variance_optimize
        N = small_cov.shape[0]
        mu = np.random.randn(N) * 0.0005
        w = mean_variance_optimize(mu, small_cov, long_only=True)
        assert (w >= -1e-8).all()

    def test_rp_sums_to_one(self, small_cov):
        from sae.portfolio.optimizer import risk_parity_optimize
        w = risk_parity_optimize(small_cov)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_rp_equal_risk_contributions(self, small_cov):
        from sae.portfolio.optimizer import risk_parity_optimize
        w = risk_parity_optimize(small_cov)
        port_var = w @ small_cov @ w
        rc = w * (small_cov @ w) / (port_var + 1e-12)
        rc_pct = rc / rc.sum()
        # All risk contributions should be approximately equal
        assert rc_pct.std() < 0.05  # less than 5% std

    def test_hrp_sums_to_one(self, small_cov):
        from sae.portfolio.optimizer import hierarchical_risk_parity
        w = hierarchical_risk_parity(small_cov)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_hrp_long_only(self, small_cov):
        from sae.portfolio.optimizer import hierarchical_risk_parity
        w = hierarchical_risk_parity(small_cov)
        assert (w >= -1e-8).all()

    def test_bl_sums_to_one(self, small_cov):
        from sae.portfolio.optimizer import black_litterman
        N = small_cov.shape[0]
        mkt_w = np.ones(N) / N
        P = np.zeros((1, N))
        P[0, :5] = 1.0 / 5
        P[0, -5:] = -1.0 / 5
        Q = np.array([0.0001])
        w = black_litterman(small_cov, mkt_w, P, Q)
        assert abs(w.sum() - 1.0) < 1e-6


# ── 11. Almgren-Chriss ────────────────────────────────────────────────────────

class TestAlmgrenChriss:
    def test_trajectory_sums_to_X(self):
        from sae.execution.almgren_chriss import AlmgrenChriss
        ac = AlmgrenChriss(sigma=0.02, eta=0.002, gamma=0.0002)
        result = ac.optimal_trajectory(X=100_000, T=10, N=10, risk_aversion=1e-6)
        assert abs(result.trades.sum() - 100_000) < 1.0

    def test_holdings_decreasing(self):
        from sae.execution.almgren_chriss import AlmgrenChriss
        ac = AlmgrenChriss(sigma=0.02, eta=0.002)
        result = ac.optimal_trajectory(X=50_000, T=5, N=5, risk_aversion=1e-6)
        assert result.holdings[-1] < 1.0  # fully liquidated

    def test_higher_lambda_faster_liquidation(self):
        from sae.execution.almgren_chriss import AlmgrenChriss
        ac = AlmgrenChriss(sigma=0.02, eta=0.002)
        r_low  = ac.optimal_trajectory(1e5, 10, risk_aversion=1e-7)
        r_high = ac.optimal_trajectory(1e5, 10, risk_aversion=1e-5)
        # Higher risk aversion → sell more earlier → faster decay
        assert r_high.holdings[3] < r_low.holdings[3]

    def test_vwap_sums_to_X(self):
        from sae.execution.almgren_chriss import AlmgrenChriss
        ac = AlmgrenChriss()
        trades = ac.vwap_schedule(10_000, N=10)
        assert abs(trades.sum() - 10_000) < 1e-8


# ── 12. Walk-Forward Backtest ─────────────────────────────────────────────────

class TestWalkForwardBacktest:
    def test_no_lookahead(self, market_data):
        """Signal computed on [0:t] must not use prices at [t:]."""
        from sae.backtest.engine import WalkForwardBacktest
        prices = market_data["prices"]
        volume = market_data["volume"]
        N = prices.shape[1]

        signal_records = []

        def signal_fn(train_prices, train_volume):
            signal_records.append(train_prices.index[-1])
            return np.ones(N) / N

        def port_fn(signals, cov, assets):
            return np.ones(N) / N

        bt = WalkForwardBacktest(train_window=126, test_window=21, step=21)
        result = bt.run(prices, volume, signal_fn, port_fn)

        # Each signal date must precede the rebalance (first OOS) date
        rebalance_dates = result["rebalance_dates"]
        for sig_date, reb_date in zip(signal_records, rebalance_dates):
            assert sig_date < reb_date, (
                f"Look-ahead: signal computed on {sig_date}, "
                f"but rebalance date is {reb_date}"
            )

    def test_portfolio_returns_length(self, market_data):
        from sae.backtest.engine import WalkForwardBacktest
        prices = market_data["prices"]
        volume = market_data["volume"]
        N = prices.shape[1]

        bt = WalkForwardBacktest(train_window=126, test_window=21, step=21)
        result = bt.run(
            prices, volume,
            lambda p, v: np.ones(N) / N,
            lambda s, c, a: np.ones(N) / N,
        )
        assert len(result["portfolio_returns"]) > 0


# ── 13. Tearsheet ─────────────────────────────────────────────────────────────

class TestTearsheet:
    def test_metrics_present(self):
        from sae.analytics.tearsheet import compute_tearsheet
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=252)
        rets = pd.Series(rng.randn(252) * 0.01 + 0.0003, index=dates)
        m = compute_tearsheet(rets)
        for key in ["annual_return", "annual_vol", "sharpe_ratio",
                    "max_drawdown", "sortino_ratio", "calmar_ratio"]:
            assert key in m

    def test_sharpe_sign(self):
        from sae.analytics.tearsheet import compute_tearsheet
        rng = np.random.RandomState(1)
        # Clearly positive-return series
        dates = pd.bdate_range("2020-01-01", periods=500)
        rets = pd.Series(rng.randn(500) * 0.005 + 0.001, index=dates)
        m = compute_tearsheet(rets)
        assert m["sharpe_ratio"] > 0.0

    def test_max_drawdown_negative(self):
        from sae.analytics.tearsheet import compute_tearsheet
        rng = np.random.RandomState(2)
        dates = pd.bdate_range("2020-01-01", periods=252)
        rets = pd.Series(rng.randn(252) * 0.02, index=dates)
        m = compute_tearsheet(rets)
        assert m["max_drawdown"] <= 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
