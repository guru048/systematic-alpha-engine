# Systematic Alpha Engine

A research-grade quantitative finance platform implementing the core techniques used at Two Sigma, Citadel, Jane Street, and AQR — built from first principles in pure Python/NumPy/SciPy.

> **46/46 unit tests pass · 4 end-to-end demo scripts · no external data required**

---

## What it implements

### Factor Research & Evaluation
| Technique | Paper / Author | Module |
|---|---|---|
| IC / ICIR with Newey-West HAC | Newey & West (1987) | `sae/evaluation/` |
| Fama-MacBeth two-pass regression | Fama & MacBeth (1973) | `sae/evaluation/statistics.py` |
| Benjamini-Hochberg FDR correction | Benjamini & Hochberg (1995) | `sae/evaluation/statistics.py` |
| Factor decay analysis | — | `sae/evaluation/metrics.py` |
| 10 technical factors (momentum, reversal, vol, skew) | — | `sae/factors/technical.py` |
| 8 fundamental factors (B/P, E/P, ROE, leverage) | Fama & French (1993) | `sae/factors/fundamental.py` |
| 5 microstructure factors (Roll, Amihud, Kyle λ) | Roll (1984), Amihud (2002), Kyle (1985) | `sae/factors/microstructure.py` |

### ML Alpha Model
| Technique | Detail |
|---|---|
| LightGBM regressor | Gradient-boosted trees predicting cross-sectional z-score of forward returns |
| Combinatorial Purged Cross-Validation | C(6,1)=6 folds, 5-day purge + 5-day embargo |
| Probabilistic Sharpe Ratio | Bailey & Lopez de Prado (2012) |
| Deflated Sharpe Ratio | Bailey & Lopez de Prado (2014) |

### Portfolio Construction
| Method | Paper |
|---|---|
| Mean-Variance Optimisation (SLSQP) | Markowitz (1952) |
| Risk Parity / Equal Risk Contribution | Roncalli (2013) |
| Black-Litterman Bayesian views | Black & Litterman (1990) |
| Hierarchical Risk Parity (Ward linkage) | Lopez de Prado (2016) |

### Covariance Estimation
| Method | Paper |
|---|---|
| Ledoit-Wolf analytical shrinkage | Ledoit & Wolf (2004) |
| Random Matrix Theory / Marchenko-Pastur denoising | Marčenko & Pastur (1967) |
| PCA statistical factor risk model (B·F·Bᵀ + Δ) | — |

### Execution & Transaction Costs
| Method | Paper |
|---|---|
| Almgren-Chriss optimal trajectory | Almgren & Chriss (2001) |
| Square-root law market impact | — |
| Newton-Raphson κ solver | — |

### Backtesting & Attribution
| Method | Paper |
|---|---|
| Walk-forward backtest (no look-ahead bias) | — |
| Brinson-Hood-Beebower sector attribution | Brinson, Hood & Beebower (1986) |
| Bootstrap Sharpe confidence intervals | — |

---

## Demo results (actual output)

### Factor Research (`scripts/run_factor_research.py`)
```
23 factors computed on 100 stocks × 5 years

Top factors by Newey-West t-statistic:
  corr_mkt_60       IC=0.016  ICIR=0.147  t_NW=3.063  ★ BH-significant
  skew_60           IC=0.014  ICIR=0.136  t_NW=2.988  ★ BH-significant
  vol_21            IC=0.011  ICIR=0.113  t_NW=2.398

2/23 factors pass Benjamini-Hochberg FDR correction (α=5%)
```

### ML Alpha Model (`scripts/run_ml_alpha.py`)
```
LightGBM + CPCV (6 folds, purge=5d, embargo=5d)
Training observations: 125,900

OOS IC:  0.0047
SR_hat:  0.188   PSR: 0.646   DSR: 0.002  (N_trials=11)

Top features by importance: corr_mkt_60, vol_21, skew_60, mom_1m_rev, rsi_14
```

### Portfolio Construction (`scripts/run_portfolio_construction.py`)
```
Covariance: Ledoit-Wolf α=0.0579, RMT signal factors=2, λ_max_noise=1.979

Portfolio             Ann.Ret  Ann.Vol  Sharpe    MaxW    EffN    Gini
-----------------------------------------------------------------
Mean-Variance           4.16%   16.74%   0.249    5.0%    21.7  0.796
Risk Parity            -0.08%   16.16%  -0.005    1.8%    93.3  0.150
Black-Litterman         1.53%   17.87%   0.086    2.0%    71.3  0.321
HRP                     0.33%   19.02%   0.018    3.4%    64.6  0.388
Equal Weight            0.00%   17.30%   0.000    1.0%   100.0  0.000

Risk Parity: all 100 assets contribute exactly 1.00% to portfolio risk (ERC achieved)

Almgren-Chriss execution ($10M, 100k shares, 10-day horizon):
  Low  risk aversion: E[cost]=$45,922 (45.9bps)  κ=0.130
  Med  risk aversion: E[cost]=$50,528 (50.5bps)  κ=0.230
  High risk aversion: E[cost]=$65,167 (65.2bps)  κ=0.405

Market impact (square-root law):
  1% ADV  → 7.2bps   5% ADV  → 10.0bps   20% ADV → 15.0bps
```

### Full Pipeline Walk-Forward Backtest (`scripts/run_full_pipeline.py`)
```
Walk-forward: 252d train, 63d test, quarterly rebalancing (2019–2023)

Strategy          Ann.Ret  Ann.Vol  Sharpe    MaxDD    DSR
---------------------------------------------------------
Risk Parity        43.38%   16.29%   2.025  -16.28%  0.9998
Mean-Variance      36.57%   15.58%   1.797  -14.61%  0.9990
Benchmark          47.58%   17.71%   2.039  -17.96%    —

Risk Parity: Sharpe 95% CI (1.11, 2.98) · Sortino 2.50 · Calmar 2.66
Brinson attribution: selection effect −3.0% annualised (strategy vs benchmark)
```

---

## Project structure

```
systematic-alpha-engine/
├── sae/
│   ├── data/generator.py          # AR(1) multi-factor synthetic market data
│   ├── factors/
│   │   ├── technical.py           # 10 price/volume factors
│   │   ├── fundamental.py         # 8 accounting factors (Fama-French proxies)
│   │   └── microstructure.py      # 5 market microstructure factors
│   ├── evaluation/
│   │   ├── metrics.py             # IC, ICIR, factor decay, turnover
│   │   └── statistics.py          # Newey-West HAC, BH FDR, Fama-MacBeth
│   ├── models/
│   │   ├── alpha_model.py         # LightGBM alpha with CPCV
│   │   ├── cross_validation.py    # Combinatorial Purged CV
│   │   └── deflated_sharpe.py     # PSR / DSR
│   ├── risk/
│   │   ├── covariance.py          # Ledoit-Wolf + RMT denoising
│   │   └── factor_model.py        # PCA factor risk model
│   ├── portfolio/
│   │   └── optimizer.py           # MVO, Risk Parity, Black-Litterman, HRP
│   ├── execution/
│   │   ├── almgren_chriss.py      # Optimal execution trajectory
│   │   └── impact.py              # Square-root law market impact
│   ├── backtest/
│   │   ├── engine.py              # Walk-forward backtest engine
│   │   └── attribution.py         # Brinson-Hood-Beebower attribution
│   └── analytics/
│       └── tearsheet.py           # Full performance tearsheet + bootstrap CI
├── scripts/
│   ├── run_factor_research.py
│   ├── run_ml_alpha.py
│   ├── run_portfolio_construction.py
│   └── run_full_pipeline.py
├── tests/
│   └── test_sae.py                # 46 unit tests
└── requirements.txt
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS only (LightGBM OpenMP):
brew install libomp
```

## Run

```bash
# Factor research
python3 scripts/run_factor_research.py

# ML alpha model
python3 scripts/run_ml_alpha.py

# Portfolio construction
python3 scripts/run_portfolio_construction.py

# Full walk-forward pipeline
python3 scripts/run_full_pipeline.py

# Tests
python3 -m pytest tests/ -v
```

---

## Key design decisions

**Synthetic data with embedded signals.** The `MarketDataGenerator` constructs returns via an AR(1) multi-factor model so that momentum, value, and quality factors have genuine positive IC — this lets the entire pipeline be validated end-to-end without market data.

**No look-ahead bias.** The walk-forward engine strictly uses `prices[:t]` for signal computation and `prices[t:t+h]` for OOS evaluation. The CPCV implementation purges and embargoes samples to prevent label overlap leakage.

**Covariance pipeline.** Ledoit-Wolf shrinkage regularises the sample covariance, then RMT denoising (Marchenko-Pastur noise threshold) replaces noise eigenvalues with their mean, preserving `K` signal factors.

**Almgren-Chriss.** The κ parameter is solved via Newton-Raphson since there is no closed-form solution for general risk aversion. The efficient frontier sweeps 50 λ values to show the cost-risk tradeoff.

**Statistical rigour.** All IC t-statistics use Newey-West HAC variance (Bartlett kernel, lag `4·(T/100)^0.25`) to account for autocorrelation. Multiple-testing correction uses Benjamini-Hochberg FDR at 5%.
