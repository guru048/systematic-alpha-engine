"""
Performance tearsheet and analytics.

Computes comprehensive performance metrics for a return series and
produces a formatted text tearsheet comparable to QuantLib / pyfolio.

Metrics
-------
  Annual return, Annualised volatility, Sharpe ratio, Sortino ratio,
  Calmar ratio, Max drawdown, Max drawdown duration, Hit rate,
  Avg win / avg loss, Skewness, Excess kurtosis, VaR (5%), CVaR (5%),
  Best/worst month, Monthly return table, Bootstrap CI for Sharpe.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple


def compute_tearsheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.045,
    frequency: int = 252,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute full performance tearsheet for a return series.

    Parameters
    ----------
    returns       : Daily return series (fractional, not %).
    benchmark     : Optional benchmark return series.
    risk_free_rate: Annual risk-free rate for Sharpe computation.
    frequency     : Trading days per year (252 for daily).
    n_bootstrap   : Bootstrap samples for CI computation.
    seed          : RNG seed for reproducibility.

    Returns
    -------
    dict of metrics.
    """
    r = returns.dropna().values
    T = len(r)
    if T < 10:
        return {}

    rf_daily = (1 + risk_free_rate) ** (1 / frequency) - 1

    # ── Core metrics ─────────────────────────────────────────────────────
    total_ret = float(np.prod(1 + r) - 1)
    ann_ret = float((1 + total_ret) ** (frequency / T) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(frequency))

    excess_r = r - rf_daily
    sharpe = float(excess_r.mean() / excess_r.std(ddof=1) * np.sqrt(frequency))

    # Sortino: only downside deviation
    downside = r[r < rf_daily] - rf_daily
    sortino_denom = np.sqrt(np.mean(downside ** 2) * frequency) if len(downside) > 0 else 1e-8
    sortino = float((ann_ret - risk_free_rate) / sortino_denom)

    # Max drawdown and duration
    cum_ret = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum_ret)
    dd = cum_ret / running_max - 1.0
    max_dd = float(dd.min())

    # Drawdown duration (longest drawdown period)
    in_dd = dd < 0
    dd_len, cur_len = 0, 0
    for x in in_dd:
        cur_len = cur_len + 1 if x else 0
        dd_len = max(dd_len, cur_len)

    # Calmar ratio
    calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-8 else np.nan

    # Win statistics
    hit_rate = float((r > 0).mean())
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-8 else np.nan

    # Higher moments
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=True))

    # VaR / CVaR
    var_5 = float(np.percentile(r, 5))
    cvar_5 = float(r[r <= var_5].mean()) if (r <= var_5).any() else var_5

    # ── Bootstrap confidence intervals for Sharpe ─────────────────────────
    rng = np.random.RandomState(seed)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(r, size=T, replace=True)
        exc = sample - rf_daily
        s = exc.mean() / (exc.std(ddof=1) + 1e-12) * np.sqrt(frequency)
        boot_sharpes.append(s)
    boot_sharpes = np.array(boot_sharpes)
    sharpe_ci_low = float(np.percentile(boot_sharpes, 2.5))
    sharpe_ci_high = float(np.percentile(boot_sharpes, 97.5))

    # ── Active return metrics vs benchmark ───────────────────────────────
    active_metrics = {}
    if benchmark is not None:
        b = benchmark.reindex(returns.index).dropna().values
        common_len = min(len(r), len(b))
        r_a, b_a = r[:common_len], b[:common_len]
        active = r_a - b_a
        active_ret = float(active.mean() * frequency)
        te = float(active.std(ddof=1) * np.sqrt(frequency))
        info_ratio = float(active_ret / te) if te > 1e-8 else np.nan
        active_metrics = {
            "active_return": active_ret,
            "tracking_error": te,
            "information_ratio": info_ratio,
            "beta": float(np.cov(r_a, b_a)[0, 1] / (np.var(b_a) + 1e-12)),
        }

    # ── Monthly return table ──────────────────────────────────────────────
    monthly = None
    if isinstance(returns.index, pd.DatetimeIndex):
        monthly_ret = (returns + 1).resample("ME").prod() - 1
        monthly = monthly_ret

    result = {
        # Headline
        "total_return": total_ret,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "sharpe_ci_95": (sharpe_ci_low, sharpe_ci_high),
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        # Drawdown
        "max_drawdown": max_dd,
        "max_dd_duration_days": dd_len,
        # Win/Loss
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        # Moments
        "skewness": skew,
        "excess_kurtosis": kurt,
        # Tail risk
        "var_5pct": var_5,
        "cvar_5pct": cvar_5,
        # Periods
        "n_days": T,
        # Active (if benchmark provided)
        **active_metrics,
        # Monthly series
        "monthly_returns": monthly,
    }
    return result


def print_tearsheet(metrics: dict, title: str = "Portfolio Tearsheet") -> None:
    """Print a formatted tearsheet to stdout."""
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    def pct(x):
        return f"{x * 100:.2f}%" if not np.isnan(x) else "N/A"

    def fmt(x, decimals=2):
        return f"{x:.{decimals}f}" if not np.isnan(x) else "N/A"

    rows = [
        ("Total Return",         pct(metrics.get("total_return", np.nan))),
        ("Annual Return",        pct(metrics.get("annual_return", np.nan))),
        ("Annual Volatility",    pct(metrics.get("annual_vol", np.nan))),
        ("Sharpe Ratio",         fmt(metrics.get("sharpe_ratio", np.nan))),
        ("  95% CI",             str(tuple(round(x, 2) for x in metrics.get("sharpe_ci_95", (np.nan, np.nan))))),
        ("Sortino Ratio",        fmt(metrics.get("sortino_ratio", np.nan))),
        ("Calmar Ratio",         fmt(metrics.get("calmar_ratio", np.nan))),
        ("Max Drawdown",         pct(metrics.get("max_drawdown", np.nan))),
        ("Max DD Duration",      f"{metrics.get('max_dd_duration_days', 0)} days"),
        ("Hit Rate",             pct(metrics.get("hit_rate", np.nan))),
        ("Win/Loss Ratio",       fmt(metrics.get("win_loss_ratio", np.nan))),
        ("Skewness",             fmt(metrics.get("skewness", np.nan))),
        ("Excess Kurtosis",      fmt(metrics.get("excess_kurtosis", np.nan))),
        ("VaR (5%)",             pct(metrics.get("var_5pct", np.nan))),
        ("CVaR (5%)",            pct(metrics.get("cvar_5pct", np.nan))),
        ("N Days",               str(metrics.get("n_days", 0))),
    ]

    if "information_ratio" in metrics:
        rows.extend([
            ("", ""),
            ("Active Return",        pct(metrics.get("active_return", np.nan))),
            ("Tracking Error",       pct(metrics.get("tracking_error", np.nan))),
            ("Information Ratio",    fmt(metrics.get("information_ratio", np.nan))),
            ("Beta",                 fmt(metrics.get("beta", np.nan))),
        ])

    for label, val in rows:
        if label == "":
            print()
        else:
            print(f"  {label:<26} {val}")
    print(sep)

    # Monthly return table
    monthly = metrics.get("monthly_returns")
    if monthly is not None and len(monthly) > 0:
        print("\n  Monthly Returns (%)")
        print("  " + "-" * 54)
        try:
            monthly_pct = (monthly * 100).round(1)
            for year, group in monthly_pct.groupby(monthly_pct.index.year):
                months = {d.month: v for d, v in group.items()}
                row = f"  {year}  "
                for m in range(1, 13):
                    v = months.get(m, None)
                    if v is None:
                        row += "   --- "
                    else:
                        row += f" {v:+6.1f}"
                print(row)
        except Exception:
            pass
        print()
