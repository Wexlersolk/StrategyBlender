"""
research/monte_carlo.py

Monte Carlo simulation on backtest results.

Two methods:
  1. Trade resampling  — shuffle trade order 1000x, compute equity paths
  2. Return bootstrap  — resample monthly returns, build equity distributions

Outputs confidence intervals for key metrics:
  - Net profit (5th, 25th, 50th, 75th, 95th percentile)
  - Max drawdown distribution
  - Sharpe ratio distribution
  - Probability of ruin (equity drops below threshold)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from engine.results import BacktestResults


@dataclass
class MonteCarloResults:
    n_simulations:  int
    profits:        np.ndarray   # (n_sims,) final profits
    max_drawdowns:  np.ndarray   # (n_sims,) max drawdown %
    sharpes:        np.ndarray   # (n_sims,)

    # Equity paths — shape (n_sims, n_trades)
    equity_paths:   np.ndarray

    initial_capital: float

    @property
    def profit_percentiles(self) -> dict:
        return {p: float(np.percentile(self.profits, p))
                for p in [5, 25, 50, 75, 95]}

    @property
    def drawdown_percentiles(self) -> dict:
        return {p: float(np.percentile(self.max_drawdowns, p))
                for p in [5, 25, 50, 75, 95]}

    @property
    def prob_profit(self) -> float:
        return float(np.mean(self.profits > 0))

    def prob_ruin(self, threshold_pct: float = 20.0) -> float:
        """Probability that max drawdown exceeds threshold_pct %."""
        return float(np.mean(self.max_drawdowns > threshold_pct))

    def summary(self) -> dict:
        pp = self.profit_percentiles
        dp = self.drawdown_percentiles
        return {
            "n_simulations":       self.n_simulations,
            "profit_p5":           round(pp[5], 2),
            "profit_p25":          round(pp[25], 2),
            "profit_median":       round(pp[50], 2),
            "profit_p75":          round(pp[75], 2),
            "profit_p95":          round(pp[95], 2),
            "drawdown_p50":        round(dp[50], 2),
            "drawdown_p95":        round(dp[95], 2),
            "prob_profit":         round(self.prob_profit, 4),
            "prob_ruin_20pct":     round(self.prob_ruin(20.0), 4),
            "sharpe_median":       round(float(np.median(self.sharpes)), 4),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  Monte Carlo ({s['n_simulations']:,} simulations)")
        print(f"{'='*50}")
        print(f"  Net Profit (5th pct):   ${s['profit_p5']:>10,.2f}")
        print(f"  Net Profit (median):    ${s['profit_median']:>10,.2f}")
        print(f"  Net Profit (95th pct):  ${s['profit_p95']:>10,.2f}")
        print(f"  Max DD (median):        {s['drawdown_p50']:>9.2f}%")
        print(f"  Max DD (95th pct):      {s['drawdown_p95']:>9.2f}%")
        print(f"  Prob. of Profit:        {s['prob_profit']*100:>9.1f}%")
        print(f"  Prob. Ruin (DD>20%):    {s['prob_ruin_20pct']*100:>9.1f}%")
        print(f"  Sharpe (median):        {s['sharpe_median']:>10.4f}")
        print(f"{'='*50}\n")


def run_monte_carlo(
    results: BacktestResults,
    n_simulations: int = 1000,
    method: str = "resample_trades",
    seed: int = 42,
) -> MonteCarloResults:
    """
    Run Monte Carlo simulation on backtest results.

    method:
        "resample_trades"  — shuffle trade order (tests sequence sensitivity)
        "bootstrap_monthly"— resample monthly returns (tests distribution)
    """
    rng    = np.random.default_rng(seed)
    trades = results.profits

    if len(trades) < 5:
        raise ValueError("Need at least 5 trades for Monte Carlo.")

    if method == "bootstrap_monthly":
        # Use monthly returns if available
        monthly = results.monthly_stats()
        if len(monthly) >= 6:
            trades = monthly["profit"].values

    n       = len(trades)
    cap     = results.initial_capital
    profits = np.zeros(n_simulations)
    dds     = np.zeros(n_simulations)
    sharpes = np.zeros(n_simulations)

    # Store all paths (cap limited to save memory)
    store_paths = min(n_simulations, 500)
    paths       = np.zeros((store_paths, n))

    for sim in range(n_simulations):
        # Resample with replacement
        sample = rng.choice(trades, size=n, replace=True)
        equity = cap + np.cumsum(sample)

        # Max drawdown
        peak  = np.maximum.accumulate(np.concatenate([[cap], equity]))
        dd    = (peak[1:] - equity) / peak[1:] * 100
        max_dd = float(dd.max())

        # Sharpe
        std = sample.std()
        sh  = float(sample.mean() / std * np.sqrt(12)) if std > 1e-8 else 0.0

        profits[sim] = float(equity[-1] - cap)
        dds[sim]     = max_dd
        sharpes[sim] = sh

        if sim < store_paths:
            paths[sim] = equity

    return MonteCarloResults(
        n_simulations   = n_simulations,
        profits         = profits,
        max_drawdowns   = dds,
        sharpes         = sharpes,
        equity_paths    = paths,
        initial_capital = cap,
    )
