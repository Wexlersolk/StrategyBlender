"""
research/walk_forward.py

Walk-Forward Optimization (WFO).

Splits data into rolling train/test windows:
  - Train on N months → find best parameters
  - Test on next M months → record out-of-sample performance
  - Slide forward, repeat

This tests whether the strategy is robust or overfitted.
A strategy that performs well out-of-sample is much more likely
to work in live trading.

Window types:
  "rolling"  — fixed-size train window slides forward
  "anchored" — train window grows from a fixed start date
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Callable
from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from engine.results import BacktestResults


@dataclass
class WFOWindow:
    window_id:      int
    train_from:     pd.Timestamp
    train_to:       pd.Timestamp
    test_from:      pd.Timestamp
    test_to:        pd.Timestamp
    best_params:    dict
    train_results:  BacktestResults
    test_results:   BacktestResults

    @property
    def is_profitable(self) -> bool:
        return self.test_results.net_profit > 0

    @property
    def efficiency(self) -> float:
        """Out-of-sample / In-sample performance ratio (1.0 = perfect)."""
        is_sharpe = self.train_results.sharpe_ratio
        oos_sharpe = self.test_results.sharpe_ratio
        if abs(is_sharpe) < 1e-8:
            return 0.0
        return oos_sharpe / is_sharpe


@dataclass
class WFOResults:
    windows:         List[WFOWindow]
    combined_trades: List          # all out-of-sample trades in sequence
    params_used:     list[dict]    # params per window

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def profitable_windows(self) -> int:
        return sum(1 for w in self.windows if w.is_profitable)

    @property
    def robustness_score(self) -> float:
        """% of windows that were profitable out-of-sample."""
        return self.profitable_windows / self.n_windows if self.n_windows else 0.0

    @property
    def avg_efficiency(self) -> float:
        effs = [w.efficiency for w in self.windows]
        return float(np.mean(effs)) if effs else 0.0

    @property
    def oos_net_profit(self) -> float:
        return sum(w.test_results.net_profit for w in self.windows)

    def summary(self) -> dict:
        return {
            "n_windows":          self.n_windows,
            "profitable_windows": self.profitable_windows,
            "robustness_score":   round(self.robustness_score, 4),
            "avg_efficiency":     round(self.avg_efficiency, 4),
            "oos_net_profit":     round(self.oos_net_profit, 2),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  Walk-Forward Results ({s['n_windows']} windows)")
        print(f"{'='*50}")
        print(f"  Profitable windows:  {s['profitable_windows']}/{s['n_windows']}")
        print(f"  Robustness score:    {s['robustness_score']*100:.1f}%")
        print(f"  Avg IS/OOS ratio:    {s['avg_efficiency']:.4f}")
        print(f"  OOS Net Profit:      ${s['oos_net_profit']:,.2f}")
        print(f"\n  Window detail:")
        for w in self.windows:
            print(f"    W{w.window_id}: "
                  f"Train {w.train_from.date()}→{w.train_to.date()} | "
                  f"Test {w.test_from.date()}→{w.test_to.date()} | "
                  f"OOS P&L: ${w.test_results.net_profit:+,.0f} | "
                  f"Eff: {w.efficiency:.2f}")
        print(f"{'='*50}\n")


def run_wfo(
    strategy_class,
    df: pd.DataFrame,
    param_grid: dict,
    train_months: int = 24,
    test_months:  int = 6,
    wfo_type:     str = "rolling",
    optimize_by:  str = "sharpe_ratio",
    backtester_kwargs: dict = None,
    n_top_params: int = 1,
    intrabar_df: pd.DataFrame | None = None,
) -> WFOResults:
    """
    Run Walk-Forward Optimization.

    Args:
        strategy_class:   BaseStrategy subclass (not instance)
        df:               Full OHLCV DataFrame
        param_grid:       Dict of param_name → list of values to test
                          e.g. {"StopLossCoef1": [1.5, 2.0, 2.5],
                                "ProfitTargetCoef1": [1.5, 2.0, 2.5]}
        train_months:     Training window size in months
        test_months:      Test window size in months
        wfo_type:         "rolling" or "anchored"
        optimize_by:      Metric to maximise: "sharpe_ratio", "net_profit",
                          "profit_factor", "recovery_factor"
        backtester_kwargs: Passed to Backtester constructor
        n_top_params:     Average results from top N parameter sets
    """
    if backtester_kwargs is None:
        backtester_kwargs = {}

    param_combination_count = _param_grid_size(param_grid)
    print(f"WFO: {param_combination_count} parameter combinations to test per window")

    windows = []
    all_oos_trades = []

    start_date = df.index[0]
    end_date   = df.index[-1]
    step       = pd.DateOffset(months=test_months)
    window_id  = 0

    # Generate window boundaries
    test_start = start_date + pd.DateOffset(months=train_months)
    while test_start + pd.DateOffset(months=test_months) <= end_date:
        test_end   = test_start + pd.DateOffset(months=test_months)

        if wfo_type == "rolling":
            train_start = test_start - pd.DateOffset(months=train_months)
        else:  # anchored
            train_start = start_date

        train_end = test_start

        # ── In-sample optimisation ────────────────────────────────────────────
        print(f"  Window {window_id}: "
              f"train {train_start.date()}→{train_end.date()} | "
              f"test {test_start.date()}→{test_end.date()}")

        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df  = df[(df.index >= test_start)  & (df.index < test_end)]
        train_intrabar_df = None
        test_intrabar_df = None
        if intrabar_df is not None:
            train_intrabar_df = intrabar_df[(intrabar_df.index >= train_start) & (intrabar_df.index < train_end)]
            test_intrabar_df = intrabar_df[(intrabar_df.index >= test_start) & (intrabar_df.index < test_end)]

        if len(train_df) < 100 or len(test_df) < 20:
            test_start += step
            window_id  += 1
            continue

        best_params, train_results = _optimise_window(
            strategy_class, train_df, param_grid,
            optimize_by, backtester_kwargs, n_top_params, intrabar_df=train_intrabar_df
        )

        # ── Out-of-sample test ────────────────────────────────────────────────
        bt  = Backtester(**backtester_kwargs)
        strat = strategy_class(**best_params)
        test_results = bt.run(strat, test_df.copy(), intrabar_df=test_intrabar_df)

        print(f"    IS  Sharpe: {train_results.sharpe_ratio:.2f} | "
              f"OOS Sharpe: {test_results.sharpe_ratio:.2f} | "
              f"OOS P&L: ${test_results.net_profit:+,.0f}")

        windows.append(WFOWindow(
            window_id     = window_id,
            train_from    = train_start,
            train_to      = train_end,
            test_from     = test_start,
            test_to       = test_end,
            best_params   = best_params,
            train_results = train_results,
            test_results  = test_results,
        ))
        all_oos_trades.extend(test_results.trades)

        test_start += step
        window_id  += 1

    return WFOResults(
        windows         = windows,
        combined_trades = all_oos_trades,
        params_used     = [w.best_params for w in windows],
    )


def _param_grid_size(param_grid: dict) -> int:
    size = 1
    for values in param_grid.values():
        size *= max(1, len(values))
    return size


def _iter_param_grid(param_grid: dict) -> Iterator[dict]:
    """Yield parameter combinations from a grid without materializing all of them."""
    import itertools
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _optimise_window(
    strategy_class, df: pd.DataFrame,
    param_grid: dict[str, list],
    optimize_by: str,
    backtester_kwargs: dict,
    n_top: int,
    intrabar_df: pd.DataFrame | None = None,
) -> tuple[dict, BacktestResults]:
    """Find the best parameter set on the training data."""
    results_list: list[tuple[float, dict, BacktestResults]] = []
    fallback_params = None

    for params in _iter_param_grid(param_grid):
        if fallback_params is None:
            fallback_params = dict(params)
        try:
            bt    = Backtester(**backtester_kwargs)
            strat = strategy_class(**params)
            r     = bt.run(strat, df.copy(), intrabar_df=intrabar_df)
            score = getattr(r, optimize_by, r.sharpe_ratio)
            results_list.append((score, params, r))
            results_list.sort(key=lambda x: x[0], reverse=True)
            if len(results_list) > max(1, int(n_top)):
                results_list = results_list[: max(1, int(n_top))]
        except Exception:
            continue

    if not results_list:
        return fallback_params or {}, None

    if n_top == 1:
        return results_list[0][1], results_list[0][2]

    # Average top N parameter sets
    top_n  = results_list[:n_top]
    keys   = list(top_n[0][1].keys())
    avg_params = {}
    for k in keys:
        vals = [p[1][k] for p in top_n]
        if all(isinstance(v, (int, float)) for v in vals):
            avg_params[k] = float(np.mean(vals))
        else:
            avg_params[k] = top_n[0][1][k]

    bt    = Backtester(**backtester_kwargs)
    strat = strategy_class(**avg_params)
    r     = bt.run(strat, df.copy(), intrabar_df=intrabar_df)
    return avg_params, r
