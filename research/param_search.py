"""
research/param_search.py

Parameter search — grid search and random search.

Runs a strategy across many parameter combinations and returns
a ranked results table. Much faster than MT5 optimizer because:
  - No UI overhead
  - Parallelised across CPU cores
  - Results available immediately in Python

Usage:
    from research.param_search import grid_search
    from strategies.strategy_1_3_45 import Strategy_1_3_45

    results = grid_search(
        Strategy_1_3_45,
        df,
        param_grid={
            "StopLossCoef1":     [1.5, 2.0, 2.5, 3.0],
            "ProfitTargetCoef1": [1.5, 2.0, 2.5, 3.0],
            "mmLots":            [45, 60, 70],
        },
        optimize_by="sharpe_ratio",
        n_jobs=4,
    )
    print(results.head(10))
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import itertools
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from engine.backtester import Backtester
from engine.results import BacktestResults


def _run_single(args):
    """Worker function for parallel execution."""
    strategy_class, params, df, backtester_kwargs = args
    try:
        bt    = Backtester(**backtester_kwargs)
        strat = strategy_class(**params)
        r     = bt.run(strat, df.copy())
        s     = r.summary()
        s.update(params)
        return s
    except Exception as e:
        return None


def grid_search(
    strategy_class,
    df: pd.DataFrame,
    param_grid: dict,
    optimize_by:       str   = "sharpe_ratio",
    n_jobs:            int   = 1,
    backtester_kwargs: dict  = None,
    date_from:         str   = None,
    date_to:           str   = None,
    min_trades:        int   = 10,
) -> pd.DataFrame:
    """
    Grid search over all parameter combinations.

    Returns a DataFrame sorted by optimize_by metric,
    with all performance metrics and parameter values.
    """
    if backtester_kwargs is None:
        backtester_kwargs = {}

    if date_from or date_to:
        if date_from:
            df = df[df.index >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df.index <= pd.Timestamp(date_to)]

    # Build all combinations
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = [dict(zip(keys, c)) for c in itertools.product(*values)]

    print(f"Grid search: {len(combos)} combinations, {n_jobs} workers")

    args_list = [(strategy_class, params, df, backtester_kwargs)
                 for params in combos]

    results = []

    if n_jobs == 1:
        for i, args in enumerate(args_list):
            r = _run_single(args)
            if r and r.get("n_trades", 0) >= min_trades:
                results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(combos)} done...")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_single, args): i
                       for i, args in enumerate(args_list)}
            done = 0
            for future in as_completed(futures):
                r = future.result()
                if r and r.get("n_trades", 0) >= min_trades:
                    results.append(r)
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(combos)} done...")

    if not results:
        print("No valid results found.")
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out.sort_values(optimize_by, ascending=False).reset_index(drop=True)

    print(f"\nGrid search complete: {len(out)} valid results")
    print(f"Best {optimize_by}: {out[optimize_by].iloc[0]:.4f}")
    print(f"Best params: { {k: out[k].iloc[0] for k in keys} }")

    return out


def random_search(
    strategy_class,
    df: pd.DataFrame,
    param_ranges: dict,
    n_trials:          int   = 200,
    optimize_by:       str   = "sharpe_ratio",
    n_jobs:            int   = 1,
    backtester_kwargs: dict  = None,
    seed:              int   = 42,
    min_trades:        int   = 10,
) -> pd.DataFrame:
    """
    Random search over parameter space.

    param_ranges: dict of param_name → (min, max) or list of values
    More efficient than grid search for large parameter spaces.
    """
    rng = np.random.default_rng(seed)

    def sample_params() -> dict:
        params = {}
        for k, v in param_ranges.items():
            if isinstance(v, (list, tuple)) and len(v) == 2 and \
               isinstance(v[0], (int, float)):
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    params[k] = int(rng.integers(lo, hi + 1))
                else:
                    params[k] = float(rng.uniform(lo, hi))
            elif isinstance(v, list):
                params[k] = rng.choice(v)
            else:
                params[k] = v
        return params

    combos = [sample_params() for _ in range(n_trials)]
    keys   = list(param_ranges.keys())

    if backtester_kwargs is None:
        backtester_kwargs = {}

    print(f"Random search: {n_trials} trials, {n_jobs} workers")

    args_list = [(strategy_class, params, df, backtester_kwargs)
                 for params in combos]

    results = []
    if n_jobs == 1:
        for i, args in enumerate(args_list):
            r = _run_single(args)
            if r and r.get("n_trades", 0) >= min_trades:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_single, args): i
                       for i, args in enumerate(args_list)}
            for future in as_completed(futures):
                r = future.result()
                if r and r.get("n_trades", 0) >= min_trades:
                    results.append(r)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values(
        optimize_by, ascending=False
    ).reset_index(drop=True)

    print(f"Random search complete: {len(out)} valid results")
    return out
