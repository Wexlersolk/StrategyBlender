"""
scripts/run_backtest.py

Quick demo of the Python backtesting engine.
Run this to verify everything works:

    cd ~/ZU/Practice/StrategyBlender
    source venv311/bin/activate
    python scripts/run_backtest.py
"""

import sys
sys.path.insert(0, '.')

from engine.backtester   import Backtester
from engine.data_loader  import load_bars
from strategies.strategy_1_3_45 import Strategy_1_3_45


def main():
    print("Loading HK50.cash data...")
    df = load_bars("HK50.cash", timeframe="H1")
    print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # ── Basic backtest ────────────────────────────────────────────────────────
    print("\nRunning backtest...")
    bt     = Backtester(
        initial_capital    = 100_000,
        commission_per_lot = 0.0,
        lot_value          = 0.1285,   # HK50 approx lot value in USD
        verbose            = False,
    )
    strat   = Strategy_1_3_45()
    results = bt.run(strat, df, date_from="2020-06-01", date_to="2026-01-01")
    results.print_summary()

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    if results.n_trades >= 10:
        print("Running Monte Carlo (1000 simulations)...")
        from research.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(results, n_simulations=1000)
        mc.print_summary()

    # ── Parameter search (small example) ─────────────────────────────────────
    print("\nRunning mini parameter search (9 combinations)...")
    from research.param_search import grid_search
    search_results = grid_search(
        Strategy_1_3_45,
        df,
        param_grid={
            "StopLossCoef1":     [2.0, 2.5, 3.0],
            "ProfitTargetCoef1": [2.0, 2.5, 3.0],
        },
        optimize_by = "sharpe_ratio",
        date_from   = "2020-06-01",
        date_to     = "2026-01-01",
    )
    if not search_results.empty:
        cols = ["sharpe_ratio", "net_profit", "max_drawdown_pct",
                "StopLossCoef1", "ProfitTargetCoef1"]
        print(search_results[cols].head(5).to_string(index=True))


if __name__ == "__main__":
    main()
