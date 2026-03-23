"""
scripts/run_wfo.py

Walk-Forward Optimization for Strategy 1.3.45.
Run with: python scripts/run_wfo.py
"""

import sys
sys.path.insert(0, '.')

from engine.data_loader  import load_bars
from engine.backtester   import Backtester
from research.walk_forward import run_wfo
from strategies.strategy_1_3_45 import Strategy_1_3_45


def main():
    print("Loading HK50.cash H1 data...")
    df = load_bars('HK50.cash', 'H1')
    print(f"Loaded {len(df)} bars\n")

    results = run_wfo(
        strategy_class = Strategy_1_3_45,
        df             = df,
        param_grid     = {
            "StopLossCoef1":     [1.5, 2.0, 2.5, 3.0],
            "ProfitTargetCoef1": [1.5, 2.0, 2.5, 3.0],
            "TrailingActCef1":   [1.0, 1.4, 1.8],
        },
        train_months   = 24,
        test_months    = 6,
        optimize_by    = "sharpe_ratio",
        backtester_kwargs = {
            "initial_capital":    100_000,
            "lot_value":          0.1285,
        },
    )

    results.print_summary()

    # Show which params were chosen per window
    print("\nParameters chosen per window:")
    for w in results.windows:
        print(f"  W{w.window_id}: SL={w.best_params.get('StopLossCoef1'):.1f} "
              f"PT={w.best_params.get('ProfitTargetCoef1'):.1f} "
              f"Trail={w.best_params.get('TrailingActCef1'):.1f} "
              f"→ OOS Sharpe: {w.test_results.sharpe_ratio:.2f} "
              f"P&L: ${w.test_results.net_profit:+,.0f}")


if __name__ == '__main__':
    main()
