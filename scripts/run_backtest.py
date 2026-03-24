"""
scripts/run_backtest.py

Demo of the Python backtesting engine with intra-bar simulation.
Run with: python scripts/run_backtest.py
"""

import sys
sys.path.insert(0, '.')

from engine.backtester   import Backtester
from engine.data_loader  import load_bars
from strategies.strategy_1_3_45 import Strategy_1_3_45


def main():
    print("Loading HK50.cash H1 data...")
    df = load_bars('HK50.cash', 'H1')
    print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # ── Standard OHLC (fast) ──────────────────────────────────────────────────
    print("\n--- Standard OHLC (1 step per bar) ---")
    bt_std  = Backtester(initial_capital=100_000, lot_value=0.1285,
                          intrabar_steps=1)
    r_std   = bt_std.run(Strategy_1_3_45(), df.copy(),
                          date_from="2020-06-01", date_to="2025-12-31")
    r_std.print_summary()

    # ── Intra-bar simulation (more accurate) ──────────────────────────────────
    print("\n--- Intra-bar simulation (60 steps per bar) ---")
    bt_ib   = Backtester(initial_capital=100_000, lot_value=0.1285,
                          intrabar_steps=60, seed=42)
    r_ib    = bt_ib.run(Strategy_1_3_45(), df.copy(),
                         date_from="2020-06-01", date_to="2025-12-31")
    r_ib.print_summary()

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n=== OHLC vs Intra-bar ===")
    print(f"{'Metric':<22} {'OHLC':>10} {'Intra-bar':>10} {'Delta':>10}")
    print("-" * 54)
    rows = [
        ("Net Profit",    f"${r_std.net_profit:,.0f}",
                          f"${r_ib.net_profit:,.0f}",
                          f"{r_ib.net_profit - r_std.net_profit:+,.0f}"),
        ("Sharpe Ratio",  f"{r_std.sharpe_ratio:.3f}",
                          f"{r_ib.sharpe_ratio:.3f}",
                          f"{r_ib.sharpe_ratio - r_std.sharpe_ratio:+.3f}"),
        ("Win Rate",      f"{r_std.win_rate*100:.1f}%",
                          f"{r_ib.win_rate*100:.1f}%",
                          f"{(r_ib.win_rate - r_std.win_rate)*100:+.1f}pp"),
        ("Max Drawdown",  f"{r_std.max_drawdown[1]:.2f}%",
                          f"{r_ib.max_drawdown[1]:.2f}%",
                          f"{r_ib.max_drawdown[1] - r_std.max_drawdown[1]:+.2f}%"),
        ("Total Trades",  str(r_std.n_trades),
                          str(r_ib.n_trades), ""),
    ]
    for name, s, ib, d in rows:
        print(f"{name:<22} {s:>10} {ib:>10} {d:>10}")

    # ── Monte Carlo on intra-bar result ───────────────────────────────────────
    if r_ib.n_trades >= 10:
        print("\nRunning Monte Carlo on intra-bar results...")
        from research.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(r_ib, n_simulations=1000)
        mc.print_summary()


if __name__ == '__main__':
    main()
