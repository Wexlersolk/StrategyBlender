"""
Compare baseline strategy vs AI-scheduled strategy using Python engine.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from engine.data_loader import load_bars
from engine.backtester  import Backtester
from strategies.strategy_1_3_45 import Strategy_1_3_45

df = load_bars('HK50.cash', 'H1')

# ── Baseline (fixed 70 lots) ──────────────────────────────────────────────────
bt_base  = Backtester(initial_capital=100000, lot_value=0.1285)
strat    = Strategy_1_3_45()
baseline = bt_base.run(strat, df.copy(),
                       date_from="2020-06-01", date_to="2026-01-01")
print("BASELINE:")
baseline.print_summary()

# ── AI-scheduled (lots from CSV) ──────────────────────────────────────────────
import csv
from pathlib import Path
schedule_path = Path.home() / ".wine/drive_c/users/wexlersolk/AppData/Roaming/MetaQuotes/Terminal/Common/Files/ml_params_schedule.csv"

schedule = {}
with open(schedule_path) as f:
    for row in csv.DictReader(f):
        ym = row['date'][:7].replace('.', '-')
        schedule[ym] = float(row['mmLots'])

class AIStrategy(Strategy_1_3_45):
    def on_bar(self, ctx):
        ym = ctx.time.strftime('%Y-%m')
        if ym in schedule:
            self.params['mmLots'] = schedule[ym]
        super().on_bar(ctx)

bt_ai  = Backtester(initial_capital=100000, lot_value=0.1285)
ai_str = AIStrategy()
ai_res = bt_ai.run(ai_str, df.copy(),
                   date_from="2020-06-01", date_to="2026-01-01")
print("\nAI-SCHEDULED:")
ai_res.print_summary()

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n=== COMPARISON ===")
print(f"{'Metric':<22} {'Baseline':>12} {'AI':>12} {'Delta':>10}")
print("-" * 58)
metrics = [
    ("Net Profit",      f"${baseline.net_profit:,.0f}",        f"${ai_res.net_profit:,.0f}",        f"{(ai_res.net_profit-baseline.net_profit)/abs(baseline.net_profit+1)*100:+.1f}%"),
    ("Sharpe Ratio",    f"{baseline.sharpe_ratio:.2f}",         f"{ai_res.sharpe_ratio:.2f}",         f"{ai_res.sharpe_ratio-baseline.sharpe_ratio:+.2f}"),
    ("Max Drawdown",    f"{baseline.max_drawdown[1]:.2f}%",     f"{ai_res.max_drawdown[1]:.2f}%",     f"{ai_res.max_drawdown[1]-baseline.max_drawdown[1]:+.2f}%"),
    ("Profit Factor",   f"{baseline.profit_factor:.2f}",        f"{ai_res.profit_factor:.2f}",        f"{ai_res.profit_factor-baseline.profit_factor:+.2f}"),
    ("Win Rate",        f"{baseline.win_rate*100:.1f}%",        f"{ai_res.win_rate*100:.1f}%",        f"{(ai_res.win_rate-baseline.win_rate)*100:+.1f}pp"),
]
for name, b, a, d in metrics:
    print(f"{name:<22} {b:>12} {a:>12} {d:>10}")
