"""
engine/results.py

Converts a list of ClosedTrade objects into a full performance report
matching MT5's strategy tester output format.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from engine.position import ClosedTrade, CloseReason
from engine.policy import DecisionRecord


@dataclass
class BacktestResults:
    trades:        List[ClosedTrade]
    decision_records: List[DecisionRecord]
    initial_capital: float
    symbol:        str
    timeframe:     str
    date_from:     pd.Timestamp
    date_to:       pd.Timestamp
    params:        dict

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def profits(self) -> np.ndarray:
        return np.array([t.net_profit for t in self.trades])

    @property
    def equity_curve(self) -> pd.Series:
        if not self.trades:
            return pd.Series([self.initial_capital], dtype=float)
        times  = [t.closed_time for t in self.trades]
        equity = self.initial_capital + np.cumsum(self.profits)
        return pd.Series(equity, index=times)

    @property
    def net_profit(self) -> float:
        return float(self.profits.sum()) if len(self.profits) else 0.0

    @property
    def gross_profit(self) -> float:
        p = self.profits
        return float(p[p > 0].sum()) if len(p) else 0.0

    @property
    def gross_loss(self) -> float:
        p = self.profits
        return float(p[p < 0].sum()) if len(p) else 0.0

    @property
    def profit_factor(self) -> float:
        gl = abs(self.gross_loss)
        return self.gross_profit / gl if gl > 0 else float('inf')

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.mean(self.profits > 0))

    @property
    def avg_profit(self) -> float:
        wins = self.profits[self.profits > 0]
        return float(wins.mean()) if len(wins) else 0.0

    @property
    def avg_loss(self) -> float:
        losses = self.profits[self.profits < 0]
        return float(losses.mean()) if len(losses) else 0.0

    @property
    def expected_payoff(self) -> float:
        return float(self.profits.mean()) if len(self.profits) else 0.0

    @property
    def max_drawdown(self) -> tuple[float, float]:
        """Returns (absolute drawdown $, relative drawdown %)."""
        eq   = self.initial_capital + np.cumsum(
            np.concatenate([[0], self.profits])
        )
        peak = np.maximum.accumulate(eq)
        dd   = peak - eq
        abs_dd = float(dd.max())
        rel_dd = float((dd / peak).max() * 100)
        return abs_dd, rel_dd

    @property
    def balance_drawdown_maximal(self) -> tuple[float, float]:
        """Closed-trade balance drawdown: ($, %)."""
        return self.max_drawdown

    @property
    def equity_drawdown_maximal(self) -> tuple[float, float]:
        """
        Equity drawdown: ($, %).
        Currently based on the same realized-trade equity series used by the engine summary.
        """
        return self.max_drawdown

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe using monthly P&L."""
        monthly = self.monthly_stats()
        if monthly.empty or len(monthly) < 2:
            return 0.0
        p   = monthly["profit"].values
        std = p.std()
        if std < 1e-8:
            return 0.0
        return float(p.mean() / std * np.sqrt(12))

    @property
    def recovery_factor(self) -> float:
        abs_dd, _ = self.max_drawdown
        return self.net_profit / abs_dd if abs_dd > 0 else 0.0

    @property
    def max_consecutive_wins(self) -> int:
        return self._max_consecutive(win=True)

    @property
    def max_consecutive_losses(self) -> int:
        return self._max_consecutive(win=False)

    def _max_consecutive(self, win: bool) -> int:
        best = cur = 0
        for p in self.profits:
            if (p > 0) == win:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def monthly_stats(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame({
            "time":   [t.closed_time for t in self.trades],
            "profit": self.profits,
        }).set_index("time")
        monthly = df.groupby(pd.Grouper(freq="ME"))["profit"].agg(
            profit="sum",
            trades="count",
            win_rate=lambda x: (x > 0).mean(),
        ).reset_index()
        monthly["year_month"] = monthly["time"].dt.strftime("%Y-%m")
        monthly["sharpe"] = monthly["profit"].apply(
            lambda p: 0.0  # per-month Sharpe needs std — computed across months
        )
        # Proper monthly Sharpe
        profits = monthly["profit"].values
        std     = profits.std() + 1e-8
        monthly["sharpe"] = profits / std * np.sqrt(12)
        return monthly.set_index("year_month")

    def summary(self) -> dict:
        abs_dd, rel_dd = self.max_drawdown
        bal_dd_abs, bal_dd_pct = self.balance_drawdown_maximal
        eq_dd_abs, eq_dd_pct = self.equity_drawdown_maximal
        return {
            "symbol":              self.symbol,
            "timeframe":           self.timeframe,
            "date_from":           str(self.date_from.date()),
            "date_to":             str(self.date_to.date()),
            "initial_capital":     self.initial_capital,
            "net_profit":          round(self.net_profit, 2),
            "gross_profit":        round(self.gross_profit, 2),
            "gross_loss":          round(self.gross_loss, 2),
            "profit_factor":       round(self.profit_factor, 4),
            "sharpe_ratio":        round(self.sharpe_ratio, 4),
            "max_drawdown_abs":    round(abs_dd, 2),
            "max_drawdown_pct":    round(rel_dd, 2),
            "balance_dd_abs":      round(bal_dd_abs, 2),
            "balance_dd_pct":      round(bal_dd_pct, 2),
            "equity_dd_abs":       round(eq_dd_abs, 2),
            "equity_dd_pct":       round(eq_dd_pct, 2),
            "recovery_factor":     round(self.recovery_factor, 4),
            "win_rate":            round(self.win_rate, 4),
            "n_trades":            self.n_trades,
            "avg_profit":          round(self.avg_profit, 2),
            "avg_loss":            round(self.avg_loss, 2),
            "expected_payoff":     round(self.expected_payoff, 2),
            "max_consec_wins":     self.max_consecutive_wins,
            "max_consec_losses":   self.max_consecutive_losses,
            "n_decisions":         len(self.decision_records),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  {s['symbol']} {s['timeframe']} | "
              f"{s['date_from']} — {s['date_to']}")
        print(f"{'='*50}")
        print(f"  Net Profit:       ${s['net_profit']:>10,.2f}")
        print(f"  Gross Profit:     ${s['gross_profit']:>10,.2f}")
        print(f"  Gross Loss:       ${s['gross_loss']:>10,.2f}")
        print(f"  Profit Factor:    {s['profit_factor']:>10.4f}")
        print(f"  Sharpe Ratio:     {s['sharpe_ratio']:>10.4f}")
        print(f"  Max Drawdown:     {s['max_drawdown_pct']:>9.2f}%")
        print(f"  Recovery Factor:  {s['recovery_factor']:>10.4f}")
        print(f"  Win Rate:         {s['win_rate']*100:>9.2f}%")
        print(f"  Total Trades:     {s['n_trades']:>10}")
        print(f"  Avg Win:          ${s['avg_profit']:>10,.2f}")
        print(f"  Avg Loss:         ${s['avg_loss']:>10,.2f}")
        print(f"  Max Consec Loss:  {s['max_consec_losses']:>10}")
        print(f"{'='*50}\n")
