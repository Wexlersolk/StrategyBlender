"""
engine/backtester.py

OHLC bar-by-bar backtesting engine with optional intra-bar simulation.

intrabar_steps > 1 enables Brownian bridge simulation:
  For each bar, generate N synthetic sub-bar price steps that:
    - Start at bar open
    - End at bar close
    - Reach the bar high and low somewhere in between
  Trailing stops are updated at each sub-bar step, giving much
  more accurate results — especially for trailing stop strategies.

  This replicates MT5's "1 Minute OHLC" mode behaviour without
  needing actual M1 data.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional
from engine.base_strategy import BaseStrategy, BarContext
from engine.position import (
    Direction, OrderType, CloseReason,
    PendingOrder, Position, ClosedTrade
)
from engine.results import BacktestResults


# ── Brownian bridge price path generator ──────────────────────────────────────

def _generate_intrabar_path(
    open_:  float,
    high:   float,
    low:    float,
    close:  float,
    steps:  int,
    rng:    np.random.Generator,  # kept for API compatibility, not used
) -> np.ndarray:
    """
    Generate a deterministic intra-bar price path.

    Uses MT5's own logic for its "1 Minute OHLC" mode:
      Bullish bar (close >= open):  open → low → high → close
      Bearish bar (close <  open):  open → high → low → close

    This reflects real price tendency:
      - In a bullish bar, price dips before rising
      - In a bearish bar, price rises before falling

    Interpolates linearly between the 4 waypoints using `steps` total points.
    """
    if steps <= 1:
        return np.array([close])

    bullish = close >= open_

    if bullish:
        waypoints = [open_, low, high, close]
    else:
        waypoints = [open_, high, low, close]

    # Distribute steps across 3 segments
    # Segment sizes: roughly proportional to price move magnitude
    seg1 = abs(waypoints[1] - waypoints[0])
    seg2 = abs(waypoints[2] - waypoints[1])
    seg3 = abs(waypoints[3] - waypoints[2])
    total = seg1 + seg2 + seg3 + 1e-10

    s1 = max(1, int(steps * seg1 / total))
    s2 = max(1, int(steps * seg2 / total))
    s3 = max(1, steps - s1 - s2)

    path = np.concatenate([
        np.linspace(waypoints[0], waypoints[1], s1, endpoint=False),
        np.linspace(waypoints[1], waypoints[2], s2, endpoint=False),
        np.linspace(waypoints[2], waypoints[3], s3),
    ])

    return path


class Backtester:

    def __init__(
        self,
        initial_capital:    float = 100_000.0,
        commission_per_lot: float = 0.0,
        spread_pips:        float = 0.0,
        tick_size:          float = 1.0,
        lot_value:          float = 1.0,
        intrabar_steps:     int   = 1,    # 1 = standard OHLC, 60 = M1-like
        seed:               int   = 42,
        verbose:            bool  = False,
    ):
        self.initial_capital    = initial_capital
        self.commission_per_lot = commission_per_lot
        self.spread_pips        = spread_pips
        self.tick_size          = tick_size
        self.lot_value          = lot_value
        self.intrabar_steps     = intrabar_steps
        self.verbose            = verbose
        self._rng               = np.random.default_rng(seed)

        self._balance:  float = initial_capital
        self._equity:   float = initial_capital
        self._order_id: int   = 0
        self._open_positions:   List[Position]      = []
        self._pending_orders:   List[PendingOrder]  = []
        self._closed_trades:    List[ClosedTrade]   = []
        self._current_strategy: Optional[BaseStrategy] = None

    def run(
        self,
        strategy:  BaseStrategy,
        df:        pd.DataFrame,
        date_from: Optional[str] = None,
        date_to:   Optional[str] = None,
    ) -> BacktestResults:

        # Reset state
        self._balance  = self.initial_capital
        self._equity   = self.initial_capital
        self._order_id = 0
        self._open_positions  = []
        self._pending_orders  = []
        self._closed_trades   = []
        self._current_strategy = strategy

        if date_from:
            df = df[df.index >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df.index <= pd.Timestamp(date_to)]

        if df.empty:
            raise ValueError("No data in the specified date range.")

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if self.verbose:
            print(f"Computing indicators for {len(df)} bars...")
        df = strategy.compute_indicators(df)
        df = df.ffill().fillna(0)

        strategy.on_start(df)

        warmup = 60
        use_intrabar = self.intrabar_steps > 1

        for i in range(1, len(df)):
            t     = df.index[i]
            bar   = df.iloc[i]
            open_ = float(bar["open"])
            high  = float(bar["high"])
            low   = float(bar["low"])
            close = float(bar["close"])

            # 1. Check pending order triggers
            self._check_pending_triggers(df, i)

            # 2. Expire unfilled orders
            self._expire_pending(i)

            if use_intrabar and self._open_positions:
                # ── Intra-bar simulation ──────────────────────────────────────
                # Generate synthetic price path for this bar
                path = _generate_intrabar_path(
                    open_, high, low, close,
                    self.intrabar_steps, self._rng
                )

                # Compute path high/low for each sub-step
                # We process in pairs: [price_prev, price_curr]
                # to detect SL/TP hits and update trailing stops
                for step in range(len(path)):
                    step_price = path[step]
                    # Use a tiny range around each step for SL/TP check
                    if step == 0:
                        step_high = max(open_, step_price)
                        step_low  = min(open_, step_price)
                    else:
                        step_high = max(path[step-1], step_price)
                        step_low  = min(path[step-1], step_price)

                    # Check SL/TP at this sub-step
                    self._check_sl_tp_prices(step_high, step_low, i, t)

                    # Update trailing stops at this sub-step
                    self._update_trailing_prices(step_high, step_low)

                    # Stop processing if all positions closed
                    if not self._open_positions:
                        break

            else:
                # ── Standard OHLC mode ────────────────────────────────────────
                self._check_sl_tp(df, i)
                self._update_trailing(df, i)

            # 3. Update equity
            self._update_equity(df, i)

            # 4. Call strategy
            if i >= warmup:
                ctx = BarContext(df, i, self)
                strategy.on_bar(ctx)

        # Close remaining positions at last bar
        last_bar   = len(df) - 1
        last_time  = df.index[last_bar]
        last_close = float(df["close"].iloc[last_bar])
        self._close_all(last_bar, last_time, last_close,
                        reason=CloseReason.END_OF_TEST)

        results = BacktestResults(
            trades          = self._closed_trades,
            initial_capital = self.initial_capital,
            symbol          = getattr(strategy, "symbol", ""),
            timeframe       = getattr(strategy, "timeframe", ""),
            date_from       = df.index[0],
            date_to         = df.index[-1],
            params          = strategy.params.copy(),
        )

        strategy.on_end(results)
        return results

    # ── Order management ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def _place_pending(
        self, order_type: OrderType, direction: Direction,
        price: float, sl: float, tp: float, lots: float,
        bar_idx: int, expiry_bars: int, comment: str
    ) -> int:
        oid = self._next_id()
        self._pending_orders.append(PendingOrder(
            id=oid, order_type=order_type, direction=direction,
            entry_price=price, stop_loss=sl, take_profit=tp,
            lots=lots, opened_bar=bar_idx,
            expiry_bars=expiry_bars, comment=comment
        ))
        return oid

    def _open_position(
        self, direction: Direction, price: float,
        sl: float, tp: float, lots: float,
        bar_idx: int, time: pd.Timestamp, comment: str = "",
        trailing_stop: float = 0.0, trail_activation: float = 0.0,
    ) -> int:
        pid = self._next_id()
        if direction == Direction.LONG:
            price += self.spread_pips
        pos = Position(
            id=pid, direction=direction, entry_price=price,
            stop_loss=sl, take_profit=tp, lots=lots,
            opened_bar=bar_idx, opened_time=time,
            trailing_stop=trailing_stop,
            trail_activation=trail_activation,
            comment=comment
        )
        self._open_positions.append(pos)
        if self.verbose:
            print(f"  [{time}] OPEN {direction.value} @ {price:.2f} "
                  f"SL:{sl:.2f} TP:{tp:.2f} Trail:{trailing_stop:.2f}")
        return pid

    def _close_position(
        self, pos: Position, exit_price: float,
        bar_idx: int, time: pd.Timestamp,
        reason: CloseReason
    ):
        gross = self._calc_profit(pos, exit_price)
        comm  = self.commission_per_lot * pos.lots * 2
        trade = ClosedTrade(
            id=pos.id, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
            lots=pos.lots, opened_bar=pos.opened_bar, closed_bar=bar_idx,
            opened_time=pos.opened_time, closed_time=time,
            close_reason=reason,
            gross_profit=gross, commission=comm, swap=0.0,
            comment=pos.comment
        )
        self._closed_trades.append(trade)
        self._balance += trade.net_profit
        self._open_positions.remove(pos)

        if self.verbose:
            print(f"  [{time}] CLOSE {pos.direction.value} "
                  f"@ {exit_price:.2f} | P&L: {trade.net_profit:+.2f} "
                  f"| {reason.value}")

    def _close_all(
        self, bar_idx: int, time: pd.Timestamp,
        price: float, reason: CloseReason = CloseReason.SIGNAL
    ):
        for pos in list(self._open_positions):
            self._close_position(pos, price, bar_idx, time, reason)

    def _calc_profit(self, pos: Position, exit_price: float) -> float:
        if pos.direction == Direction.LONG:
            pips = exit_price - pos.entry_price
        else:
            pips = pos.entry_price - exit_price
        return pips * pos.lots * self.lot_value

    # ── OHLC simulation ───────────────────────────────────────────────────────

    def _check_pending_triggers(self, df: pd.DataFrame, i: int):
        bar   = df.iloc[i]
        high  = float(bar["high"])
        low   = float(bar["low"])
        open_ = float(bar["open"])
        t     = df.index[i]

        for order in list(self._pending_orders):
            triggered  = False
            fill_price = order.entry_price

            if order.order_type == OrderType.BUY_STOP:
                if high >= order.entry_price:
                    fill_price = max(order.entry_price, open_)
                    triggered  = True
            elif order.order_type == OrderType.SELL_STOP:
                if low <= order.entry_price:
                    fill_price = min(order.entry_price, open_)
                    triggered  = True
            elif order.order_type == OrderType.BUY_LIMIT:
                if low <= order.entry_price:
                    fill_price = order.entry_price
                    triggered  = True
            elif order.order_type == OrderType.SELL_LIMIT:
                if high >= order.entry_price:
                    fill_price = order.entry_price
                    triggered  = True

            if triggered:
                self._pending_orders.remove(order)
                trail_dist = getattr(self._current_strategy,
                                     '_next_trail_dist', 0.0)
                trail_act  = getattr(self._current_strategy,
                                     '_next_trail_activation', 0.0)
                self._open_position(
                    order.direction, fill_price,
                    order.stop_loss, order.take_profit,
                    order.lots, i, t, order.comment,
                    trailing_stop=trail_dist,
                    trail_activation=trail_act,
                )

    def _check_sl_tp(self, df: pd.DataFrame, i: int):
        """Standard OHLC SL/TP check."""
        bar  = df.iloc[i]
        high = float(bar["high"])
        low  = float(bar["low"])
        t    = df.index[i]
        self._check_sl_tp_prices(high, low, i, t)

    def _check_sl_tp_prices(self, high: float, low: float,
                             bar_idx: int, time: pd.Timestamp):
        """Check SL/TP against given high/low prices."""
        for pos in list(self._open_positions):
            sl_hit = False
            tp_hit = False

            if pos.direction == Direction.LONG:
                if pos.stop_loss > 0 and low <= pos.stop_loss:
                    sl_hit = True
                if pos.take_profit > 0 and high >= pos.take_profit:
                    tp_hit = True
            else:
                if pos.stop_loss > 0 and high >= pos.stop_loss:
                    sl_hit = True
                if pos.take_profit > 0 and low <= pos.take_profit:
                    tp_hit = True

            # Conservative: SL wins if both hit
            if sl_hit:
                self._close_position(
                    pos, pos.stop_loss, bar_idx, time, CloseReason.STOP_LOSS
                )
            elif tp_hit:
                self._close_position(
                    pos, pos.take_profit, bar_idx, time, CloseReason.TAKE_PROFIT
                )

    def _update_trailing(self, df: pd.DataFrame, i: int):
        """Standard OHLC trailing stop update."""
        bar  = df.iloc[i]
        high = float(bar["high"])
        low  = float(bar["low"])
        self._update_trailing_prices(high, low)

    def _update_trailing_prices(self, high: float, low: float):
        """Update trailing stops given high/low prices."""
        for pos in self._open_positions:
            if pos.trailing_stop <= 0:
                continue

            if pos.direction == Direction.LONG:
                profit_pts = high - pos.entry_price
                if pos.trail_activation > 0 and profit_pts < pos.trail_activation:
                    continue
                new_sl = high - pos.trailing_stop
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
            else:
                profit_pts = pos.entry_price - low
                if pos.trail_activation > 0 and profit_pts < pos.trail_activation:
                    continue
                new_sl = low + pos.trailing_stop
                if pos.stop_loss == 0 or new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl

    def _expire_pending(self, current_bar: int):
        to_remove = [
            o for o in self._pending_orders
            if o.expiry_bars > 0 and current_bar - o.opened_bar >= o.expiry_bars
        ]
        for order in to_remove:
            self._pending_orders.remove(order)

    def _update_equity(self, df: pd.DataFrame, i: int):
        close    = float(df["close"].iloc[i])
        floating = sum(
            self._calc_profit(pos, close)
            for pos in self._open_positions
        )
        self._equity = self._balance + floating
