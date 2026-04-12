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
from engine.policy import (
    DecisionRecord,
    NullOverlayPolicy,
    OverlayPolicy,
    OverlayDecision,
    TradeIntent,
)
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
        slippage_pips:      float = 0.0,
        tick_size:          float = 1.0,
        tick_value:         float = 0.0,
        contract_size:      float = 0.0,
        swap_per_lot_long:  float = 0.0,
        swap_per_lot_short: float = 0.0,
        session_timezone_offset_hours: float = 0.0,
        use_bar_spread:     bool  = False,
        lot_value:          float = 1.0,
        intrabar_steps:     int   = 1,    # 1 = standard OHLC, 60 = M1-like
        seed:               int   = 42,
        verbose:            bool  = False,
        overlay_policy: Optional[OverlayPolicy] = None,
    ):
        self.initial_capital    = initial_capital
        self.commission_per_lot = commission_per_lot
        self.spread_pips        = spread_pips
        self.slippage_pips      = slippage_pips
        self.tick_size          = tick_size
        self.tick_value         = tick_value
        self.contract_size      = contract_size
        self.swap_per_lot_long  = swap_per_lot_long
        self.swap_per_lot_short = swap_per_lot_short
        self.session_timezone_offset_hours = session_timezone_offset_hours
        self.use_bar_spread     = use_bar_spread
        self.lot_value          = lot_value
        self.intrabar_steps     = intrabar_steps
        self.verbose            = verbose
        self._rng               = np.random.default_rng(seed)
        self.overlay_policy     = overlay_policy or NullOverlayPolicy()

        self._balance:  float = initial_capital
        self._equity:   float = initial_capital
        self._order_id: int   = 0
        self._decision_id: int = 0
        self._open_positions:   List[Position]      = []
        self._pending_orders:   List[PendingOrder]  = []
        self._closed_trades:    List[ClosedTrade]   = []
        self._decision_records: List[DecisionRecord] = []
        self._current_strategy: Optional[BaseStrategy] = None
        self._price_df: Optional[pd.DataFrame] = None
        self._debug_events: list[dict[str, object]] = []
        self._balance_curve_points: list[float] = []
        self._equity_curve_points: list[float] = []
        self._equity_worst_curve_points: list[float] = []
        self._curve_times: list[pd.Timestamp] = []

    def _bar_interval(self) -> pd.Timedelta | None:
        timeframe = str(getattr(self._current_strategy, "timeframe", "") or "").upper()
        mapping = {
            "M1": pd.Timedelta(minutes=1),
            "M5": pd.Timedelta(minutes=5),
            "M15": pd.Timedelta(minutes=15),
            "M30": pd.Timedelta(minutes=30),
            "H1": pd.Timedelta(hours=1),
            "H4": pd.Timedelta(hours=4),
            "D1": pd.Timedelta(days=1),
        }
        return mapping.get(timeframe)

    def _normalized_symbol(self) -> str:
        symbol = getattr(self._current_strategy, "symbol", "") or ""
        symbol = symbol.strip().upper()
        symbol = symbol.split("(")[0]
        for suffix in ["_FTMO", ".FTMO", "_MT5", ".MT5"]:
            if symbol.endswith(suffix):
                symbol = symbol[: -len(suffix)]
        if symbol in {"HKG50_MT5IMPORT", "HK50", "HK50.CASH"}:
            return "HK50.CASH"
        if symbol in {"US30_MT5", "US30", "US30.CASH"}:
            return "US30.CASH"
        return symbol

    def _profit_multiplier(self, exit_price: float) -> float:
        if self.tick_size > 0 and self.tick_value > 0:
            return float(self.tick_value) / float(self.tick_size)

        symbol = self._normalized_symbol()

        if symbol == "XAUUSD":
            return float(self.contract_size) if self.contract_size > 0 else 100.0
        if symbol == "XAGUSD":
            return 5_000.0

        if len(symbol) == 6 and symbol.isalpha():
            contract_size = float(self.contract_size) if self.contract_size > 0 else 100_000.0
            base = symbol[:3]
            quote = symbol[3:]
            if quote == "USD":
                return contract_size
            if base == "USD":
                return contract_size / max(float(exit_price), 1e-9)

        return self.lot_value

    def session_time(self, ts: pd.Timestamp) -> pd.Timestamp:
        if abs(float(self.session_timezone_offset_hours)) < 1e-9:
            return ts
        return ts + pd.to_timedelta(float(self.session_timezone_offset_hours), unit="h")

    def _bar_spread(self, bar_idx: int) -> float:
        if not self.use_bar_spread or self._price_df is None or "spread" not in self._price_df.columns:
            return float(self.spread_pips)
        if bar_idx < 0 or bar_idx >= len(self._price_df.index):
            return float(self.spread_pips)
        raw = pd.to_numeric(self._price_df["spread"].iloc[bar_idx], errors="coerce")
        if pd.isna(raw):
            return float(self.spread_pips)
        spread = float(raw) * float(self.tick_size)
        return spread if spread > 0.0 else float(self.spread_pips)

    def _apply_entry_execution_price(
        self,
        direction: Direction,
        price: float,
        *,
        spread: float = 0.0,
    ) -> float:
        if direction == Direction.LONG:
            price = float(price) + float(spread)
        if direction == Direction.LONG:
            return float(price) + float(self.slippage_pips)
        return float(price) - float(self.slippage_pips)

    def _apply_exit_execution_price(
        self,
        direction: Direction,
        price: float,
        *,
        spread: float = 0.0,
    ) -> float:
        if direction == Direction.SHORT:
            price = float(price) + float(spread)
        if direction == Direction.LONG:
            return float(price) - float(self.slippage_pips)
        return float(price) + float(self.slippage_pips)

    def run(
        self,
        strategy:  BaseStrategy,
        df:        pd.DataFrame,
        intrabar_df: Optional[pd.DataFrame] = None,
        date_from: Optional[str] = None,
        date_to:   Optional[str] = None,
    ) -> BacktestResults:

        # Reset state
        self._balance  = self.initial_capital
        self._equity   = self.initial_capital
        self._order_id = 0
        self._decision_id = 0
        self._open_positions  = []
        self._pending_orders  = []
        self._closed_trades   = []
        self._decision_records = []
        self._current_strategy = strategy
        self._debug_events = []
        self._balance_curve_points = []
        self._equity_curve_points = []
        self._equity_worst_curve_points = []
        self._curve_times = []

        sim_date_from = pd.Timestamp(date_from) if date_from else None
        sim_date_to = pd.Timestamp(date_to) if date_to else None
        if sim_date_to is not None:
            df = df[df.index <= sim_date_to]
        if intrabar_df is not None:
            intrabar_df = intrabar_df.copy()
            intrabar_df.columns = [c.lower() for c in intrabar_df.columns]
            if sim_date_to is not None:
                intrabar_df = intrabar_df[intrabar_df.index <= sim_date_to]

        if df.empty:
            raise ValueError("No data in the specified date range.")

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        self._price_df = df

        if self.verbose:
            print(f"Computing indicators for {len(df)} bars...")
        df = strategy.compute_indicators(df)
        df = df.ffill().fillna(0)

        strategy.on_start(df)
        start_pos = 0
        if sim_date_from is not None:
            candidates = np.flatnonzero(df.index >= sim_date_from)
            if len(candidates) == 0:
                raise ValueError("No data in the specified date range.")
            start_pos = int(candidates[0])
        end_pos = len(df) - 1
        if sim_date_to is not None:
            candidates = np.flatnonzero(df.index <= sim_date_to)
            if len(candidates) == 0:
                raise ValueError("No data in the specified date range.")
            end_pos = int(candidates[-1])
        if start_pos >= len(df) or start_pos > end_pos:
            raise ValueError("No data in the specified date range.")

        self._record_curve_point(
            df.index[start_pos],
            float(df["close"].iloc[start_pos]),
            float(df["low"].iloc[start_pos]),
            float(df["high"].iloc[start_pos]),
            start_pos,
        )

        warmup = 60
        use_intrabar = self.intrabar_steps > 1

        for i in range(max(1, start_pos), end_pos + 1):
            t     = df.index[i]
            bar   = df.iloc[i]
            open_ = float(bar["open"])
            high  = float(bar["high"])
            low   = float(bar["low"])
            close = float(bar["close"])

            # 1. Expire unfilled orders / time-based exits at bar open
            self._expire_pending(i, t)
            self._expire_positions(i, t, open_)

            # 2. Strategy acts at bar open like MT5 EAs on a new bar tick.
            if i >= warmup:
                ctx = BarContext(df, i, self)
                strategy.on_bar(ctx)

            # 3. Replay bar path for all active positions/orders, including those created at this bar open.
            if use_intrabar and (self._open_positions or self._pending_orders):
                next_t = (
                    df.index[i + 1]
                    if i + 1 < len(df)
                    else t + (df.index[i] - df.index[i - 1])
                )
                minute_slice = None
                if intrabar_df is not None:
                    minute_slice = intrabar_df[(intrabar_df.index >= t) & (intrabar_df.index < next_t)]

                if minute_slice is not None and not minute_slice.empty:
                    for minute_time, minute_bar in minute_slice.iterrows():
                        self._check_pending_triggers_prices(
                            float(minute_bar["high"]),
                            float(minute_bar["low"]),
                            float(minute_bar["open"]),
                            i,
                            minute_time,
                        )
                        self._check_sl_tp_prices(
                            float(minute_bar["high"]),
                            float(minute_bar["low"]),
                            i,
                            minute_time,
                        )
                        self._update_trailing_prices(
                            float(minute_bar["high"]),
                            float(minute_bar["low"]),
                        )
                        if not self._open_positions and not self._pending_orders:
                            break
                elif self._open_positions:
                    # Fallback when minute data is unavailable.
                    path = _generate_intrabar_path(
                        open_, high, low, close,
                        self.intrabar_steps, self._rng
                    )
                    for step in range(len(path)):
                        step_price = path[step]
                        if step == 0:
                            step_high = max(open_, step_price)
                            step_low  = min(open_, step_price)
                        else:
                            step_high = max(path[step-1], step_price)
                            step_low  = min(path[step-1], step_price)

                        self._check_sl_tp_prices(step_high, step_low, i, t)
                        self._update_trailing_prices(step_high, step_low)
                        if not self._open_positions:
                            break

            else:
                # ── Standard OHLC mode ────────────────────────────────────────
                self._check_pending_triggers(df, i)
                self._check_sl_tp(df, i)
                self._update_trailing(df, i)

            # 4. Update equity and curves
            self._update_equity(df, i)
            self._record_curve_point(t, close, low, high, i)

        # Close remaining positions at last bar
        last_bar   = end_pos
        last_time  = df.index[last_bar]
        last_close = float(df["close"].iloc[last_bar])
        self._close_all(last_bar, last_time, last_close,
                        reason=CloseReason.END_OF_TEST)
        self._update_equity(df, last_bar)
        self._record_curve_point(last_time, last_close, float(df["low"].iloc[last_bar]), float(df["high"].iloc[last_bar]), last_bar)

        balance_curve = pd.Series(self._balance_curve_points, index=self._curve_times, dtype=float)
        equity_curve = pd.Series(self._equity_curve_points, index=self._curve_times, dtype=float)
        equity_worst_curve = pd.Series(self._equity_worst_curve_points, index=self._curve_times, dtype=float)
        balance_curve = balance_curve[~balance_curve.index.duplicated(keep="last")]
        equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]
        equity_worst_curve = equity_worst_curve[~equity_worst_curve.index.duplicated(keep="last")]

        results = BacktestResults(
            trades          = self._closed_trades,
            decision_records = self._decision_records,
            initial_capital = self.initial_capital,
            symbol          = getattr(strategy, "symbol", ""),
            timeframe       = getattr(strategy, "timeframe", ""),
            date_from       = df.index[start_pos],
            date_to         = df.index[end_pos],
            params          = strategy.params.copy(),
            balance_curve_series=balance_curve,
            equity_curve_series=equity_curve,
            equity_worst_curve_series=equity_worst_curve,
            debug_events=list(self._debug_events),
        )

        strategy.on_end(results)
        return results

    # ── Order management ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def _next_decision_id(self) -> int:
        self._decision_id += 1
        return self._decision_id

    def _market_state(self, bar_idx: int) -> dict:
        if self._price_df is None or bar_idx <= 0:
            return {
                "realized_vol": 0.0,
                "balance_drawdown_pct": 0.0,
                "equity_drawdown_pct": 0.0,
                "open_positions": len(self._open_positions),
                "pending_orders": len(self._pending_orders),
                "closed_trades": len(self._closed_trades),
                "equity": self._equity,
                "balance": self._balance,
            }

        closes = self._price_df["close"].iloc[: bar_idx + 1].astype(float)
        rets = closes.pct_change().dropna()
        realized_vol = float(rets.tail(20).std(ddof=0) * np.sqrt(252)) if len(rets) >= 2 else 0.0

        if self._closed_trades:
            closed_equity = self.initial_capital + np.cumsum(
                np.concatenate([[0.0], [t.net_profit for t in self._closed_trades]])
            )
        else:
            closed_equity = np.array([self.initial_capital], dtype=float)
        closed_peak = np.maximum.accumulate(closed_equity)
        balance_drawdown_pct = float(((closed_peak - closed_equity) / np.maximum(closed_peak, 1e-9)).max() * 100.0) if len(closed_equity) else 0.0

        return {
            "realized_vol": realized_vol,
            "balance_drawdown_pct": balance_drawdown_pct,
            "open_positions": len(self._open_positions),
            "pending_orders": len(self._pending_orders),
            "closed_trades": len(self._closed_trades),
            "equity": self._equity,
            "balance": self._balance,
        }

    def _submit_trade_intent(self, intent: TradeIntent) -> int:
        state = self._market_state(intent.bar_idx)
        decision = self.overlay_policy.evaluate(intent, state) if self.overlay_policy else OverlayDecision()
        final_lots = float(
            decision.adjusted_lots
            if decision.adjusted_lots is not None
            else intent.lots * float(decision.size_multiplier)
        )
        if final_lots <= 0:
            decision.allow_trade = False

        decision_id = self._next_decision_id()
        self._decision_records.append(DecisionRecord(
            decision_id=decision_id,
            time=intent.time,
            bar_idx=intent.bar_idx,
            symbol=getattr(self._current_strategy, "symbol", ""),
            timeframe=getattr(self._current_strategy, "timeframe", ""),
            order_type=intent.order_type.value,
            direction=intent.direction.value,
            requested_lots=float(intent.lots),
            final_lots=float(final_lots if decision.allow_trade else 0.0),
            allow_trade=bool(decision.allow_trade),
            policy_name=self.overlay_policy.name if self.overlay_policy else "None",
            policy_tag=decision.tag,
            comment=intent.comment,
            market_features={
                "realized_vol": float(state.get("realized_vol", 0.0)),
                "balance_drawdown_pct": float(state.get("balance_drawdown_pct", 0.0)),
                "equity": float(state.get("equity", self._equity)),
                "balance": float(state.get("balance", self._balance)),
            },
            policy_notes=dict(decision.notes or {}),
        ))
        if not decision.allow_trade:
            return 0

        executable = TradeIntent(
            order_type=intent.order_type,
            direction=intent.direction,
            price=float(intent.price),
            stop_loss=float(intent.stop_loss),
            take_profit=float(intent.take_profit),
            lots=float(final_lots),
            bar_idx=int(intent.bar_idx),
            time=intent.time,
            expiry_bars=int(intent.expiry_bars),
            comment=intent.comment,
            trailing_stop=float(intent.trailing_stop),
            trail_activation=float(intent.trail_activation),
            exit_after_bars=int(intent.exit_after_bars),
        )
        return self._execute_trade_intent(executable, decision_id=decision_id, requested_lots=float(intent.lots))

    def _execute_trade_intent(self, intent: TradeIntent, *, decision_id: int, requested_lots: float) -> int:
        if intent.order_type == OrderType.MARKET:
            return self._open_position(
                intent.direction,
                intent.price,
                intent.stop_loss,
                intent.take_profit,
                intent.lots,
                intent.bar_idx,
                intent.time,
                intent.comment,
                trailing_stop=intent.trailing_stop,
                trail_activation=intent.trail_activation,
                exit_after_bars=intent.exit_after_bars,
                decision_id=decision_id,
                requested_lots=requested_lots,
                preserve_exit_offsets=True,
            )
        return self._place_pending(
            intent.order_type,
            intent.direction,
            intent.price,
            intent.stop_loss,
            intent.take_profit,
            intent.lots,
            intent.bar_idx,
            intent.time,
            intent.expiry_bars,
            intent.comment,
            decision_id=decision_id,
            requested_lots=requested_lots,
        )

    def _place_pending(
        self, order_type: OrderType, direction: Direction,
        price: float, sl: float, tp: float, lots: float,
        bar_idx: int, time: pd.Timestamp, expiry_bars: int, comment: str,
        decision_id: int = 0, requested_lots: float = 0.0,
    ) -> int:
        oid = self._next_id()
        self._pending_orders.append(PendingOrder(
            id=oid, order_type=order_type, direction=direction,
            entry_price=price, stop_loss=sl, take_profit=tp,
            lots=lots, opened_bar=bar_idx, opened_time=time,
            expiry_bars=expiry_bars, comment=comment
            , decision_id=decision_id, requested_lots=requested_lots or lots
        ))
        self._debug_events.append(
            {
                "kind": "pending_placed",
                "order_id": oid,
                "time": time,
                "bar_idx": bar_idx,
                "order_type": order_type.value,
                "direction": direction.value,
                "entry_price": float(price),
                "stop_loss": float(sl),
                "take_profit": float(tp),
                "comment": comment,
            }
        )
        return oid

    def _open_position(
        self, direction: Direction, price: float,
        sl: float, tp: float, lots: float,
        bar_idx: int, time: pd.Timestamp, comment: str = "",
        trailing_stop: float = 0.0, trail_activation: float = 0.0,
        exit_after_bars: int = 0,
        decision_id: int = 0,
        requested_lots: float = 0.0,
        preserve_exit_offsets: bool = False,
    ) -> int:
        pid = self._next_id()
        spread = self._bar_spread(bar_idx)
        requested_entry_price = float(price)
        price = self._apply_entry_execution_price(direction, requested_entry_price, spread=spread)
        if preserve_exit_offsets:
            execution_delta = float(price) - requested_entry_price
            if sl:
                sl = float(sl) + execution_delta
            if tp:
                tp = float(tp) + execution_delta
        pos = Position(
            id=pid, direction=direction, entry_price=price,
            stop_loss=sl, take_profit=tp, lots=lots,
            opened_bar=bar_idx, opened_time=time,
            trailing_stop=trailing_stop,
            trail_activation=trail_activation,
            exit_after_bars=exit_after_bars,
            comment=comment,
            decision_id=decision_id,
            requested_lots=requested_lots or lots,
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
        spread = self._bar_spread(bar_idx)
        exit_price = self._apply_exit_execution_price(pos.direction, exit_price, spread=spread)
        gross = self._calc_profit(pos, exit_price)
        comm  = self.commission_per_lot * pos.lots * 2
        hold_days = max(0.0, (time - pos.opened_time).total_seconds() / 86400.0)
        swap_rate = self.swap_per_lot_long if pos.direction == Direction.LONG else self.swap_per_lot_short
        swap = float(swap_rate) * float(pos.lots) * hold_days
        trade = ClosedTrade(
            id=pos.id, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
            lots=pos.lots, opened_bar=pos.opened_bar, closed_bar=bar_idx,
            opened_time=pos.opened_time, closed_time=time,
            close_reason=reason,
            gross_profit=gross, commission=comm, swap=swap,
            comment=pos.comment,
            decision_id=pos.decision_id,
            requested_lots=pos.requested_lots,
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
        return pips * pos.lots * self._profit_multiplier(exit_price)

    # ── OHLC simulation ───────────────────────────────────────────────────────

    def _check_pending_triggers(self, df: pd.DataFrame, i: int):
        bar   = df.iloc[i]
        high  = float(bar["high"])
        low   = float(bar["low"])
        open_ = float(bar["open"])
        t     = df.index[i]
        self._check_pending_triggers_prices(high, low, open_, i, t)

    def _check_pending_triggers_prices(
        self,
        high: float,
        low: float,
        open_: float,
        bar_idx: int,
        time: pd.Timestamp,
    ):
        t = time

        for order in list(self._pending_orders):
            if self.intrabar_steps > 1 and time <= order.opened_time:
                continue
            if order.first_eligible_time is None:
                order.first_eligible_time = time
                self._debug_events.append(
                    {
                        "kind": "pending_first_eligible",
                        "order_id": order.id,
                        "time": time,
                        "bar_idx": bar_idx,
                        "order_type": order.order_type.value,
                        "direction": order.direction.value,
                        "entry_price": float(order.entry_price),
                        "comment": order.comment,
                    }
                )
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
                self._debug_events.append(
                    {
                        "kind": "pending_filled",
                        "order_id": order.id,
                        "time": t,
                        "bar_idx": bar_idx,
                        "order_type": order.order_type.value,
                        "direction": order.direction.value,
                        "entry_price": float(order.entry_price),
                        "fill_price": float(fill_price),
                        "first_eligible_time": order.first_eligible_time,
                        "comment": order.comment,
                    }
                )
                trail_dist = getattr(self._current_strategy,
                                     '_next_trail_dist', 0.0)
                trail_act  = getattr(self._current_strategy,
                                     '_next_trail_activation', 0.0)
                exit_after_bars = getattr(
                    self._current_strategy,
                    '_next_exit_after_bars',
                    0,
                )
                self._open_position(
                    order.direction, fill_price,
                    order.stop_loss, order.take_profit,
                    order.lots, bar_idx, t, order.comment,
                    trailing_stop=trail_dist,
                    trail_activation=trail_act,
                    exit_after_bars=exit_after_bars,
                    decision_id=order.decision_id,
                    requested_lots=order.requested_lots,
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

    def _expire_pending(self, current_bar: int, time: pd.Timestamp):
        bar_interval = self._bar_interval()
        to_remove = []
        for order in self._pending_orders:
            if order.expiry_bars <= 0:
                continue
            expired = False
            if bar_interval is not None:
                expires_at = order.opened_time + (bar_interval * int(order.expiry_bars))
                expired = time >= expires_at
            else:
                expired = current_bar - order.opened_bar >= order.expiry_bars
            if expired:
                to_remove.append(order)
        for order in to_remove:
            self._pending_orders.remove(order)

    def _expire_positions(self, current_bar: int, time: pd.Timestamp, open_price: float):
        bar_interval = self._bar_interval()
        to_close = []
        for pos in self._open_positions:
            if pos.exit_after_bars <= 0:
                continue
            expired = False
            if bar_interval is not None:
                expires_at = pos.opened_time + (bar_interval * int(pos.exit_after_bars))
                expired = time >= expires_at
            else:
                expired = current_bar - pos.opened_bar >= pos.exit_after_bars
            if expired:
                to_close.append(pos)
        for pos in to_close:
            self._close_position(pos, open_price, current_bar, time, CloseReason.SIGNAL)

    def _update_equity(self, df: pd.DataFrame, i: int):
        close    = float(df["close"].iloc[i])
        spread = self._bar_spread(i)
        floating = sum(
            self._calc_profit(
                pos,
                self._apply_exit_execution_price(pos.direction, close, spread=spread),
            )
            for pos in self._open_positions
        )
        self._equity = self._balance + floating

    def _worst_case_equity(self, low: float, high: float, bar_idx: int) -> float:
        spread = self._bar_spread(bar_idx)
        floating = 0.0
        for pos in self._open_positions:
            worst_price = low if pos.direction == Direction.LONG else high
            floating += self._calc_profit(
                pos,
                self._apply_exit_execution_price(pos.direction, worst_price, spread=spread),
            )
        return self._balance + floating

    def _record_curve_point(self, time: pd.Timestamp, close: float, low: float, high: float, bar_idx: int):
        spread = self._bar_spread(bar_idx)
        floating_close = sum(
            self._calc_profit(
                pos,
                self._apply_exit_execution_price(pos.direction, close, spread=spread),
            )
            for pos in self._open_positions
        )
        self._curve_times.append(time)
        self._balance_curve_points.append(float(self._balance))
        self._equity_curve_points.append(float(self._balance + floating_close))
        self._equity_worst_curve_points.append(float(self._worst_case_equity(low, high, bar_idx)))
