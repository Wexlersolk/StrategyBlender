"""
engine/base_strategy.py

All strategies inherit from BaseStrategy.
Subclasses implement:
    compute_indicators(df)  — add columns to OHLCV DataFrame
    on_bar(ctx)             — called on every bar, place/close orders here
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from engine.position import (
    Direction, OrderType, PendingOrder, Position
)


class BarContext:
    """
    Passed to on_bar() on every bar.
    Provides clean API to read prices and place orders.
    """

    def __init__(self, df: pd.DataFrame, bar_idx: int, backtester):
        self._df  = df
        self._i   = bar_idx
        self._bt  = backtester

    # ── Current bar data ──────────────────────────────────────────────────────

    @property
    def open(self)   -> float: return float(self._df["open"].iloc[self._i])
    @property
    def high(self)   -> float: return float(self._df["high"].iloc[self._i])
    @property
    def low(self)    -> float: return float(self._df["low"].iloc[self._i])
    @property
    def close(self)  -> float: return float(self._df["close"].iloc[self._i])
    @property
    def volume(self) -> float: return float(self._df["volume"].iloc[self._i])
    @property
    def time(self)   -> pd.Timestamp: return self._df.index[self._i]
    @property
    def bar_index(self) -> int: return self._i

    def indicator(self, name: str, shift: int = 0) -> float:
        """Get indicator value. shift=1 means previous bar."""
        idx = self._i - shift
        if idx < 0 or name not in self._df.columns:
            return float('nan')
        return float(self._df[name].iloc[idx])

    def indicators(self, name: str, shift: int = 0) -> pd.Series:
        """Get full indicator series up to current bar."""
        if name not in self._df.columns:
            return pd.Series(dtype=float)
        return self._df[name].iloc[:self._i + 1 - shift]

    # ── Order placement ───────────────────────────────────────────────────────

    def buy_stop(self, price: float, sl: float, tp: float,
                 lots: float, expiry_bars: int = 1, comment: str = "") -> int:
        """Place a buy stop order (triggers when price rises to `price`)."""
        return self._bt._place_pending(
            OrderType.BUY_STOP, Direction.LONG,
            price, sl, tp, lots, self._i, expiry_bars, comment
        )

    def buy_limit(self, price: float, sl: float, tp: float,
                  lots: float, expiry_bars: int = 1, comment: str = "") -> int:
        """Place a buy limit order (triggers when price falls to `price`)."""
        return self._bt._place_pending(
            OrderType.BUY_LIMIT, Direction.LONG,
            price, sl, tp, lots, self._i, expiry_bars, comment
        )

    def sell_stop(self, price: float, sl: float, tp: float,
                  lots: float, expiry_bars: int = 1, comment: str = "") -> int:
        """Place a sell stop order (triggers when price falls to `price`)."""
        return self._bt._place_pending(
            OrderType.SELL_STOP, Direction.SHORT,
            price, sl, tp, lots, self._i, expiry_bars, comment
        )

    def sell_limit(self, price: float, sl: float, tp: float,
                   lots: float, expiry_bars: int = 1, comment: str = "") -> int:
        """Place a sell limit order (triggers when price rises to `price`)."""
        return self._bt._place_pending(
            OrderType.SELL_LIMIT, Direction.SHORT,
            price, sl, tp, lots, self._i, expiry_bars, comment
        )

    def buy_market(self, sl: float, tp: float,
                   lots: float, comment: str = "") -> int:
        """Open a long position at current bar open."""
        trailing_stop = getattr(self._bt._current_strategy, "_next_trail_dist", 0.0)
        trail_activation = getattr(self._bt._current_strategy, "_next_trail_activation", 0.0)
        exit_after_bars = getattr(self._bt._current_strategy, "_next_exit_after_bars", 0)
        return self._bt._open_position(
            Direction.LONG,
            self.open,
            sl,
            tp,
            lots,
            self._i,
            self.time,
            comment,
            trailing_stop=trailing_stop,
            trail_activation=trail_activation,
            exit_after_bars=exit_after_bars,
        )

    def sell_market(self, sl: float, tp: float,
                    lots: float, comment: str = "") -> int:
        """Open a short position at current bar open."""
        trailing_stop = getattr(self._bt._current_strategy, "_next_trail_dist", 0.0)
        trail_activation = getattr(self._bt._current_strategy, "_next_trail_activation", 0.0)
        exit_after_bars = getattr(self._bt._current_strategy, "_next_exit_after_bars", 0)
        return self._bt._open_position(
            Direction.SHORT,
            self.open,
            sl,
            tp,
            lots,
            self._i,
            self.time,
            comment,
            trailing_stop=trailing_stop,
            trail_activation=trail_activation,
            exit_after_bars=exit_after_bars,
        )

    def close_all(self):
        """Close all open positions at current bar open."""
        self._bt._close_all(self._i, self.time, self.open)

    def cancel_pending(self):
        """Cancel all pending orders."""
        self._bt._pending_orders.clear()

    # ── State queries ─────────────────────────────────────────────────────────

    @property
    def has_long(self) -> bool:
        return any(p.direction == Direction.LONG
                   for p in self._bt._open_positions)

    @property
    def has_short(self) -> bool:
        return any(p.direction == Direction.SHORT
                   for p in self._bt._open_positions)

    @property
    def has_position(self) -> bool:
        return bool(self._bt._open_positions)

    @property
    def has_pending(self) -> bool:
        return bool(self._bt._pending_orders)

    @property
    def equity(self) -> float:
        return self._bt._equity

    @property
    def balance(self) -> float:
        return self._bt._balance


class BaseStrategy:
    """
    Base class for all strategies.

    Subclasses must implement:
        compute_indicators(df) -> pd.DataFrame
        on_bar(ctx: BarContext)

    Optionally override:
        on_start(df)    — called once before first bar
        on_end(results) — called after last bar
    """

    # Override in subclass
    name:   str  = "BaseStrategy"
    params: dict = {}

    def __init__(self, **param_overrides):
        # Deep copy params so each instance is independent
        import copy
        self.params = copy.deepcopy(self.__class__.params)
        self.params.update(param_overrides)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator columns to df. Called once before backtesting starts.
        Must return the modified df.
        """
        raise NotImplementedError

    def on_bar(self, ctx: BarContext):
        """
        Called on every bar (at bar open).
        Use ctx to read prices and place orders.
        """
        raise NotImplementedError

    def on_start(self, df: pd.DataFrame):
        """Called once before the first bar. Override if needed."""
        pass

    def on_end(self, results):
        """Called after the last bar. Override if needed."""
        pass

    # ── Helper methods available to subclasses ────────────────────────────────

    def p(self, name: str):
        """Shorthand for self.params[name]."""
        return self.params[name]

    def crosses_above(self, series: pd.Series, level: float, i: int) -> bool:
        """True if series crossed above level at bar i."""
        if i < 1:
            return False
        return series.iloc[i - 1] <= level < series.iloc[i]

    def crosses_below(self, series: pd.Series, level: float, i: int) -> bool:
        """True if series crossed below level at bar i."""
        if i < 1:
            return False
        return series.iloc[i - 1] >= level > series.iloc[i]

    def crosses_above_series(self, s1: pd.Series, s2: pd.Series, i: int) -> bool:
        """True if s1 crossed above s2 at bar i."""
        if i < 1:
            return False
        return s1.iloc[i - 1] <= s2.iloc[i - 1] and s1.iloc[i] > s2.iloc[i]

    def crosses_below_series(self, s1: pd.Series, s2: pd.Series, i: int) -> bool:
        """True if s1 crossed below s2 at bar i."""
        if i < 1:
            return False
        return s1.iloc[i - 1] >= s2.iloc[i - 1] and s1.iloc[i] < s2.iloc[i]
