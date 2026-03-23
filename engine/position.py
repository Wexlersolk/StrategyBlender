"""
engine/position.py

Data classes for positions, pending orders, and closed trades.
Mirrors MT5's order/position model closely.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class Direction(Enum):
    LONG  = "long"
    SHORT = "short"


class OrderType(Enum):
    MARKET      = "market"
    BUY_STOP    = "buy_stop"     # entry above market
    SELL_STOP   = "sell_stop"    # entry below market
    BUY_LIMIT   = "buy_limit"    # entry below market
    SELL_LIMIT  = "sell_limit"   # entry above market


class CloseReason(Enum):
    STOP_LOSS    = "sl"
    TAKE_PROFIT  = "tp"
    TRAILING     = "trailing"
    SIGNAL       = "signal"
    END_OF_TEST  = "end"


@dataclass
class PendingOrder:
    """A stop/limit order waiting to be triggered."""
    id:          int
    order_type:  OrderType
    direction:   Direction
    entry_price: float          # price at which order triggers
    stop_loss:   float
    take_profit: float
    lots:        float
    opened_bar:  int            # bar index when order was placed
    expiry_bars: int  = 1       # cancel after N bars (0 = GTC)
    comment:     str  = ""


@dataclass
class Position:
    """An open position."""
    id:            int
    direction:     Direction
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    lots:          float
    opened_bar:    int
    opened_time:   pd.Timestamp
    trailing_stop: float = 0.0  # current trailing stop distance in price
    trail_activation: float = 0.0  # profit needed before trailing activates
    comment:       str   = ""


@dataclass
class ClosedTrade:
    """A completed trade with full statistics."""
    id:           int
    direction:    Direction
    entry_price:  float
    exit_price:   float
    stop_loss:    float
    take_profit:  float
    lots:         float
    opened_bar:   int
    closed_bar:   int
    opened_time:  pd.Timestamp
    closed_time:  pd.Timestamp
    close_reason: CloseReason
    gross_profit: float         # before commission/swap
    commission:   float
    swap:         float
    comment:      str = ""

    @property
    def net_profit(self) -> float:
        return self.gross_profit - self.commission - self.swap

    @property
    def duration_bars(self) -> int:
        return self.closed_bar - self.opened_bar

    @property
    def pips(self) -> float:
        if self.direction == Direction.LONG:
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price
