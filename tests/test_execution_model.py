import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy


class SingleMarketTradeStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.bar_index == 60 and not ctx.has_position:
            ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 2.0, lots=1.0, comment="entry")


def _bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=90, freq="1h")
    base = pd.Series(100.0, index=index)
    frame = pd.DataFrame(
        {
            "open": base.values,
            "high": (base + 0.25).values,
            "low": (base - 0.25).values,
            "close": base.values,
            "volume": [1.0] * len(index),
        },
        index=index,
    )
    frame.iloc[61, frame.columns.get_loc("high")] = frame.iloc[60]["open"] + 2.5
    return frame


def test_execution_costs_reduce_trade_profit():
    df = _bars()
    base = Backtester(initial_capital=100_000).run(SingleMarketTradeStrategy(), df)
    costed = Backtester(
        initial_capital=100_000,
        commission_per_lot=5.0,
        spread_pips=0.2,
        slippage_pips=0.1,
    ).run(SingleMarketTradeStrategy(), df)

    assert base.n_trades == 1
    assert costed.n_trades == 1
    assert costed.trades[0].entry_price > base.trades[0].entry_price
    assert costed.trades[0].exit_price < base.trades[0].exit_price
    assert costed.net_profit < base.net_profit
