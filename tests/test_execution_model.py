import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from engine.indicators import wma
from engine.position import Direction


class SingleMarketTradeStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.bar_index == 60 and not ctx.has_position:
            ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 2.0, lots=1.0, comment="entry")


class SameBarStopStrategy(BaseStrategy):
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.bar_index == 60 and not ctx.has_position:
            ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 5.0, lots=1.0, comment="same_bar_stop")


class SessionTimeStrategy(BaseStrategy):
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.time.hour == 6 and ctx.bar_index >= 60 and not ctx.has_position:
            ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 1.0, lots=1.0, comment="session")


class PendingStopStrategy(BaseStrategy):
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.bar_index == 60 and not ctx.has_position and not ctx.has_pending:
            ctx.buy_stop(
                price=ctx.open + 1.0,
                sl=ctx.open - 1.0,
                tp=ctx.open + 3.0,
                lots=1.0,
                expiry_bars=2,
                comment="pending",
            )


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
    assert costed.trades[0].take_profit > base.trades[0].take_profit
    assert costed.net_profit < base.net_profit


def test_bar_spread_is_applied_to_xau_long_entries_and_short_exits():
    index = pd.date_range("2024-01-01", periods=3, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.2, 100.2, 100.2],
            "low": [99.8, 99.8, 99.8],
            "close": [100.0, 100.0, 100.0],
            "spread": [20.0, 20.0, 20.0],
        },
        index=index,
    )

    bt = Backtester(initial_capital=100_000, tick_size=0.01, spread_pips=0.05, use_bar_spread=True)
    bt._price_df = df

    long_entry = bt._apply_entry_execution_price(Direction.LONG, 100.0, spread=bt._bar_spread(1))
    short_exit = bt._apply_exit_execution_price(Direction.SHORT, 100.0, spread=bt._bar_spread(1))

    assert long_entry == 100.2
    assert short_exit == 100.2


def test_same_bar_market_entries_can_hit_stop_loss():
    df = _bars()
    df.iloc[60, df.columns.get_loc("low")] = df.iloc[60]["open"] - 1.5

    result = Backtester(initial_capital=100_000).run(SameBarStopStrategy(), df)

    assert result.n_trades == 1
    assert result.trades[0].close_reason.value == "sl"
    assert result.trades[0].closed_bar == 60


def test_market_entry_shifts_sl_tp_to_executed_fill():
    df = _bars()

    result = Backtester(
        initial_capital=100_000,
        spread_pips=0.2,
        slippage_pips=0.1,
    ).run(SingleMarketTradeStrategy(), df)

    trade = result.trades[0]

    assert trade.entry_price == 100.3
    assert trade.stop_loss == 99.3
    assert trade.take_profit == 102.3


def test_session_time_uses_configured_timezone_offset():
    index = pd.date_range("2024-01-01", periods=90, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [101.5] * len(index),
            "low": [99.5] * len(index),
            "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
        },
        index=index,
    )

    base = Backtester(initial_capital=100_000).run(SessionTimeStrategy(), df)
    shifted = Backtester(initial_capital=100_000, session_timezone_offset_hours=6.0).run(SessionTimeStrategy(), df)

    assert base.n_trades > 0
    assert shifted.n_trades > 0
    assert base.trades[0].opened_bar == 78
    assert shifted.trades[0].opened_bar == 72


class FloatingDrawdownStrategy(BaseStrategy):
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.bar_index == 60 and not ctx.has_position:
            ctx.buy_market(sl=0.0, tp=0.0, lots=1.0, comment="float_dd")


def test_equity_drawdown_captures_floating_bar_pain():
    index = pd.date_range("2024-01-01", periods=90, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0] * len(index),
            "high": [100.2] * len(index),
            "low": [99.8] * len(index),
            "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
        },
        index=index,
    )
    df.iloc[60, df.columns.get_loc("low")] = 80.0
    df.iloc[-1, df.columns.get_loc("close")] = 100.1

    result = Backtester(initial_capital=100_000).run(FloatingDrawdownStrategy(), df)

    bal_dd_abs, bal_dd_pct = result.balance_drawdown_maximal
    eq_dd_abs, eq_dd_pct = result.equity_drawdown_maximal

    assert eq_dd_abs > bal_dd_abs
    assert eq_dd_pct > bal_dd_pct


def test_equity_marking_includes_exit_side_execution_costs():
    df = _bars()
    df.iloc[60, df.columns.get_loc("low")] = 99.0
    df.iloc[60, df.columns.get_loc("close")] = 99.5

    base = Backtester(initial_capital=100_000).run(FloatingDrawdownStrategy(), df)
    costed = Backtester(initial_capital=100_000, spread_pips=0.2, slippage_pips=0.1).run(FloatingDrawdownStrategy(), df)

    assert costed.equity_drawdown_maximal[0] > base.equity_drawdown_maximal[0]


def test_wma_matches_manual_weighted_average():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = wma(series, 5)
    expected = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (1 + 2 + 3 + 4 + 5)
    assert round(float(result.iloc[-1]), 10) == round(float(expected), 10)


def test_intrabar_pending_stop_waits_until_after_opening_minute():
    df = _bars()
    intrabar_index = pd.date_range(df.index[0], df.index[-1] + pd.Timedelta(minutes=59), freq="1min")
    intrabar_df = pd.DataFrame(
        {
            "open": [100.0] * len(intrabar_index),
            "high": [100.1] * len(intrabar_index),
            "low": [99.9] * len(intrabar_index),
            "close": [100.0] * len(intrabar_index),
            "volume": [1.0] * len(intrabar_index),
        },
        index=intrabar_index,
    )

    entry_bar_time = df.index[60]
    intrabar_df.loc[entry_bar_time, ["open", "high", "low", "close"]] = [100.0, 101.5, 99.9, 100.2]
    intrabar_df.loc[entry_bar_time + pd.Timedelta(minutes=1), ["open", "high", "low", "close"]] = [100.2, 101.6, 100.0, 101.2]

    result = Backtester(initial_capital=100_000, intrabar_steps=60).run(
        PendingStopStrategy(),
        df,
        intrabar_df=intrabar_df,
    )

    assert result.n_trades == 1
    assert result.trades[0].opened_time == entry_bar_time + pd.Timedelta(minutes=1)


def test_pending_expiry_uses_elapsed_time_not_sparse_bar_count():
    index = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
        ]
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.2, 100.2, 101.5, 100.2],
            "low": [99.8, 99.8, 99.8, 99.8],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1.0] * 4,
        },
        index=index,
    )

    result = Backtester(initial_capital=100_000, intrabar_steps=1).run(PendingStopStrategy(), df)

    assert result.n_trades == 0
