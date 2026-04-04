import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from engine.policy import DrawdownThrottlePolicy, VolatilityTargetPolicy
from research.trade_dataset import build_trade_dataset


class PendingSizingStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.has_position or ctx.has_pending:
            return
        if ctx.bar_index == 60:
            ctx.buy_stop(
                price=ctx.open + 0.5,
                sl=ctx.open - 1.0,
                tp=ctx.open + 1.0,
                lots=1.0,
                comment="pending_sized",
            )


class LossThenBlockedPendingStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.bar_index == 60 and not ctx.has_position:
            ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 2.0, lots=1.0, comment="loss_first")
        if ctx.bar_index == 62 and not ctx.has_position and not ctx.has_pending:
            ctx.buy_stop(
                price=ctx.open + 0.5,
                sl=ctx.open - 1.0,
                tp=ctx.open + 1.0,
                lots=1.0,
                comment="pending_blocked",
            )


def _pending_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=140, freq="1h")
    base = pd.Series(100.0, index=index)
    frame = pd.DataFrame(
        {
            "open": base.values,
            "high": (base + 0.2).values,
            "low": (base - 0.2).values,
            "close": base.values,
            "volume": [1.0] * len(index),
        },
        index=index,
    )
    frame.iloc[61, frame.columns.get_loc("high")] = frame.iloc[61]["open"] + 1.2
    return frame


def _loss_then_pending_df() -> pd.DataFrame:
    frame = _pending_df()
    frame.iloc[61, frame.columns.get_loc("low")] = frame.iloc[60]["open"] - 1.5
    frame.iloc[63, frame.columns.get_loc("high")] = frame.iloc[62]["open"] + 1.5
    return frame


def test_pending_order_keeps_overlay_adjusted_lot_size():
    policy = VolatilityTargetPolicy(target_vol=0.01, min_multiplier=0.25, max_multiplier=0.5)
    result = Backtester(initial_capital=100_000, overlay_policy=policy).run(PendingSizingStrategy(), _pending_df())

    assert len(result.decision_records) == 1
    record = result.decision_records[0]
    assert record.order_type == "buy_stop"
    assert record.allow_trade is True
    assert record.final_lots < record.requested_lots
    assert result.trades
    assert result.trades[0].lots == record.final_lots
    assert result.trades[0].requested_lots == record.requested_lots


def test_pending_order_can_be_blocked_after_drawdown():
    policy = DrawdownThrottlePolicy(soft_drawdown_pct=0.01, hard_drawdown_pct=0.05, soft_multiplier=0.5)
    result = Backtester(initial_capital=1_000, overlay_policy=policy).run(LossThenBlockedPendingStrategy(), _loss_then_pending_df())
    dataset = build_trade_dataset(result)

    assert len(result.decision_records) >= 2
    assert result.decision_records[1].order_type == "buy_stop"
    assert result.decision_records[1].allow_trade is False
    assert "dd_block" in result.decision_records[1].policy_tag
    assert len(result.trades) == 1
    assert int(dataset["trade_taken"].sum()) == 1
