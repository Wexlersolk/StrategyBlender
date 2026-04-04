import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from engine.policy import CompositeOverlayPolicy, DrawdownThrottlePolicy, VolatilityTargetPolicy
from research.trade_dataset import build_trade_dataset


class SingleTradeStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.has_position or ctx.has_pending or ctx.bar_index != 60:
            return
        ctx.buy_market(
            sl=ctx.open - 1.0,
            tp=ctx.open + 2.0,
            lots=float(self.p("lots")),
            comment="single_trade",
        )


def _sample_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=120, freq="1h")
    close = pd.Series(range(120), index=index, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "open": close.values,
            "high": (close + 0.5).values,
            "low": (close - 0.5).values,
            "close": close.values,
            "volume": [1.0] * len(index),
        },
        index=index,
    )


def test_overlay_decision_records_and_trade_dataset():
    policy = CompositeOverlayPolicy([
        VolatilityTargetPolicy(target_vol=0.01, lookback=20, min_multiplier=0.25, max_multiplier=1.0),
        DrawdownThrottlePolicy(soft_drawdown_pct=50.0, hard_drawdown_pct=80.0, soft_multiplier=0.5),
    ])
    result = Backtester(initial_capital=100_000, overlay_policy=policy).run(SingleTradeStrategy(), _sample_df())

    assert len(result.decision_records) == 1
    record = result.decision_records[0]
    assert record.allow_trade is True
    assert 0.0 < record.final_lots <= 1.0
    assert "realized_vol" in record.market_features

    dataset = build_trade_dataset(result)
    assert len(dataset) == 1
    assert dataset.loc[0, "trade_taken"] == 1
    assert dataset.loc[0, "realized_trade_count"] == 1


def test_drawdown_policy_can_block_trade():
    class LosingThenTradeStrategy(BaseStrategy):
        params = {"lots": 1.0}

        def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

        def on_bar(self, ctx):
            if ctx.bar_index == 60 and not ctx.has_position:
                ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 2.0, lots=1.0, comment="loss")
            if ctx.bar_index == 62 and not ctx.has_position:
                ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 2.0, lots=1.0, comment="blocked")

    df = _sample_df()
    df.iloc[61, df.columns.get_loc("low")] = df.iloc[60]["open"] - 2.0
    policy = DrawdownThrottlePolicy(soft_drawdown_pct=0.05, hard_drawdown_pct=0.09, soft_multiplier=0.5)
    result = Backtester(initial_capital=1_000, overlay_policy=policy).run(LosingThenTradeStrategy(), df)

    assert len(result.decision_records) >= 2
    assert any(record.allow_trade is False for record in result.decision_records[1:])
