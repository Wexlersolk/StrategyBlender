import pandas as pd
import pytest

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from engine.policy import DrawdownThrottlePolicy, ModelOverlayPolicy, NullOverlayPolicy
from research.meta_models import fit_filter_model, fit_sizing_model, predict_filter_probabilities
from research.trade_dataset import build_trade_dataset


class AlternatingTradeStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.has_position or ctx.has_pending:
            return
        if ctx.bar_index in {60, 62, 64, 66, 68, 70}:
            ctx.buy_market(
                sl=ctx.open - 1.0,
                tp=ctx.open + 1.0,
                lots=1.0,
                comment=f"bar_{ctx.bar_index}",
            )


def _sample_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=140, freq="1h")
    close = pd.Series(100.0, index=index)
    frame = pd.DataFrame(
        {
            "open": close.values,
            "high": (close + 0.25).values,
            "low": (close - 0.25).values,
            "close": close.values,
            "volume": [1.0] * len(index),
        },
        index=index,
    )
    for bar_idx in [61, 65, 69]:
        frame.iloc[bar_idx, frame.columns.get_loc("high")] = frame.iloc[bar_idx]["open"] + 1.5
    for bar_idx in [63, 67, 71]:
        frame.iloc[bar_idx, frame.columns.get_loc("low")] = frame.iloc[bar_idx]["open"] - 1.5
    return frame


def test_fit_meta_models_on_trade_dataset():
    result = Backtester(initial_capital=100_000, overlay_policy=NullOverlayPolicy()).run(AlternatingTradeStrategy(), _sample_df())
    dataset = build_trade_dataset(result)

    filter_model = fit_filter_model(dataset, threshold=0.55)
    sizing_model = fit_sizing_model(dataset, min_multiplier=0.5, max_multiplier=1.5)
    probabilities = predict_filter_probabilities(dataset, filter_model)

    assert len(probabilities) == len(dataset)
    assert 0.0 <= float(probabilities.min()) <= 1.0
    assert 0.0 <= float(probabilities.max()) <= 1.0
    assert filter_model["sample_size"] == len(dataset)
    assert sizing_model["sample_size"] == len(dataset)


def test_model_overlay_policy_records_ml_notes():
    result = Backtester(initial_capital=100_000).run(AlternatingTradeStrategy(), _sample_df())
    dataset = build_trade_dataset(result)
    filter_model = fit_filter_model(dataset, threshold=0.55)
    sizing_model = fit_sizing_model(dataset, min_multiplier=0.5, max_multiplier=1.5)

    policy = ModelOverlayPolicy(filter_model=filter_model, sizing_model=sizing_model)
    overlay_result = Backtester(initial_capital=100_000, overlay_policy=policy).run(AlternatingTradeStrategy(), _sample_df())

    assert len(overlay_result.decision_records) == len(result.decision_records)
    assert any("ml_" in record.policy_tag for record in overlay_result.decision_records)
    assert all("filter_probability" in record.policy_notes for record in overlay_result.decision_records)


def test_fit_models_fail_on_invalid_training_sets():
    result = Backtester(initial_capital=100_000).run(AlternatingTradeStrategy(), _sample_df())
    dataset = build_trade_dataset(result)

    all_positive = dataset.copy()
    all_positive["realized_net_profit"] = 10.0
    with pytest.raises(ValueError, match="both positive and negative"):
        fit_filter_model(all_positive)

    too_small = dataset.head(2).copy()
    with pytest.raises(ValueError, match="at least 3 samples"):
        fit_sizing_model(too_small)


def test_model_overlay_policy_respects_blocking_base_policy():
    result = Backtester(initial_capital=100_000).run(AlternatingTradeStrategy(), _sample_df())
    dataset = build_trade_dataset(result)
    filter_model = fit_filter_model(dataset, threshold=0.55)
    sizing_model = fit_sizing_model(dataset, min_multiplier=0.5, max_multiplier=1.5)

    blocking_base = DrawdownThrottlePolicy(soft_drawdown_pct=0.01, hard_drawdown_pct=0.01, soft_multiplier=0.5)
    policy = ModelOverlayPolicy(filter_model=filter_model, sizing_model=sizing_model, base_policy=blocking_base)
    overlay_result = Backtester(initial_capital=1_000, overlay_policy=policy).run(AlternatingTradeStrategy(), _sample_df())

    assert any(record.allow_trade is False for record in overlay_result.decision_records)
    assert any("dd_block" in record.policy_tag for record in overlay_result.decision_records)
