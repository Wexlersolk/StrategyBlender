import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from engine.policy import ModelOverlayPolicy, NullOverlayPolicy, VolatilityTargetPolicy
from research.meta_models import fit_filter_model, fit_sizing_model
from research.purged_walk_forward import generate_purged_splits, run_purged_walk_forward
from research.trade_dataset import build_trade_dataset


class PassiveStrategy(BaseStrategy):
    params = {}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        return


class AlternatingTradeStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.has_position or ctx.has_pending:
            return
        if ctx.bar_index in {60, 62, 64, 66, 68, 70, 72, 74}:
            ctx.buy_market(
                sl=ctx.open - 1.0,
                tp=ctx.open + 1.0,
                lots=1.0,
                comment=f"bar_{ctx.bar_index}",
            )


def _bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=180, freq="1h")
    base = pd.Series(range(180), index=index, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "open": base.values,
            "high": (base + 1.0).values,
            "low": (base - 1.0).values,
            "close": base.values,
            "volume": [1.0] * len(index),
        },
        index=index,
    )


def _bars_with_alternating_outcomes() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=260, freq="1h")
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
    for bar_idx in [61, 65, 69, 73]:
        frame.iloc[bar_idx, frame.columns.get_loc("high")] = frame.iloc[bar_idx]["open"] + 1.5
    for bar_idx in [63, 67, 71, 75]:
        frame.iloc[bar_idx, frame.columns.get_loc("low")] = frame.iloc[bar_idx]["open"] - 1.5
    for offset in [80, 100, 120, 140, 160, 180, 200, 220]:
        if offset + 1 < len(frame):
            frame.iloc[offset + 1, frame.columns.get_loc("high")] = frame.iloc[offset + 1]["open"] + 1.5
        if offset + 3 < len(frame):
            frame.iloc[offset + 3, frame.columns.get_loc("low")] = frame.iloc[offset + 3]["open"] - 1.5
    return frame


def test_generate_purged_splits_respects_embargo():
    df = _bars()
    splits = generate_purged_splits(df.index, train_bars=60, test_bars=20, embargo_bars=5)

    assert splits
    first = splits[0]
    assert first.train_end < first.test_start
    assert first.embargo_start is not None
    assert first.embargo_end is not None
    assert (first.test_start - first.embargo_end).total_seconds() == 3600


def test_run_purged_walk_forward_returns_window_results():
    df = _bars()

    def make_bt(policy):
        return Backtester(initial_capital=10_000, overlay_policy=policy)

    def policy_factory(_train_df, _train_result):
        return VolatilityTargetPolicy(target_vol=0.15)

    windows = run_purged_walk_forward(
        df=df,
        strategy_class=PassiveStrategy,
        backtester_factory=make_bt,
        policy_factory=policy_factory,
        train_bars=60,
        test_bars=20,
        embargo_bars=5,
    )

    assert windows
    assert all(window.overlay_name == "VolatilityTargetPolicy" for window in windows)
    assert all(window.train_result.n_trades == 0 for window in windows)
    assert all(window.test_result.n_trades == 0 for window in windows)


def test_run_purged_walk_forward_can_train_ml_overlay_policy():
    df = _bars_with_alternating_outcomes()

    def make_bt(policy):
        return Backtester(initial_capital=10_000, overlay_policy=policy)

    def policy_factory(_train_df, train_result):
        train_dataset = build_trade_dataset(train_result)
        filter_model = fit_filter_model(train_dataset, threshold=0.55)
        sizing_model = fit_sizing_model(train_dataset, min_multiplier=0.5, max_multiplier=1.5)
        return ModelOverlayPolicy(filter_model=filter_model, sizing_model=sizing_model)

    windows = run_purged_walk_forward(
        df=df,
        strategy_class=AlternatingTradeStrategy,
        backtester_factory=make_bt,
        policy_factory=policy_factory,
        train_bars=100,
        test_bars=80,
        embargo_bars=5,
    )

    assert windows
    assert all(window.overlay_name == "ModelOverlayPolicy" for window in windows)
    non_empty_test_sets = [window.test_dataset for window in windows if not window.test_dataset.empty]
    assert non_empty_test_sets
    assert all("policy_filter_probability" in dataset.columns for dataset in non_empty_test_sets)
    assert any(dataset["policy_tag"].astype(str).str.contains("ml_").any() for dataset in non_empty_test_sets)
