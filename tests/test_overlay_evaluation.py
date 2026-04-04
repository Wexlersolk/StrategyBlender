import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from engine.policy import NullOverlayPolicy
from research.meta_models import fit_filter_model, fit_sizing_model
from research.overlay_evaluation import (
    evaluate_overlay_walk_forward_suite,
    evaluate_portfolio_overlay_walk_forward,
    make_ml_policy_factory,
)
from research.trade_dataset import build_trade_dataset


class AlternatingTradeStrategy(BaseStrategy):
    params = {"lots": 1.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        if ctx.has_position or ctx.has_pending:
            return
        if ctx.bar_index in {60, 62, 64, 66, 68, 70, 72, 74}:
            ctx.buy_market(sl=ctx.open - 1.0, tp=ctx.open + 1.0, lots=1.0, comment=f"bar_{ctx.bar_index}")


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


def test_evaluate_overlay_walk_forward_suite_compares_all_tracks():
    df = _bars_with_alternating_outcomes()
    seed_result = Backtester(initial_capital=100_000).run(AlternatingTradeStrategy(), df)
    seed_dataset = build_trade_dataset(seed_result)
    artifact = {
        "filter_model": fit_filter_model(seed_dataset, threshold=0.55),
        "sizing_model": fit_sizing_model(seed_dataset, min_multiplier=0.5, max_multiplier=1.5),
        "benchmark_name": "baseline",
        "benchmark_settings": {
            "target_vol": 0.2,
            "min_multiplier": 0.5,
            "max_multiplier": 1.5,
            "soft_dd": 5.0,
            "hard_dd": 10.0,
            "soft_multiplier": 0.5,
        },
    }

    report = evaluate_overlay_walk_forward_suite(
        df=df,
        strategy_class=AlternatingTradeStrategy,
        backtester_factory=lambda policy: Backtester(initial_capital=100_000, overlay_policy=policy),
        benchmark_factory=lambda _train_df, _train_result: NullOverlayPolicy(),
        ml_policy_factory=make_ml_policy_factory(artifact),
        train_bars=100,
        test_bars=80,
        embargo_bars=5,
    )

    assert not report.split_metrics.empty
    assert not report.aggregate_metrics.empty
    assert set(report.split_metrics["overlay"]) == {"Baseline", "Benchmark", "ML Overlay"}
    assert set(report.aggregate_metrics["overlay"]) == {"Baseline", "Benchmark", "ML Overlay"}
    assert "validation" in report.metadata
    assert "window_details" in report.metadata
    assert "trade_logs" in report.metadata
    assert "equity_curves" in report.metadata


def test_portfolio_overlay_evaluation_aggregates_assets():
    df = _bars_with_alternating_outcomes()
    seed_result = Backtester(initial_capital=100_000).run(AlternatingTradeStrategy(), df)
    seed_dataset = build_trade_dataset(seed_result)
    artifact = {
        "filter_model": fit_filter_model(seed_dataset, threshold=0.55),
        "sizing_model": fit_sizing_model(seed_dataset, min_multiplier=0.5, max_multiplier=1.5),
        "benchmark_name": "baseline",
        "benchmark_settings": {
            "target_vol": 0.2,
            "min_multiplier": 0.5,
            "max_multiplier": 1.5,
            "soft_dd": 5.0,
            "hard_dd": 10.0,
            "soft_multiplier": 0.5,
        },
    }
    report_a = evaluate_overlay_walk_forward_suite(
        df=df,
        strategy_class=AlternatingTradeStrategy,
        backtester_factory=lambda policy: Backtester(initial_capital=100_000, overlay_policy=policy),
        benchmark_factory=lambda _train_df, _train_result: NullOverlayPolicy(),
        ml_policy_factory=make_ml_policy_factory(artifact),
        train_bars=100,
        test_bars=80,
        embargo_bars=5,
    )
    report_b = evaluate_overlay_walk_forward_suite(
        df=df,
        strategy_class=AlternatingTradeStrategy,
        backtester_factory=lambda policy: Backtester(initial_capital=50_000, overlay_policy=policy),
        benchmark_factory=lambda _train_df, _train_result: NullOverlayPolicy(),
        ml_policy_factory=make_ml_policy_factory(artifact),
        train_bars=100,
        test_bars=80,
        embargo_bars=5,
    )

    portfolio = evaluate_portfolio_overlay_walk_forward(
        reports_by_asset={"ea_a": report_a, "ea_b": report_b}
    )

    assert not portfolio.split_metrics.empty
    assert not portfolio.aggregate_metrics.empty
    assert set(portfolio.metadata["asset_ids"]) == {"ea_a", "ea_b"}
    assert set(portfolio.aggregate_metrics["overlay"]) == {"Baseline", "Benchmark", "ML Overlay"}
    assert "validation" in portfolio.metadata
