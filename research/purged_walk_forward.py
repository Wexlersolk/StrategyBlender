from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from engine.policy import NullOverlayPolicy, OverlayPolicy
from engine.results import BacktestResults
from research.trade_dataset import build_trade_dataset


@dataclass
class PurgedSplit:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_start: pd.Timestamp | None
    embargo_end: pd.Timestamp | None


@dataclass
class PurgedWindowResult:
    split: PurgedSplit
    train_result: BacktestResults
    test_result: BacktestResults
    overlay_name: str
    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame


def generate_purged_splits(
    index: pd.Index,
    *,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
) -> list[PurgedSplit]:
    dt_index = pd.DatetimeIndex(index)
    splits: list[PurgedSplit] = []
    start = 0
    window_id = 0

    while start + train_bars + embargo_bars + test_bars <= len(dt_index):
        train_start = dt_index[start]
        train_end = dt_index[start + train_bars - 1]
        test_start_idx = start + train_bars + embargo_bars
        test_end_idx = test_start_idx + test_bars - 1
        embargo_start = dt_index[start + train_bars] if embargo_bars > 0 else None
        embargo_end = dt_index[test_start_idx - 1] if embargo_bars > 0 else None
        splits.append(PurgedSplit(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=dt_index[test_start_idx],
            test_end=dt_index[test_end_idx],
            embargo_start=embargo_start,
            embargo_end=embargo_end,
        ))
        start += test_bars
        window_id += 1

    return splits


def run_purged_walk_forward(
    *,
    df: pd.DataFrame,
    strategy_class: type,
    backtester_factory: Callable[[OverlayPolicy], object],
    policy_factory: Callable[[pd.DataFrame, BacktestResults], OverlayPolicy] | None = None,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
) -> list[PurgedWindowResult]:
    splits = generate_purged_splits(
        df.index,
        train_bars=train_bars,
        test_bars=test_bars,
        embargo_bars=embargo_bars,
    )
    results: list[PurgedWindowResult] = []

    for split in splits:
        train_df = df[(df.index >= split.train_start) & (df.index <= split.train_end)].copy()
        test_df = df[(df.index >= split.test_start) & (df.index <= split.test_end)].copy()
        if train_df.empty or test_df.empty:
            continue

        train_bt = backtester_factory(NullOverlayPolicy())
        train_result = train_bt.run(strategy_class(), train_df)

        overlay = policy_factory(train_df, train_result) if policy_factory else NullOverlayPolicy()

        test_bt = backtester_factory(overlay)
        test_result = test_bt.run(strategy_class(), test_df)

        results.append(PurgedWindowResult(
            split=split,
            train_result=train_result,
            test_result=test_result,
            overlay_name=overlay.name,
            train_dataset=build_trade_dataset(train_result),
            test_dataset=build_trade_dataset(test_result),
        ))

    return results
