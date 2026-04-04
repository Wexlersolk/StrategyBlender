from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from engine.policy import (
    CompositeOverlayPolicy,
    DrawdownThrottlePolicy,
    ModelOverlayPolicy,
    NullOverlayPolicy,
    OverlayPolicy,
    VolatilityTargetPolicy,
)
from engine.results import BacktestResults
from research.meta_models import fit_filter_model, fit_sizing_model
from research.purged_walk_forward import PurgedSplit, generate_purged_splits
from research.trade_dataset import build_trade_dataset


@dataclass
class OverlayEvaluationWindow:
    overlay_label: str
    split: PurgedSplit
    train_result: BacktestResults
    test_result: BacktestResults
    test_dataset: pd.DataFrame
    train_dataset: pd.DataFrame = field(default_factory=pd.DataFrame)
    asset_id: str = ""


@dataclass
class OverlayEvaluationReport:
    windows: list[OverlayEvaluationWindow]
    split_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def build_benchmark_policy(name: str, settings: dict) -> OverlayPolicy:
    if name == "baseline":
        return NullOverlayPolicy()
    if name == "vol_target":
        return VolatilityTargetPolicy(
            target_vol=float(settings["target_vol"]),
            min_multiplier=float(settings["min_multiplier"]),
            max_multiplier=float(settings["max_multiplier"]),
        )
    if name == "drawdown_throttle":
        return DrawdownThrottlePolicy(
            soft_drawdown_pct=float(settings["soft_dd"]),
            hard_drawdown_pct=float(settings["hard_dd"]),
            soft_multiplier=float(settings["soft_multiplier"]),
        )
    if name == "composite":
        return CompositeOverlayPolicy(
            [
                VolatilityTargetPolicy(
                    target_vol=float(settings["target_vol"]),
                    min_multiplier=float(settings["min_multiplier"]),
                    max_multiplier=float(settings["max_multiplier"]),
                ),
                DrawdownThrottlePolicy(
                    soft_drawdown_pct=float(settings["soft_dd"]),
                    hard_drawdown_pct=float(settings["hard_dd"]),
                    soft_multiplier=float(settings["soft_multiplier"]),
                ),
            ]
        )
    raise ValueError(f"Unknown benchmark policy: {name}")


def make_ml_policy_factory(artifact: dict) -> Callable[[pd.DataFrame, BacktestResults], OverlayPolicy]:
    filter_template = artifact["filter_model"]
    sizing_template = artifact["sizing_model"]
    benchmark_name = artifact["benchmark_name"]
    benchmark_settings = artifact["benchmark_settings"]

    def _factory(_train_df: pd.DataFrame, train_result: BacktestResults) -> OverlayPolicy:
        train_dataset = build_trade_dataset(train_result)
        filter_model = fit_filter_model(
            train_dataset,
            threshold=float(filter_template["probability_threshold"]),
            positive_return_cutoff=float(filter_template.get("positive_return_cutoff", 0.0)),
        )
        sizing_model = fit_sizing_model(
            train_dataset,
            min_multiplier=float(sizing_template["min_multiplier"]),
            max_multiplier=float(sizing_template["max_multiplier"]),
        )
        return ModelOverlayPolicy(
            filter_model=filter_model,
            sizing_model=sizing_model,
            base_policy=build_benchmark_policy(benchmark_name, benchmark_settings),
        )

    return _factory


def _split_metric_row(window: OverlayEvaluationWindow) -> dict[str, float | int | str]:
    test_summary = window.test_result.summary()
    test_dataset = window.test_dataset if isinstance(window.test_dataset, pd.DataFrame) else pd.DataFrame()
    trade_rate = float(test_dataset["trade_taken"].mean()) if not test_dataset.empty else 0.0
    decision_win_rate = (
        float((pd.to_numeric(test_dataset["realized_net_profit"], errors="coerce").fillna(0.0) > 0).mean())
        if not test_dataset.empty
        else 0.0
    )
    avg_probability = (
        float(pd.to_numeric(test_dataset["policy_filter_probability"], errors="coerce").mean())
        if not test_dataset.empty and "policy_filter_probability" in test_dataset.columns
        else 0.0
    )
    return {
        "overlay": window.overlay_label,
        "window_id": int(window.split.window_id),
        "train_start": str(window.split.train_start.date()),
        "train_end": str(window.split.train_end.date()),
        "test_start": str(window.split.test_start.date()),
        "test_end": str(window.split.test_end.date()),
        "oos_net_profit": float(test_summary["net_profit"]),
        "oos_profit_factor": float(test_summary["profit_factor"]),
        "oos_max_drawdown_pct": float(test_summary["max_drawdown_pct"]),
        "oos_win_rate": float(test_summary["win_rate"]),
        "oos_trades": int(test_summary["n_trades"]),
        "oos_decisions": int(test_summary["n_decisions"]),
        "oos_trade_rate": trade_rate,
        "oos_decision_win_rate": decision_win_rate,
        "avg_filter_probability": avg_probability,
    }


def _aggregate_split_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    grouped = rows.groupby("overlay", as_index=False).agg(
        windows=("window_id", "count"),
        total_oos_net_profit=("oos_net_profit", "sum"),
        avg_oos_net_profit=("oos_net_profit", "mean"),
        median_oos_net_profit=("oos_net_profit", "median"),
        win_window_rate=("oos_net_profit", lambda x: float((x > 0).mean())),
        avg_oos_profit_factor=("oos_profit_factor", "mean"),
        avg_oos_max_drawdown_pct=("oos_max_drawdown_pct", "mean"),
        total_oos_trades=("oos_trades", "sum"),
        total_oos_decisions=("oos_decisions", "sum"),
        avg_oos_trade_rate=("oos_trade_rate", "mean"),
        avg_oos_decision_win_rate=("oos_decision_win_rate", "mean"),
        avg_filter_probability=("avg_filter_probability", "mean"),
    )
    return grouped.sort_values("overlay").reset_index(drop=True)


def _regime_label(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "unknown"
    if "close" in frame.columns:
        base_series = pd.to_numeric(frame["close"], errors="coerce")
        returns = base_series.pct_change().dropna()
    elif "realized_net_profit" in frame.columns:
        returns = pd.to_numeric(frame["realized_net_profit"], errors="coerce").fillna(0.0)
    else:
        returns = pd.Series(dtype=float)
    if returns.empty:
        return "unknown"
    drift = float(returns.mean())
    vol = float(returns.std(ddof=0))
    direction = "uptrend" if drift > 0 else "downtrend"
    volatility = "high_vol" if vol > returns.median() else "low_vol"
    return f"{direction}|{volatility}"


def _result_trade_log(result: BacktestResults, *, overlay: str, window_id: int, asset_id: str) -> pd.DataFrame:
    rows = []
    for trade in result.trades:
        rows.append(
            {
                "asset_id": asset_id,
                "overlay": overlay,
                "window_id": int(window_id),
                "trade_id": trade.id,
                "decision_id": trade.decision_id,
                "opened_time": trade.opened_time,
                "closed_time": trade.closed_time,
                "direction": trade.direction.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "lots": trade.lots,
                "net_profit": trade.net_profit,
                "close_reason": trade.close_reason.value,
                "comment": trade.comment,
            }
        )
    return pd.DataFrame(rows)


def _result_equity_curve(result: BacktestResults, *, overlay: str, window_id: int, asset_id: str) -> pd.DataFrame:
    curve = result.equity_curve
    if curve is None or curve.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "asset_id": asset_id,
            "overlay": overlay,
            "window_id": int(window_id),
            "time": curve.index,
            "equity": curve.values,
        }
    )


def _window_detail_row(window: OverlayEvaluationWindow) -> dict[str, float | int | str]:
    test_dataset = window.test_dataset if isinstance(window.test_dataset, pd.DataFrame) else pd.DataFrame()
    train_dataset = window.train_dataset if isinstance(window.train_dataset, pd.DataFrame) else pd.DataFrame()
    return {
        "asset_id": window.asset_id,
        "overlay": window.overlay_label,
        "window_id": int(window.split.window_id),
        "train_rows": int(len(train_dataset)),
        "test_rows": int(len(test_dataset)),
        "train_trades": int(window.train_result.n_trades),
        "test_trades": int(window.test_result.n_trades),
        "train_regime": _regime_label(train_dataset),
        "test_regime": _regime_label(test_dataset),
        "train_net_profit": float(window.train_result.net_profit),
        "test_net_profit": float(window.test_result.net_profit),
    }


def _calibration_frame(windows: list[OverlayEvaluationWindow]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window in windows:
        if window.overlay_label != "ML Overlay":
            continue
        dataset = window.test_dataset
        if dataset is None or dataset.empty or "policy_filter_probability" not in dataset.columns:
            continue
        frame = dataset.copy()
        frame["asset_id"] = window.asset_id
        frame["window_id"] = int(window.split.window_id)
        frame["probability"] = pd.to_numeric(frame["policy_filter_probability"], errors="coerce")
        frame["realized_win"] = (pd.to_numeric(frame["realized_net_profit"], errors="coerce").fillna(0.0) > 0).astype(int)
        frame = frame.dropna(subset=["probability"])
        if not frame.empty:
            rows.append(frame[["asset_id", "window_id", "probability", "realized_win"]])
    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined["bin"] = pd.cut(combined["probability"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
    return (
        combined.groupby("bin", as_index=False, observed=False)
        .agg(
            samples=("probability", "count"),
            avg_predicted_probability=("probability", "mean"),
            realized_win_rate=("realized_win", "mean"),
        )
        .assign(calibration_gap=lambda df: df["realized_win_rate"] - df["avg_predicted_probability"])
    )


def _feature_importance(artifact: dict | None) -> list[dict]:
    if not isinstance(artifact, dict):
        return []
    rows: list[dict] = []
    for model_key, label in [("filter_model", "filter"), ("sizing_model", "sizing")]:
        model = artifact.get(model_key, {})
        features = list(model.get("feature_names", []))
        coefs = list(model.get("coefficients", []))
        for feature, coef in zip(features, coefs, strict=False):
            rows.append(
                {
                    "model": label,
                    "feature": str(feature),
                    "coefficient": float(coef),
                    "abs_coefficient": abs(float(coef)),
                }
            )
    rows.sort(key=lambda row: (row["model"], -row["abs_coefficient"], row["feature"]))
    return rows


def _validation_diagnostics(
    *,
    split_metrics: pd.DataFrame,
    splits: list[PurgedSplit],
    artifact: dict | None = None,
    windows: list[OverlayEvaluationWindow] | None = None,
) -> dict:
    leakage_violations = 0
    for split in splits:
        if split.train_end >= split.test_start:
            leakage_violations += 1
        if split.embargo_end is not None and split.embargo_end >= split.test_start:
            leakage_violations += 1

    benchmark_present = bool(not split_metrics.empty and "Benchmark" in set(split_metrics["overlay"]))
    regime_summary = {}
    window_details = pd.DataFrame([_window_detail_row(window) for window in (windows or [])])
    if not window_details.empty:
        regime_summary = (
            window_details.groupby(["overlay", "test_regime"], as_index=False)
            .agg(
                windows=("window_id", "count"),
                avg_test_net_profit=("test_net_profit", "mean"),
            )
            .to_dict(orient="records")
        )

    outliers: list[dict] = []
    if not split_metrics.empty:
        for overlay, frame in split_metrics.groupby("overlay"):
            profits = pd.to_numeric(frame["oos_net_profit"], errors="coerce").fillna(0.0)
            q1 = float(profits.quantile(0.25))
            q3 = float(profits.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            flagged = frame[(profits < lower) | (profits > upper)]
            for _, row in flagged.iterrows():
                outliers.append(
                    {
                        "overlay": overlay,
                        "window_id": int(row["window_id"]),
                        "oos_net_profit": float(row["oos_net_profit"]),
                    }
                )

    return {
        "leakage_checks": {
            "windows_checked": len(splits),
            "violations": int(leakage_violations),
            "passed": leakage_violations == 0,
        },
        "walk_forward_integrity": {
            "window_count": len(splits),
            "has_windows": len(splits) > 0,
            "benchmark_present": benchmark_present,
            "baseline_present": bool(not split_metrics.empty and "Baseline" in set(split_metrics["overlay"])),
            "ml_overlay_present": bool(not split_metrics.empty and "ML Overlay" in set(split_metrics["overlay"])),
        },
        "regime_summary": regime_summary,
        "outlier_windows": outliers,
        "feature_importance": _feature_importance(artifact),
    }


def evaluate_overlay_walk_forward_suite(
    *,
    df: pd.DataFrame,
    strategy_class: type,
    backtester_factory: Callable[[OverlayPolicy], object],
    benchmark_factory: Callable[[pd.DataFrame, BacktestResults], OverlayPolicy],
    ml_policy_factory: Callable[[pd.DataFrame, BacktestResults], OverlayPolicy],
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
) -> OverlayEvaluationReport:
    splits = generate_purged_splits(
        df.index,
        train_bars=int(train_bars),
        test_bars=int(test_bars),
        embargo_bars=int(embargo_bars),
    )

    windows: list[OverlayEvaluationWindow] = []
    for split in splits:
        train_df = df[(df.index >= split.train_start) & (df.index <= split.train_end)].copy()
        test_df = df[(df.index >= split.test_start) & (df.index <= split.test_end)].copy()
        if train_df.empty or test_df.empty:
            continue

        train_bt = backtester_factory(NullOverlayPolicy())
        train_result = train_bt.run(strategy_class(), train_df)
        train_dataset = build_trade_dataset(train_result)

        overlay_specs = [
            ("Baseline", NullOverlayPolicy()),
            ("Benchmark", benchmark_factory(train_df, train_result)),
            ("ML Overlay", ml_policy_factory(train_df, train_result)),
        ]

        for label, overlay in overlay_specs:
            test_bt = backtester_factory(overlay)
            test_result = test_bt.run(strategy_class(), test_df)
            windows.append(
                OverlayEvaluationWindow(
                    overlay_label=label,
                    split=split,
                    train_result=train_result,
                    test_result=test_result,
                    train_dataset=train_dataset,
                    test_dataset=build_trade_dataset(test_result),
                    asset_id="",
                )
            )

    split_metrics = pd.DataFrame([_split_metric_row(window) for window in windows])
    aggregate_metrics = _aggregate_split_metrics(split_metrics)
    window_details = pd.DataFrame([_window_detail_row(window) for window in windows])
    trade_logs = pd.concat(
        [_result_trade_log(window.test_result, overlay=window.overlay_label, window_id=window.split.window_id, asset_id=window.asset_id) for window in windows],
        ignore_index=True,
    ) if windows else pd.DataFrame()
    equity_curves = pd.concat(
        [_result_equity_curve(window.test_result, overlay=window.overlay_label, window_id=window.split.window_id, asset_id=window.asset_id) for window in windows],
        ignore_index=True,
    ) if windows else pd.DataFrame()
    calibration = _calibration_frame(windows)
    return OverlayEvaluationReport(
        windows=windows,
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_metrics,
        metadata={
            "window_details": window_details,
            "trade_logs": trade_logs,
            "equity_curves": equity_curves,
            "calibration": calibration,
            "validation": _validation_diagnostics(
                split_metrics=split_metrics,
                splits=splits,
                windows=windows,
            ),
        },
    )


def evaluate_portfolio_overlay_walk_forward(
    *,
    reports_by_asset: dict[str, OverlayEvaluationReport],
) -> OverlayEvaluationReport:
    split_frames: list[pd.DataFrame] = []
    window_rows: list[OverlayEvaluationWindow] = []

    for asset_id, report in reports_by_asset.items():
        if report.split_metrics is not None and not report.split_metrics.empty:
            frame = report.split_metrics.copy()
            frame["asset_id"] = asset_id
            split_frames.append(frame)
        for window in report.windows:
            window.asset_id = asset_id
            window_rows.append(window)

    if not split_frames:
        return OverlayEvaluationReport(
            windows=[],
            split_metrics=pd.DataFrame(),
            aggregate_metrics=pd.DataFrame(),
            metadata={"asset_ids": list(reports_by_asset.keys())},
        )

    portfolio_split = pd.concat(split_frames, ignore_index=True)
    portfolio_summary = (
        portfolio_split.groupby(["overlay", "window_id"], as_index=False)
        .agg(
            assets=("asset_id", "nunique"),
            train_start=("train_start", "min"),
            train_end=("train_end", "max"),
            test_start=("test_start", "min"),
            test_end=("test_end", "max"),
            oos_net_profit=("oos_net_profit", "sum"),
            oos_profit_factor=("oos_profit_factor", "mean"),
            oos_max_drawdown_pct=("oos_max_drawdown_pct", "mean"),
            oos_win_rate=("oos_win_rate", "mean"),
            oos_trades=("oos_trades", "sum"),
            oos_decisions=("oos_decisions", "sum"),
            oos_trade_rate=("oos_trade_rate", "mean"),
            oos_decision_win_rate=("oos_decision_win_rate", "mean"),
            avg_filter_probability=("avg_filter_probability", "mean"),
        )
        .sort_values(["overlay", "window_id"])
        .reset_index(drop=True)
    )
    aggregate = _aggregate_split_metrics(portfolio_summary)
    window_details = pd.concat(
        [report.metadata.get("window_details", pd.DataFrame()) for report in reports_by_asset.values() if isinstance(report.metadata.get("window_details"), pd.DataFrame)],
        ignore_index=True,
    ) if reports_by_asset else pd.DataFrame()
    trade_logs = pd.concat(
        [report.metadata.get("trade_logs", pd.DataFrame()) for report in reports_by_asset.values() if isinstance(report.metadata.get("trade_logs"), pd.DataFrame)],
        ignore_index=True,
    ) if reports_by_asset else pd.DataFrame()
    equity_curves = pd.concat(
        [report.metadata.get("equity_curves", pd.DataFrame()) for report in reports_by_asset.values() if isinstance(report.metadata.get("equity_curves"), pd.DataFrame)],
        ignore_index=True,
    ) if reports_by_asset else pd.DataFrame()
    calibration = pd.concat(
        [report.metadata.get("calibration", pd.DataFrame()) for report in reports_by_asset.values() if isinstance(report.metadata.get("calibration"), pd.DataFrame)],
        ignore_index=True,
    ) if reports_by_asset else pd.DataFrame()
    return OverlayEvaluationReport(
        windows=window_rows,
        split_metrics=portfolio_summary,
        aggregate_metrics=aggregate,
        metadata={
            "asset_ids": list(reports_by_asset.keys()),
            "window_details": window_details,
            "trade_logs": trade_logs,
            "equity_curves": equity_curves,
            "calibration": calibration,
            "validation": _validation_diagnostics(
                split_metrics=portfolio_summary,
                splits=[window.split for window in window_rows],
                windows=window_rows,
            ),
        },
    )
