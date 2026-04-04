from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from engine.policy import (
    CompositeOverlayPolicy,
    DrawdownThrottlePolicy,
    ModelOverlayPolicy,
    NullOverlayPolicy,
    POLICY_FEATURE_NAMES,
    VolatilityTargetPolicy,
)
from research.experiment_registry import list_experiment_records, save_overlay_snapshot
from research.meta_models import (
    OverlayResearchArtifact,
    fit_filter_model,
    fit_sizing_model,
    predict_filter_probabilities,
    predict_sizing_signal,
)
from research.overlay_evaluation import (
    build_benchmark_policy,
    evaluate_overlay_walk_forward_suite,
    evaluate_portfolio_overlay_walk_forward,
    make_ml_policy_factory,
)
from research.trade_dataset import build_trade_dataset
from research.state_store import (
    canonical_hash,
    load_artifact_snapshot,
    load_dataset_snapshot,
    load_experiment_manifest,
    save_artifact_snapshot,
    save_dataset_snapshot,
    text_hash,
)
from services.backtest_service import backtest_result_payload, run_backtest
from services.conversion_service import convert_ea_source, normalize_symbol, persist_converted_ea
from ui.components.common import empty_state, error_box, section_header
from ui.state import autosave


ROOT = Path(__file__).resolve().parent.parent.parent


def _stable_hash(payload) -> str:
    return canonical_hash(payload)


def _strategy_version_record(ea: dict) -> dict:
    return {
        "ea_id": ea.get("id", ""),
        "name": ea.get("name", ""),
        "symbol": ea.get("symbol", ""),
        "timeframe": ea.get("timeframe", ""),
        "source_hash": text_hash(ea.get("source")),
        "engine_source_hash": text_hash(ea.get("engine_source")),
        "review_source_hash": text_hash(ea.get("review_source")),
        "strategy_module": ea.get("strategy_module", ""),
        "strategy_class": ea.get("strategy_class", ""),
    }


def _strategy_snapshot_record(ea: dict) -> dict:
    return {
        **_strategy_version_record(ea),
        "source": ea.get("source", ""),
        "engine_source": ea.get("engine_source", ""),
        "review_source": ea.get("review_source", ""),
        "strategy_path": ea.get("strategy_path", ""),
    }


def _backtest_input_record(ea: dict, *, date_from: str, date_to: str, intrabar_steps: int, execution_config: dict | None = None) -> dict:
    return {
        "ea_id": ea.get("id", ""),
        "name": ea.get("name", ""),
        "symbol": ea.get("symbol", ""),
        "timeframe": ea.get("timeframe", ""),
        "date_from": str(date_from),
        "date_to": str(date_to),
        "intrabar_steps": int(intrabar_steps),
        "execution_config": dict(execution_config or {}),
        "strategy_version": _strategy_version_record(ea),
    }


def _dataset_feature_config(dataset: pd.DataFrame) -> dict:
    feature_columns = [str(col) for col in dataset.columns if str(col).startswith("feature_")]
    return {
        "policy_feature_names": list(POLICY_FEATURE_NAMES),
        "dataset_feature_columns": feature_columns,
        "feature_config_hash": _stable_hash(
            {
                "policy_feature_names": list(POLICY_FEATURE_NAMES),
                "dataset_feature_columns": feature_columns,
            }
        ),
    }


def _current_dataset_snapshot() -> dict | None:
    dataset_id = st.session_state.get("_ai_dataset_id", "")
    return load_dataset_snapshot(dataset_id) if dataset_id else None


def _current_artifact_snapshot() -> dict | None:
    artifact_id = st.session_state.get("_ai_training_artifact_id", "")
    return load_artifact_snapshot(artifact_id) if artifact_id else None


def _report_artifacts(report, artifact: dict | None = None) -> dict:
    metadata = dict(getattr(report, "metadata", {}) or {})
    artifacts = {}
    for key in ["window_details", "trade_logs", "equity_curves", "calibration"]:
        value = metadata.get(key)
        if isinstance(value, pd.DataFrame):
            artifacts[key] = value
    validation = metadata.get("validation", {})
    if artifact:
        validation = dict(validation)
        validation.setdefault("feature_importance", [])
        if not validation["feature_importance"]:
            for model_key, label in [("filter_model", "filter"), ("sizing_model", "sizing")]:
                model = artifact.get(model_key, {})
                for feature, coef in zip(model.get("feature_names", []), model.get("coefficients", []), strict=False):
                    validation["feature_importance"].append(
                        {
                            "model": label,
                            "feature": str(feature),
                            "coefficient": float(coef),
                            "abs_coefficient": abs(float(coef)),
                        }
                    )
    if validation:
        artifacts["validation"] = validation
    return artifacts


def render():
    section_header(
        "AI Training",
        "Decision-level research workflow: build trade datasets, benchmark overlay policies, and train ML meta-filter/sizing models.",
    )

    eas = st.session_state.get("eas", {})
    if not eas:
        empty_state("Load at least one EA first so the decision research workflow has something to run.", "🧠")
        return

    tab_dataset, tab_benchmarks, tab_models = st.tabs(
        [
            "🗂️ Trade Dataset",
            "📊 Benchmark Overlays",
            "🤖 ML Meta-Models",
        ]
    )

    with tab_dataset:
        _render_dataset_tab(eas)
    with tab_benchmarks:
        _render_benchmark_tab(eas)
    with tab_models:
        _render_model_tab(eas)


def _default_date_range(selected_ids: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    results = st.session_state.get("backtest_results", {})
    defaults: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for ea_id in selected_ids:
        summary = results.get(ea_id, {}).get("summary", {})
        try:
            start = pd.Timestamp(summary.get("date_from"))
            end = pd.Timestamp(summary.get("date_to"))
        except Exception:
            continue
        if pd.notna(start) and pd.notna(end):
            defaults.append((start, end))
    if defaults:
        return min(x[0] for x in defaults), max(x[1] for x in defaults)
    return pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")


def _load_strategy_class(ea: dict):
    if not ea:
        raise ValueError("EA is missing.")

    strategy_module = ea.get("strategy_module")
    strategy_class = ea.get("strategy_class")
    if not strategy_module or not strategy_class:
        refreshed = convert_ea_source(
            source=ea["source"],
            strategy_name=ea["name"],
            symbol=normalize_symbol(ea["symbol"]),
            timeframe=ea["timeframe"],
            ea_id=ea["id"],
        )
        paths = persist_converted_ea(refreshed, ea["name"])
        ea["strategy_module"] = refreshed.strategy_module
        ea["strategy_class"] = refreshed.strategy_class
        ea["strategy_path"] = paths["engine_path"]
        strategy_module = refreshed.strategy_module
        strategy_class = refreshed.strategy_class

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module = importlib.reload(sys.modules[strategy_module]) if strategy_module in sys.modules else importlib.import_module(strategy_module)
    return getattr(module, strategy_class)


def _load_frame_from_record(record: dict, key: str, preview_key: str) -> pd.DataFrame:
    files = record.get("files", {}) or {}
    path = files.get(key)
    if path and Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    preview = record.get(preview_key, [])
    return pd.DataFrame(preview)


def _run_dataset_backtests(
    eas: dict,
    selected_ids: list[str],
    *,
    date_from: str,
    date_to: str,
    intrabar_steps: int,
    execution_config: dict | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    raw_results: dict[str, object] = {}
    dataset_frames: list[pd.DataFrame] = []

    for ea_id in selected_ids:
        ea = eas[ea_id]
        strat_cls = _load_strategy_class(ea)
        result = run_backtest(
            strat_cls,
            symbol=ea["symbol"],
            timeframe=ea["timeframe"],
            date_from=date_from,
            date_to=date_to,
            intrabar_steps=intrabar_steps,
            execution_config=execution_config,
        )
        raw_results[ea_id] = result

        payload = backtest_result_payload(result)
        payload["meta"] = {
            "ea_id": ea_id,
            "name": ea["name"],
            "symbol": ea["symbol"],
            "timeframe": ea["timeframe"],
        }
        st.session_state.setdefault("backtest_results", {})[ea_id] = payload

        trade_dataset = build_trade_dataset(result)
        if trade_dataset.empty:
            continue
        trade_dataset = trade_dataset.copy()
        trade_dataset["ea_id"] = ea_id
        trade_dataset["name"] = ea["name"]
        trade_dataset["symbol"] = ea["symbol"]
        trade_dataset["timeframe"] = ea["timeframe"]
        dataset_frames.append(trade_dataset)

    combined = pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame()
    return raw_results, combined


def _dataset_summary(dataset: pd.DataFrame) -> dict[str, float | int | str]:
    if dataset.empty:
        return {}
    return {
        "strategies": int(dataset["ea_id"].nunique()),
        "decisions": int(len(dataset)),
        "trades_taken": int(dataset["trade_taken"].sum()),
        "trade_rate": float(dataset["trade_taken"].mean()),
        "win_rate": float((dataset["realized_net_profit"] > 0).mean()),
        "avg_profit": float(dataset["realized_net_profit"].mean()),
        "date_from": str(pd.to_datetime(dataset["time"]).min().date()),
        "date_to": str(pd.to_datetime(dataset["time"]).max().date()),
    }


def _render_dataset_metrics(dataset: pd.DataFrame):
    summary = _dataset_summary(dataset)
    if not summary:
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strategies", summary["strategies"])
    c2.metric("Decisions", f"{summary['decisions']:,}")
    c3.metric("Trade Rate", f"{summary['trade_rate'] * 100:.1f}%")
    c4.metric("Decision Win Rate", f"{summary['win_rate'] * 100:.1f}%")
    c5.metric("Avg Net Profit", f"${summary['avg_profit']:,.2f}")


def _dataset_scope_selector(eas: dict, *, key_prefix: str) -> tuple[list[str], pd.Timestamp, pd.Timestamp, int]:
    options = {ea_id: f"{ea['name']} | {ea['symbol']} {ea['timeframe']}" for ea_id, ea in eas.items()}
    default_ids = list(options.keys())[: min(3, len(options))]
    selected_ids = st.multiselect(
        "Strategies",
        list(options.keys()),
        default=default_ids,
        format_func=lambda ea_id: options[ea_id],
        key=f"{key_prefix}_eas",
    )
    default_from, default_to = _default_date_range(selected_ids or list(options.keys()))
    col1, col2, col3 = st.columns(3)
    with col1:
        date_from = st.date_input("From", value=default_from, key=f"{key_prefix}_from")
    with col2:
        date_to = st.date_input("To", value=default_to, key=f"{key_prefix}_to")
    with col3:
        use_intrabar = st.checkbox("Use M1 intrabar", value=False, key=f"{key_prefix}_intrabar")
    intrabar_steps = 60 if use_intrabar else 1
    return selected_ids, pd.Timestamp(date_from), pd.Timestamp(date_to), intrabar_steps


def _execution_settings_form(prefix: str) -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        commission_per_lot = st.number_input(
            "Commission / lot",
            value=float(settings.DEFAULT_COMMISSION_PER_LOT),
            step=0.1,
            key=f"{prefix}_commission",
        )
    with c2:
        spread_pips = st.number_input(
            "Spread",
            value=float(settings.DEFAULT_SPREAD_PIPS),
            step=0.1,
            key=f"{prefix}_spread",
        )
    with c3:
        slippage_pips = st.number_input(
            "Slippage",
            value=float(settings.DEFAULT_SLIPPAGE_PIPS),
            step=0.1,
            key=f"{prefix}_slippage",
        )
    return {
        "commission_per_lot": float(commission_per_lot),
        "spread_pips": float(spread_pips),
        "slippage_pips": float(slippage_pips),
    }


def _render_experiment_registry():
    registry = list_experiment_records()
    if not registry:
        return
    st.markdown("#### Experiment Registry")
    st.dataframe(pd.DataFrame(registry), use_container_width=True, hide_index=True)


def _save_overlay_report(
    *,
    title: str,
    metadata: dict,
    lineage: dict | None,
    report,
    portfolio_ids: list[str],
) -> dict:
    record, snapshot = save_overlay_snapshot(
        title=title,
        metadata=metadata,
        lineage=lineage,
        aggregate_metrics=report.aggregate_metrics,
        split_metrics=report.split_metrics,
        windows=report.windows,
        artifacts=_report_artifacts(report, metadata.get("artifact")),
    )
    st.session_state["experiment_registry"] = list_experiment_records()[:100]
    st.session_state["_ai_model_wfo"] = snapshot
    st.session_state["_ai_model_wfo_run_id"] = record["run_id"]
    return snapshot


def _render_dataset_tab(eas: dict):
    st.markdown("#### Build decision-level trade dataset")
    st.caption(
        "This reruns selected strategies and captures each trade decision with realized outcome labels. "
        "That becomes the canonical research dataset for overlays and ML."
    )

    selected_ids, date_from, date_to, intrabar_steps = _dataset_scope_selector(eas, key_prefix="ai_dataset")
    execution_config = _execution_settings_form("ai_dataset_exec")
    if st.button("Build Dataset", type="primary", use_container_width=True, key="ai_dataset_build"):
        if not selected_ids:
            error_box("Select at least one strategy.")
            return
        with st.spinner("Running baseline backtests and assembling the decision dataset..."):
            raw_results, dataset = _run_dataset_backtests(
                eas,
                selected_ids,
                date_from=str(date_from.date()),
                date_to=str(date_to.date()),
                intrabar_steps=intrabar_steps,
                execution_config=execution_config,
            )
        feature_config = _dataset_feature_config(dataset)
        dataset_snapshot = save_dataset_snapshot(
            dataset=dataset,
            metadata={
                "scope": {
                    "ea_ids": list(selected_ids),
                    "date_from": str(date_from.date()),
                    "date_to": str(date_to.date()),
                    "intrabar_steps": intrabar_steps,
                },
                "execution_config": execution_config,
                "strategy_versions": [_strategy_version_record(eas[ea_id]) for ea_id in selected_ids if ea_id in eas],
                "strategy_snapshots": [_strategy_snapshot_record(eas[ea_id]) for ea_id in selected_ids if ea_id in eas],
                "backtest_inputs": [
                    _backtest_input_record(
                        eas[ea_id],
                        date_from=str(date_from.date()),
                        date_to=str(date_to.date()),
                        intrabar_steps=intrabar_steps,
                        execution_config=execution_config,
                    )
                    for ea_id in selected_ids
                    if ea_id in eas
                ],
                "feature_config": feature_config,
            },
        )
        st.session_state["_ai_research_runs"] = raw_results
        st.session_state["_ai_trade_dataset"] = dataset
        st.session_state["_ai_dataset_id"] = dataset_snapshot["dataset_id"]
        st.session_state["_ai_dataset_scope"] = {
            "ea_ids": list(selected_ids),
            "date_from": str(date_from.date()),
            "date_to": str(date_to.date()),
            "intrabar_steps": intrabar_steps,
            "execution_config": execution_config,
            "dataset_id": dataset_snapshot["dataset_id"],
            "dataset_hash": dataset_snapshot["dataset_hash"],
            "feature_config": feature_config,
            "backtest_inputs": dataset_snapshot.get("metadata", {}).get("backtest_inputs", []),
        }
        autosave()

    dataset = st.session_state.get("_ai_trade_dataset", pd.DataFrame())
    if (dataset is None or dataset.empty) and st.session_state.get("_ai_dataset_id"):
        dataset_snapshot = _current_dataset_snapshot()
        if dataset_snapshot:
            dataset = dataset_snapshot["dataset"]
            st.session_state["_ai_trade_dataset"] = dataset
            scope = dict(st.session_state.get("_ai_dataset_scope", {}) or {})
            scope.setdefault("dataset_id", dataset_snapshot["dataset_id"])
            scope.setdefault("dataset_hash", dataset_snapshot["dataset_hash"])
            scope.setdefault("feature_config", dataset_snapshot.get("metadata", {}).get("feature_config", {}))
            st.session_state["_ai_dataset_scope"] = scope
    if dataset is None or dataset.empty:
        empty_state("Build a trade dataset to unlock overlay benchmarks and ML models.", "🗂️")
        return

    _render_dataset_metrics(dataset)
    grouped = (
        dataset.groupby(["name", "symbol", "timeframe"], as_index=False)
        .agg(
            decisions=("decision_id", "count"),
            trade_rate=("trade_taken", "mean"),
            decision_win_rate=("realized_win", "mean"),
            net_profit=("realized_net_profit", "sum"),
        )
        .sort_values("decisions", ascending=False)
    )
    st.markdown("#### Strategy coverage")
    st.dataframe(grouped, use_container_width=True, hide_index=True)

    st.markdown("#### Dataset preview")
    preview_cols = [
        "time",
        "name",
        "symbol",
        "direction",
        "order_type",
        "requested_lots",
        "trade_taken",
        "realized_trade_count",
        "realized_net_profit",
        "feature_realized_vol",
        "feature_balance_drawdown_pct",
        "policy_tag",
    ]
    existing_cols = [col for col in preview_cols if col in dataset.columns]
    st.dataframe(dataset[existing_cols], use_container_width=True, hide_index=True)

    st.markdown("#### Outcome distribution")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=dataset["realized_net_profit"],
            marker_color="#2E75B6",
            nbinsx=60,
            name="Realized net profit",
        )
    )
    fig.add_vline(x=0.0, line_color="#E74C3C", line_dash="dash")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _benchmark_settings_form(prefix: str) -> dict:
    c1, c2, c3, c4, c5 = st.columns(5)
    settings = {}
    with c1:
        settings["target_vol"] = st.number_input("Target vol", min_value=0.01, max_value=1.0, value=0.20, step=0.01, key=f"{prefix}_target_vol")
    with c2:
        settings["min_multiplier"] = st.number_input("Min mult", min_value=0.1, max_value=2.0, value=0.50, step=0.05, key=f"{prefix}_min_mult")
    with c3:
        settings["max_multiplier"] = st.number_input("Max mult", min_value=0.2, max_value=3.0, value=1.50, step=0.05, key=f"{prefix}_max_mult")
    with c4:
        settings["soft_dd"] = st.number_input("Soft DD %", min_value=0.1, max_value=50.0, value=5.0, step=0.1, key=f"{prefix}_soft_dd")
    with c5:
        settings["hard_dd"] = st.number_input("Hard DD %", min_value=0.2, max_value=80.0, value=10.0, step=0.1, key=f"{prefix}_hard_dd")
    settings["soft_multiplier"] = st.slider(
        "Soft DD multiplier",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        key=f"{prefix}_soft_dd_mult",
    )
    return settings


def _result_row(label: str, result) -> dict[str, float | str | int]:
    summary = result.summary()
    return {
        "overlay": label,
        "net_profit": float(summary["net_profit"]),
        "profit_factor": float(summary["profit_factor"]),
        "max_drawdown_pct": float(summary["max_drawdown_pct"]),
        "win_rate": float(summary["win_rate"]),
        "n_trades": int(summary["n_trades"]),
        "n_decisions": int(summary["n_decisions"]),
    }


def _run_walk_forward_comparison(
    ea: dict,
    artifact: dict,
    *,
    date_from: str,
    date_to: str,
    train_bars: int,
    test_bars: int,
    embargo_bars: int,
    execution_config: dict | None = None,
):
    from engine.backtester import Backtester
    from engine.data_loader import load_bars

    strat_cls = _load_strategy_class(ea)
    bars = load_bars(ea["symbol"], ea["timeframe"], date_from=date_from, date_to=date_to)
    if bars is None or bars.empty:
        raise ValueError("No bars available for the selected walk-forward range.")

    lot_value = getattr(strat_cls, "lot_value", 1.0)
    return evaluate_overlay_walk_forward_suite(
        df=bars,
        strategy_class=strat_cls,
        backtester_factory=lambda policy: Backtester(
            initial_capital=100_000,
            commission_per_lot=float((execution_config or {}).get("commission_per_lot", settings.DEFAULT_COMMISSION_PER_LOT)),
            spread_pips=float((execution_config or {}).get("spread_pips", settings.DEFAULT_SPREAD_PIPS)),
            slippage_pips=float((execution_config or {}).get("slippage_pips", settings.DEFAULT_SLIPPAGE_PIPS)),
            lot_value=lot_value,
            overlay_policy=policy,
        ),
        benchmark_factory=lambda _train_df, _train_result: build_benchmark_policy(
            artifact["benchmark_name"],
            artifact["benchmark_settings"],
        ),
        ml_policy_factory=make_ml_policy_factory(artifact),
        train_bars=int(train_bars),
        test_bars=int(test_bars),
        embargo_bars=int(embargo_bars),
    )


def _render_benchmark_tab(eas: dict):
    st.markdown("#### Benchmark overlay lab")
    st.caption(
        "Run deterministic overlay policies on top of the same strategies before introducing ML. "
        "This gives you a clean baseline for what simple vol-targeting and drawdown throttles already achieve."
    )

    scope = st.session_state.get("_ai_dataset_scope")
    if not scope:
        empty_state("Build a trade dataset first so the benchmark lab knows what strategy/date scope to use.", "📊")
        return

    ea_options = {ea_id: f"{eas[ea_id]['name']} | {eas[ea_id]['symbol']} {eas[ea_id]['timeframe']}" for ea_id in scope["ea_ids"] if ea_id in eas}
    selected_ea_id = st.selectbox(
        "Benchmark strategy",
        list(ea_options.keys()),
        format_func=lambda ea_id: ea_options[ea_id],
        key="ai_benchmark_ea",
    )
    benchmark_name = st.selectbox(
        "Overlay set",
        ["vol_target", "drawdown_throttle", "composite"],
        format_func=lambda name: {
            "vol_target": "Volatility targeting",
            "drawdown_throttle": "Drawdown throttle",
            "composite": "Composite benchmark",
        }[name],
        key="ai_benchmark_name",
    )
    settings = _benchmark_settings_form("ai_benchmark")
    execution_config = _execution_settings_form("ai_benchmark_exec")

    if st.button("Run Benchmark Overlay", type="primary", use_container_width=True, key="ai_benchmark_run"):
        ea = eas[selected_ea_id]
        strat_cls = _load_strategy_class(ea)
        with st.spinner("Running baseline and benchmark overlay backtests..."):
            baseline = run_backtest(
                strat_cls,
                symbol=ea["symbol"],
                timeframe=ea["timeframe"],
                date_from=scope["date_from"],
                date_to=scope["date_to"],
                intrabar_steps=int(scope["intrabar_steps"]),
                execution_config=execution_config,
            )
            overlay = build_benchmark_policy(benchmark_name, settings)
            benchmark = run_backtest(
                strat_cls,
                symbol=ea["symbol"],
                timeframe=ea["timeframe"],
                date_from=scope["date_from"],
                date_to=scope["date_to"],
                intrabar_steps=int(scope["intrabar_steps"]),
                overlay_policy=overlay,
                execution_config=execution_config,
            )
        st.session_state["_ai_benchmark_results"] = {
            "strategy_id": selected_ea_id,
            "benchmark_name": benchmark_name,
            "settings": settings,
            "execution_config": execution_config,
            "baseline": baseline,
            "benchmark": benchmark,
        }

    stored = st.session_state.get("_ai_benchmark_results")
    if not stored:
        empty_state("Run a benchmark overlay to compare it against the base strategy.", "📈")
        return

    comparison = pd.DataFrame(
        [
            _result_row("Baseline", stored["baseline"]),
            _result_row("Benchmark", stored["benchmark"]),
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    curve = go.Figure()
    for label, result, color in [
        ("Baseline", stored["baseline"], "#2E75B6"),
        ("Benchmark", stored["benchmark"], "#2ECC71"),
    ]:
        equity = result.equity_curve
        curve.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name=label, line=dict(color=color, width=2)))
    curve.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Equity ($)",
    )
    st.plotly_chart(curve, use_container_width=True)

    decision_frame = build_trade_dataset(stored["benchmark"])
    if not decision_frame.empty:
        decision_frame = decision_frame.copy()
        st.markdown("#### Benchmark decision tags")
        tags = decision_frame["policy_tag"].fillna("unknown").value_counts().rename_axis("policy_tag").reset_index(name="count")
        st.dataframe(tags, use_container_width=True, hide_index=True)


def _coef_table(model: dict) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": model.get("feature_names", []),
                "coefficient": model.get("coefficients", []),
            }
        )
        .assign(abs_coef=lambda df: df["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns=["abs_coef"])
    )


def _render_model_tab(eas: dict):
    st.markdown("#### ML meta-filter and sizing models")
    st.caption(
        "Train a probability-based trade filter and a linear sizing model directly on the decision dataset, "
        "then backtest them as an overlay on top of either baseline execution or a benchmark policy."
    )

    dataset = st.session_state.get("_ai_trade_dataset", pd.DataFrame())
    scope = st.session_state.get("_ai_dataset_scope")
    if (dataset is None or dataset.empty) and st.session_state.get("_ai_dataset_id"):
        dataset_snapshot = _current_dataset_snapshot()
        if dataset_snapshot:
            dataset = dataset_snapshot["dataset"]
            st.session_state["_ai_trade_dataset"] = dataset
            scope = dict(scope or {})
            scope.setdefault("dataset_id", dataset_snapshot["dataset_id"])
            scope.setdefault("dataset_hash", dataset_snapshot["dataset_hash"])
            scope.setdefault("feature_config", dataset_snapshot.get("metadata", {}).get("feature_config", {}))
            st.session_state["_ai_dataset_scope"] = scope
    if dataset is None or dataset.empty or not scope:
        empty_state("Build a trade dataset first.", "🤖")
        return

    options = {ea_id: f"{ea['name']} | {ea['symbol']} {ea['timeframe']}" for ea_id, ea in eas.items() if ea_id in scope["ea_ids"]}
    eligible_ids = list(options.keys())
    selected_train_ids = st.multiselect(
        "Training scope",
        eligible_ids,
        default=eligible_ids,
        format_func=lambda ea_id: options[ea_id],
        key="ai_model_train_ids",
    )
    train_df = dataset[dataset["ea_id"].isin(selected_train_ids)].copy() if selected_train_ids else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        positive_cutoff = st.number_input("Filter profit cutoff", value=0.0, step=10.0, key="ai_model_cutoff")
    with c2:
        probability_threshold = st.slider("Filter threshold", min_value=0.50, max_value=0.90, value=0.55, step=0.01, key="ai_model_prob")
    with c3:
        min_multiplier = st.number_input("Min size mult", min_value=0.1, max_value=2.0, value=0.50, step=0.05, key="ai_model_min_mult")
    with c4:
        max_multiplier = st.number_input("Max size mult", min_value=0.2, max_value=3.0, value=1.50, step=0.05, key="ai_model_max_mult")

    benchmark_choice = st.selectbox(
        "Base overlay under the ML models",
        ["baseline", "vol_target", "drawdown_throttle", "composite"],
        format_func=lambda name: {
            "baseline": "No benchmark overlay",
            "vol_target": "Volatility targeting",
            "drawdown_throttle": "Drawdown throttle",
            "composite": "Composite benchmark",
        }[name],
        key="ai_model_base_overlay",
    )
    benchmark_settings = _benchmark_settings_form("ai_model_base")
    execution_config = _execution_settings_form("ai_model_exec")

    if st.button("Train Meta-Models", type="primary", use_container_width=True, key="ai_model_train"):
        if train_df.empty:
            error_box("Select at least one strategy with decision data.")
            return
        with st.spinner("Fitting local filter and sizing models..."):
            filter_model = fit_filter_model(
                train_df,
                threshold=float(probability_threshold),
                positive_return_cutoff=float(positive_cutoff),
            )
            sizing_model = fit_sizing_model(
                train_df,
                min_multiplier=float(min_multiplier),
                max_multiplier=float(max_multiplier),
            )
        artifact = OverlayResearchArtifact(
            filter_model=filter_model,
            sizing_model=sizing_model,
            dataset_summary=_dataset_summary(train_df),
            benchmark_name=benchmark_choice,
            benchmark_settings=benchmark_settings,
        ).to_dict()
        artifact_snapshot = save_artifact_snapshot(
            artifact=artifact,
            metadata={
                "dataset_id": scope.get("dataset_id", ""),
                "dataset_hash": scope.get("dataset_hash", ""),
                "training_scope_ids": list(selected_train_ids),
                "feature_config": scope.get("feature_config", _dataset_feature_config(train_df)),
                "model_parameters": {
                    "positive_cutoff": float(positive_cutoff),
                    "probability_threshold": float(probability_threshold),
                    "min_multiplier": float(min_multiplier),
                    "max_multiplier": float(max_multiplier),
                },
                "benchmark_name": benchmark_choice,
                "benchmark_settings": benchmark_settings,
                "execution_config": execution_config,
            },
        )
        st.session_state["training_artifact"] = artifact
        st.session_state["_ai_training_artifact_id"] = artifact_snapshot["artifact_id"]
        st.session_state["training_log"] = [
            f"Dataset rows: {len(train_df)}",
            f"Strategies: {train_df['ea_id'].nunique()}",
            f"Filter accuracy: {filter_model['metrics']['accuracy']:.3f}",
            f"Sizing R2: {sizing_model['metrics']['r2']:.3f}",
        ]
        st.session_state["model_trained"] = True
        autosave()

    artifact = st.session_state.get("training_artifact")
    if not artifact and st.session_state.get("_ai_training_artifact_id"):
        artifact_snapshot = _current_artifact_snapshot()
        if artifact_snapshot:
            artifact = artifact_snapshot["artifact"]
            st.session_state["training_artifact"] = artifact
    if not artifact:
        empty_state("Train the models to inspect coefficients and run overlay comparisons.", "🧪")
        return

    filter_model = artifact["filter_model"]
    sizing_model = artifact["sizing_model"]
    probs = predict_filter_probabilities(train_df, filter_model) if not train_df.empty else pd.Series(dtype=float)
    signals = predict_sizing_signal(train_df, sizing_model) if not train_df.empty else pd.Series(dtype=float)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filter Accuracy", f"{filter_model['metrics']['accuracy'] * 100:.1f}%")
    c2.metric("Filter Precision", f"{filter_model['metrics']['precision'] * 100:.1f}%")
    c3.metric("Sizing R²", f"{sizing_model['metrics']['r2']:.3f}")
    c4.metric("Train Rows", f"{filter_model['sample_size']:,}")

    st.markdown("#### Model coefficients")
    left, right = st.columns(2)
    with left:
        st.caption("Meta-filter")
        st.dataframe(_coef_table(filter_model), use_container_width=True, hide_index=True)
    with right:
        st.caption("Sizing model")
        st.dataframe(_coef_table(sizing_model), use_container_width=True, hide_index=True)

    if not train_df.empty:
        scored = train_df.copy()
        scored["filter_probability"] = probs
        scored["sizing_signal"] = signals
        st.markdown("#### Training-set scores")
        st.dataframe(
            scored[
                [
                    "time",
                    "name",
                    "direction",
                    "requested_lots",
                    "realized_net_profit",
                    "filter_probability",
                    "sizing_signal",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    eval_ids = st.multiselect(
        "Walk-forward portfolio scope",
        eligible_ids,
        default=eligible_ids,
        format_func=lambda ea_id: options[ea_id],
        key="ai_model_wfo_ids",
    )
    st.markdown("#### Default validation: purged walk-forward")
    st.caption(
        "This is the primary validation path for the ML overlay. Each split refits the filter and sizing models on the train window, "
        "applies an embargo, and reports only out-of-sample test performance. When multiple strategies are selected, "
        "the app also computes a portfolio-level overlay comparison."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        train_bars = int(st.number_input("Train bars", min_value=80, max_value=5000, value=120, step=20, key="ai_model_wfo_train_bars"))
    with c2:
        test_bars = int(st.number_input("Test bars", min_value=60, max_value=2000, value=80, step=20, key="ai_model_wfo_test_bars"))
    with c3:
        embargo_bars = int(st.number_input("Embargo bars", min_value=0, max_value=500, value=5, step=1, key="ai_model_wfo_embargo_bars"))

    if st.button("Run Walk-Forward ML Evaluation", type="primary", use_container_width=True, key="ai_model_wfo_run"):
        scope_start = scope["date_from"]
        scope_end = scope["date_to"]
        selected_eval_ids = list(eval_ids or [])
        if not selected_eval_ids:
            error_box("Select at least one strategy for walk-forward evaluation.")
            return
        try:
            with st.spinner("Running purged walk-forward ML evaluation..."):
                reports_by_asset = {}
                for ea_id in selected_eval_ids:
                    reports_by_asset[ea_id] = _run_walk_forward_comparison(
                        eas[ea_id],
                        artifact,
                        date_from=scope_start,
                        date_to=scope_end,
                        train_bars=train_bars,
                        test_bars=test_bars,
                        embargo_bars=embargo_bars,
                        execution_config=execution_config,
                    )
                if len(reports_by_asset) == 1:
                    portfolio_report = next(iter(reports_by_asset.values()))
                else:
                    portfolio_report = evaluate_portfolio_overlay_walk_forward(reports_by_asset=reports_by_asset)
            if portfolio_report.split_metrics.empty:
                raise ValueError("Walk-forward run completed but produced no out-of-sample windows. Increase the date range or reduce train/test bars.")
            metadata = {
                "portfolio_ids": selected_eval_ids,
                "scope": {"date_from": scope_start, "date_to": scope_end},
                "train_bars": train_bars,
                "test_bars": test_bars,
                "embargo_bars": embargo_bars,
                "execution_config": execution_config,
                "benchmark_name": artifact["benchmark_name"],
                "benchmark_settings": artifact["benchmark_settings"],
                "artifact": artifact,
                "model_summary": {
                    "filter_threshold": artifact["filter_model"].get("probability_threshold"),
                    "positive_return_cutoff": artifact["filter_model"].get("positive_return_cutoff"),
                    "min_multiplier": artifact["sizing_model"].get("min_multiplier"),
                    "max_multiplier": artifact["sizing_model"].get("max_multiplier"),
                },
            }
            dataset_snapshot = _current_dataset_snapshot()
            artifact_snapshot = _current_artifact_snapshot()
            lineage = {
                "lineage_version": 1,
                "strategy_versions": [_strategy_version_record(eas[ea_id]) for ea_id in selected_eval_ids if ea_id in eas],
                "dataset_snapshot": {
                    "dataset_id": scope.get("dataset_id", ""),
                    "dataset_hash": scope.get("dataset_hash", ""),
                    "metadata_hash": dataset_snapshot.get("metadata_hash", "") if dataset_snapshot else "",
                },
                "dataset_backtest_inputs": dataset_snapshot.get("metadata", {}).get("backtest_inputs", []) if dataset_snapshot else scope.get("backtest_inputs", []),
                "feature_config": scope.get("feature_config", _dataset_feature_config(dataset)),
                "execution_assumptions": execution_config,
                "execution_assumptions_hash": _stable_hash(execution_config),
                "model_artifact": {
                    "artifact_id": artifact_snapshot["artifact_id"] if artifact_snapshot else "",
                    "artifact_hash": artifact_snapshot["artifact_hash"] if artifact_snapshot else _stable_hash(artifact),
                    "metadata_hash": artifact_snapshot.get("metadata_hash", "") if artifact_snapshot else "",
                },
            }
            title = "Portfolio OOS Overlay Evaluation" if len(selected_eval_ids) > 1 else f"OOS Overlay Evaluation - {options[selected_eval_ids[0]]}"
            _save_overlay_report(
                title=title,
                metadata=metadata,
                lineage=lineage,
                report=portfolio_report,
                portfolio_ids=selected_eval_ids,
            )
            autosave()
        except Exception as exc:
            error_box(f"Walk-forward evaluation failed: {exc}")

    walk_forward = st.session_state.get("_ai_model_wfo")
    if walk_forward is None and st.session_state.get("_ai_model_wfo_run_id"):
        manifest = load_experiment_manifest(st.session_state["_ai_model_wfo_run_id"])
        if manifest:
            walk_forward = {
                "run_id": manifest.get("run_id", ""),
                "title": manifest.get("title", ""),
                "portfolio_ids": manifest.get("metadata", {}).get("portfolio_ids", []),
                "export_dir": manifest.get("export_dir", ""),
                "files": manifest.get("files", {}),
                "metadata": manifest.get("metadata", {}),
                "lineage": manifest.get("lineage", {}),
                "split_metrics": _load_frame_from_record(manifest, "split_metrics", "split_preview"),
                "aggregate_metrics": _load_frame_from_record(manifest, "aggregate_metrics", "aggregate_preview"),
            }
            st.session_state["_ai_model_wfo"] = walk_forward
    if walk_forward:
        st.markdown("#### Aggregate OOS metrics")
        st.dataframe(walk_forward["aggregate_metrics"], use_container_width=True, hide_index=True)
        if walk_forward.get("export_dir"):
            st.caption(f"Export bundle: {walk_forward['export_dir']}")

        st.markdown("#### Split-level OOS metrics")
        st.dataframe(walk_forward["split_metrics"], use_container_width=True, hide_index=True)

        wf_fig = go.Figure()
        split_metrics = walk_forward["split_metrics"]
        colors = {
            "Baseline": "#2E75B6",
            "Benchmark": "#2ECC71",
            "ML Overlay": "#F39C12",
        }
        for overlay in ["Baseline", "Benchmark", "ML Overlay"]:
            overlay_rows = split_metrics[split_metrics["overlay"] == overlay]
            if overlay_rows.empty:
                continue
            wf_fig.add_trace(
                go.Scatter(
                    x=overlay_rows["window_id"],
                    y=overlay_rows["oos_net_profit"],
                    mode="lines+markers",
                    name=overlay,
                    line=dict(color=colors[overlay], width=2),
                )
            )
        wf_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Window",
            yaxis_title="OOS Net Profit ($)",
            showlegend=False,
        )
        st.plotly_chart(wf_fig, use_container_width=True)

        artifacts_path = (walk_forward.get("files", {}) or {}).get("artifacts", "")
        if artifacts_path and Path(artifacts_path).exists():
            artifacts = json.loads(Path(artifacts_path).read_text(encoding="utf-8"))
            validation = artifacts.get("validation", {})
            if validation:
                st.markdown("#### Validation Diagnostics")
                st.code(json.dumps(validation, indent=2, default=str), language="json")

    st.markdown("#### Secondary validation: single-range backtest")
    st.caption("Use this only as a spot check after reviewing the walk-forward results above.")
    eval_ea_id = st.selectbox(
        "Evaluation strategy",
        eligible_ids,
        format_func=lambda ea_id: options[ea_id],
        key="ai_model_eval_ea",
    )
    if st.button("Run ML Overlay Backtest", type="secondary", use_container_width=True, key="ai_model_eval_run"):
        ea = eas[eval_ea_id]
        strat_cls = _load_strategy_class(ea)
        base_policy = build_benchmark_policy(artifact["benchmark_name"], artifact["benchmark_settings"])
        ml_policy = ModelOverlayPolicy(
            filter_model=artifact["filter_model"],
            sizing_model=artifact["sizing_model"],
            base_policy=base_policy,
        )
        with st.spinner("Running baseline and ML overlay backtests..."):
            baseline = run_backtest(
                strat_cls,
                symbol=ea["symbol"],
                timeframe=ea["timeframe"],
                date_from=scope["date_from"],
                date_to=scope["date_to"],
                intrabar_steps=int(scope["intrabar_steps"]),
                execution_config=execution_config,
            )
            ml_result = run_backtest(
                strat_cls,
                symbol=ea["symbol"],
                timeframe=ea["timeframe"],
                date_from=scope["date_from"],
                date_to=scope["date_to"],
                intrabar_steps=int(scope["intrabar_steps"]),
                overlay_policy=ml_policy,
                execution_config=execution_config,
            )
        st.session_state["_ai_model_eval"] = {
            "ea_id": eval_ea_id,
            "baseline": baseline,
            "ml": ml_result,
            "base_overlay_name": artifact["benchmark_name"],
        }

    evaluation = st.session_state.get("_ai_model_eval")
    if not evaluation:
        return

    comparison = pd.DataFrame(
        [
            _result_row("Baseline", evaluation["baseline"]),
            _result_row("ML Overlay", evaluation["ml"]),
        ]
    )
    st.markdown("#### Backtest comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    curve = go.Figure()
    for label, result, color in [
        ("Baseline", evaluation["baseline"], "#2E75B6"),
        ("ML Overlay", evaluation["ml"], "#F39C12"),
    ]:
        equity = result.equity_curve
        curve.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name=label, line=dict(color=color, width=2)))
    curve.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Equity ($)",
    )
    st.plotly_chart(curve, use_container_width=True)

    decisions = build_trade_dataset(evaluation["ml"])
    if not decisions.empty:
        st.markdown("#### ML overlay decisions")
        decision_cols = [
            "time",
            "direction",
            "requested_lots",
            "final_lots",
            "allow_trade",
            "policy_tag",
            "policy_filter_probability",
            "policy_predicted_profit",
            "policy_sizing_multiplier",
            "realized_net_profit",
        ]
        shown = [col for col in decision_cols if col in decisions.columns]
        st.dataframe(decisions[shown], use_container_width=True, hide_index=True)

    st.caption("Saved runs and reproduction controls now live in the `Experiments` page.")
