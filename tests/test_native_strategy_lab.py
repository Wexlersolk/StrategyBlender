import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.native_strategy_lab import (
    create_batch_run,
    default_scoring_profile_for_template,
    generate_batch_candidates,
    CATALOG_PATH,
    BATCH_RUNS_PATH,
    GENERATOR_RUNS_PATH,
    LAB_DIR,
    PRESET_DIR,
    default_parameter_mutation_space,
    default_promotion_policy_for_template,
    default_structural_mutation_space,
    default_wfo_param_grid_for_template,
    _score_candidate,
    evaluate_native_strategy,
    get_native_strategy_record,
    get_generator_run,
    list_generator_runs,
    rank_batch_candidates,
    regenerate_native_strategy_version,
    register_generated_strategy,
    run_generator_session,
    run_batch_generation,
    save_custom_preset,
    strategy_payload_diff,
    update_native_strategy_mt5_validation,
    update_native_strategy_status,
)


def test_generate_batch_candidates_from_mutation_space():
    candidates = generate_batch_candidates(
        template_name="trend_pullback",
        base_payload={
            "name": "Batch Base",
            "symbol": "EURUSD",
            "timeframe": "H1",
            "params": {"FastEMA": 21, "SlowEMA": 55, "mmLots": 1.0},
        },
        mutation_space={"FastEMA": [13, 21], "SlowEMA": [34, 55]},
        limit=10,
    )

    assert len(candidates) == 4
    assert all(candidate["template_name"] == "trend_pullback" for candidate in candidates)
    assert all("mutations" in candidate for candidate in candidates)


def test_generate_batch_candidates_supports_progressive_search_mode():
    base_payload = {
        "name": "Batch Base",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "params": {"FastEMA": 21, "SlowEMA": 55, "mmLots": 1.0},
    }
    candidates = generate_batch_candidates(
        template_name="trend_pullback",
        base_payload=base_payload,
        mutation_space={"FastEMA": [13, 21], "SlowEMA": [34, 55]},
        limit=4,
        search_mode="progressive",
    )
    assert len(candidates) == 4
    distances = [item["mutation_distance"] for item in candidates]
    assert distances == sorted(distances)


def test_generate_batch_candidates_can_include_structural_mutations():
    base_payload = {
        "name": "Batch Base",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "direction": "long",
        "min_bars": 55,
        "params": {"FastEMA": 21, "SlowEMA": 55, "mmLots": 1.0},
    }
    candidates = generate_batch_candidates(
        template_name="trend_pullback",
        base_payload=base_payload,
        mutation_space={"FastEMA": [21]},
        limit=10,
        search_mode="progressive",
        include_structural_mutations=True,
    )
    assert candidates
    assert any(candidate["payload"].get("direction") == "short" for candidate in candidates)
    assert any(candidate.get("structural_changes") for candidate in candidates)


def test_structural_mutation_space_ignores_min_bars_below_engine_warmup():
    structural = default_structural_mutation_space(
        "breakout_confirm",
        {
            "name": "Breakout Base",
            "symbol": "US30.cash",
            "timeframe": "H1",
            "direction": "long",
            "expiry_bars": 3,
            "min_bars": 25,
            "params": {
                "HighestPeriod": 20,
                "LowestPeriod": 20,
                "ATRPeriod": 14,
                "BBWRPeriod": 20,
                "MaxDistancePct": 4.0,
            },
        },
    )
    assert "min_bars" not in structural
    assert "expiry_bars" in structural


def test_parameter_mutation_space_is_template_specific():
    params = default_parameter_mutation_space(
        "trend_pullback",
        {
            "name": "Trend Base",
            "symbol": "EURUSD",
            "timeframe": "H1",
            "params": {
                "FastEMA": 21,
                "SlowEMA": 55,
                "ATRPeriod": 14,
                "ExitAfterBars": 12,
            },
        },
    )
    assert "params.FastEMA" in params
    assert "params.SlowEMA" in params
    assert "direction" not in params


def test_default_wfo_param_grid_for_template_uses_family_specific_keys():
    grid = default_wfo_param_grid_for_template(
        "breakout_confirm",
        {
            "HighestPeriod": 20,
            "LowestPeriod": 20,
            "ATRPeriod": 14,
            "BBWRPeriod": 20,
            "MaxDistancePct": 4.0,
            "mmLots": 1.0,
        },
    )
    assert set(grid) == {"HighestPeriod", "LowestPeriod", "ATRPeriod", "BBWRPeriod", "MaxDistancePct"}
    assert "mmLots" not in grid


def test_default_scoring_profile_for_template_is_family_specific():
    breakout = default_scoring_profile_for_template("breakout_confirm")
    reversion = default_scoring_profile_for_template("reversion")
    assert breakout["profit_factor_weight"] > reversion["profit_factor_weight"]
    assert reversion["drawdown_weight"] > breakout["drawdown_weight"]


def test_score_candidate_uses_family_specific_weights():
    metrics = {
        "backtest": {
            "sharpe_mean": 1.0,
            "profit_factor": 1.3,
            "max_drawdown_pct": 12.0,
            "num_trades": 18,
        },
        "wfo": {"robustness_score": 0.55},
        "monte_carlo": {"prob_profit": 0.6},
        "stability_score": 0.7,
    }
    breakout_score, breakout_components, _ = _score_candidate(
        template_name="breakout_confirm",
        backtest=metrics["backtest"],
        wfo=metrics["wfo"],
        monte_carlo=metrics["monte_carlo"],
        stability_score=metrics["stability_score"],
    )
    reversion_score, reversion_components, _ = _score_candidate(
        template_name="reversion",
        backtest=metrics["backtest"],
        wfo=metrics["wfo"],
        monte_carlo=metrics["monte_carlo"],
        stability_score=metrics["stability_score"],
    )
    assert breakout_score != reversion_score
    assert breakout_components["profit_factor"] > reversion_components["profit_factor"]
    assert reversion_components["drawdown"] > breakout_components["drawdown"]


def test_rank_batch_candidates_assigns_quality_bucket():
    ranked = rank_batch_candidates(
        [
            {
                "strategy_id": "a",
                "score": 60.0,
                "accepted": True,
                "backtest": {"profit_factor": 1.4, "max_drawdown_pct": 12.0},
                "wfo": {"robustness_score": 0.6},
            },
            {
                "strategy_id": "b",
                "score": 35.0,
                "accepted": False,
                "backtest": {"profit_factor": 1.1, "max_drawdown_pct": 20.0},
                "wfo": {"robustness_score": 0.2},
            },
        ]
    )
    assert ranked[0]["quality_bucket"] == "promote"
    assert ranked[1]["quality_bucket"] in {"watch", "reject"}


def test_template_defaults_expose_family_specific_policy_and_structural_mutations():
    policy = default_promotion_policy_for_template("sqx_usdjpy_vwap_wt")
    assert policy["min_profit_factor"] >= 1.16

    structural = default_structural_mutation_space(
        "sqx_usdjpy_vwap_wt",
        {
            "name": "USDJPY Test",
            "symbol": "USDJPY",
            "timeframe": "H4",
            "params": {"HighestPeriod": 20, "ExitAfterBars1": 14},
        },
    )
    assert structural == {}

    params = default_parameter_mutation_space(
        "sqx_usdjpy_vwap_wt",
        {
            "name": "USDJPY Test",
            "symbol": "USDJPY",
            "timeframe": "H4",
            "params": {"HighestPeriod": 20, "ExitAfterBars1": 14},
        },
    )
    assert "params.HighestPeriod" in params
    assert "params.ExitAfterBars1" in params


def test_xau_breakout_family_defaults_are_exposed():
    policy = default_promotion_policy_for_template("xau_breakout_session")
    assert policy["min_profit_factor"] >= 1.16

    params = default_parameter_mutation_space(
        "xau_breakout_session",
        {
            "name": "XAU Breakout",
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "params": {"HighestPeriod": 24, "ATRPeriod": 14, "ExitAfterBars": 18},
        },
    )
    assert "params.HighestPeriod" in params
    assert "params.ExitAfterBars" in params

    grid = default_wfo_param_grid_for_template(
        "xau_breakout_session",
        {"HighestPeriod": 24, "LowestPeriod": 24, "ATRPeriod": 14, "BBWRPeriod": 24, "BBWRMin": 0.015},
    )
    assert "HighestPeriod" in grid
    assert "BBWRMin" in grid


def test_xau_discovery_family_exposes_structural_grammar():
    policy = default_promotion_policy_for_template("xau_discovery_grammar")
    assert policy["min_profit_factor"] >= 1.12

    structural = default_structural_mutation_space(
        "xau_discovery_grammar",
        {
            "name": "XAU Discovery",
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "entry_archetype": "breakout_stop",
            "volatility_filter": "bb_width",
            "session_filter": "london_ny",
            "stop_model": "atr",
            "target_model": "fixed_rr",
            "exit_model": "session_close",
            "params": {"HighestPeriod": 24, "ATRPeriod": 14},
        },
    )
    assert "entry_archetype" in structural
    assert "stop_model" in structural
    assert "exit_model" in structural
    assert "break_even_then_trail" in structural["exit_model"]
    assert "atr_pullback_limit" in structural["entry_archetype"]


def test_run_generator_session_persists_ranked_candidates(monkeypatch):
    original_generators = GENERATOR_RUNS_PATH.read_text(encoding="utf-8") if GENERATOR_RUNS_PATH.exists() else None
    try:
        monkeypatch.setattr(
            "services.native_strategy_lab.register_generated_strategy",
            lambda **kwargs: (
                {"strategy_id": kwargs["strategy_id"], "name": kwargs["payload"]["name"]},
                {},
            ),
        )
        monkeypatch.setattr(
            "services.native_strategy_lab.evaluate_native_strategy",
            lambda strategy_id, **kwargs: {
                "score": 50.0 if strategy_id.endswith("0000") else 42.0,
                "accepted": strategy_id.endswith("0000"),
                "backtest": {"profit_factor": 1.2, "max_drawdown_pct": 10.0, "num_trades": 20, "sharpe_mean": 0.8},
                "wfo": {},
                "parameter_stability": {},
                "monte_carlo": {},
            },
        )
        monkeypatch.setattr("services.native_strategy_lab._load_fast_filter_df", lambda symbol, timeframe: None)
        monkeypatch.setattr(
            "services.native_strategy_lab._behavioral_fingerprint",
            lambda payload, template_name, df: str(payload.get("params", {}).get("HighestPeriod", 0)) + str(payload.get("params", {}).get("ATRPeriod", 0)),
        )

        run = run_generator_session(
            template_name="xau_breakout_session",
            base_payload={
                "name": "XAU Breakout",
                "symbol": "XAUUSD",
                "timeframe": "H1",
                "params": {"HighestPeriod": 24, "ATRPeriod": 14},
            },
            mutation_space={
                "params.HighestPeriod": [20, 24, 30],
                "params.ATRPeriod": [10, 14, 18],
            },
            date_from="2024-01-01",
            date_to="2024-12-31",
            max_candidates=2,
            include_structural_mutations=False,
        )

        assert run["evaluated"] == 2
        assert run["accepted"] == 1
        assert len(run["leaderboard"]) == 2
        assert get_generator_run(run["run_id"]) is not None
        assert any(item["run_id"] == run["run_id"] for item in list_generator_runs())
    finally:
        if original_generators is None:
            GENERATOR_RUNS_PATH.unlink(missing_ok=True)
        else:
            GENERATOR_RUNS_PATH.write_text(original_generators, encoding="utf-8")


def test_create_batch_run_filters_behavioral_duplicates_before_persisting(monkeypatch):
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    original_runs = BATCH_RUNS_PATH.read_text(encoding="utf-8") if BATCH_RUNS_PATH.exists() else None
    created_paths: list[Path] = []

    monkeypatch.setattr("services.native_strategy_lab._load_fast_filter_df", lambda symbol, timeframe: None)
    monkeypatch.setattr("services.native_strategy_lab._behavioral_fingerprint", lambda payload, template_name, df: "same-fingerprint")

    try:
        run = create_batch_run(
            template_name="trend_pullback",
            base_payload={
                "name": "Batch Base",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "direction": "long",
                "params": {
                    "FastEMA": 21,
                    "SlowEMA": 55,
                    "ATRPeriod": 14,
                    "WaveTrendChannel": 9,
                    "WaveTrendAverage": 21,
                    "ExitAfterBars": 12,
                    "mmLots": 1.0,
                    "StopLossATR": 1.8,
                    "ProfitTargetATR": 2.7,
                },
            },
            mutation_space={"FastEMA": [13, 21]},
            limit=10,
        )

        assert run["requested_candidates"] == 2
        assert run["generated_candidates"] == 1
        assert len(run["candidates"]) == 1
        assert len(run["duplicate_candidates"]) == 1

        record = get_native_strategy_record(run["candidates"][0]["strategy_id"])
        assert record is not None
        created_paths.extend([Path(record["strategy_path"]), Path(record["spec_path"])])
    finally:
        for path in created_paths:
            path.unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")
        if original_runs is None:
            BATCH_RUNS_PATH.unlink(missing_ok=True)
        else:
            BATCH_RUNS_PATH.write_text(original_runs, encoding="utf-8")


def test_run_batch_generation_stops_after_fast_filter(monkeypatch):
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    original_runs = BATCH_RUNS_PATH.read_text(encoding="utf-8") if BATCH_RUNS_PATH.exists() else None
    created_paths: list[Path] = []

    monkeypatch.setattr("services.native_strategy_lab._load_fast_filter_df", lambda symbol, timeframe: None)
    monkeypatch.setattr(
        "services.native_strategy_lab._behavioral_fingerprint",
        lambda payload, template_name, df: str(payload["params"]["FastEMA"]),
    )

    try:
        run = run_batch_generation(
            template_name="trend_pullback",
            base_payload={
                "name": "Batch Base",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "direction": "long",
                "params": {
                    "FastEMA": 21,
                    "SlowEMA": 55,
                    "ATRPeriod": 14,
                    "WaveTrendChannel": 9,
                    "WaveTrendAverage": 21,
                    "ExitAfterBars": 12,
                    "mmLots": 1.0,
                    "StopLossATR": 1.8,
                    "ProfitTargetATR": 2.7,
                },
            },
            mutation_space={"FastEMA": [13, 21]},
            date_from="2021-01-01",
            date_to="2024-01-01",
            limit=10,
        )

        assert run["stage"] == "filtered"
        assert all(item["evaluated"] is False for item in run["candidates"])
        assert "date_from" not in run

        for item in run["candidates"]:
            record = get_native_strategy_record(item["strategy_id"])
            assert record is not None
            created_paths.extend([Path(record["strategy_path"]), Path(record["spec_path"])])
    finally:
        for path in created_paths:
            path.unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")
        if original_runs is None:
            BATCH_RUNS_PATH.unlink(missing_ok=True)
        else:
            BATCH_RUNS_PATH.write_text(original_runs, encoding="utf-8")


def test_evaluate_native_strategy_uses_template_specific_wfo_grid(monkeypatch):
    strategy_id = f"pytest_{uuid.uuid4().hex[:8]}"
    payload = {
        "name": "Breakout WFO Grid",
        "symbol": "US30.cash",
        "timeframe": "H1",
        "direction": "long",
        "params": {
            "HighestPeriod": 20,
            "LowestPeriod": 20,
            "ATRPeriod": 14,
            "BBWRPeriod": 20,
            "BBWRMin": 0.0,
            "MaxDistancePct": 4.0,
            "mmLots": 1.0,
            "StopLossATR": 1.7,
            "ProfitTargetATR": 2.8,
        },
        "expiry_bars": 3,
        "min_bars": 25,
        "lot_value": 1.0,
    }
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    captured: dict[str, object] = {}

    class _DummyWFO:
        params_used = [{"HighestPeriod": 20, "LowestPeriod": 20}]

        def summary(self):
            return {
                "n_windows": 1,
                "profitable_windows": 1,
                "robustness_score": 0.5,
                "avg_efficiency": 1.0,
                "oos_net_profit": 100.0,
            }

    def _fake_run_backtest(*args, **kwargs):
        class _Result:
            n_trades = 0
        return _Result()

    def _fake_backtest_result_payload(result):
        return {
            "summary": {
                "sharpe_mean": 0.5,
                "profit_factor": 1.2,
                "max_drawdown_pct": 10.0,
                "num_trades": 12,
            }
        }

    def _fake_load_bars(symbol, timeframe, date_from=None, date_to=None):
        import pandas as pd

        index = pd.date_range("2023-01-01", periods=200, freq="1h")
        return pd.DataFrame(
            {
                "open": [100.0] * len(index),
                "high": [101.0] * len(index),
                "low": [99.0] * len(index),
                "close": [100.5] * len(index),
                "volume": [1] * len(index),
            },
            index=index,
        )

    def _fake_run_wfo(strat_cls, df, *, param_grid, **kwargs):
        captured["param_grid"] = param_grid
        return _DummyWFO()

    monkeypatch.setattr("services.native_strategy_lab.run_backtest", _fake_run_backtest)
    monkeypatch.setattr("services.native_strategy_lab.backtest_result_payload", _fake_backtest_result_payload)
    monkeypatch.setattr("services.native_strategy_lab.load_bars", _fake_load_bars)
    monkeypatch.setattr("services.native_strategy_lab.run_wfo", _fake_run_wfo)

    try:
        register_generated_strategy(
            template_name="breakout_confirm",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
        )

        evaluation = evaluate_native_strategy(
            strategy_id,
            date_from="2023-01-01",
            date_to="2023-12-31",
            run_mc=False,
            run_wfo_checks=True,
        )

        assert set(captured["param_grid"]) == {"HighestPeriod", "LowestPeriod", "ATRPeriod", "BBWRPeriod", "MaxDistancePct"}
        assert evaluation["wfo_param_grid"] == captured["param_grid"]
    finally:
        record = get_native_strategy_record(strategy_id)
        if record:
            Path(record["strategy_path"]).unlink(missing_ok=True)
            Path(record["spec_path"]).unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")


def test_evaluate_native_strategy_persists_family_scoring_profile(monkeypatch):
    strategy_id = f"pytest_{uuid.uuid4().hex[:8]}"
    payload = {
        "name": "Reversion Score Profile",
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "params": {
            "BBPeriod": 20,
            "BBDev": 2.0,
            "WPRPeriod": 14,
            "ATRPeriod": 14,
            "WPROversold": -80.0,
            "WPROverbought": -20.0,
            "mmLots": 1.0,
            "StopLossATR": 1.5,
            "ProfitTargetATR": 2.0,
        },
        "min_bars": 20,
        "lot_value": 1.0,
    }
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None

    def _fake_run_backtest(*args, **kwargs):
        class _Result:
            n_trades = 0
        return _Result()

    def _fake_backtest_result_payload(result):
        return {
            "summary": {
                "sharpe_mean": 0.9,
                "profit_factor": 1.15,
                "max_drawdown_pct": 9.0,
                "num_trades": 16,
            }
        }

    monkeypatch.setattr("services.native_strategy_lab.run_backtest", _fake_run_backtest)
    monkeypatch.setattr("services.native_strategy_lab.backtest_result_payload", _fake_backtest_result_payload)

    try:
        register_generated_strategy(
            template_name="reversion",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
        )

        evaluation = evaluate_native_strategy(
            strategy_id,
            date_from="2023-01-01",
            date_to="2023-12-31",
            run_mc=False,
            run_wfo_checks=False,
        )

        assert evaluation["scoring_profile"]["drawdown_weight"] == default_scoring_profile_for_template("reversion")["drawdown_weight"]
        assert set(evaluation["score_components"]) == {"sharpe", "profit_factor", "drawdown", "trade_count", "wfo", "monte_carlo", "stability"}
    finally:
        record = get_native_strategy_record(strategy_id)
        if record:
            Path(record["strategy_path"]).unlink(missing_ok=True)
            Path(record["spec_path"]).unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")


def test_evaluate_native_strategy_defaults_xau_h1_to_intrabar_replay(monkeypatch):
    strategy_id = f"pytest_{uuid.uuid4().hex[:8]}"
    payload = {
        "name": "XAU Intrabar Default",
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "entry_archetype": "ema_reclaim",
        "volatility_filter": "bb_width",
        "session_filter": "london_ny",
        "stop_model": "atr",
        "target_model": "fixed_rr",
        "exit_model": "session_close",
        "params": {"mmLots": 1.0},
    }
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    captured: dict[str, object] = {}

    def _fake_run_backtest(*args, **kwargs):
        captured["intrabar_steps"] = kwargs.get("intrabar_steps")
        class _Result:
            n_trades = 0
        return _Result()

    def _fake_backtest_result_payload(result):
        return {"summary": {"sharpe_mean": 0.9, "profit_factor": 1.15, "max_drawdown_pct": 9.0, "num_trades": 16}}

    monkeypatch.setattr("services.native_strategy_lab.run_backtest", _fake_run_backtest)
    monkeypatch.setattr("services.native_strategy_lab.backtest_result_payload", _fake_backtest_result_payload)

    try:
        register_generated_strategy(
            template_name="xau_discovery_grammar",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
        )
        evaluate_native_strategy(
            strategy_id,
            date_from="2023-01-01",
            date_to="2023-12-31",
            intrabar_steps=1,
            run_mc=False,
            run_wfo_checks=False,
        )
        assert captured["intrabar_steps"] == 60
    finally:
        record = get_native_strategy_record(strategy_id)
        if record:
            Path(record["strategy_path"]).unlink(missing_ok=True)
            Path(record["spec_path"]).unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")


def test_update_native_strategy_mt5_validation_persists_gate():
    strategy_id = f"pytest_{uuid.uuid4().hex[:8]}"
    payload = {
        "name": "MT5 Gate",
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "entry_archetype": "ema_reclaim",
        "volatility_filter": "bb_width",
        "session_filter": "london_ny",
        "stop_model": "atr",
        "target_model": "fixed_rr",
        "exit_model": "session_close",
        "params": {"mmLots": 1.0},
    }
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    try:
        register_generated_strategy(
            template_name="xau_discovery_grammar",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
        )
        updated = update_native_strategy_mt5_validation(
            strategy_id,
            report_path="mt5/generated/test/report.html",
            metrics={"profit_factor": 1.2},
            comparison={"profit_ratio_mt5_to_native": 0.9},
            accepted=True,
        )
        assert updated is not None
        assert updated["mt5_validation"]["accepted"] is True
    finally:
        record = get_native_strategy_record(strategy_id)
        if record:
            Path(record["strategy_path"]).unlink(missing_ok=True)
            Path(record["spec_path"]).unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")


def test_register_strategy_and_update_catalog_status():
    strategy_id = f"pytest_{uuid.uuid4().hex[:8]}"
    payload = {
        "name": "Catalog Test",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "direction": "long",
        "params": {
            "FastEMA": 21,
            "SlowEMA": 55,
            "ATRPeriod": 14,
            "mmLots": 1.0,
            "StopLossATR": 1.8,
            "ProfitTargetATR": 2.7,
            "WaveTrendChannel": 9,
            "WaveTrendAverage": 21,
            "ExitAfterBars": 12,
        },
        "min_bars": 55,
        "lot_value": 1.0,
    }

    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    try:
        record, meta = register_generated_strategy(
            template_name="trend_pullback",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
            tags=["pytest"],
        )

        assert record["strategy_id"] == strategy_id
        assert Path(record["strategy_path"]).exists()
        assert Path(record["spec_path"]).exists()

        updated = update_native_strategy_status(strategy_id, status="promoted", notes="pytest")
        loaded = get_native_strategy_record(strategy_id)

        assert updated is not None
        assert loaded is not None
        assert loaded["status"] == "promoted"
        assert loaded["notes"] == "pytest"
    finally:
        Path(meta["paths"]["strategy_path"]).unlink(missing_ok=True)
        Path(meta["paths"]["spec_path"]).unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")


def test_regenerate_native_strategy_version_creates_child_record():
    strategy_id = f"pytest_{uuid.uuid4().hex[:8]}"
    payload = {
        "name": "Versioned Test",
        "symbol": "USDJPY",
        "timeframe": "H4",
        "params": {
            "EMAPeriod1": 20,
            "VWAPPeriod1": 77,
            "WaveTrendChannel": 9,
            "WaveTrendAverage": 60,
            "IndicatorCrsMAPrd1": 35,
            "ExitAfterBars1": 14,
            "ProfitTargetCoef1": 3.0,
            "StopLossCoef1": 2.8,
            "ATRPeriod1": 19,
            "ATRPeriod2": 14,
            "HighestPeriod": 20,
            "mmLots": 1.0
        },
    }
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    try:
        parent, parent_meta = register_generated_strategy(
            template_name="sqx_usdjpy_vwap_wt",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
        )
        child, child_meta = regenerate_native_strategy_version(
            strategy_id,
            payload={**payload, "name": "Versioned Test V2"},
        )
        assert child["strategy_id"] != parent["strategy_id"]
        assert child["lineage"]["parent_strategy_id"] == parent["strategy_id"]
        assert Path(child["strategy_path"]).exists()
        diff = strategy_payload_diff(parent["strategy_id"], child["strategy_id"])
        assert diff["payload_changed"] is True
        assert any(change["param"] == "EMAPeriod1" for change in diff["param_changes"]) is False
    finally:
        Path(parent_meta["paths"]["strategy_path"]).unlink(missing_ok=True)
        Path(parent_meta["paths"]["spec_path"]).unlink(missing_ok=True)
        if 'child_meta' in locals():
            Path(child_meta["paths"]["strategy_path"]).unlink(missing_ok=True)
            Path(child_meta["paths"]["spec_path"]).unlink(missing_ok=True)
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")


def test_save_custom_preset_writes_json_file():
    preset_name = f"pytest_preset_{uuid.uuid4().hex[:8]}"
    path = Path(
        save_custom_preset(
            label=preset_name,
            description="pytest preset",
            template_name="reversion",
            payload={"name": "Preset Test", "symbol": "XAUUSD", "timeframe": "H1", "params": {}},
        )
    )
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["label"] == preset_name
        assert body["template"] == "reversion"
    finally:
        path.unlink(missing_ok=True)
