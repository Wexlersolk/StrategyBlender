from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy
from research.meta_models import fit_filter_model, fit_sizing_model
from research.trade_dataset import build_trade_dataset
from ui.views import ai_training


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


class _FakeContainer:
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, *args, **kwargs):
        return self.parent.metric(*args, **kwargs)


class FakeStreamlit:
    def __init__(self, *, session_state: dict | None = None, buttons: dict | None = None):
        self.session_state = session_state if session_state is not None else {}
        self.buttons = buttons or {}
        self.dataframes: list[pd.DataFrame] = []
        self.metrics: list[tuple] = []
        self.markdowns: list[str] = []

    def markdown(self, text, **kwargs):
        self.markdowns.append(text)

    def caption(self, *args, **kwargs):
        return None

    def dataframe(self, df, **kwargs):
        self.dataframes.append(df.copy() if isinstance(df, pd.DataFrame) else df)

    def plotly_chart(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def columns(self, count):
        return [_FakeContainer(self) for _ in range(count)]

    def tabs(self, labels):
        return [_FakeContainer(self) for _ in labels]

    def multiselect(self, label, options, default=None, **kwargs):
        return list(default or [])

    def date_input(self, label, value=None, **kwargs):
        return value

    def checkbox(self, label, value=False, **kwargs):
        return value

    def button(self, label, key=None, **kwargs):
        return bool(self.buttons.get(key, False))

    def selectbox(self, label, options, index=0, **kwargs):
        if not options:
            return None
        return options[index]

    def number_input(self, label, value=0.0, **kwargs):
        return value

    def slider(self, label, value=None, **kwargs):
        return value

    @contextmanager
    def spinner(self, text):
        yield


def _dataset_for_ui() -> pd.DataFrame:
    result = Backtester(initial_capital=100_000).run(AlternatingTradeStrategy(), _sample_df())
    dataset = build_trade_dataset(result)
    dataset["ea_id"] = "ea1"
    dataset["name"] = "Alt"
    dataset["symbol"] = "XAUUSD"
    dataset["timeframe"] = "H1"
    return dataset


def test_render_shows_empty_state_when_no_eas(monkeypatch):
    fake_st = FakeStreamlit(session_state={"eas": {}})
    seen: list[str] = []

    monkeypatch.setattr(ai_training, "st", fake_st)
    monkeypatch.setattr(ai_training, "empty_state", lambda message, icon="": seen.append(message))

    ai_training.render()

    assert seen
    assert "Load at least one EA first" in seen[0]


def test_dataset_tab_builds_and_stores_research_state(monkeypatch):
    fake_st = FakeStreamlit(session_state={}, buttons={"ai_dataset_build": True})
    dataset = _dataset_for_ui()
    raw_results = {"ea1": object()}

    monkeypatch.setattr(ai_training, "st", fake_st)
    monkeypatch.setattr(
        ai_training,
        "_dataset_scope_selector",
        lambda eas, key_prefix: (["ea1"], pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"), 1),
    )
    monkeypatch.setattr(ai_training, "_run_dataset_backtests", lambda *args, **kwargs: (raw_results, dataset))
    monkeypatch.setattr(
        ai_training,
        "save_dataset_snapshot",
        lambda **kwargs: {"dataset_id": "dataset_1", "dataset_hash": "hash_1", "metadata": {"feature_config": {"feature_config_hash": "cfg_1"}}},
    )
    monkeypatch.setattr(ai_training, "autosave", lambda: None)

    ai_training._render_dataset_tab({"ea1": {"name": "Alt", "symbol": "XAUUSD", "timeframe": "H1"}})

    assert fake_st.session_state["_ai_research_runs"] == raw_results
    assert fake_st.session_state["_ai_trade_dataset"].equals(dataset)
    assert fake_st.session_state["_ai_dataset_scope"]["ea_ids"] == ["ea1"]
    assert fake_st.session_state["_ai_dataset_id"] == "dataset_1"


def test_model_tab_trains_and_persists_artifact(monkeypatch):
    dataset = _dataset_for_ui()
    session_state = {
        "_ai_trade_dataset": dataset,
        "_ai_dataset_scope": {
            "ea_ids": ["ea1"],
            "date_from": "2024-01-01",
            "date_to": "2024-01-31",
            "intrabar_steps": 1,
        },
    }
    fake_st = FakeStreamlit(session_state=session_state, buttons={"ai_model_train": True, "ai_model_eval_run": False})

    monkeypatch.setattr(ai_training, "st", fake_st)
    monkeypatch.setattr(
        ai_training,
        "save_artifact_snapshot",
        lambda **kwargs: {"artifact_id": "artifact_1", "artifact_hash": "artifact_hash_1"},
    )
    monkeypatch.setattr(ai_training, "autosave", lambda: None)

    ai_training._render_model_tab({"ea1": {"name": "Alt", "symbol": "XAUUSD", "timeframe": "H1"}})

    artifact = fake_st.session_state["training_artifact"]
    assert fake_st.session_state["model_trained"] is True
    assert artifact["filter_model"]["kind"] == "filter"
    assert artifact["sizing_model"]["kind"] == "sizing"
    assert fake_st.session_state["training_log"]
    assert fake_st.session_state["_ai_training_artifact_id"] == "artifact_1"


def test_model_tab_stores_walk_forward_oos_summary(monkeypatch):
    dataset = _dataset_for_ui()
    filter_model = fit_filter_model(dataset, threshold=0.55)
    sizing_model = fit_sizing_model(dataset, min_multiplier=0.5, max_multiplier=1.5)
    session_state = {
        "_ai_trade_dataset": dataset,
        "_ai_dataset_scope": {
            "ea_ids": ["ea1"],
            "date_from": "2024-01-01",
            "date_to": "2024-01-31",
            "intrabar_steps": 1,
            "dataset_id": "dataset_1",
            "dataset_hash": "hash_1",
            "feature_config": {"feature_config_hash": "cfg_1"},
        },
        "_ai_training_artifact_id": "artifact_1",
        "training_artifact": {
            "filter_model": filter_model,
            "sizing_model": sizing_model,
            "dataset_summary": {},
            "benchmark_name": "baseline",
            "benchmark_settings": {"target_vol": 0.2, "min_multiplier": 0.5, "max_multiplier": 1.5, "soft_dd": 5.0, "hard_dd": 10.0, "soft_multiplier": 0.5},
        },
    }
    fake_st = FakeStreamlit(session_state=session_state, buttons={"ai_model_train": False, "ai_model_wfo_run": True, "ai_model_eval_run": False})

    class _FakeResult:
        def summary(self):
            return {
                "net_profit": 120.0,
                "profit_factor": 1.4,
                "max_drawdown_pct": 3.2,
                "win_rate": 0.55,
                "n_trades": 4,
                "n_decisions": 6,
            }

    fake_report = SimpleNamespace(
        windows=[object()],
        split_metrics=pd.DataFrame(
            [
                {"overlay": "Baseline", "window_id": 0, "oos_net_profit": 80.0},
                {"overlay": "Benchmark", "window_id": 0, "oos_net_profit": 100.0},
                {"overlay": "ML Overlay", "window_id": 0, "oos_net_profit": 120.0},
            ]
        ),
        aggregate_metrics=pd.DataFrame(
            [
                {"overlay": "Baseline", "windows": 1, "total_oos_net_profit": 80.0},
                {"overlay": "Benchmark", "windows": 1, "total_oos_net_profit": 100.0},
                {"overlay": "ML Overlay", "windows": 1, "total_oos_net_profit": 120.0},
            ]
        ),
    )

    monkeypatch.setattr(ai_training, "st", fake_st)
    monkeypatch.setattr(ai_training, "autosave", lambda: None)
    monkeypatch.setattr(
        ai_training,
        "_current_artifact_snapshot",
        lambda: {"artifact_id": "artifact_1", "artifact_hash": "artifact_hash_1", "artifact": fake_st.session_state["training_artifact"]},
    )
    def _fake_save_overlay_report(**kwargs):
        snapshot = {
            "run_id": "overlay_wfo_2",
            "portfolio_ids": ["ea1"],
            "split_metrics": fake_report.split_metrics,
            "aggregate_metrics": fake_report.aggregate_metrics,
        }
        fake_st.session_state["_ai_model_wfo"] = snapshot
        fake_st.session_state["experiment_registry"] = [{"run_id": "overlay_wfo_2"}]
        return snapshot
    monkeypatch.setattr(
        ai_training,
        "_save_overlay_report",
        _fake_save_overlay_report,
    )
    monkeypatch.setattr(ai_training, "_run_walk_forward_comparison", lambda *args, **kwargs: fake_report)

    ai_training._render_model_tab({"ea1": {"name": "Alt", "symbol": "XAUUSD", "timeframe": "H1"}})

    stored = fake_st.session_state["_ai_model_wfo"]
    assert set(stored["aggregate_metrics"]["overlay"]) == {"Baseline", "Benchmark", "ML Overlay"}
    assert float(stored["aggregate_metrics"].loc[stored["aggregate_metrics"]["overlay"] == "ML Overlay", "total_oos_net_profit"].iloc[0]) == 120.0
    assert set(stored["split_metrics"]["overlay"]) == {"Baseline", "Benchmark", "ML Overlay"}
    assert fake_st.session_state["experiment_registry"]
