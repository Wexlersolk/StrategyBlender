from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

from ui.views import experiments


class _FakeContainer:
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeStreamlit:
    def __init__(self, *, session_state: dict | None = None, buttons: dict | None = None):
        self.session_state = session_state if session_state is not None else {}
        self.buttons = buttons or {}
        self.rerun_called = False
        self.dataframes: list[pd.DataFrame] = []
        self.codes: list[str] = []
        self.progress_calls: list[tuple[float, str | None]] = []

    def markdown(self, *args, **kwargs):
        return None

    def dataframe(self, df, **kwargs):
        self.dataframes.append(df.copy() if isinstance(df, pd.DataFrame) else df)

    def code(self, text, **kwargs):
        self.codes.append(text)

    def progress(self, value, text=None):
        self.progress_calls.append((value, text))

    def selectbox(self, label, options, index=0, **kwargs):
        return options[index]

    def button(self, label, key=None, **kwargs):
        return bool(self.buttons.get(key, False))

    def columns(self, count):
        return [_FakeContainer(self) for _ in range(count)]

    def rerun(self):
        self.rerun_called = True

    @contextmanager
    def spinner(self, text):
        yield


def _record() -> dict:
    return {
        "run_id": "overlay_wfo_1",
        "title": "Saved run",
        "kind": "overlay_wfo",
        "created_at": "2026-04-04T00:00:00Z",
        "export_dir": "/tmp/export",
        "files": {},
        "metadata": {
            "portfolio_ids": ["ea1"],
            "scope": {"date_from": "2024-01-01", "date_to": "2024-01-31"},
            "train_bars": 120,
            "test_bars": 80,
            "embargo_bars": 5,
            "execution_config": {"commission_per_lot": 0.0, "spread_pips": 0.0, "slippage_pips": 0.0},
            "artifact": {"filter_model": {}, "sizing_model": {}, "benchmark_name": "baseline", "benchmark_settings": {}},
        },
        "aggregate_preview": [{"overlay": "ML Overlay", "total_oos_net_profit": 123.0}],
        "split_preview": [{"overlay": "ML Overlay", "window_id": 0, "oos_net_profit": 123.0}],
    }


def test_experiments_page_loads_saved_run_into_ai_validation(monkeypatch):
    fake_st = FakeStreamlit(
        session_state={},
        buttons={"experiment_load": True},
    )

    monkeypatch.setattr(experiments, "st", fake_st)
    monkeypatch.setattr(experiments, "section_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiments, "autosave", lambda: None)
    monkeypatch.setattr(experiments, "list_experiment_records", lambda: [_record()])
    monkeypatch.setattr(experiments, "list_jobs", lambda: [])
    monkeypatch.setattr(experiments, "poll_background_jobs", lambda *args, **kwargs: False)

    experiments.render()

    assert fake_st.session_state["page"] == "AI Training"
    assert fake_st.session_state["_ai_model_wfo"]["run_id"] == "overlay_wfo_1"
    assert fake_st.rerun_called is True


def test_experiments_page_queues_rerun_job(monkeypatch):
    fake_st = FakeStreamlit(
        session_state={"eas": {"ea1": {"name": "EA"}}},
        buttons={"experiment_rerun": True},
    )

    monkeypatch.setattr(experiments, "st", fake_st)
    monkeypatch.setattr(experiments, "section_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiments, "autosave", lambda: None)
    monkeypatch.setattr(experiments, "list_experiment_records", lambda: [_record()])
    monkeypatch.setattr(experiments, "list_jobs", lambda: [])
    monkeypatch.setattr(experiments, "poll_background_jobs", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        experiments,
        "_queue_rerun_job",
        lambda record, eas: {"job_id": "job_2"},
    )

    experiments.render()

    assert fake_st.session_state["_selected_experiment_job_id"] == "job_2"
    assert fake_st.rerun_called is True


def test_integrate_completed_job_registers_rerun_and_selects_new_run(monkeypatch):
    fake_st = FakeStreamlit(session_state={})
    monkeypatch.setattr(experiments, "st", fake_st)

    job = {"job_id": "job_2", "result_run_id": ""}
    result = {"run_id": "overlay_wfo_2"}

    experiments._integrate_completed_job(job, result)

    assert fake_st.session_state["_selected_experiment_run_id"] == "overlay_wfo_2"
