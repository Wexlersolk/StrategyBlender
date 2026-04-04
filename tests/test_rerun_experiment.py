from __future__ import annotations

import pandas as pd

from scripts import rerun_experiment


def test_rerun_saved_experiment_uses_persisted_strategy_snapshots(monkeypatch):
    record = {
        "run_id": "overlay_wfo_1",
        "title": "Saved run",
        "metadata": {
            "portfolio_ids": ["ea1"],
            "scope": {"date_from": "2024-01-01", "date_to": "2024-01-31"},
            "train_bars": 120,
            "test_bars": 80,
            "embargo_bars": 5,
            "execution_config": {"commission_per_lot": 0.0},
            "artifact": {"filter_model": {}, "sizing_model": {}, "benchmark_name": "baseline", "benchmark_settings": {}},
        },
        "lineage": {"dataset_snapshot": {"dataset_id": "dataset_1"}},
    }
    dataset_snapshot = {
        "metadata": {
            "strategy_snapshots": [
                {
                    "ea_id": "ea1",
                    "name": "EA",
                    "symbol": "XAUUSD",
                    "timeframe": "H1",
                    "source": "source",
                    "engine_source": "engine",
                    "strategy_module": "strategies.example",
                    "strategy_class": "ExampleStrategy",
                }
            ]
        }
    }

    class _FakeReport:
        def __init__(self):
            self.aggregate_metrics = pd.DataFrame([{"overlay": "ML Overlay", "total_oos_net_profit": 1.0}])
            self.split_metrics = pd.DataFrame([{"overlay": "ML Overlay", "window_id": 0, "oos_net_profit": 1.0}])
            self.windows = []
            self.metadata = {}

    seen = {}

    monkeypatch.setattr(rerun_experiment, "get_experiment_record", lambda run_id: record)
    monkeypatch.setattr(rerun_experiment, "load_dataset_snapshot", lambda dataset_id: dataset_snapshot)
    def _fake_run_walk_forward(ea, *args, **kwargs):
        seen["ea"] = ea
        return _FakeReport()

    monkeypatch.setattr(rerun_experiment.ai_training, "_run_walk_forward_comparison", _fake_run_walk_forward)
    monkeypatch.setattr(rerun_experiment.ai_training, "_report_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rerun_experiment,
        "save_overlay_snapshot",
        lambda **kwargs: ({"run_id": "overlay_wfo_2", "title": kwargs["title"]}, {"run_id": "overlay_wfo_2"}),
    )

    snapshot = rerun_experiment.rerun_saved_experiment("overlay_wfo_1")

    assert snapshot["run_id"] == "overlay_wfo_2"
    assert seen["ea"]["strategy_module"] == "strategies.example"
