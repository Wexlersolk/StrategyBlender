from __future__ import annotations

from pathlib import Path

import json
import pandas as pd

from research.experiment_registry import create_experiment_record, export_overlay_report_bundle


def test_export_overlay_report_bundle_writes_files(tmp_path):
    aggregate = pd.DataFrame([{"overlay": "ML Overlay", "total_oos_net_profit": 123.0}])
    split = pd.DataFrame([{"overlay": "ML Overlay", "window_id": 0, "oos_net_profit": 123.0}])
    window_details = pd.DataFrame([{"overlay": "ML Overlay", "window_id": 0, "test_net_profit": 123.0}])
    record = create_experiment_record(
        kind="overlay_wfo",
        title="Test",
        metadata={"foo": "bar"},
        aggregate_metrics=aggregate,
        split_metrics=split,
    )

    files = export_overlay_report_bundle(
        record=record,
        aggregate_metrics=aggregate,
        split_metrics=split,
        metadata={"foo": "bar"},
        artifacts={"window_details": window_details, "validation": {"passed": True}},
        output_root=tmp_path,
    )

    assert Path(files["aggregate_metrics"]).exists()
    assert Path(files["split_metrics"]).exists()
    assert Path(files["metadata"]).exists()
    assert Path(files["artifacts"]).exists()
    assert json.loads(Path(files["metadata"]).read_text(encoding="utf-8"))["foo"] == "bar"
