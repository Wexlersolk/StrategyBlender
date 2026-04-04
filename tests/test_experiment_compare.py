from __future__ import annotations

import pandas as pd

from research.experiment_compare import compare_experiments


def test_compare_experiments_surfaces_metric_and_parameter_diffs(tmp_path):
    base_dir = tmp_path / "base"
    cand_dir = tmp_path / "cand"
    base_dir.mkdir()
    cand_dir.mkdir()

    base_agg = pd.DataFrame([{"overlay": "ML Overlay", "total_oos_net_profit": 100.0, "win_window_rate": 0.5}])
    cand_agg = pd.DataFrame([{"overlay": "ML Overlay", "total_oos_net_profit": 140.0, "win_window_rate": 1.0}])
    base_path = base_dir / "aggregate_metrics.csv"
    cand_path = cand_dir / "aggregate_metrics.csv"
    base_agg.to_csv(base_path, index=False)
    cand_agg.to_csv(cand_path, index=False)

    base = {
        "files": {"aggregate_metrics": str(base_path)},
        "metadata": {"train_bars": 120},
        "lineage": {"dataset_snapshot": {"dataset_id": "dataset_a"}},
    }
    candidate = {
        "files": {"aggregate_metrics": str(cand_path)},
        "metadata": {"train_bars": 160},
        "lineage": {"dataset_snapshot": {"dataset_id": "dataset_b"}},
    }

    comparison = compare_experiments(base, candidate)

    assert float(comparison["metric_diff"]["delta_total_oos_net_profit"].iloc[0]) == 40.0
    assert not comparison["parameter_changes"].empty
    assert "dataset" in comparison["explanation"]
