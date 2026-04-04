from __future__ import annotations

import pandas as pd

from ui import profile_manager


def test_workspace_profile_roundtrip_persists_walk_forward_report(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "profiles"
    key_file = tmp_path / ".workspace_key"
    workspace_profile = profiles_dir / "workspace_local.enc"

    monkeypatch.setattr(profile_manager, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(profile_manager, "KEY_FILE", key_file)
    monkeypatch.setattr(profile_manager, "WORKSPACE_PROFILE", workspace_profile)

    state = {
        "eas": {},
        "backtest_results": {},
        "ai_experiment_results": {},
        "model_trained": True,
        "training_log": ["ok"],
        "training_artifact": {"filter_model": {"kind": "filter"}},
        "_ai_dataset_id": "dataset_1",
        "_ai_training_artifact_id": "artifact_1",
        "schedule_path": "",
        "ai_saved_reports": [],
        "_ai_model_wfo": {
            "ea_id": "ea1",
            "split_metrics": pd.DataFrame(
                [
                    {"overlay": "Baseline", "window_id": 0, "oos_net_profit": 80.0},
                    {"overlay": "ML Overlay", "window_id": 0, "oos_net_profit": 120.0},
                ]
            ),
            "aggregate_metrics": pd.DataFrame(
                [
                    {"overlay": "Baseline", "windows": 1, "total_oos_net_profit": 80.0},
                    {"overlay": "ML Overlay", "windows": 1, "total_oos_net_profit": 120.0},
                ]
            ),
        },
        "_ai_model_wfo_run_id": "overlay_wfo_test",
        "experiment_registry": [{"run_id": "overlay_wfo_test", "title": "Saved report"}],
        "job_registry": [{"job_id": "job_1", "status": "running"}],
    }

    profile_manager.save_workspace_profile(state)
    loaded = profile_manager.load_workspace_profile()

    assert loaded is not None
    assert loaded["ai_model_wfo"]["ea_id"] == "ea1"
    assert set(loaded["ai_model_wfo"]["split_metrics"]["overlay"]) == {"Baseline", "ML Overlay"}
    assert float(
        loaded["ai_model_wfo"]["aggregate_metrics"]
        .loc[loaded["ai_model_wfo"]["aggregate_metrics"]["overlay"] == "ML Overlay", "total_oos_net_profit"]
        .iloc[0]
    ) == 120.0
    assert loaded["ai_dataset_id"] == "dataset_1"
    assert loaded["ai_training_artifact_id"] == "artifact_1"
    assert loaded["ai_model_wfo_run_id"] == "overlay_wfo_test"
    assert loaded["experiment_registry"][0]["run_id"] == "overlay_wfo_test"
    assert loaded["job_registry"][0]["job_id"] == "job_1"
