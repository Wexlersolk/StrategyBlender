from __future__ import annotations

from pathlib import Path

from research import state_store
from services import research_worker


def _patch_store(monkeypatch, tmp_path):
    root = tmp_path / "research_state"
    monkeypatch.setattr(state_store, "STORE_ROOT", root)
    monkeypatch.setattr(state_store, "JOBS_DIR", root / "jobs")
    monkeypatch.setattr(state_store, "EXPERIMENTS_DIR", root / "experiments")
    monkeypatch.setattr(state_store, "DATASETS_DIR", root / "datasets")
    monkeypatch.setattr(state_store, "ARTIFACTS_DIR", root / "artifacts")
    monkeypatch.setattr(state_store, "LOGS_DIR", root / "logs")
    monkeypatch.setattr(state_store, "DB_PATH", root / "research_state.db")
    monkeypatch.setattr(state_store, "AUDIT_LOG_PATH", root / "logs" / "audit.jsonl")


def test_worker_run_once_processes_queued_job_and_persists_success(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGYBLENDER_USER", "tester")
    _patch_store(monkeypatch, tmp_path)
    state_store.ensure_store()

    job = {
        "job_id": "job_1",
        "owner_id": "tester",
        "submitted_seq": 1,
        "kind": "overlay_rerun",
        "title": "Rerun",
        "status": "queued",
        "stage": "queued",
        "progress_pct": 0.0,
        "submitted_at": "",
        "started_at": "",
        "finished_at": "",
        "error": "",
        "payload": {"source_run_id": "overlay_wfo_1"},
        "result_run_id": "",
        "events": [],
        "attempt": 0,
        "max_retries": 0,
        "cancel_requested": False,
        "notified_at": "",
        "worker_id": "",
    }
    state_store.save_job(job)
    monkeypatch.setattr(research_worker, "process_job", lambda queued_job: {"run_id": "overlay_wfo_2"})

    result = research_worker.run_once(worker_id="worker_test", owner_id="tester")

    assert result is not None
    persisted = state_store.load_job("job_1", owner_id="tester")
    assert persisted is not None
    assert persisted["status"] == "succeeded"
    assert persisted["result_run_id"] == "overlay_wfo_2"
    assert Path(state_store.AUDIT_LOG_PATH).exists()


def test_recover_orphaned_running_job_requeues_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGYBLENDER_USER", "tester")
    _patch_store(monkeypatch, tmp_path)
    state_store.ensure_store()

    state_store.save_job(
        {
            "job_id": "job_1",
            "owner_id": "tester",
            "submitted_seq": 1,
            "kind": "overlay_rerun",
            "title": "Rerun",
            "status": "running",
            "stage": "running",
            "progress_pct": 50.0,
            "submitted_at": "",
            "started_at": "",
            "finished_at": "",
            "error": "",
            "payload": {"source_run_id": "overlay_wfo_1"},
            "result_run_id": "",
            "events": [],
            "attempt": 1,
            "max_retries": 0,
            "cancel_requested": False,
            "notified_at": "",
            "worker_id": "worker_dead",
        }
    )

    recovered = state_store.recover_orphaned_jobs(owner_id="tester", stale_after_seconds=0)

    assert recovered == 1
    persisted = state_store.load_job("job_1", owner_id="tester")
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["stage"] == "queued"
