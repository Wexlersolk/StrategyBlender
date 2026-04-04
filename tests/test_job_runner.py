from __future__ import annotations

from ui import job_runner


def _job_record(**overrides) -> dict:
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
        "max_retries": 1,
        "cancel_requested": False,
        "notified_at": "",
        "worker_id": "",
    }
    job.update(overrides)
    return job


def test_enqueue_background_job_persists_and_starts_worker(monkeypatch):
    seen = {}
    monkeypatch.setattr(job_runner, "save_job", lambda job: seen.setdefault("job", dict(job)) or dict(job))
    monkeypatch.setattr(job_runner, "_ensure_worker_service", lambda: seen.setdefault("worker_started", True))

    job = job_runner.enqueue_background_job(_job_record())

    assert job["job_id"] == "job_1"
    assert seen["worker_started"] is True
    assert seen["job"]["job_id"] == "job_1"


def test_poll_background_jobs_notifies_succeeded_job_once(monkeypatch):
    jobs = {"job_1": _job_record(status="succeeded", result_run_id="overlay_wfo_2", notified_at="")}
    monkeypatch.setattr(job_runner, "autosave", lambda: None)
    monkeypatch.setattr(job_runner, "recover_orphaned_jobs", lambda owner_id=None: 0)
    monkeypatch.setattr(job_runner, "_ensure_worker_service", lambda: None)
    monkeypatch.setattr(job_runner, "current_user_id", lambda: "tester")
    monkeypatch.setattr(job_runner, "list_jobs", lambda owner_id=None: [jobs["job_1"]])
    monkeypatch.setattr(job_runner, "save_job", lambda job: jobs.__setitem__(job["job_id"], dict(job)) or dict(job))

    seen = {}
    changed = job_runner.poll_background_jobs(
        {"job_registry": []},
        on_success=lambda job, result: seen.update({"job": job, "result": result}),
    )

    assert changed is True
    assert seen["result"]["run_id"] == "overlay_wfo_2"
    assert jobs["job_1"]["notified_at"] != ""


def test_cancel_background_job_marks_running_job_for_cancellation(monkeypatch):
    jobs = {"job_1": _job_record(status="running", stage="running")}
    monkeypatch.setattr(job_runner, "autosave", lambda: None)
    monkeypatch.setattr(job_runner, "load_job", lambda job_id: jobs.get(job_id))
    monkeypatch.setattr(job_runner, "save_job", lambda job: jobs.__setitem__(job["job_id"], dict(job)) or dict(job))

    job_runner.cancel_background_job("job_1")

    assert jobs["job_1"]["cancel_requested"] is True
    assert jobs["job_1"]["stage"] == "cancelling"
