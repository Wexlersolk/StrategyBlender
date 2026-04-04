from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import uuid

from config.settings import RESEARCH_WORKER_POLL_INTERVAL, RESEARCH_WORKER_POOL_SIZE
from research.state_store import (
    current_user_id,
    list_workers,
    list_audit_events,
    list_jobs,
    load_job,
    next_sequence,
    recover_orphaned_jobs,
    save_job,
    worker_is_healthy,
)
from ui.state import autosave


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_JOB_STATUSES = {"queued", "running"}
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
_MANAGED_WORKERS: dict[tuple[str, str], subprocess.Popen] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(level: str, stage: str, message: str) -> dict:
    return {
        "timestamp": _utc_now(),
        "level": level,
        "stage": stage,
        "message": message,
    }


def _prune_worker_pool(owner_id: str) -> None:
    stale_keys = []
    for key, proc in _MANAGED_WORKERS.items():
        if key[0] != owner_id:
            continue
        if proc.poll() is not None:
            stale_keys.append(key)
    for key in stale_keys:
        _MANAGED_WORKERS.pop(key, None)


def _healthy_worker_count(owner_id: str) -> int:
    count = 0
    now = datetime.now(timezone.utc)
    for worker in list_workers(owner_id=owner_id):
        heartbeat = worker.get("heartbeat_at", "")
        if not heartbeat:
            continue
        stamp = datetime.fromisoformat(heartbeat)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if (now - stamp).total_seconds() <= 30:
            count += 1
    return count


def _spawn_worker(owner_id: str) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    worker_id = f"worker_{uuid.uuid4().hex[:8]}"
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts" / "research_worker.py"),
            "--owner-id",
            owner_id,
            "--worker-id",
            worker_id,
            "--poll-interval",
            str(RESEARCH_WORKER_POLL_INTERVAL),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    _MANAGED_WORKERS[(owner_id, worker_id)] = proc


def _ensure_worker_service() -> None:
    owner_id = current_user_id()
    _prune_worker_pool(owner_id)
    healthy_count = _healthy_worker_count(owner_id)
    target_size = max(1, int(RESEARCH_WORKER_POOL_SIZE))
    inflight_count = sum(1 for key in _MANAGED_WORKERS if key[0] == owner_id)
    missing = max(0, target_size - max(healthy_count, inflight_count))
    if missing == 0 and (healthy_count > 0 or worker_is_healthy(owner_id=owner_id)):
        return
    for _ in range(missing or (0 if healthy_count else 1)):
        _spawn_worker(owner_id)


def new_job_record(*, kind: str, title: str, payload: dict | None = None, max_retries: int = 1) -> dict:
    owner = current_user_id()
    return {
        "job_id": f"job_{uuid.uuid4().hex[:10]}",
        "owner_id": owner,
        "submitted_seq": next_sequence("jobs"),
        "kind": kind,
        "title": title,
        "status": "queued",
        "stage": "queued",
        "progress_pct": 0.0,
        "submitted_at": _utc_now(),
        "started_at": "",
        "finished_at": "",
        "error": "",
        "payload": dict(payload or {}),
        "result_run_id": "",
        "events": [_event("info", "queued", "Job submitted.")],
        "attempt": 0,
        "max_retries": max(0, int(max_retries)),
        "cancel_requested": False,
        "notified_at": "",
        "worker_id": "",
    }


def enqueue_background_job(job: dict, _fn=None, *args, **kwargs) -> dict:
    del _fn, args, kwargs
    saved = save_job(job)
    _ensure_worker_service()
    return saved


def cancel_background_job(job_id: str) -> dict | None:
    job = load_job(job_id)
    if not job or job.get("status") in TERMINAL_JOB_STATUSES:
        return job
    if job.get("status") == "queued":
        job["status"] = "cancelled"
        job["stage"] = "cancelled"
        job["finished_at"] = _utc_now()
        job["events"] = list(job.get("events", [])) + [_event("warning", "cancelled", "Queued job was cancelled.")]
    else:
        job["cancel_requested"] = True
        job["stage"] = "cancelling"
        job["events"] = list(job.get("events", [])) + [_event("warning", "cancelling", "Cancellation requested.")]
    save_job(job)
    autosave()
    return job


def poll_background_jobs(session_state: dict, *, context: dict | None = None, on_success=None) -> bool:
    del context
    changed = bool(recover_orphaned_jobs(owner_id=current_user_id()))
    _ensure_worker_service()
    jobs = list_jobs(owner_id=current_user_id())
    for job in jobs:
        if job.get("status") != "succeeded" or job.get("notified_at"):
            continue
        if on_success:
            on_success(job, {"run_id": job.get("result_run_id", "")})
        job["notified_at"] = _utc_now()
        save_job(job)
        changed = True
    session_state["job_registry"] = list_jobs(owner_id=current_user_id())[:100]
    if changed:
        autosave()
    return changed


def job_audit_log(job_id: str, *, limit: int = 100) -> list[dict]:
    return list_audit_events(entity_type="job", entity_id=job_id, owner_id=current_user_id(), limit=limit)
