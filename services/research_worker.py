from __future__ import annotations

from concurrent.futures import CancelledError
from datetime import datetime, timezone
import os
import time
import uuid

from research.job_execution import JOB_HANDLERS, JobReporter
from research.state_store import (
    claim_next_job,
    current_user_id,
    load_job,
    recover_orphaned_jobs,
    save_audit_event,
    save_job,
    save_worker_heartbeat,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_job(job: dict) -> dict:
    reporter = JobReporter(job["job_id"])
    handler = JOB_HANDLERS.get(job.get("kind"))
    if handler is None:
        raise ValueError(f"Unsupported job kind: {job.get('kind')}")
    return handler(job, reporter)


def run_once(*, worker_id: str | None = None, owner_id: str | None = None) -> dict | None:
    owner = owner_id or current_user_id()
    wid = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
    save_worker_heartbeat(worker_id=wid, pid=os.getpid(), status="idle", claimed_job_id="", owner_id=owner)
    recover_orphaned_jobs(owner_id=owner)
    job = claim_next_job(worker_id=wid, owner_id=owner)
    if not job:
        save_worker_heartbeat(worker_id=wid, pid=os.getpid(), status="idle", claimed_job_id="", owner_id=owner)
        return None

    save_worker_heartbeat(worker_id=wid, pid=os.getpid(), status="running", claimed_job_id=job["job_id"], owner_id=owner)
    try:
        result = process_job(job)
        latest = load_job(job["job_id"]) or job
        latest["status"] = "succeeded"
        latest["stage"] = "succeeded"
        latest["progress_pct"] = 100.0
        latest["finished_at"] = utc_now()
        if isinstance(result, dict):
            latest["result_run_id"] = str(result.get("run_id", latest.get("result_run_id", "")))
        save_job(latest)
        save_audit_event(entity_type="job", entity_id=job["job_id"], event_type="job_succeeded", payload={"result_run_id": latest.get("result_run_id", "")}, owner_id=owner)
        return latest
    except CancelledError as exc:
        latest = load_job(job["job_id"]) or job
        latest["status"] = "cancelled"
        latest["stage"] = "cancelled"
        latest["finished_at"] = utc_now()
        latest["error"] = str(exc)
        save_job(latest)
        save_audit_event(entity_type="job", entity_id=job["job_id"], event_type="job_cancelled", level="warning", payload={"error": str(exc)}, owner_id=owner)
        return latest
    except Exception as exc:
        latest = load_job(job["job_id"]) or job
        latest["error"] = str(exc)
        latest["worker_id"] = wid
        if int(latest.get("attempt", 0)) <= int(latest.get("max_retries", 0)):
            latest["status"] = "queued"
            latest["stage"] = "queued"
            latest["progress_pct"] = 0.0
            latest["finished_at"] = ""
            save_audit_event(entity_type="job", entity_id=job["job_id"], event_type="job_retry_scheduled", level="warning", payload={"error": str(exc), "attempt": latest.get("attempt", 0)}, owner_id=owner)
        else:
            latest["status"] = "failed"
            latest["stage"] = "failed"
            latest["finished_at"] = utc_now()
            save_audit_event(entity_type="job", entity_id=job["job_id"], event_type="job_failed", level="error", payload={"error": str(exc)}, owner_id=owner)
        save_job(latest)
        return latest
    finally:
        save_worker_heartbeat(worker_id=wid, pid=os.getpid(), status="idle", claimed_job_id="", owner_id=owner)


def run_forever(*, poll_interval_seconds: float = 2.0, worker_id: str | None = None, owner_id: str | None = None) -> None:
    owner = owner_id or current_user_id()
    wid = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
    while True:
        save_worker_heartbeat(worker_id=wid, pid=os.getpid(), status="idle", claimed_job_id="", owner_id=owner)
        run_once(worker_id=wid, owner_id=owner)
        time.sleep(float(poll_interval_seconds))
