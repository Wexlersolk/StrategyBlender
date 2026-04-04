from __future__ import annotations

from ui import job_runner


def test_ensure_worker_service_maintains_target_pool(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(job_runner, "current_user_id", lambda: "tester")
    monkeypatch.setattr(job_runner, "_prune_worker_pool", lambda owner_id: None)
    monkeypatch.setattr(job_runner, "_healthy_worker_count", lambda owner_id: 0)
    monkeypatch.setattr(job_runner, "worker_is_healthy", lambda owner_id=None: False)
    monkeypatch.setattr(job_runner, "RESEARCH_WORKER_POOL_SIZE", 3)
    monkeypatch.setattr(job_runner, "_MANAGED_WORKERS", {})
    monkeypatch.setattr(job_runner, "_spawn_worker", lambda owner_id: calls.append(owner_id))

    job_runner._ensure_worker_service()

    assert calls == ["tester", "tester", "tester"]
