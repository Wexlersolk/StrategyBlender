from __future__ import annotations

from research import auth, state_store


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


def test_authenticate_user_issues_resolvable_session(tmp_path, monkeypatch):
    _patch_store(monkeypatch, tmp_path)
    state_store.ensure_store()
    auth.ensure_user(user_id="analyst", password="secret", role="researcher")

    session = auth.authenticate_user(user_id="analyst", password="secret")

    assert session is not None
    resolved = auth.resolve_session(session["token"])
    assert resolved is not None
    assert resolved["user_id"] == "analyst"


def test_bound_user_overrides_legacy_local_user(tmp_path, monkeypatch):
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setenv("STRATEGYBLENDER_USER", "local-user")

    auth.bind_user("analyst")
    assert state_store.current_user_id() == "analyst"
    auth.bind_user(None)
