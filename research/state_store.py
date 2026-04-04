from __future__ import annotations

import getpass
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
import uuid

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
STORE_ROOT = ROOT / "data" / "research_state"
JOBS_DIR = STORE_ROOT / "jobs"
EXPERIMENTS_DIR = STORE_ROOT / "experiments"
DATASETS_DIR = STORE_ROOT / "datasets"
ARTIFACTS_DIR = STORE_ROOT / "artifacts"
LOGS_DIR = STORE_ROOT / "logs"
DB_PATH = STORE_ROOT / "research_state.db"
AUDIT_LOG_PATH = LOGS_DIR / "audit.jsonl"

_STORE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bootstrap_user_id() -> str:
    return os.environ.get("STRATEGYBLENDER_BOOTSTRAP_USER", "admin").strip() or "admin"


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def current_user_id() -> str:
    from research import auth

    bound = auth.bound_user_id()
    if bound:
        return bound
    return os.environ.get("STRATEGYBLENDER_USER", getpass.getuser())


def ensure_store() -> None:
    for path in [STORE_ROOT, JOBS_DIR, EXPERIMENTS_DIR, DATASETS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                password_hash TEXT NOT NULL DEFAULT '',
                password_salt TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sequences (
                namespace TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                submitted_seq INTEGER NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                immutable INTEGER NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                dataset_hash TEXT NOT NULL,
                metadata_hash TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                metadata_hash TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                level TEXT NOT NULL,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                pid INTEGER NOT NULL,
                status TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                claimed_job_id TEXT NOT NULL,
                owner_id TEXT NOT NULL
            );
            """
        )
        _ensure_column(conn, "users", "password_hash", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "users", "password_salt", "TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, role, created_at) VALUES (?, ?, ?)",
            (_bootstrap_user_id(), "admin", _utc_now()),
        )
        conn.commit()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _atomic_write_text(path: Path, text: str) -> None:
    ensure_store()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _write_json(path: Path, payload: dict | list) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def _record_from_row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return json.loads(row["record_json"])


def canonical_hash(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def dataframe_hash(frame: pd.DataFrame) -> str:
    if frame is None:
        return canonical_hash(None)
    normalized = frame.copy()
    normalized.columns = [str(col) for col in normalized.columns]
    normalized = normalized.sort_index(axis=1)
    json_blob = normalized.to_json(orient="records", date_format="iso")
    return hashlib.sha256(json_blob.encode("utf-8")).hexdigest()


def text_hash(value: str | None) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def save_audit_event(*, entity_type: str, entity_id: str, event_type: str, payload: dict, level: str = "info", owner_id: str | None = None) -> None:
    ensure_store()
    owner = owner_id or current_user_id()
    envelope = {
        "owner_id": owner,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "event_type": event_type,
        "level": level,
        "created_at": _utc_now(),
        "payload": dict(payload or {}),
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_events (owner_id, entity_type, entity_id, level, event_type, created_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner,
                entity_type,
                entity_id,
                level,
                event_type,
                envelope["created_at"],
                json.dumps(envelope["payload"], default=str),
            ),
        )
        conn.commit()
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, default=str) + "\n")


def list_audit_events(*, entity_type: str | None = None, entity_id: str | None = None, owner_id: str | None = None, limit: int = 200) -> list[dict]:
    ensure_store()
    owner = owner_id or current_user_id()
    clauses = ["owner_id = ?"]
    params: list[object] = [owner]
    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if entity_id:
        clauses.append("entity_id = ?")
        params.append(entity_id)
    params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT owner_id, entity_type, entity_id, level, event_type, created_at, payload_json
            FROM audit_events
            WHERE {' AND '.join(clauses)}
            ORDER BY audit_id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [
        {
            "owner_id": row["owner_id"],
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "level": row["level"],
            "event_type": row["event_type"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload_json"]),
        }
        for row in rows
    ]


def next_sequence(namespace: str) -> int:
    ensure_store()
    with _STORE_LOCK, _connect() as conn:
        row = conn.execute("SELECT value FROM sequences WHERE namespace = ?", (namespace,)).fetchone()
        current = int(row["value"]) + 1 if row else 1
        conn.execute(
            "INSERT INTO sequences (namespace, value) VALUES (?, ?) ON CONFLICT(namespace) DO UPDATE SET value = excluded.value",
            (namespace, current),
        )
        conn.commit()
        return current


def _save_record(table: str, id_field: str, record: dict, *, immutable: bool = False) -> dict:
    ensure_store()
    owner = record.setdefault("owner_id", current_user_id())
    now = _utc_now()
    payload = json.dumps(record, default=str)
    with _connect() as conn:
        if immutable:
            existing = conn.execute(f"SELECT {id_field} FROM {table} WHERE {id_field} = ?", (record[id_field],)).fetchone()
            if existing:
                raise ValueError(f"{table[:-1].title()} already exists: {record[id_field]}")
        if table == "jobs":
            conn.execute(
                """
                INSERT INTO jobs (job_id, owner_id, submitted_seq, kind, title, status, updated_at, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    submitted_seq = excluded.submitted_seq,
                    kind = excluded.kind,
                    title = excluded.title,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    record_json = excluded.record_json
                """,
                (
                    record["job_id"],
                    owner,
                    int(record.get("submitted_seq", 0)),
                    record.get("kind", ""),
                    record.get("title", ""),
                    record.get("status", ""),
                    now,
                    payload,
                ),
            )
        elif table == "experiments":
            conn.execute(
                """
                INSERT INTO experiments (run_id, owner_id, created_at, kind, title, immutable, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["run_id"],
                    owner,
                    record.get("created_at", now),
                    record.get("kind", ""),
                    record.get("title", ""),
                    1,
                    payload,
                ),
            )
        elif table == "datasets":
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets (dataset_id, owner_id, created_at, dataset_hash, metadata_hash, record_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["dataset_id"],
                    owner,
                    record.get("created_at", now),
                    record.get("dataset_hash", ""),
                    record.get("metadata_hash", ""),
                    payload,
                ),
            )
        elif table == "artifacts":
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (artifact_id, owner_id, created_at, artifact_hash, metadata_hash, record_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["artifact_id"],
                    owner,
                    record.get("created_at", now),
                    record.get("artifact_hash", ""),
                    record.get("metadata_hash", ""),
                    payload,
                ),
            )
        conn.commit()
    save_audit_event(entity_type=table[:-1], entity_id=record[id_field], event_type=f"{table[:-1]}_saved", payload={"status": record.get("status", ""), "title": record.get("title", "")}, owner_id=owner)
    return record


def save_job(job: dict) -> dict:
    return _save_record("jobs", "job_id", dict(job))


def load_job(job_id: str, *, owner_id: str | None = None) -> dict | None:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        row = conn.execute("SELECT record_json FROM jobs WHERE job_id = ? AND owner_id = ?", (job_id, owner)).fetchone()
    return _record_from_row(row)


def list_jobs(*, owner_id: str | None = None) -> list[dict]:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT record_json FROM jobs WHERE owner_id = ? ORDER BY submitted_seq DESC, updated_at DESC",
            (owner,),
        ).fetchall()
    return [_record_from_row(row) for row in rows if _record_from_row(row) is not None]


def save_experiment_manifest(record: dict) -> dict:
    payload = dict(record)
    payload.setdefault("owner_id", current_user_id())
    payload.setdefault("access", {"visibility": "private", "writers": [payload["owner_id"]], "readers": [payload["owner_id"]]})
    payload.setdefault("immutable", True)
    return _save_record("experiments", "run_id", payload, immutable=True)


def load_experiment_manifest(run_id: str, *, owner_id: str | None = None) -> dict | None:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        row = conn.execute("SELECT record_json FROM experiments WHERE run_id = ? AND owner_id = ?", (run_id, owner)).fetchone()
    return _record_from_row(row)


def list_experiment_manifests(*, owner_id: str | None = None) -> list[dict]:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT record_json FROM experiments WHERE owner_id = ? ORDER BY created_at DESC",
            (owner,),
        ).fetchall()
    return [_record_from_row(row) for row in rows if _record_from_row(row) is not None]


def save_dataset_snapshot(*, dataset: pd.DataFrame, metadata: dict) -> dict:
    ensure_store()
    dataset_id = f"dataset_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    export_dir = DATASETS_DIR / dataset_id
    export_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = export_dir / "dataset.csv"
    metadata_path = export_dir / "metadata.json"
    dataset.to_csv(dataset_path, index=False)

    payload = {
        "dataset_id": dataset_id,
        "owner_id": current_user_id(),
        "created_at": _utc_now(),
        "rows": int(len(dataset)),
        "columns": [str(col) for col in dataset.columns],
        "dataset_hash": dataframe_hash(dataset),
        "metadata_hash": canonical_hash(metadata),
        "files": {
            "dataset": str(dataset_path),
            "metadata": str(metadata_path),
        },
        "metadata": dict(metadata or {}),
    }
    _write_json(metadata_path, payload)
    _save_record("datasets", "dataset_id", payload)
    return payload


def load_dataset_snapshot(dataset_id: str, *, owner_id: str | None = None) -> dict | None:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        row = conn.execute("SELECT record_json FROM datasets WHERE dataset_id = ? AND owner_id = ?", (dataset_id, owner)).fetchone()
    payload = _record_from_row(row)
    if not payload:
        return None
    dataset_path = payload.get("files", {}).get("dataset")
    frame = pd.read_csv(dataset_path) if dataset_path and Path(dataset_path).exists() else pd.DataFrame()
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    payload["dataset"] = frame
    return payload


def save_artifact_snapshot(*, artifact: dict, metadata: dict) -> dict:
    ensure_store()
    artifact_id = f"artifact_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    export_dir = ARTIFACTS_DIR / artifact_id
    export_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = export_dir / "artifact.json"
    metadata_path = export_dir / "metadata.json"
    _write_json(artifact_path, artifact)
    payload = {
        "artifact_id": artifact_id,
        "owner_id": current_user_id(),
        "created_at": _utc_now(),
        "artifact_hash": canonical_hash(artifact),
        "metadata_hash": canonical_hash(metadata),
        "files": {
            "artifact": str(artifact_path),
            "metadata": str(metadata_path),
        },
        "metadata": dict(metadata or {}),
    }
    _write_json(metadata_path, payload)
    _save_record("artifacts", "artifact_id", payload)
    return payload


def load_artifact_snapshot(artifact_id: str, *, owner_id: str | None = None) -> dict | None:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        row = conn.execute("SELECT record_json FROM artifacts WHERE artifact_id = ? AND owner_id = ?", (artifact_id, owner)).fetchone()
    payload = _record_from_row(row)
    if not payload:
        return None
    artifact_path = payload.get("files", {}).get("artifact")
    payload["artifact"] = json.loads(Path(artifact_path).read_text(encoding="utf-8")) if artifact_path and Path(artifact_path).exists() else {}
    return payload


def save_worker_heartbeat(*, worker_id: str, pid: int, status: str, claimed_job_id: str = "", owner_id: str | None = None) -> None:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO workers (worker_id, pid, status, heartbeat_at, claimed_job_id, owner_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                pid = excluded.pid,
                status = excluded.status,
                heartbeat_at = excluded.heartbeat_at,
                claimed_job_id = excluded.claimed_job_id,
                owner_id = excluded.owner_id
            """,
            (worker_id, int(pid), status, _utc_now(), claimed_job_id, owner),
        )
        conn.commit()


def list_workers(*, owner_id: str | None = None) -> list[dict]:
    ensure_store()
    owner = owner_id or current_user_id()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT worker_id, pid, status, heartbeat_at, claimed_job_id, owner_id FROM workers WHERE owner_id = ? ORDER BY heartbeat_at DESC",
            (owner,),
        ).fetchall()
    return [dict(row) for row in rows]


def worker_is_healthy(*, owner_id: str | None = None, max_age_seconds: int = 30) -> bool:
    threshold = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    for worker in list_workers(owner_id=owner_id):
        heartbeat = pd.Timestamp(worker["heartbeat_at"])
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.tz_localize("UTC")
        if heartbeat.to_pydatetime() >= threshold:
            return True
    return False


def claim_next_job(*, worker_id: str, owner_id: str | None = None) -> dict | None:
    ensure_store()
    owner = owner_id or current_user_id()
    with _STORE_LOCK, _connect() as conn:
        row = conn.execute(
            "SELECT record_json FROM jobs WHERE owner_id = ? AND status = 'queued' ORDER BY submitted_seq ASC LIMIT 1",
            (owner,),
        ).fetchone()
        record = _record_from_row(row)
        if not record:
            return None
        record["status"] = "running"
        record["stage"] = "running"
        record["worker_id"] = worker_id
        record["started_at"] = record.get("started_at") or _utc_now()
        record["attempt"] = int(record.get("attempt", 0)) + 1
        save_job(record)
        save_audit_event(entity_type="job", entity_id=record["job_id"], event_type="job_claimed", payload={"worker_id": worker_id, "attempt": record["attempt"]}, owner_id=owner)
        return record


def recover_orphaned_jobs(*, owner_id: str | None = None, stale_after_seconds: int = 45) -> int:
    ensure_store()
    owner = owner_id or current_user_id()
    workers = {worker["worker_id"]: worker for worker in list_workers(owner_id=owner)}
    threshold = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    recovered = 0
    for job in list_jobs(owner_id=owner):
        if job.get("status") != "running":
            continue
        worker = workers.get(job.get("worker_id", ""))
        heartbeat_ok = False
        if worker:
            heartbeat = pd.Timestamp(worker["heartbeat_at"])
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.tz_localize("UTC")
            heartbeat_ok = heartbeat.to_pydatetime() >= threshold
        if heartbeat_ok:
            continue
        if job.get("cancel_requested"):
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["finished_at"] = _utc_now()
        else:
            job["status"] = "queued"
            job["stage"] = "queued"
            job["progress_pct"] = 0.0
            job["worker_id"] = ""
        save_job(job)
        save_audit_event(entity_type="job", entity_id=job["job_id"], event_type="job_recovered", payload={"status": job["status"]}, level="warning", owner_id=owner)
        recovered += 1
    return recovered
