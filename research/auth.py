from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone

from config.settings import AUTH_SESSION_TTL_HOURS
from research import state_store


_CURRENT_USER_ID: ContextVar[str | None] = ContextVar("strategyblender_current_user_id", default=None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_secret(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def bind_user(user_id: str | None) -> None:
    _CURRENT_USER_ID.set(str(user_id).strip() if user_id else None)


def bound_user_id() -> str | None:
    user_id = _CURRENT_USER_ID.get()
    if user_id:
        return user_id
    token = os.environ.get("STRATEGYBLENDER_AUTH_TOKEN", "").strip()
    if token:
        session = resolve_session(token)
        if session:
            return session["user_id"]
    return None


def ensure_auth_tables() -> None:
    state_store.ensure_store()


def ensure_user(*, user_id: str, password: str, role: str = "researcher") -> dict:
    ensure_auth_tables()
    normalized = str(user_id).strip()
    if not normalized:
        raise ValueError("user_id is required")
    salt = secrets.token_hex(16)
    password_hash = _hash_secret(password, salt)
    with state_store._connect() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO users (user_id, role, created_at, password_hash, password_salt)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                role = excluded.role,
                password_hash = excluded.password_hash,
                password_salt = excluded.password_salt
            """,
            (normalized, role, _utc_now(), password_hash, salt),
        )
        conn.commit()
    return {"user_id": normalized, "role": role}


def bootstrap_default_admin() -> dict:
    ensure_auth_tables()
    user_id = os.environ.get("STRATEGYBLENDER_BOOTSTRAP_USER", "admin").strip() or "admin"
    password = os.environ.get("STRATEGYBLENDER_BOOTSTRAP_PASSWORD", "admin")
    with state_store._connect() as conn:  # noqa: SLF001
        row = conn.execute(
            "SELECT user_id, role, password_hash, password_salt FROM users ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
    if row:
        if not row["password_hash"] or not row["password_salt"]:
            return ensure_user(user_id=row["user_id"], password=password, role=row["role"])
        return {"user_id": row["user_id"], "role": row["role"]}
    return ensure_user(user_id=user_id, password=password, role="admin")


def authenticate_user(*, user_id: str, password: str) -> dict | None:
    ensure_auth_tables()
    normalized = str(user_id).strip()
    with state_store._connect() as conn:  # noqa: SLF001
        row = conn.execute(
            "SELECT user_id, role, password_hash, password_salt FROM users WHERE user_id = ?",
            (normalized,),
        ).fetchone()
    if not row:
        return None
    expected = row["password_hash"] or ""
    salt = row["password_salt"] or ""
    if not expected or not salt:
        return None
    actual = _hash_secret(password, salt)
    if not hmac.compare_digest(actual, expected):
        return None
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(hours=AUTH_SESSION_TTL_HOURS)
    with state_store._connect() as conn:  # noqa: SLF001
        conn.execute(
            """
            INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at, revoked_at)
            VALUES (?, ?, ?, ?, '')
            """,
            (token_hash, normalized, created_at.isoformat(), expires_at.isoformat()),
        )
        conn.commit()
    return {"token": token, "user_id": normalized, "role": row["role"], "expires_at": expires_at.isoformat()}


def resolve_session(token: str) -> dict | None:
    if not token:
        return None
    ensure_auth_tables()
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with state_store._connect() as conn:  # noqa: SLF001
        row = conn.execute(
            """
            SELECT user_id, created_at, expires_at, revoked_at
            FROM auth_sessions
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
    if not row or row["revoked_at"]:
        return None
    expires = datetime.fromisoformat(row["expires_at"])
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires <= datetime.now(timezone.utc):
        return None
    return {
        "user_id": row["user_id"],
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
    }


def revoke_session(token: str) -> None:
    if not token:
        return
    ensure_auth_tables()
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with state_store._connect() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ?",
            (_utc_now(), token_hash),
        )
        conn.commit()
