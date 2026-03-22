"""
ui/profile_manager.py

Encrypted per-account profile storage.
Each MT5 account gets its own encrypted JSON file containing:
  - credentials
  - EAs (source, params, symbol, timeframe)
  - backtest results (serialised)
  - AI training state

File naming: config/profiles/{login}_{server}.enc
"""

import os
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from cryptography.fernet import Fernet

PROFILES_DIR = Path(__file__).parent.parent / "config" / "profiles"
KEY_FILE     = Path(__file__).parent.parent / "config" / ".mt5_key"


# ── Encryption helpers ────────────────────────────────────────────────────────

def _get_key() -> bytes:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    KEY_FILE.chmod(0o600)
    return key


def _profile_path(login: int, server: str) -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    safe_server = server.replace(" ", "_").replace("/", "-")
    return PROFILES_DIR / f"{login}_{safe_server}.enc"


def profile_id(login: int, server: str) -> str:
    """Canonical string ID for an account."""
    return f"{login}_{server}"


# ── Serialisation helpers ─────────────────────────────────────────────────────
# DataFrames are not JSON-serialisable by default, so we convert them.

def _df_to_dict(df: pd.DataFrame | None) -> dict | None:
    if df is None or df.empty:
        return None
    return {"index": list(df.index.astype(str)), "data": df.to_dict(orient="list")}


def _dict_to_df(d: dict | None) -> pd.DataFrame:
    if not d:
        return pd.DataFrame()
    df = pd.DataFrame(d["data"], index=d["index"])
    return df


def _serialise_results(results: dict) -> dict:
    """Convert backtest_results dict to JSON-safe format."""
    out = {}
    for ea_id, r in results.items():
        out[ea_id] = {
            "summary":    r.get("summary", {}),
            "monthly_df": _df_to_dict(r.get("monthly_df")),
            "deals_df":   _df_to_dict(r.get("deals_df")),
        }
    return out


def _deserialise_results(data: dict) -> dict:
    """Restore backtest_results from JSON-safe format."""
    out = {}
    for ea_id, r in data.items():
        out[ea_id] = {
            "summary":    r.get("summary", {}),
            "monthly_df": _dict_to_df(r.get("monthly_df")),
            "deals_df":   _dict_to_df(r.get("deals_df")),
        }
    return out


# ── Core API ──────────────────────────────────────────────────────────────────

def save_profile(login: int, server: str, state: dict):
    """
    Save the full session state for one account.
    Called automatically on every meaningful change.
    """
    payload = {
        "login":             login,
        "server":            server,
        "password":          state.get("mt5_creds", {}).get("password", ""),
        "path":              state.get("mt5_creds", {}).get("path", ""),
        "eas":               state.get("eas", {}),
        "backtest_results":  _serialise_results(state.get("backtest_results", {})),
        "model_trained":     state.get("model_trained", False),
        "training_log":      state.get("training_log", []),
        "schedule_path":     state.get("schedule_path", ""),
    }

    raw  = json.dumps(payload, default=str).encode()
    fernet = Fernet(_get_key())
    enc  = fernet.encrypt(raw)

    path = _profile_path(login, server)
    path.write_bytes(enc)
    path.chmod(0o600)


def load_profile(login: int, server: str) -> dict | None:
    """
    Load a saved profile. Returns None if not found.
    """
    path = _profile_path(login, server)
    if not path.exists():
        return None
    try:
        fernet  = Fernet(_get_key())
        raw     = fernet.decrypt(path.read_bytes())
        payload = json.loads(raw.decode())
        # Restore DataFrames
        payload["backtest_results"] = _deserialise_results(
            payload.get("backtest_results", {})
        )
        return payload
    except Exception:
        return None


def list_profiles() -> list[dict]:
    """
    Return a list of all saved profiles as metadata dicts:
      [{ "login": 12345, "server": "FTMO-Demo2", "id": "12345_FTMO-Demo2",
         "n_eas": 3, "model_trained": True }, ...]
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profiles = []
    for enc_file in sorted(PROFILES_DIR.glob("*.enc")):
        try:
            fernet  = Fernet(_get_key())
            raw     = fernet.decrypt(enc_file.read_bytes())
            payload = json.loads(raw.decode())
            profiles.append({
                "login":         payload["login"],
                "server":        payload["server"],
                "id":            profile_id(payload["login"], payload["server"]),
                "n_eas":         len(payload.get("eas", {})),
                "model_trained": payload.get("model_trained", False),
            })
        except Exception:
            continue
    return profiles


def delete_profile(login: int, server: str):
    """Delete a saved profile."""
    path = _profile_path(login, server)
    if path.exists():
        path.unlink()


def apply_profile_to_state(profile: dict, st_state: dict):
    """
    Write a loaded profile into Streamlit session state.
    Called when the user switches accounts.
    """
    st_state["mt5_creds"] = {
        "login":    profile["login"],
        "password": profile["password"],
        "server":   profile["server"],
        "path":     profile.get("path", ""),
    }
    st_state["eas"]               = profile.get("eas", {})
    st_state["backtest_results"]  = profile.get("backtest_results", {})
    st_state["model_trained"]     = profile.get("model_trained", False)
    st_state["training_log"]      = profile.get("training_log", [])
    st_state["schedule_path"]     = profile.get("schedule_path", "")
    st_state["active_profile_id"] = profile_id(profile["login"], profile["server"])
