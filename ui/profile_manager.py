"""
Local workspace persistence.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet


PROFILES_DIR = Path(__file__).parent.parent / "config" / "profiles"
KEY_FILE = Path(__file__).parent.parent / "config" / ".workspace_key"
WORKSPACE_PROFILE = PROFILES_DIR / "workspace_local.enc"


def _get_key() -> bytes:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    KEY_FILE.chmod(0o600)
    return key


def _df_to_dict(df: pd.DataFrame | None) -> dict | None:
    if df is None or df.empty:
        return None
    return {"index": list(df.index.astype(str)), "data": df.to_dict(orient="list")}


def _dict_to_df(d: dict | None) -> pd.DataFrame:
    if not d:
        return pd.DataFrame()
    frame = pd.DataFrame(d["data"], index=d["index"])
    if frame.empty:
        return frame

    raw_index = pd.Index(frame.index)
    parsed_index = pd.to_datetime(raw_index, errors="coerce")
    valid_ratio = float(parsed_index.notna().mean()) if len(parsed_index) else 0.0
    if valid_ratio >= 0.8:
        frame.index = pd.DatetimeIndex(parsed_index)
    else:
        frame.index = raw_index
    return frame


def _serialise_results(results: dict) -> dict:
    out = {}
    for ea_id, result in results.items():
        out[ea_id] = {
            "summary": result.get("summary", {}),
            "monthly_df": _df_to_dict(result.get("monthly_df")),
            "deals_df": _df_to_dict(result.get("deals_df")),
            "balance_curve_df": _df_to_dict(result.get("balance_curve_df")),
            "meta": result.get("meta", {}),
        }
    return out


def _deserialise_results(data: dict) -> dict:
    out = {}
    for ea_id, result in data.items():
        out[ea_id] = {
            "summary": result.get("summary", {}),
            "monthly_df": _dict_to_df(result.get("monthly_df")),
            "deals_df": _dict_to_df(result.get("deals_df")),
            "balance_curve_df": _dict_to_df(result.get("balance_curve_df")),
            "meta": result.get("meta", {}),
        }
    return out


def save_workspace_profile(state: dict):
    training_artifact = state.get("training_artifact")
    if isinstance(training_artifact, dict):
        training_artifact = copy.deepcopy(training_artifact)
        # Raw end-to-end payloads embed DataFrames that do not survive JSON round-trips.
        # Saved AI reports keep a compact snapshot already, so do not persist the raw payload here.
        training_artifact["actual_compare"] = None

    payload = {
        "eas": state.get("eas", {}),
        "backtest_results": _serialise_results(state.get("backtest_results", {})),
        "ai_experiment_results": _serialise_results(state.get("ai_experiment_results", {})),
        "model_trained": state.get("model_trained", False),
        "training_log": state.get("training_log", []),
        "training_artifact": training_artifact,
        "schedule_path": state.get("schedule_path", ""),
        "ai_saved_reports": state.get("ai_saved_reports", []),
    }
    raw = json.dumps(payload, default=str).encode()
    enc = Fernet(_get_key()).encrypt(raw)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_PROFILE.write_bytes(enc)
    WORKSPACE_PROFILE.chmod(0o600)


def load_workspace_profile() -> dict | None:
    if not WORKSPACE_PROFILE.exists():
        return None
    try:
        raw = Fernet(_get_key()).decrypt(WORKSPACE_PROFILE.read_bytes())
        payload = json.loads(raw.decode())
        payload["backtest_results"] = _deserialise_results(
            payload.get("backtest_results", {})
        )
        payload["ai_experiment_results"] = _deserialise_results(
            payload.get("ai_experiment_results", {})
        )
        return payload
    except Exception:
        return None
