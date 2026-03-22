"""
ui/mt5_auth.py

Handles MT5 connection with persistent login.
Credentials are saved locally (encrypted) so the user logs in once,
exactly like MT5 itself remembers the last account.
"""

import os
import json
import base64
import platform
import streamlit as st
from pathlib import Path
from cryptography.fernet import Fernet

# ── Credential storage ────────────────────────────────────────────────────────
CREDS_FILE = Path(__file__).parent.parent / "config" / ".mt5_credentials"
KEY_FILE   = Path(__file__).parent.parent / "config" / ".mt5_key"


def _get_or_create_key() -> bytes:
    """Get or create a machine-specific encryption key."""
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    KEY_FILE.chmod(0o600)
    return key


def save_credentials(login: int, password: str, server: str, path: str = ""):
    """Save MT5 credentials encrypted to disk."""
    key  = _get_or_create_key()
    f    = Fernet(key)
    data = json.dumps({
        "login":    login,
        "password": password,
        "server":   server,
        "path":     path,
    }).encode()
    CREDS_FILE.write_bytes(f.encrypt(data))
    CREDS_FILE.chmod(0o600)


def load_credentials() -> dict | None:
    """Load saved credentials. Returns None if none saved."""
    if not CREDS_FILE.exists() or not KEY_FILE.exists():
        return None
    try:
        key  = _get_or_create_key()
        f    = Fernet(key)
        data = f.decrypt(CREDS_FILE.read_bytes())
        return json.loads(data.decode())
    except Exception:
        return None


def clear_credentials():
    """Remove saved credentials (logout)."""
    if CREDS_FILE.exists():
        CREDS_FILE.unlink()


# ── MT5 connection ────────────────────────────────────────────────────────────
def get_mt5():
    """Return a connected mt5 instance for this OS."""
    if platform.system() == "Windows":
        import MetaTrader5 as mt5
        return mt5
    else:
        from mt5linux import MetaTrader5
        return MetaTrader5(host='localhost', port=18812)


def connect(login: int, password: str, server: str, path: str = "") -> tuple[bool, str]:
    """
    Connect to MT5. Returns (success, message).
    Passes credentials directly to initialize() which handles new accounts
    that have never connected to this terminal before.
    """
    mt5 = get_mt5()

    # First attempt: pass credentials directly to initialize()
    # This works for both existing and brand new accounts
    init_kwargs = dict(login=login, password=password, server=server)
    if path:
        init_kwargs["path"] = path

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        # If initialize with credentials failed, try bare initialize + login
        # (fallback for some broker configurations)
        if not mt5.initialize(**({"path": path} if path else {})):
            return False, f"MT5 initialization failed: {mt5.last_error()}"
        authorized = mt5.login(login=login, password=password, server=server)
        if not authorized:
            err = mt5.last_error()
            mt5.shutdown()
            # Friendly message for the common (1, 'Success') case
            if err[0] == 1:
                return False, (
                    "Authorization failed. This usually means:\n"
                    "• The account has never connected to this MT5 terminal before — "
                    "open MT5 manually, go to File → Open an Account, find your broker "
                    "and log in once.\n"
                    "• Wrong password or login number.\n"
                    "• The broker server name is incorrect."
                )
            return False, f"Login failed: {err}"

    info = mt5.account_info()
    if info is None:
        mt5.shutdown()
        return False, "Connected but could not retrieve account info. Check credentials."

    return True, f"Connected: {info.name} | {server} | Balance: {info.balance:.2f} {info.currency}"


def disconnect():
    """Disconnect from MT5."""
    try:
        mt5 = get_mt5()
        mt5.shutdown()
    except Exception:
        pass


def get_account_info() -> dict | None:
    """Return current account info as a dict, or None if not connected."""
    try:
        mt5   = get_mt5()
        info  = mt5.account_info()
        if info is None:
            return None
        return {
            "login":    info.login,
            "name":     info.name,
            "server":   info.server,
            "balance":  info.balance,
            "equity":   info.equity,
            "currency": info.currency,
            "leverage": info.leverage,
            "margin":   info.margin_free,
        }
    except Exception:
        return None


# ── Auto-connect on app start ─────────────────────────────────────────────────
def auto_connect() -> bool:
    """
    Try to connect using saved credentials.
    Returns True if successful. Called once on app startup.
    """
    if st.session_state.get("mt5_connected"):
        return True

    creds = load_credentials()
    if not creds:
        return False

    ok, msg = connect(
        login=creds["login"],
        password=creds["password"],
        server=creds["server"],
        path=creds.get("path", ""),
    )

    if ok:
        st.session_state["mt5_connected"] = True
        st.session_state["mt5_message"]   = msg
        st.session_state["mt5_creds"]     = creds
    else:
        st.session_state["mt5_connected"] = False

    return ok
