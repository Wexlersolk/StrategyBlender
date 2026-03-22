"""
ui/state.py

Session state initialisation + auto-save helper.
"""

import streamlit as st


def init_state():
    defaults = {
        # MT5 connection
        "mt5_connected":      False,
        "mt5_message":        "",
        "mt5_creds":          {},

        # Active account
        "active_profile_id":  None,   # e.g. "12345_FTMO-Demo2"

        # EA manager
        "eas":                {},     # { ea_id: { name, symbol, tf, params, source } }
        "ea_counter":         0,

        # Backtests
        "backtest_results":   {},     # { ea_id: { monthly_df, deals_df, summary } }

        # AI training
        "model_trained":      False,
        "training_log":       [],
        "schedule_path":      "",

        # Navigation
        "page":               "Dashboard",

        # Internal flags
        "_auto_connect_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def autosave():
    """
    Save the current session state to the active profile.
    Call this after any meaningful change (EA added, backtest done, etc).
    """
    from ui.profile_manager import save_profile

    creds = st.session_state.get("mt5_creds", {})
    login  = creds.get("login")
    server = creds.get("server")

    if not login or not server:
        return  # nothing to save yet

    save_profile(login, server, dict(st.session_state))
