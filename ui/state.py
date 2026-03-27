"""
ui/state.py

Session state initialisation + auto-save helper.
"""

import streamlit as st


def init_state():
    defaults = {
        # EA manager
        "eas":                {},     # { ea_id: { name, symbol, timeframe, source, python_source, engine_source, ... } }
        "ea_counter":         0,

        # Backtests
        "backtest_results":   {},     # { ea_id: { monthly_df, deals_df, summary } }
        "ai_experiment_results": {},  # { exp_id: { monthly_df, deals_df, summary, meta } }

        # AI training
        "model_trained":      False,
        "training_log":       [],
        "training_artifact":  None,
        "schedule_path":      "",
        "ai_saved_reports":   [],

        # Navigation
        "page":               "Dashboard",

        # Internal flags
        "_workspace_loaded":  False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def autosave():
    """
    Save the current session state to the active profile.
    Call this after any meaningful change (EA added, backtest done, etc).
    """
    from ui.profile_manager import save_workspace_profile
    save_workspace_profile(dict(st.session_state))
