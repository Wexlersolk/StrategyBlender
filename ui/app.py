"""
ui/app.py — StrategyBlender main entry point

Run with:
    python -m streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui.state import init_state, autosave
from ui.mt5_auth import (
    auto_connect, connect, disconnect,
    save_credentials, clear_credentials, load_credentials
)
from ui.profile_manager import (
    list_profiles, load_profile, delete_profile,
    apply_profile_to_state, profile_id
)

st.set_page_config(
    page_title="StrategyBlender",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1A1F2E;
        border-right: 1px solid #2A3044;
    }
    /* Hide "Press Enter to apply" tooltip on form inputs */
    .stTextInput div[data-baseweb="input"] + div small,
    [data-testid="InputInstructions"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Account card styling */
    .account-card {
        background: #0E1117;
        border: 1px solid #2A3044;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 3px 0;
        cursor: pointer;
    }
    .account-card.active {
        border-color: #2E75B6;
        background: #162032;
    }
</style>
""", unsafe_allow_html=True)


def main():
    init_state()

    # Auto-connect once on startup
    if not st.session_state.get("_auto_connect_done"):
        _try_auto_connect()
        st.session_state["_auto_connect_done"] = True

    with st.sidebar:
        _render_sidebar()

    _render_page()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar():
    # Logo
    st.markdown("""
    <div style='padding:1rem 0 1rem 0;'>
        <h1 style='color:#2E75B6;font-size:1.5rem;margin:0;'>⚡ StrategyBlender</h1>
        <p style='color:#8B9BB4;font-size:0.75rem;margin:0;'>AI-Powered Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Account switcher ──────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#8B9BB4;font-size:0.75rem;letter-spacing:0.08em;"
        "text-transform:uppercase;margin:0 0 6px 0;'>Accounts</p>",
        unsafe_allow_html=True
    )

    profiles = list_profiles()
    active   = st.session_state.get("active_profile_id")

    for p in profiles:
        is_active = p["id"] == active
        col_btn, col_del = st.columns([5, 1])

        with col_btn:
            label = (
                f"{'🟢' if is_active else '⚪'}  **{p['login']}** "
                f"{p['server']}\n"
                f"<small style='color:#8B9BB4;'>"
                f"{p['n_eas']} EA{'s' if p['n_eas'] != 1 else ''}  "
                f"{'· AI ✅' if p['model_trained'] else ''}</small>"
            )
            if st.button(
                f"{'🟢' if is_active else '⚪'}  {p['login']}  {p['server']}",
                key=f"switch_{p['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    _switch_account(p["login"], p["server"])

        with col_del:
            if st.button("✕", key=f"del_profile_{p['id']}", help="Remove account"):
                _remove_account(p["login"], p["server"], is_active)

    # Add account button
    if st.button("＋  Add account", use_container_width=True, key="add_account"):
        st.session_state["_show_login"] = True

    # Login form (shown when adding new account or not connected)
    connected = st.session_state.get("mt5_connected", False)
    if st.session_state.get("_show_login") or not connected:
        with st.expander("Login to MT5", expanded=not connected):
            _render_login_form()

    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#8B9BB4;font-size:0.75rem;letter-spacing:0.08em;"
        "text-transform:uppercase;margin:0 0 6px 0;'>Navigation</p>",
        unsafe_allow_html=True
    )

    pages = {
        "Dashboard":   "📊",
        "EA Manager":  "🤖",
        "Backtests":   "▶️",
        "AI Training": "🧠",
        "Research":    "🔬",
    }
    for page_name, icon in pages.items():
        is_active_page = st.session_state.get("page") == page_name
        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active_page else "secondary",
        ):
            st.session_state["page"] = page_name
            st.rerun()

    st.markdown("---")

    # ── Status summary ────────────────────────────────────────────────────────
    n_eas    = len(st.session_state.get("eas", {}))
    n_tested = len(st.session_state.get("backtest_results", {}))
    trained  = st.session_state.get("model_trained", False)

    st.markdown(f"""
    <div style='color:#8B9BB4;font-size:0.8rem;line-height:2;'>
        EAs loaded:&nbsp; <b style='color:#E8EDF3;'>{n_eas}</b><br>
        Backtested:&nbsp; <b style='color:#E8EDF3;'>{n_tested}</b><br>
        AI trained:&nbsp; <b style='color:{"#2ECC71" if trained else "#E74C3C"};'>
            {"Yes ✅" if trained else "No"}</b>
    </div>
    """, unsafe_allow_html=True)

    # Disconnect button
    if connected:
        st.markdown("")
        if st.button("Disconnect", use_container_width=True, type="secondary"):
            autosave()
            disconnect()
            st.session_state["mt5_connected"] = False
            st.session_state["mt5_creds"]     = {}
            st.session_state["active_profile_id"] = None
            st.rerun()


# ── Account actions ───────────────────────────────────────────────────────────

def _switch_account(login: int, server: str):
    """Save current state, load new account, reconnect."""
    # Auto-save current
    autosave()

    # Load the target profile
    profile = load_profile(login, server)
    if profile is None:
        st.error(f"Could not load profile for {login} @ {server}")
        return

    # Disconnect current MT5 session
    disconnect()
    st.session_state["mt5_connected"] = False

    # Apply profile to state
    apply_profile_to_state(profile, st.session_state)

    # Reconnect with new credentials
    ok, msg = connect(
        login=login,
        password=profile["password"],
        server=server,
        path=profile.get("path", ""),
    )
    st.session_state["mt5_connected"] = ok
    st.session_state["mt5_message"]   = msg
    st.session_state["_show_login"]   = False
    st.rerun()


def _remove_account(login: int, server: str, is_active: bool):
    """Delete a saved profile."""
    delete_profile(login, server)
    if is_active:
        disconnect()
        st.session_state["mt5_connected"]    = False
        st.session_state["mt5_creds"]        = {}
        st.session_state["active_profile_id"] = None
        st.session_state["eas"]              = {}
        st.session_state["backtest_results"] = {}
        st.session_state["model_trained"]    = False
    st.rerun()


def _try_auto_connect():
    """On startup, try to reconnect the last active account."""
    profiles = list_profiles()
    if not profiles:
        return

    # Try the first (most recently saved) profile
    p = profiles[0]
    profile = load_profile(p["login"], p["server"])
    if profile is None:
        return

    ok, msg = connect(
        login=profile["login"],
        password=profile["password"],
        server=profile["server"],
        path=profile.get("path", ""),
    )

    if ok:
        apply_profile_to_state(profile, st.session_state)
        st.session_state["mt5_connected"] = True
        st.session_state["mt5_message"]   = msg


# ── Login form ────────────────────────────────────────────────────────────────

def _render_login_form():
    with st.form("mt5_login_form", clear_on_submit=False):
        login    = st.text_input("Account login", placeholder="e.g. 1234567")
        password = st.text_input("Password", type="password")
        server   = st.text_input("Server", placeholder="e.g. FTMO-Demo2")
        path     = st.text_input("MT5 path (optional)", placeholder="Leave blank for auto-detect")
        submitted = st.form_submit_button("Connect & Save", use_container_width=True,
                                           type="primary")

        if submitted:
            if not login.strip() or not password or not server:
                st.error("Fill in all required fields.")
                return

            try:
                login_int = int(login.strip())
            except ValueError:
                st.error("Account login must be a number.")
                return

            with st.spinner("Connecting..."):
                ok, msg = connect(
                    login=login_int, password=password,
                    server=server, path=path or ""
                )

            if ok:
                st.session_state["mt5_connected"]    = True
                st.session_state["mt5_message"]      = msg
                st.session_state["mt5_creds"]        = {
                    "login": login_int, "password": password,
                    "server": server, "path": path or ""
                }
                st.session_state["active_profile_id"] = profile_id(login_int, server)
                st.session_state["_show_login"]       = False
                autosave()
                st.rerun()
            else:
                st.error(msg)


# ── Page router ───────────────────────────────────────────────────────────────

def _render_page():
    page = st.session_state.get("page", "Dashboard")

    if page == "Dashboard":
        from ui.pages.dashboard import render
        render()
    elif page == "EA Manager":
        from ui.pages.ea_manager import render
        render()
    elif page == "Backtests":
        from ui.pages.backtests import render
        render()
    elif page == "AI Training":
        from ui.pages.ai_training import render
        render()
    elif page == "Research":
        from ui.pages.research import render
        render()


if __name__ == "__main__":
    main()
