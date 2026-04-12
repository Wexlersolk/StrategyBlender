"""
ui/app.py — StrategyBlender main entry point

Run with:
    python -m streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from research import auth
from ui.state import init_state
from ui.profile_manager import (
    load_workspace_profile
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
    auth.bootstrap_default_admin()
    _bind_auth_context()
    if not st.session_state.get("auth_user_id"):
        with st.sidebar:
            _render_login()
        st.info("Sign in to access persisted research jobs and experiments.")
        return
    if not st.session_state.get("_workspace_loaded"):
        _load_workspace_state()
        st.session_state["_workspace_loaded"] = True

    with st.sidebar:
        _render_sidebar()

    _render_page()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar():
    # Logo
    st.markdown("""
    <div style='padding:1rem 0 1rem 0;'>
        <h1 style='color:#2E75B6;font-size:1.5rem;margin:0;'>⚡ StrategyBlender</h1>
        <p style='color:#8B9BB4;font-size:0.75rem;margin:0;'>Local Strategy Conversion and Research Platform</p>
    </div>
    """, unsafe_allow_html=True)

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
        "Strategy Builder": "🧩",
        "Backtests":   "▶️",
        "AI Training": "🧠",
        "Experiments": "🧪",
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

    user_id = st.session_state.get("auth_user_id", "")
    st.caption(f"Signed in as `{user_id}`")
    if st.button("Sign Out", key="sign_out", use_container_width=True, type="secondary"):
        token = st.session_state.get("auth_session_token", "")
        auth.revoke_session(token)
        st.session_state["auth_user_id"] = ""
        st.session_state["auth_session_token"] = ""
        auth.bind_user(None)
        st.rerun()

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

def _load_workspace_state():
    profile = load_workspace_profile()
    if not profile:
        return
    st.session_state["eas"] = profile.get("eas", {})
    st.session_state["backtest_results"] = profile.get("backtest_results", {})
    st.session_state["ai_experiment_results"] = profile.get("ai_experiment_results", {})
    st.session_state["model_trained"] = profile.get("model_trained", False)
    st.session_state["training_log"] = profile.get("training_log", [])
    st.session_state["training_artifact"] = profile.get("training_artifact")
    st.session_state["_ai_dataset_id"] = profile.get("ai_dataset_id", "")
    st.session_state["_ai_training_artifact_id"] = profile.get("ai_training_artifact_id", "")
    st.session_state["schedule_path"] = profile.get("schedule_path", "")
    st.session_state["ai_saved_reports"] = profile.get("ai_saved_reports", [])
    st.session_state["_ai_model_wfo"] = profile.get("ai_model_wfo")
    st.session_state["_ai_model_wfo_run_id"] = profile.get("ai_model_wfo_run_id", "")
    st.session_state["experiment_registry"] = profile.get("experiment_registry", [])
    st.session_state["job_registry"] = profile.get("job_registry", [])
    if not st.session_state.get("auth_user_id"):
        st.session_state["auth_user_id"] = profile.get("auth_user_id", "")


def _bind_auth_context():
    token = st.session_state.get("auth_session_token", "")
    user_id = st.session_state.get("auth_user_id", "")
    if token:
        session = auth.resolve_session(token)
        if session:
            user_id = session["user_id"]
            st.session_state["auth_user_id"] = user_id
        else:
            st.session_state["auth_user_id"] = ""
            st.session_state["auth_session_token"] = ""
            user_id = ""
    auth.bind_user(user_id or None)


def _render_login():
    st.markdown("### Sign In")
    username = st.text_input("User ID", key="login_user_id")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Sign In", key="login_submit", use_container_width=True, type="primary"):
        session = auth.authenticate_user(user_id=username, password=password)
        if not session:
            st.error("Invalid credentials.")
            return
        st.session_state["auth_user_id"] = session["user_id"]
        st.session_state["auth_session_token"] = session["token"]
        auth.bind_user(session["user_id"])
        st.rerun()


# ── Page router ───────────────────────────────────────────────────────────────

def _render_page():
    page = st.session_state.get("page", "Dashboard")

    if page == "Dashboard":
        from ui.views.dashboard import render
        render()
    elif page == "EA Manager":
        from ui.views.ea_manager import render
        render()
    elif page == "Strategy Builder":
        from ui.views.strategy_builder import render
        render()
    elif page == "Backtests":
        from ui.views.backtests import render
        render()
    elif page == "AI Training":
        from ui.views.ai_training import render
        render()
    elif page == "Experiments":
        from ui.views.experiments import render
        render()
    elif page == "Research":
        from ui.views.research import render
        render()


if __name__ == "__main__":
    main()
