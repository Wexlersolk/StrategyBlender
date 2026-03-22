"""
ui/components/common.py

Reusable UI building blocks used across all pages.
"""

import streamlit as st


def metric_card(label: str, value: str, delta: str = "", delta_color: str = "normal"):
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta if delta else None,
              delta_color=delta_color)


def status_badge(connected: bool) -> str:
    """Return a colored status string."""
    if connected:
        return "🟢 Connected"
    return "🔴 Disconnected"


def section_header(title: str, subtitle: str = ""):
    """Render a page section header."""
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<p style='color:#8B9BB4;margin-top:-12px;'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.markdown("---")


def info_box(message: str):
    st.info(message, icon="ℹ️")


def success_box(message: str):
    st.success(message, icon="✅")


def error_box(message: str):
    st.error(message, icon="🚨")


def warning_box(message: str):
    st.warning(message, icon="⚠️")


def empty_state(message: str, icon: str = "📭"):
    st.markdown(
        f"""<div style='text-align:center;padding:3rem;color:#8B9BB4;'>
            <div style='font-size:3rem;'>{icon}</div>
            <p style='margin-top:1rem;font-size:1.1rem;'>{message}</p>
        </div>""",
        unsafe_allow_html=True,
    )


TIMEFRAMES = ["M1","M5","M15","M30","H1","H4","D1","W1","MN"]

SYMBOLS_COMMON = [
    "XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURGBP", "EURJPY",
    "US30.cash", "US500.cash", "NAS100.cash", "HK50.cash",
    "UK100.cash", "GER40.cash", "USOIL.cash", "BTCUSD",
]
