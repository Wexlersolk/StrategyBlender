"""
ui/pages/dashboard.py — Portfolio overview dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from ui.components.common import section_header, empty_state, metric_card
from ui.mt5_auth import get_account_info


def render():
    section_header("Dashboard", "Portfolio overview and account status")

    # ── Account info row ──────────────────────────────────────────────────────
    info = get_account_info() if st.session_state.get("mt5_connected") else None

    if info:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Balance", f"{info['balance']:,.2f} {info['currency']}")
        with c2:
            st.metric("Equity", f"{info['equity']:,.2f} {info['currency']}")
        with c3:
            st.metric("Free Margin", f"{info['margin']:,.2f}")
        with c4:
            st.metric("Leverage", f"1:{info['leverage']}")
        with c5:
            st.metric("Account", str(info["login"]))
    else:
        st.warning("Not connected to MT5. Go to the sidebar to log in.", icon="⚠️")

    st.markdown("---")

    # ── EA summary ────────────────────────────────────────────────────────────
    eas = st.session_state.get("eas", {})
    results = st.session_state.get("backtest_results", {})

    if not eas:
        empty_state("No EAs loaded yet. Go to EA Manager to get started.", "🤖")
        return

    # ── EA status table ───────────────────────────────────────────────────────
    st.markdown("#### Loaded Strategies")

    rows = []
    for ea_id, ea in eas.items():
        r = results.get(ea_id, {})
        summary = r.get("summary", {})
        rows.append({
            "Strategy":     ea["name"],
            "Symbol":       ea["symbol"],
            "Timeframe":    ea["timeframe"],
            "Net Profit":   f"${summary.get('total_profit', 0):,.0f}" if summary else "—",
            "Sharpe":       f"{summary.get('sharpe_mean', 0):.2f}" if summary else "—",
            "AI":           "✅ On" if ea.get("ai_enabled") else "⬜ Off",
            "Status":       "Backtested" if summary else "Pending backtest",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Portfolio equity curve (if backtests exist) ───────────────────────────
    if not results:
        st.markdown("---")
        st.info("Run backtests to see the portfolio equity curve.", icon="ℹ️")
        return

    st.markdown("---")
    st.markdown("#### Portfolio Equity Curve")
    _render_portfolio_curve(results)

    # ── Per-strategy Sharpe comparison ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Strategy Comparison")
    _render_comparison_bars(eas, results)


def _render_portfolio_curve(results: dict):
    """Combine all strategy monthly P&L into one equity curve."""
    all_dfs = []
    for ea_id, r in results.items():
        df = r.get("monthly_df")
        if df is not None and not df.empty and "total_profit" in df.columns:
            tmp = df[["total_profit"]].copy()
            tmp.index = pd.to_datetime(tmp.index)
            all_dfs.append(tmp)

    if not all_dfs:
        st.info("No monthly data available yet.")
        return

    combined = pd.concat(all_dfs, axis=1).fillna(0)
    combined.columns = [f"EA_{i}" for i in range(len(combined.columns))]
    combined["portfolio"] = combined.sum(axis=1)
    combined["equity"]    = 100_000 + combined["portfolio"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined.index, y=combined["equity"],
        mode="lines", name="Portfolio Equity",
        line=dict(color="#2E75B6", width=2),
        fill="tozeroy", fillcolor="rgba(46,117,182,0.1)"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis_title="Equity ($)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_comparison_bars(eas: dict, results: dict):
    """Bar chart comparing Sharpe ratios across strategies."""
    names, sharpes, colors = [], [], []
    for ea_id, ea in eas.items():
        summary = results.get(ea_id, {}).get("summary", {})
        if summary:
            names.append(f"{ea['name']} / {ea['symbol']}")
            sharpes.append(summary.get("sharpe_mean", 0))
            colors.append("#2E75B6" if ea.get("ai_enabled") else "#5A6478")

    if not names:
        st.info("Backtest at least one strategy to see comparisons.")
        return

    fig = px.bar(
        x=names, y=sharpes, color=names,
        color_discrete_sequence=colors,
        labels={"x": "Strategy", "y": "Sharpe Ratio"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
