"""
ui/views/research.py

Research tab — backtest, WFO, Monte Carlo, AI comparison.
All tools use the Python engine (no MT5 needed).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from ui.components.common import section_header, empty_state, error_box
from services.backtest_service import (
    available_backtest_symbols,
    discover_strategies,
    run_backtest,
)


def get_available_symbols() -> list[str]:
    return available_backtest_symbols("H1")


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    section_header("Research", "Backtest, Walk-Forward, Monte Carlo and AI comparison")

    strategies = discover_strategies()
    if not strategies:
        empty_state(
            "No strategies found. Add a Python strategy to the strategies/ folder.",
            "🔬"
        )
        return

    tab_bt, tab_wfo, tab_mc, tab_ai = st.tabs([
        "▶️ Backtest",
        "📈 Walk-Forward",
        "🎲 Monte Carlo",
        "🤖 AI Comparison",
    ])

    with tab_bt:
        _render_backtest_tab(strategies)
    with tab_wfo:
        _render_wfo_tab(strategies)
    with tab_mc:
        _render_mc_tab()
    with tab_ai:
        _render_ai_tab()


# ── Backtest tab ──────────────────────────────────────────────────────────────

def _render_backtest_tab(strategies: dict):
    st.markdown("#### Run Backtest")
    st.markdown(
        "<p style='color:#8B9BB4;'>Runs entirely in Python — no MT5 needed. "
        "Results are approximate vs MT5 (~90% accuracy).</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        strat_name = st.selectbox("Strategy", list(strategies.keys()),
                                   key="bt_strategy")
        symbols    = get_available_symbols()
        symbol     = st.selectbox("Symbol", symbols, key="bt_symbol")
    with col2:
        date_from = st.date_input("From", value=pd.Timestamp("2020-06-01"),
                                   key="bt_from")
        date_to   = st.date_input("To",   value=pd.Timestamp("2025-12-31"),
                                   key="bt_to")

    # Parameter overrides
    strat_cls = strategies[strat_name]
    default_params = strat_cls.params.copy()

    with st.expander("⚙️ Parameter overrides (optional)"):
        overrides = {}
        cols = st.columns(3)
        numeric_params = {k: v for k, v in default_params.items()
                          if isinstance(v, (int, float))}
        for idx, (k, v) in enumerate(numeric_params.items()):
            with cols[idx % 3]:
                overrides[k] = st.number_input(
                    k, value=float(v), step=0.1 if isinstance(v, float) else 1.0,
                    key=f"bt_param_{k}"
                )

    intrabar = st.checkbox("Use intra-bar simulation (slower, more accurate)",
                            value=False, key="bt_intrabar")

    if st.button("▶️ Run Backtest", type="primary", use_container_width=True,
                  key="bt_run"):
        _run_and_show_backtest(
            strat_cls, symbol, str(date_from), str(date_to),
            overrides, intrabar_steps=60 if intrabar else 1
        )

    # Show cached result
    if "bt_result" in st.session_state and st.session_state["bt_result"]:
        _show_backtest_results(st.session_state["bt_result"])


def _run_and_show_backtest(strat_cls, symbol, date_from, date_to,
                            overrides, intrabar_steps=1):
    with st.spinner("Running backtest..."):
        try:
            r = run_backtest(
                strat_cls,
                symbol=symbol,
                timeframe="H1",
                date_from=date_from,
                date_to=date_to,
                overrides=overrides,
                intrabar_steps=intrabar_steps,
            )
            st.session_state["bt_result"] = r
            st.session_state["bt_last_strat"] = strat_cls.__name__
            st.rerun()
        except Exception as e:
            import traceback
            error_box(f"Backtest failed: {e}\n{traceback.format_exc()}")


def _show_backtest_results(r):
    st.markdown("---")
    st.markdown("##### Results")

    abs_dd, rel_dd = r.max_drawdown
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Net Profit",     f"${r.net_profit:,.0f}")
    c2.metric("Sharpe",         f"{r.sharpe_ratio:.2f}")
    c3.metric("Max Drawdown",   f"{rel_dd:.2f}%")
    c4.metric("Profit Factor",  f"{r.profit_factor:.2f}")
    c5.metric("Win Rate",       f"{r.win_rate*100:.1f}%")
    c6.metric("Trades",         str(r.n_trades))

    # Equity curve
    eq = r.equity_curve
    if len(eq) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines",
            line=dict(color="#2E75B6", width=2),
            fill="tozeroy", fillcolor="rgba(46,117,182,0.1)"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Equity ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Monthly P&L
    monthly = r.monthly_stats()
    if not monthly.empty:
        st.markdown("##### Monthly P&L")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=monthly.index,
            y=monthly["profit"],
            marker_color=["#2ECC71" if v > 0 else "#E74C3C"
                          for v in monthly["profit"]]
        ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=220,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Walk-Forward tab ──────────────────────────────────────────────────────────

def _render_wfo_tab(strategies: dict):
    st.markdown("#### Walk-Forward Optimization")
    st.markdown(
        "<p style='color:#8B9BB4;'>Tests whether optimised parameters remain "
        "robust out-of-sample. Robustness score ≥60% indicates the strategy "
        "is likely not overfitted.</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        strat_name   = st.selectbox("Strategy", list(strategies.keys()),
                                     key="wfo_strategy")
        symbols      = get_available_symbols()
        symbol       = st.selectbox("Symbol", symbols, key="wfo_symbol")
        train_months = st.slider("Train window (months)", 12, 48, 24, 6,
                                  key="wfo_train")
    with col2:
        test_months  = st.slider("Test window (months)", 3, 12, 6, 3,
                                  key="wfo_test")
        optimize_by  = st.selectbox("Optimise by",
                                     ["sharpe_ratio", "net_profit",
                                      "profit_factor", "recovery_factor"],
                                     key="wfo_metric")

    st.markdown("##### Parameter grid to search")
    strat_cls      = strategies[strat_name]
    default_params = strat_cls.params.copy()

    # Let user define ranges for numeric params
    param_grid = {}
    numeric_params = {k: v for k, v in default_params.items()
                      if isinstance(v, (int, float))}

    st.markdown(
        "<p style='color:#8B9BB4;font-size:0.85rem;'>Enter comma-separated values "
        "to search, or leave blank to use the default.</p>",
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    for idx, (k, v) in enumerate(list(numeric_params.items())[:9]):
        with cols[idx % 3]:
            raw = st.text_input(
                k, value="", placeholder=f"default: {v}",
                key=f"wfo_grid_{k}"
            )
            if raw.strip():
                try:
                    vals = [float(x.strip()) for x in raw.split(",")]
                    if len(vals) > 1:
                        param_grid[k] = vals
                except ValueError:
                    pass

    if not param_grid:
        st.info("Define at least one parameter range above to run WFO.", icon="ℹ️")

    if st.button("📈 Run Walk-Forward", type="primary",
                  use_container_width=True, key="wfo_run",
                  disabled=not param_grid):
        _run_wfo(strat_cls, symbol, param_grid, train_months,
                 test_months, optimize_by)

    if "wfo_result" in st.session_state and st.session_state["wfo_result"]:
        _show_wfo_results(st.session_state["wfo_result"])


def _run_wfo(strat_cls, symbol, param_grid, train_months, test_months, optimize_by):
    n_combos  = 1
    for v in param_grid.values():
        n_combos *= len(v)
    est_windows = max(1, 72 // test_months)
    est_bt      = n_combos * est_windows

    with st.spinner(f"Running WFO ({n_combos} combinations × ~{est_windows} windows "
                    f"= ~{est_bt} backtests)..."):
        try:
            from engine.data_loader    import load_bars
            from research.walk_forward import run_wfo

            df = load_bars(symbol, "H1")
            results = run_wfo(
                strategy_class    = strat_cls,
                df                = df,
                param_grid        = param_grid,
                train_months      = train_months,
                test_months       = test_months,
                optimize_by       = optimize_by,
                backtester_kwargs = {
                    "initial_capital": 100_000,
                    "lot_value":       getattr(strat_cls, "lot_value", 1.0),
                },
            )
            st.session_state["wfo_result"] = results
            st.rerun()
        except Exception as e:
            import traceback
            error_box(f"WFO failed: {e}\n{traceback.format_exc()}")


def _show_wfo_results(results):
    st.markdown("---")
    s = results.summary()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Windows",        str(s["n_windows"]))
    c2.metric("Profitable",     f"{s['profitable_windows']}/{s['n_windows']}")
    c3.metric("Robustness",     f"{s['robustness_score']*100:.0f}%",
              delta="Good" if s["robustness_score"] >= 0.6 else "Weak",
              delta_color="normal" if s["robustness_score"] >= 0.6 else "inverse")
    c4.metric("OOS Net Profit", f"${s['oos_net_profit']:,.0f}")

    # OOS equity curve
    if results.combined_trades:
        profits = np.array([t.net_profit for t in results.combined_trades])
        equity  = 100_000 + np.cumsum(profits)
        times   = [t.closed_time for t in results.combined_trades]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=equity, mode="lines", name="OOS Equity",
            line=dict(color="#2E75B6", width=2),
            fill="tozeroy", fillcolor="rgba(46,117,182,0.1)"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            title="Out-of-Sample Combined Equity Curve"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Window detail table
    st.markdown("##### Window Details")
    rows = []
    for w in results.windows:
        rows.append({
            "Window":       f"W{w.window_id}",
            "Train":        f"{w.train_from.date()} → {w.train_to.date()}",
            "Test":         f"{w.test_from.date()} → {w.test_to.date()}",
            "IS Sharpe":    round(w.train_results.sharpe_ratio, 2),
            "OOS Sharpe":   round(w.test_results.sharpe_ratio, 2),
            "OOS P&L":      f"${w.test_results.net_profit:+,.0f}",
            "Profitable":   "✅" if w.is_profitable else "❌",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Best params per window
    with st.expander("Parameters chosen per window"):
        for w in results.windows:
            params_str = "  |  ".join(
                f"{k}={v:.2f}" for k, v in w.best_params.items()
                if k in ("StopLossCoef1","ProfitTargetCoef1","TrailingActCef1","mmLots")
            )
            st.markdown(
                f"**W{w.window_id}** ({w.test_from.date()}) → "
                f"{params_str} → OOS: ${w.test_results.net_profit:+,.0f}"
            )


# ── Monte Carlo tab ───────────────────────────────────────────────────────────

def _render_mc_tab():
    st.markdown("#### Monte Carlo Simulation")
    st.markdown(
        "<p style='color:#8B9BB4;'>Resamples trade sequences to show the "
        "distribution of possible outcomes. Requires a backtest to be run first.</p>",
        unsafe_allow_html=True
    )

    if "bt_result" not in st.session_state or not st.session_state["bt_result"]:
        st.info("Run a backtest in the Backtest tab first.", icon="ℹ️")
        return

    r = st.session_state["bt_result"]
    st.success(f"Using backtest result: {r.n_trades} trades", icon="✅")

    col1, col2 = st.columns(2)
    with col1:
        n_sims  = st.select_slider("Simulations",
                                    options=[500, 1000, 2000, 5000], value=1000,
                                    key="mc_sims")
    with col2:
        ruin_threshold = st.slider("Ruin threshold (drawdown %)", 10, 50, 20,
                                    key="mc_ruin")

    if st.button("🎲 Run Monte Carlo", type="primary",
                  use_container_width=True, key="mc_run"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            try:
                from research.monte_carlo import run_monte_carlo
                mc = run_monte_carlo(r, n_simulations=n_sims)
                st.session_state["mc_result"]    = mc
                st.session_state["mc_ruin_pct"]  = ruin_threshold
                st.rerun()
            except Exception as e:
                error_box(f"Monte Carlo failed: {e}")

    if "mc_result" in st.session_state and st.session_state["mc_result"]:
        _show_mc_results(st.session_state["mc_result"],
                          st.session_state.get("mc_ruin_pct", 20))


def _show_mc_results(mc, ruin_threshold):
    st.markdown("---")
    pp = mc.profit_percentiles
    dp = mc.drawdown_percentiles

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Prob. of Profit",     f"{mc.prob_profit*100:.1f}%")
    c2.metric(f"Prob. Ruin (DD>{ruin_threshold}%)",
               f"{mc.prob_ruin(ruin_threshold)*100:.1f}%")
    c3.metric("Median Net Profit",  f"${pp[50]:,.0f}")
    c4.metric("Median Max DD",       f"{dp[50]:.2f}%")

    # Equity path fan chart
    paths  = mc.equity_paths
    n_show = min(200, len(paths))
    idx    = np.random.choice(len(paths), n_show, replace=False)

    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(
            y=paths[i], mode="lines",
            line=dict(color="rgba(46,117,182,0.05)", width=1),
            showlegend=False
        ))

    # Percentile bands
    p5  = mc.initial_capital + np.percentile([p[-1] for p in paths], 5)
    p50 = mc.initial_capital + np.percentile([p[-1] for p in paths], 50)
    p95 = mc.initial_capital + np.percentile([p[-1] for p in paths], 95)

    fig.add_hline(y=p5,  line_color="#E74C3C", line_dash="dash",
                  annotation_text=f"5th pct: ${p5-mc.initial_capital:+,.0f}")
    fig.add_hline(y=p50, line_color="#2E75B6", line_dash="dash",
                  annotation_text=f"Median: ${p50-mc.initial_capital:+,.0f}")
    fig.add_hline(y=p95, line_color="#2ECC71", line_dash="dash",
                  annotation_text=f"95th pct: ${p95-mc.initial_capital:+,.0f}")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Equity ($)", xaxis_title="Trade #"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Profit distribution histogram
    st.markdown("##### Final Profit Distribution")
    fig2 = px.histogram(
        x=mc.profits, nbins=60,
        color_discrete_sequence=["#2E75B6"],
        labels={"x": "Net Profit ($)"}
    )
    fig2.add_vline(x=0, line_color="#E74C3C", line_dash="dash")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=240,
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Full stats table
    with st.expander("Full statistics"):
        s = mc.summary()
        rows = [(k, str(v)) for k, v in s.items()]
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]),
                     use_container_width=True, hide_index=True)


# ── AI Comparison tab ─────────────────────────────────────────────────────────

def _render_ai_tab():
    st.markdown("#### AI vs Baseline Comparison")
    st.markdown(
        "<p style='color:#8B9BB4;'>Compares fixed parameters against "
        "AI-scheduled lot sizing from the trained model.</p>",
        unsafe_allow_html=True
    )

    if "bt_result" not in st.session_state or not st.session_state["bt_result"]:
        st.info("Run a backtest in the Backtest tab first.", icon="ℹ️")
        return

    import config.settings as settings
    if not Path(settings.MODEL_SAVE_PATH).exists():
        st.warning("No trained model found. Go to AI Training and train the model first.",
                    icon="⚠️")
        return

    r = st.session_state["bt_result"]
    strat_name = st.session_state.get("bt_last_strat", "")
    st.success(f"Baseline: {r.n_trades} trades | Sharpe: {r.sharpe_ratio:.2f} | "
               f"Net: ${r.net_profit:,.0f}", icon="📊")

    if st.button("🤖 Run AI Comparison", type="primary",
                  use_container_width=True, key="ai_run"):
        _run_ai_comparison(r)

    if "ai_result" in st.session_state and st.session_state["ai_result"]:
        _show_ai_comparison(r, st.session_state["ai_result"])


def _run_ai_comparison(baseline):
    with st.spinner("Loading AI schedule and running comparison..."):
        try:
            import csv, torch
            import config.settings as settings
            from engine.data_loader  import load_bars
            from engine.backtester   import Backtester
            from engine.base_strategy import BaseStrategy

            # Load schedule CSV
            schedule_path = (
                Path.home() / ".wine/drive_c/users" /
                os.environ.get("USER", "user") /
                "AppData/Roaming/MetaQuotes/Terminal/Common/Files/ml_params_schedule.csv"
            )
            if not schedule_path.exists():
                error_box(f"Schedule CSV not found at {schedule_path}. "
                          "Run Generate Schedule in AI Training.")
                return

            schedule = {}
            with open(schedule_path) as f:
                for row in csv.DictReader(f):
                    ym = row["date"][:7].replace(".", "-")
                    schedule[ym] = float(row["mmLots"])

            # Get the strategy class used for baseline
            strategies   = discover_strategies()
            strat_name   = st.session_state.get("bt_last_strat", "")
            strat_cls    = next(
                (cls for cls in strategies.values()
                 if cls.__name__ == strat_name),
                list(strategies.values())[0]
            )

            # Build AI strategy that reads lot size from schedule
            class AIScheduledStrategy(strat_cls):
                def on_bar(self, ctx):
                    ym = ctx.time.strftime("%Y-%m")
                    if ym in schedule:
                        self.params["mmLots"] = schedule[ym]
                    super().on_bar(ctx)

            # Load same data and run
            df  = load_bars(baseline.symbol or "HK50.cash", "H1")
            bt  = Backtester(
                initial_capital = baseline.initial_capital,
                lot_value       = getattr(strat_cls, "lot_value", 1.0),
            )
            strat  = AIScheduledStrategy()
            ai_res = bt.run(
                strat, df,
                date_from = str(baseline.date_from.date()),
                date_to   = str(baseline.date_to.date()),
            )
            st.session_state["ai_result"] = ai_res
            st.rerun()

        except Exception as e:
            import traceback
            error_box(f"AI comparison failed: {e}\n{traceback.format_exc()}")


def _show_ai_comparison(baseline, ai_res):
    st.markdown("---")
    st.markdown("##### Metrics Comparison")

    abs_dd_b, rel_dd_b = baseline.max_drawdown
    abs_dd_a, rel_dd_a = ai_res.max_drawdown

    metrics = [
        ("Net Profit",     f"${baseline.net_profit:,.0f}",
                           f"${ai_res.net_profit:,.0f}",
                           ai_res.net_profit > baseline.net_profit),
        ("Sharpe Ratio",   f"{baseline.sharpe_ratio:.2f}",
                           f"{ai_res.sharpe_ratio:.2f}",
                           ai_res.sharpe_ratio > baseline.sharpe_ratio),
        ("Max Drawdown",   f"{rel_dd_b:.2f}%",
                           f"{rel_dd_a:.2f}%",
                           rel_dd_a < rel_dd_b),
        ("Profit Factor",  f"{baseline.profit_factor:.2f}",
                           f"{ai_res.profit_factor:.2f}",
                           ai_res.profit_factor > baseline.profit_factor),
        ("Win Rate",       f"{baseline.win_rate*100:.1f}%",
                           f"{ai_res.win_rate*100:.1f}%",
                           ai_res.win_rate > baseline.win_rate),
        ("Gross Loss",     f"${baseline.gross_loss:,.0f}",
                           f"${ai_res.gross_loss:,.0f}",
                           ai_res.gross_loss > baseline.gross_loss),  # less loss = better
    ]

    cols = st.columns(len(metrics))
    for col, (name, bval, aval, ai_better) in zip(cols, metrics):
        col.metric(
            label      = name,
            value      = aval,
            delta      = f"Base: {bval}",
            delta_color = "normal" if ai_better else "inverse"
        )

    # Equity curves
    st.markdown("##### Equity Curves")
    b_eq = baseline.equity_curve
    a_eq = ai_res.equity_curve

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=b_eq.index, y=b_eq.values, mode="lines",
        name="Baseline", line=dict(color="#8B9BB4", width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=a_eq.index, y=a_eq.values, mode="lines",
        name="AI-Scheduled", line=dict(color="#2E75B6", width=2.5),
        fill="tonexty", fillcolor="rgba(46,117,182,0.07)"
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=300,
        margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Equity ($)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown comparison
    st.markdown("##### Drawdown")
    for label, eq, color in [("Baseline", b_eq, "#E74C3C"),
                               ("AI", a_eq, "#F39C12")]:
        peak = np.maximum.accumulate(eq.values)
        dd   = (peak - eq.values) / peak * 100
        fig2 = go.Figure() if label == "Baseline" else fig2
        fig2.add_trace(go.Scatter(
            x=eq.index, y=-dd, mode="lines", name=label,
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(color[i:i+2],16)) for i in (1,3,5))},0.1)"
        ))

    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=220,
        margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Drawdown (%)"
    )
    st.plotly_chart(fig2, use_container_width=True)
