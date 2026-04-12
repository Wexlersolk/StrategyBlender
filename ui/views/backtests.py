"""
ui/views/backtests.py

Strategy conversion workspace:
- shows generated local Python review scaffolds for uploaded MT5 EAs
- allows exporting converted files
- runs local backtests on generated engine strategies using exported MT5 data
"""

from pathlib import Path

import importlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components.common import section_header, empty_state
from ui.state import autosave
from services.backtest_service import (
    available_backtest_symbols,
    backtest_result_payload,
    default_execution_config,
    run_backtest,
)
from services.conversion_service import convert_ea_source, normalize_symbol, persist_converted_ea


GENERATED_DIR = Path(__file__).parent.parent.parent / "convert" / "generated"


def _unique_series_name(base: str, used: set[str]) -> str:
    candidate = (base or "Strategy").strip()
    if candidate not in used:
        used.add(candidate)
        return candidate

    idx = 2
    while True:
        numbered = f"{candidate} ({idx})"
        if numbered not in used:
            used.add(numbered)
            return numbered
        idx += 1


def render():
    section_header("Backtests", "Convert MT5 EAs and run them on local market data")

    eas = st.session_state.get("eas", {})

    tab_convert, tab_results, tab_portfolio = st.tabs([
        "🔁 MT5 -> Python", "📊 Individual Results", "📈 Portfolio View"
    ])

    with tab_convert:
        _render_conversion_tab(eas)

    with tab_results:
        _render_results_tab(eas)

    with tab_portfolio:
        _render_portfolio_tab(eas)


def _render_conversion_tab(eas: dict):
    st.markdown("#### Conversion and execution flow")
    st.info(
        "**MT5 path** — Upload your `.mq5` EA in `EA Manager` and StrategyBlender converts it into "
        "local Python artifacts.\n\n"
        "**Native Python path** — Use `Strategy Builder` to generate a strategy family directly from "
        "templates and presets.\n\n"
        "**Execution** — Run either type below against the local market data already stored in "
        "`data/exports/MT5 data export/`.",
        icon="ℹ️",
    )

    if not eas:
        empty_state("No EAs loaded yet. Upload a `.mq5` file in EA Manager first.", "🤖")
        return

    st.markdown(f"Generated files are written to `{GENERATED_DIR}` and `strategies/generated/`.")

    ea_options = {
        ea_id: f"{ea['name']} — {ea['symbol']} {ea['timeframe']}"
        for ea_id, ea in eas.items()
    }
    selected = st.selectbox(
        "Select converted EA",
        list(ea_options.keys()),
        format_func=lambda x: ea_options[x],
    )
    ea = eas[selected]
    review_source = ea.get("review_source") or ea.get("python_source", "")
    engine_source = ea.get("engine_source", "")

    col1, col2, col3 = st.columns(3)
    col1.metric("Inputs", len(ea.get("params", {})))
    col2.metric("Functions", len(ea.get("conversion_functions", [])))
    col3.metric("Warnings", len(ea.get("conversion_warnings", [])))

    if ea.get("conversion_warnings"):
        for warning in ea["conversion_warnings"]:
            st.warning(warning, icon="⚠️")

    available_syms = available_backtest_symbols(ea["timeframe"])
    default_symbol = ea["symbol"] if ea["symbol"] in available_syms else available_syms[0]
    default_idx = available_syms.index(default_symbol)

    st.markdown("##### Run local backtest")
    col_cfg_1, col_cfg_2, col_cfg_3 = st.columns(3)
    with col_cfg_1:
        run_symbol = st.selectbox(
            "Symbol",
            available_syms,
            index=default_idx,
            key=f"run_symbol_{selected}",
        )
    with col_cfg_2:
        date_from = st.date_input(
            "From",
            value=pd.Timestamp("2020-01-01"),
            key=f"run_from_{selected}",
        )
    with col_cfg_3:
        date_to = st.date_input(
            "To",
            value=pd.Timestamp("2025-12-31"),
            key=f"run_to_{selected}",
        )

    intrabar = st.checkbox(
        "Use actual M1 intrabar data",
        value=False,
        key=f"intrabar_{selected}",
        help="Replays real minute bars inside each higher-timeframe candle when M1 export data is available.",
    )
    default_execution = default_execution_config(run_symbol)
    st.markdown("##### Execution model")
    st.caption("XAUUSD defaults use conservative commission/slippage and can use the MT5 spread column when available.")
    exec_col_1, exec_col_2, exec_col_3 = st.columns(3)
    with exec_col_1:
        commission_per_lot = st.number_input(
            "Commission / lot",
            min_value=0.0,
            value=float(st.session_state.get(f"run_commission_{selected}_{run_symbol}", default_execution["commission_per_lot"])),
            step=0.1,
            key=f"run_commission_{selected}_{run_symbol}",
        )
    with exec_col_2:
        spread_pips = st.number_input(
            "Fallback spread",
            min_value=0.0,
            value=float(st.session_state.get(f"run_spread_{selected}_{run_symbol}", default_execution["spread_pips"])),
            step=0.01,
            key=f"run_spread_{selected}_{run_symbol}",
        )
    with exec_col_3:
        slippage_pips = st.number_input(
            "Slippage",
            min_value=0.0,
            value=float(st.session_state.get(f"run_slippage_{selected}_{run_symbol}", default_execution["slippage_pips"])),
            step=0.01,
            key=f"run_slippage_{selected}_{run_symbol}",
        )
    exec_col_4, exec_col_5 = st.columns(2)
    with exec_col_4:
        tick_size = st.number_input(
            "Tick size",
            min_value=0.00001,
            value=float(st.session_state.get(f"run_tick_size_{selected}_{run_symbol}", default_execution.get("tick_size", 1.0))),
            step=0.01,
            format="%.5f",
            key=f"run_tick_size_{selected}_{run_symbol}",
        )
    with exec_col_5:
        use_bar_spread = st.checkbox(
            "Use MT5 spread column",
            value=bool(st.session_state.get(f"run_bar_spread_{selected}_{run_symbol}", default_execution.get("use_bar_spread", False))),
            key=f"run_bar_spread_{selected}_{run_symbol}",
            help="When enabled, the backtester converts MT5 <SPREAD> values into price spread using the configured tick size.",
        )
    if st.button("Run backtest", type="primary", use_container_width=True, key=f"run_bt_{selected}"):
        with st.spinner("Running generated strategy on local data..."):
            try:
                strat_cls = _load_generated_strategy_class(ea)
                execution_config = {
                    "commission_per_lot": float(commission_per_lot),
                    "spread_pips": float(spread_pips),
                    "slippage_pips": float(slippage_pips),
                    "tick_size": float(tick_size),
                    "use_bar_spread": bool(use_bar_spread),
                }
                result = run_backtest(
                    strat_cls,
                    symbol=run_symbol,
                    timeframe=ea["timeframe"],
                    date_from=str(date_from),
                    date_to=str(date_to),
                    overrides=ea.get("params", {}),
                    intrabar_steps=60 if intrabar else 1,
                    execution_config=execution_config,
                )
                st.session_state.setdefault("backtest_results", {})[selected] = backtest_result_payload(result)
                st.session_state["eas"][selected]["last_backtest_symbol"] = run_symbol
                st.session_state["eas"][selected]["last_execution_config"] = execution_config
                autosave()
                st.success(
                    f"Backtest finished: {result.n_trades} trades, net profit ${result.net_profit:,.0f}, "
                    f"Sharpe {result.sharpe_ratio:.2f}",
                    icon="✅",
                )
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")

    with st.expander("Generated local engine strategy", expanded=False):
        st.code(engine_source or "# No generated engine strategy found", language="python")

    with st.expander("Generated Python review scaffold / payload", expanded=False):
        st.code(review_source or "# No generated review scaffold or payload found", language="python")

    filename = f"{ea['name']}.py"
    col_save, col_download = st.columns(2)
    with col_save:
        if st.button("Save review scaffold", type="secondary", use_container_width=True):
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            output_path = GENERATED_DIR / filename
            output_path.write_text(review_source, encoding="utf-8")
            st.session_state["eas"][selected]["review_path"] = str(output_path)
            autosave()
            st.success(f"Saved `{output_path}`", icon="✅")

    with col_download:
        st.download_button(
            "Download review scaffold",
            data=review_source.encode("utf-8"),
            file_name=filename,
            mime="text/x-python",
            use_container_width=True,
        )

    if ea.get("origin") == "python_template":
        st.markdown("##### Template payload")
        st.text_area(
            "Template payload JSON",
            value=ea.get("review_source", ""),
            height=220,
            disabled=True,
        )
    else:
        st.markdown("##### Source mapping")
        st.text_area(
            "Original MQL5 source",
            value=ea.get("source", ""),
            height=220,
            disabled=True,
        )


def _load_generated_strategy_class(ea: dict):
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if ea.get("origin") == "python_template":
        module_name = ea["strategy_module"]
        mod = importlib.reload(sys.modules[module_name]) if module_name in sys.modules else importlib.import_module(module_name)
        return getattr(mod, ea["strategy_class"])

    normalized_symbol = normalize_symbol(ea["symbol"])
    ea["symbol"] = normalized_symbol
    refreshed = convert_ea_source(
        source=ea["source"],
        strategy_name=ea["name"],
        symbol=normalized_symbol,
        timeframe=ea["timeframe"],
        ea_id=ea["id"],
    )
    paths = persist_converted_ea(refreshed, ea["name"])
    ea["params"] = refreshed.params
    ea["review_source"] = refreshed.review_source
    ea["engine_source"] = refreshed.engine_source
    ea["strategy_path"] = paths["engine_path"]
    ea["review_path"] = paths["review_path"]
    ea["strategy_module"] = refreshed.strategy_module
    ea["strategy_class"] = refreshed.strategy_class
    ea["conversion_warnings"] = refreshed.warnings
    ea["conversion_functions"] = refreshed.functions

    module_name = ea["strategy_module"]
    mod = importlib.reload(sys.modules[module_name]) if module_name in sys.modules else importlib.import_module(module_name)
    return getattr(mod, ea["strategy_class"])


def _render_results_tab(eas: dict):
    results = st.session_state.get("backtest_results", {})
    if not results:
        empty_state(
            "No backtest results yet. Convert/export a strategy first, then run your backtest pipeline.",
            "📊",
        )
        return

    ea_options = {
        ea_id: f"{eas[ea_id]['name']} — {eas[ea_id]['symbol']}"
        for ea_id in results if ea_id in eas
    }
    if not ea_options:
        empty_state("No matching EAs found.", "📊")
        return

    selected = st.selectbox(
        "Select strategy",
        list(ea_options.keys()),
        format_func=lambda x: ea_options[x],
    )
    result = results[selected]
    summary = result["summary"]
    monthly = result["monthly_df"].copy()
    balance_curve = result.get("balance_curve_df", pd.DataFrame()).copy()
    if "total_profit" not in monthly.columns and "profit" in monthly.columns:
        monthly["total_profit"] = monthly["profit"]
    if "num_trades" not in monthly.columns and "trades" in monthly.columns:
        monthly["num_trades"] = monthly["trades"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Profit", f"${summary['total_profit']:,.0f}")
    c2.metric("Avg Sharpe", f"{summary['sharpe_mean']:.2f}")
    c3.metric("Avg Win Rate", f"{summary['win_rate'] * 100:.1f}%")
    c4.metric("Total Trades", str(summary["num_trades"]))

    c5, c6 = st.columns(2)
    c5.metric(
        "Balance Drawdown Maximal",
        f"${summary.get('balance_dd_abs', 0):,.0f}",
        delta=f"{summary.get('balance_dd_pct', 0):.2f}%",
        delta_color="inverse",
    )
    c6.metric(
        "Equity Drawdown Maximal",
        f"${summary.get('equity_dd_abs', 0):,.0f}",
        delta=f"{summary.get('equity_dd_pct', 0):.2f}%",
        delta_color="inverse",
    )

    st.markdown("---")
    st.markdown("##### Balance Graph")
    if balance_curve.empty:
        st.info("No balance curve is available for this run.", icon="ℹ️")
    else:
        if not isinstance(balance_curve.index, pd.DatetimeIndex):
            balance_curve.index = pd.to_datetime(balance_curve.index, errors="coerce")
        balance_curve = balance_curve[balance_curve.index.notna()].sort_index()

        if len(balance_curve) == 1:
            start_time = pd.Timestamp(summary.get("date_from")) if summary.get("date_from") else balance_curve.index[0]
            if pd.notna(start_time):
                balance_curve.loc[start_time] = float(balance_curve["balance"].iloc[0])
                balance_curve = balance_curve.sort_index()

        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(
            x=balance_curve.index,
            y=balance_curve["balance"],
            mode="lines",
            name="Balance",
            line=dict(color="#2ECC71", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.12)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Balance: $%{y:,.2f}<extra></extra>",
        ))
        fig_balance.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            xaxis_title=None,
            yaxis_title="Account Balance",
            hovermode="x unified",
        )
        fig_balance.update_xaxes(showgrid=False)
        fig_balance.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_balance, use_container_width=True)

        c_curve_1, c_curve_2, c_curve_3 = st.columns(3)
        c_curve_1.metric("Start Balance", f"${float(balance_curve['balance'].iloc[0]):,.0f}")
        c_curve_2.metric("End Balance", f"${float(balance_curve['balance'].iloc[-1]):,.0f}")
        c_curve_3.metric("Net Change", f"${float(balance_curve['balance'].iloc[-1] - balance_curve['balance'].iloc[0]):,.0f}")

    st.markdown("---")
    st.markdown("##### Monthly Sharpe Ratio")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly.index,
        y=monthly["sharpe"],
        marker_color=["#2ECC71" if v > 0 else "#E74C3C" for v in monthly["sharpe"]],
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Monthly Profit")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthly.index,
        y=monthly["total_profit"],
        marker_color=["#2E75B6" if v > 0 else "#C0392B" for v in monthly["total_profit"]],
    ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Monthly data table"):
        st.dataframe(monthly, use_container_width=True)


def _render_portfolio_tab(eas: dict):
    results = st.session_state.get("backtest_results", {})
    if len(results) < 2:
        st.info(
            "Add backtest results for at least 2 strategies to see portfolio analytics.",
            icon="ℹ️",
        )
        return

    st.markdown("##### Combined Portfolio Equity Curve")

    dfs = []
    used_names: set[str] = set()
    for ea_id, result in results.items():
        df = result.get("monthly_df")
        if df is not None and not df.empty:
            df = df.copy()
            if "total_profit" not in df.columns and "profit" in df.columns:
                df["total_profit"] = df["profit"]
            tmp = df[["total_profit"]].copy()
            ea_meta = eas.get(ea_id, {})
            base_name = ea_meta.get("name") or f"{ea_meta.get('symbol', '')}_{ea_meta.get('timeframe', '')}" or ea_id
            tmp.columns = [_unique_series_name(base_name, used_names)]
            tmp.index = pd.to_datetime(tmp.index)
            dfs.append(tmp)

    if not dfs:
        return

    combined = pd.concat(dfs, axis=1).fillna(0)
    combined["Total"] = combined.sum(axis=1)
    combined["Equity"] = 100_000 + combined["Total"].cumsum()

    fig = go.Figure()
    for col in combined.columns[:-2]:
        fig.add_trace(go.Scatter(
            x=combined.index,
            y=100_000 + combined[col].cumsum(),
            mode="lines",
            name=col,
            line=dict(width=1),
            opacity=0.4,
        ))
    fig.add_trace(go.Scatter(
        x=combined.index,
        y=combined["Equity"],
        mode="lines",
        name="Portfolio",
        line=dict(color="#2E75B6", width=3),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    profits = combined["Total"].values
    mean_m = np.mean(profits)
    std_m = np.std(profits) + 1e-8
    p_sharpe = float(mean_m / std_m * np.sqrt(12))
    equity = combined["Equity"].values
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100
    max_dd = float(dd.max())

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Sharpe", f"{p_sharpe:.2f}")
    c2.metric("Max Drawdown", f"{max_dd:.2f}%")
    c3.metric("Total Net Profit", f"${profits.sum():,.0f}")
