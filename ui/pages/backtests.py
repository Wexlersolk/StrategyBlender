"""
ui/pages/backtests.py — Backtest results viewer with auto-import
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from ui.components.common import section_header, empty_state, error_box
from ui.mt5_watcher import (
    get_watch_dirs, find_all_reports, is_mt5_report,
    extract_report_metadata, MT5ReportWatcher
)


def render():
    section_header("Backtests", "Import and view MT5 strategy tester results")

    eas = st.session_state.get("eas", {})

    tab_import, tab_results, tab_portfolio = st.tabs([
        "📥 Import Results", "📊 Individual Results", "📈 Portfolio View"
    ])

    with tab_import:
        _render_import_tab(eas)

    with tab_results:
        _render_results_tab(eas)

    with tab_portfolio:
        _render_portfolio_tab(eas)


# ── Import tab ────────────────────────────────────────────────────────────────

def _render_import_tab(eas: dict):
    st.markdown("#### How to run a backtest")

    sb_reports = Path(__file__).parent.parent.parent / "reports"
    sb_reports.mkdir(exist_ok=True)

    st.info(
        f"**Step 1** — Run your backtest in MT5 Strategy Tester as normal.\n\n"
        f"**Step 2** — Right-click the Results tab → **Save as Report** → "
        f"save it to:\n\n"
        f"```\n{sb_reports}\n```\n\n"
        f"**Step 3** — Click **Scan for New Reports** below.",
        icon="ℹ️"
    )

    # Show watch directories so user knows where to save
    watch_dirs = get_watch_dirs()
    with st.expander("📁 Where to save your reports (MT5 watches these folders)"):
        if watch_dirs:
            for d in watch_dirs:
                st.markdown(f"`{d}`")
        else:
            st.warning("No MT5 folders detected. Make sure MT5 is installed.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Scan for New Reports", type="primary",
                     use_container_width=True):
            with st.spinner("Scanning for MT5 reports..."):
                all_reports = find_all_reports()
            mt5_reports = [p for p in all_reports if is_mt5_report(p)]

            if not mt5_reports:
                st.warning(
                    "No reports found. Save your MT5 report to the "
                    "`reports/` folder first.",
                    icon="⚠️"
                )
                st.session_state["_scanned_reports"] = []
            else:
                # Store paths as strings so they survive reruns
                st.session_state["_scanned_reports"] = [str(p) for p in mt5_reports]
                st.success(f"Found {len(mt5_reports)} report(s)", icon="✅")

    with col2:
        if st.button("🗑️ Clear scan", use_container_width=True):
            st.session_state["_scanned_reports"] = []
            st.rerun()

    # Render persisted scan results
    scanned = st.session_state.get("_scanned_reports", [])
    for path_str in scanned:
        report_path = Path(path_str)
        if report_path.exists():
            meta = extract_report_metadata(report_path)
            _show_report_assignment(report_path, meta, eas)

    st.markdown("---")
    st.markdown("#### Or upload directly")

    uploaded = st.file_uploader(
        "Upload HTML report",
        type=["html", "htm"],
        accept_multiple_files=True,
        key="report_uploader",
    )
    if uploaded:
        st.session_state["_pending_uploads"] = [
            {"name": f.name, "data": f.read()} for f in uploaded
        ]

    pending = st.session_state.get("_pending_uploads", [])
    if pending:
        st.markdown(f"**{len(pending)} file(s) ready to import**")
        if eas:
            ea_options = {
                ea_id: f"{ea['name']} — {ea['symbol']} {ea['timeframe']}"
                for ea_id, ea in eas.items()
            }
            selected_ea = st.selectbox(
                "Assign to EA",
                list(ea_options.keys()),
                format_func=lambda x: ea_options[x],
                key="upload_ea_select",
            )
            if st.button("Import uploaded files", type="primary",
                         use_container_width=True):
                import tempfile
                for f in pending:
                    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                    tmp.write(f["data"])
                    tmp.close()
                    _parse_file_and_store(Path(tmp.name), selected_ea)
                    os.unlink(tmp.name)
                st.session_state["_pending_uploads"] = []
                st.rerun()
        else:
            st.warning("Add EAs in EA Manager first.")

    # Show already-imported reports
    results = st.session_state.get("backtest_results", {})
    if results:
        st.markdown("---")
        st.markdown(f"#### {len(results)} strategy/strategies imported")
        for ea_id, r in results.items():
            ea   = eas.get(ea_id, {})
            name = ea.get("name", ea_id)
            s    = r.get("summary", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"**{name}**")
            col2.metric("Profit",   f"${s.get('total_profit', 0):,.0f}")
            col3.metric("Sharpe",   f"{s.get('sharpe_mean', 0):.2f}")
            col4.metric("Trades",   str(int(s.get('num_trades', 0))))


def _show_report_assignment(report_path: Path, meta: dict, eas: dict):
    """Show one found report and let user assign it to an EA."""
    key_prefix = report_path.stem

    with st.expander(
        f"📄 {report_path.name}  —  "
        f"{meta.get('symbol', '?')} {meta.get('timeframe', '?')}",
        expanded=True
    ):
        st.markdown(f"**Path:** `{report_path}`")
        st.markdown(
            f"**Detected:** Symbol `{meta.get('symbol', '?')}` | "
            f"TF `{meta.get('timeframe', '?')}` | "
            f"EA `{meta.get('ea_name', '?')}`"
        )

        if not eas:
            st.warning("No EAs loaded. Add EAs in EA Manager first.")
            return

        ea_options = {
            ea_id: f"{ea['name']} — {ea['symbol']} {ea['timeframe']}"
            for ea_id, ea in eas.items()
        }

        # Auto-match by symbol — store default in session state once
        state_key = f"assign_{key_prefix}"
        if state_key not in st.session_state:
            default_id = list(ea_options.keys())[0]
            for ea_id, ea in eas.items():
                if ea.get("symbol") == meta.get("symbol"):
                    default_id = ea_id
                    break
            st.session_state[state_key] = default_id

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_ea = st.selectbox(
                "Assign to EA",
                list(ea_options.keys()),
                index=list(ea_options.keys()).index(st.session_state[state_key]),
                format_func=lambda x: ea_options[x],
                key=f"select_{key_prefix}",
            )
            st.session_state[state_key] = selected_ea

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ Import", key=f"import_{key_prefix}", type="primary",
                         use_container_width=True):
                _parse_file_and_store(report_path, selected_ea)
                st.rerun()




# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_file_and_store(report_path: Path, ea_id: str):
    """Parse one HTML report file and store results in session state."""
    try:
        from bs4 import BeautifulSoup

        raw = report_path.read_bytes()
        try:
            content = raw.decode("utf-16-le", errors="replace")
        except Exception:
            content = raw.decode("utf-8", errors="replace")

        soup   = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")
        if len(tables) < 2:
            st.error(f"Could not parse {report_path.name} — unexpected format")
            return

        # Parse deals
        rows        = tables[1].find_all("tr")
        deals_start = None
        for i, row in enumerate(rows):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if "Deals" in cells:
                deals_start = i + 2
                break

        if deals_start is None:
            st.error(f"No deals section found in {report_path.name}")
            return

        deal_rows = []
        for row in rows[deals_start:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 11 or cells[4] != "out" or not cells[2]:
                continue
            try:
                t = pd.to_datetime(cells[0], format="%Y.%m.%d %H:%M:%S")
                p = float(cells[10].replace(" ", "").replace(",", ""))
                deal_rows.append({"time": t, "profit": p})
            except (ValueError, IndexError):
                continue

        if not deal_rows:
            st.error(f"No completed trades found in {report_path.name}")
            return

        deals_df = pd.DataFrame(deal_rows).set_index("time").sort_index()

        # Monthly stats
        monthly_rows = []
        for period, group in deals_df.groupby(pd.Grouper(freq="ME")):
            profits = group["profit"].values
            if len(profits) < 2:
                continue
            mean_p = np.mean(profits)
            std_p  = np.std(profits) + 1e-8
            monthly_rows.append({
                "year_month":   period.strftime("%Y-%m"),
                "sharpe":       round(float(mean_p / std_p * np.sqrt(240)), 4),
                "total_profit": round(float(profits.sum()), 2),
                "win_rate":     round(float(np.mean(profits > 0)), 4),
                "num_trades":   len(profits),
            })

        if not monthly_rows:
            st.error("Could not compute monthly stats — not enough data.")
            return

        monthly_df = pd.DataFrame(monthly_rows).set_index("year_month")
        summary    = {
            "total_profit": float(deals_df["profit"].sum()),
            "sharpe_mean":  float(monthly_df["sharpe"].mean()),
            "win_rate":     float(monthly_df["win_rate"].mean()),
            "num_months":   len(monthly_df),
            "num_trades":   int(monthly_df["num_trades"].sum()),
        }

        if "backtest_results" not in st.session_state:
            st.session_state["backtest_results"] = {}

        # Merge with existing results for this EA (multiple reports)
        existing = st.session_state["backtest_results"].get(ea_id)
        if existing and existing.get("monthly_df") is not None:
            combined_monthly = pd.concat([existing["monthly_df"], monthly_df])
            combined_monthly = combined_monthly[~combined_monthly.index.duplicated(keep="last")]
            combined_deals   = pd.concat([existing["deals_df"], deals_df]).sort_index()
            monthly_df       = combined_monthly
            deals_df         = combined_deals
            summary["total_profit"] = float(deals_df["profit"].sum())
            summary["sharpe_mean"]  = float(monthly_df["sharpe"].mean())
            summary["num_trades"]   = int(monthly_df["num_trades"].sum())

        st.session_state["backtest_results"][ea_id] = {
            "monthly_df": monthly_df,
            "deals_df":   deals_df,
            "summary":    summary,
        }

        from ui.state import autosave
        autosave()

        st.success(
            f"Imported {report_path.name} — "
            f"{int(summary['num_trades'])} trades, "
            f"{len(monthly_df)} months, "
            f"Sharpe {summary['sharpe_mean']:.2f}",
            icon="✅"
        )

    except Exception as e:
        import traceback
        st.error(f"Parse error: {e}\n{traceback.format_exc()}")


# ── Results tab ───────────────────────────────────────────────────────────────

def _render_results_tab(eas: dict):
    results = st.session_state.get("backtest_results", {})
    if not results:
        empty_state("No results yet. Import reports in the Import tab.", "📊")
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
        format_func=lambda x: ea_options[x]
    )
    r       = results[selected]
    summary = r["summary"]
    monthly = r["monthly_df"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Profit",  f"${summary['total_profit']:,.0f}")
    c2.metric("Avg Sharpe",    f"{summary['sharpe_mean']:.2f}")
    c3.metric("Avg Win Rate",  f"{summary['win_rate']*100:.1f}%")
    c4.metric("Total Trades",  str(summary["num_trades"]))

    st.markdown("---")
    st.markdown("##### Monthly Sharpe Ratio")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly["sharpe"],
        marker_color=["#2ECC71" if v > 0 else "#E74C3C" for v in monthly["sharpe"]],
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=280,
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Monthly Profit")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=monthly.index, y=monthly["total_profit"],
        marker_color=["#2E75B6" if v > 0 else "#C0392B" for v in monthly["total_profit"]],
    ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=260,
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Monthly data table"):
        st.dataframe(monthly, use_container_width=True)


# ── Portfolio tab ─────────────────────────────────────────────────────────────

def _render_portfolio_tab(eas: dict):
    results = st.session_state.get("backtest_results", {})
    if len(results) < 2:
        st.info(
            "Add backtest results for at least 2 strategies to see portfolio analytics.",
            icon="ℹ️"
        )
        return

    st.markdown("##### Combined Portfolio Equity Curve")

    dfs = []
    for ea_id, r in results.items():
        df = r.get("monthly_df")
        if df is not None and not df.empty:
            tmp = df[["total_profit"]].copy()
            tmp.columns = [eas.get(ea_id, {}).get("name", ea_id)]
            tmp.index   = pd.to_datetime(tmp.index)
            dfs.append(tmp)

    if not dfs:
        return

    combined           = pd.concat(dfs, axis=1).fillna(0)
    combined["Total"]  = combined.sum(axis=1)
    combined["Equity"] = 100_000 + combined["Total"].cumsum()

    fig = go.Figure()
    for col in combined.columns[:-2]:
        fig.add_trace(go.Scatter(
            x=combined.index, y=100_000 + combined[col].cumsum(),
            mode="lines", name=col, line=dict(width=1), opacity=0.4
        ))
    fig.add_trace(go.Scatter(
        x=combined.index, y=combined["Equity"],
        mode="lines", name="Portfolio",
        line=dict(color="#2E75B6", width=3)
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=360,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    profits  = combined["Total"].values
    mean_m   = np.mean(profits)
    std_m    = np.std(profits) + 1e-8
    p_sharpe = float(mean_m / std_m * np.sqrt(12))
    equity   = combined["Equity"].values
    peak     = np.maximum.accumulate(equity)
    dd       = (peak - equity) / peak * 100
    max_dd   = float(dd.max())

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Sharpe", f"{p_sharpe:.2f}")
    c2.metric("Max Drawdown",     f"{max_dd:.2f}%")
    c3.metric("Total Net Profit", f"${profits.sum():,.0f}")
