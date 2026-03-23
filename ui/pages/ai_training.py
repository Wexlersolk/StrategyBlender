"""
ui/pages/ai_training.py — Train AI model and compare results
"""

import sys
import os
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ui.components.common import section_header, empty_state, error_box


def render():
    section_header("AI Training", "Train the regime model and compare AI vs baseline")

    results = st.session_state.get("backtest_results", {})
    eas     = st.session_state.get("eas", {})

    tab_train, tab_compare, tab_schedule = st.tabs([
        "🧠 Train Model", "📊 Compare Results", "📅 Parameter Schedule"
    ])

    with tab_train:
        _render_train_tab(results, eas)

    with tab_compare:
        _render_compare_tab(eas)

    with tab_schedule:
        _render_schedule_tab()


def _render_train_tab(results: dict, eas: dict):
    st.markdown("#### Train AI Model")
    st.markdown(
        "<p style='color:#8B9BB4;'>The AI learns which market conditions preceded good "
        "vs bad months for your strategies, then scales position sizes accordingly.</p>",
        unsafe_allow_html=True
    )

    if not results:
        empty_state("No backtest data found. Upload reports in the Backtests tab first.", "📂")
        return

    # ── EA selector ───────────────────────────────────────────────────────────
    st.markdown("#### Select Strategy to Train")

    ea_options = {
        ea_id: f"{eas[ea_id]['name']} — {eas[ea_id]['symbol']} {eas[ea_id]['timeframe']}"
        for ea_id in results if ea_id in eas
    }

    if not ea_options:
        empty_state("No matched EAs found. Import backtest reports first.", "📂")
        return

    # Add "All strategies combined" option
    ea_options = {"__all__": "🔀 All strategies combined"} | ea_options

    selected_ea_id = st.selectbox(
        "Strategy",
        list(ea_options.keys()),
        format_func=lambda x: ea_options[x],
        key="train_ea_select",
    )

    # Filter results and symbols based on selection
    if selected_ea_id == "__all__":
        filtered_results = results
        filtered_symbols = list(set(
            eas[ea_id]["symbol"] for ea_id in results if ea_id in eas
        ))
    else:
        filtered_results = {selected_ea_id: results[selected_ea_id]}
        filtered_symbols = [eas[selected_ea_id]["symbol"]]

    # Store selection for use in _run_training
    st.session_state["_train_ea_id"]      = selected_ea_id
    st.session_state["_train_results"]    = filtered_results
    st.session_state["_train_symbols"]    = filtered_symbols

    st.markdown("---")

    total_months = sum(
        len(r["monthly_df"]) for r in filtered_results.values()
        if r.get("monthly_df") is not None
    )
    total_reports = len(filtered_results)

    # ── Optimizer results section ─────────────────────────────────────────────
    st.markdown("#### Data Sources")

    col_html, col_opt = st.columns(2)

    with col_html:
        st.metric("Backtest reports", str(total_reports))
        st.metric("Monthly samples", str(total_months))

    with col_opt:
        opt_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "exports", "optimizer_results.csv"
        )
        opt_exists = os.path.exists(opt_path)
        if opt_exists:
            import pandas as pd
            opt_df     = pd.read_csv(opt_path)
            opt_rows   = len(opt_df)
            est_total  = opt_rows * max(total_months, 70)
            st.metric("Optimizer sets", str(opt_rows))
            st.metric("Est. total samples", f"~{est_total:,}")
        else:
            st.metric("Optimizer sets", "0")
            st.caption("No optimizer data yet")

    # Optimizer upload
    with st.expander("📊 Import MT5 Optimizer Results (recommended)", expanded=not opt_exists):
        st.markdown(
            "<p style='color:#8B9BB4;'>Run MT5's optimizer on your EA, "
            "then right-click Optimization Results → Save. "
            "Upload the XML file here to massively increase training data.</p>",
            unsafe_allow_html=True
        )
        opt_file = st.file_uploader(
            "Upload optimizer results XML",
            type=["xml"],
            key="optimizer_xml"
        )
        if opt_file:
            if st.button("Parse Optimizer Results", type="secondary"):
                _parse_optimizer_upload(opt_file)

    st.markdown("---")
    st.markdown("#### Training Configuration")

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Training epochs", 500, 5000, 3000, 500)
    with col2:
        lr = st.select_slider(
            "Learning rate",
            options=[0.001, 0.003, 0.005, 0.01],
            value=0.005
        )

    st.markdown("#### Parameter scaling bounds")
    st.markdown(
        "<p style='color:#8B9BB4;font-size:0.9rem;'>These control how aggressively the AI "
        "can scale parameters. Values are multipliers on your base parameters.</p>",
        unsafe_allow_html=True
    )

    bounds = {}
    param_defs = {
        "mmLots":            (0.2, 1.0),
        "StopLossCoef1":     (1.0, 1.0),
        "ProfitTargetCoef1": (1.0, 1.0),
        "StopLossCoef2":     (1.0, 1.0),
        "ProfitTargetCoef2": (1.0, 1.0),
        "TrailingActCef1":   (1.0, 1.0),
    }

    for param, (lo, hi) in param_defs.items():
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"`{param}`")
        with c2:
            bounds[param] = (
                st.number_input(f"Min##{param}", value=lo, step=0.05,
                                min_value=0.01, max_value=2.0, key=f"lo_{param}"),
            )
        with c3:
            bounds[param] = (
                bounds[param][0],
                st.number_input(f"Max##{param}", value=hi, step=0.05,
                                min_value=0.01, max_value=3.0, key=f"hi_{param}")
            )

    st.markdown("---")

    if st.button("🧠 Train AI Model", type="primary", use_container_width=True):
        _run_training(
            results  = st.session_state.get("_train_results", filtered_results),
            eas      = eas,
            epochs   = epochs,
            lr       = lr,
            bounds   = bounds,
            symbols  = st.session_state.get("_train_symbols", filtered_symbols),
        )


def _run_training(results: dict, eas: dict, epochs: int, lr: float,
                  bounds: dict, symbols: list = None):
    """Call the existing train_meta pipeline from the UI."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    progress_bar = st.progress(0, text="Preparing training data...")
    log_container = st.empty()
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        log_container.code("\n".join(log_lines[-20:]), language="")

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from data.storage import DataStorage
        from data.features import compute_features, get_feature_columns

        # Combine all monthly data
        all_monthly = []
        for ea_id, r in results.items():
            df = r.get("monthly_df")
            if df is not None and not df.empty:
                all_monthly.append(df[["sharpe"]].copy())

        if not all_monthly:
            error_box("No monthly data available.")
            return

        monthly_df = pd.concat(all_monthly).groupby(level=0)["sharpe"].mean().to_frame()
        log(f"Training samples: {len(monthly_df)}")

        # Load market data
        storage  = DataStorage()
        features = get_feature_columns()
        # Use passed symbols, fall back to deriving from results
        if symbols:
            train_symbols = symbols
        else:
            train_symbols = list(set(ea["symbol"] for ea in eas.values()))

        all_data = {}
        for sym in train_symbols:
            df = storage.load_bars(sym)
            if not df.empty:
                df = compute_features(df)
                all_data[sym] = df
                log(f"Loaded {len(df)} bars for {sym}")

        if not all_data:
            error_box("No market data. Run update_data.py first.")
            return

        import datetime

        X_list, y_list = [], []
        for ym, row in monthly_df.iterrows():
            sharpe = row["sharpe"]
            target = float((np.clip(sharpe, -10, 10) + 10) / 20)
            year, month = int(ym[:4]), int(ym[5:7])
            month_dt    = datetime.datetime(year, month, 1)
            prior_end   = month_dt
            prior_start = month_dt - datetime.timedelta(days=60)

            feats_list = []
            for sym, df in all_data.items():
                mask = (df.index >= prior_start) & (df.index < prior_end)
                sl   = df[mask]
                if sl.empty:
                    feats_list.append(np.zeros(len(features)))
                else:
                    f = sl[features].mean().values.astype(np.float32)
                    std = f.std()
                    if std > 1e-8:
                        f = (f - f.mean()) / std
                    feats_list.append(f)

            if not feats_list:
                continue
            combined = np.concatenate(feats_list)
            if np.any(~np.isfinite(combined)):
                continue

            X_list.append(combined)
            y_list.append(target)

        if len(X_list) < 5:
            error_box(f"Only {len(X_list)} valid training samples. Need more data.")
            return

        log(f"Building model: {len(X_list)} samples, {len(X_list[0])} features")
        progress_bar.progress(0.2, text="Building model...")

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
        y = y.expand(-1, len(bounds))

        class RegimeModel(nn.Module):
            def __init__(self, in_d, out_d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_d, 32), nn.Tanh(),
                    nn.Linear(32, 16), nn.Tanh(),
                    nn.Linear(16, out_d)
                )
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.uniform_(m.weight, -0.05, 0.05)
                        nn.init.zeros_(m.bias)

            def forward(self, x):
                return torch.sigmoid(self.net(x))

        model     = RegimeModel(X.shape[1], len(bounds))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn   = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

            if epoch % (epochs // 20) == 0:
                pct  = 0.2 + 0.7 * (epoch / epochs)
                corr = float(np.corrcoef(
                    model(X).detach().numpy()[:, 0], y[:, 0].numpy()
                )[0, 1])
                progress_bar.progress(pct, text=f"Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")
                log(f"Epoch {epoch:4d} | loss: {loss.item():.6f} | corr: {corr:.3f}")

        progress_bar.progress(0.95, text="Saving model...")

        import config.settings as settings

        base_params = {
            "mmLots": 70.0, "StopLossCoef1": 2.0, "ProfitTargetCoef1": 1.5,
            "StopLossCoef2": 2.4, "ProfitTargetCoef2": 1.5, "TrailingActCef1": 1.4
        }

        torch.save({
            "model_state":  model.state_dict(),
            "input_dim":    X.shape[1],
            "output_dim":   len(bounds),
            "base_params":  base_params,
            "scale_bounds": {k: list(v) for k, v in bounds.items()},
            "param_names":  list(bounds.keys()),
            "symbols":      train_symbols,
            "feature_cols": features,
        }, settings.MODEL_SAVE_PATH)

        progress_bar.progress(1.0, text="Done!")
        log("Model saved successfully.")

        st.session_state["model_trained"] = True
        st.session_state["training_log"]  = log_lines
        st.success("AI model trained successfully! Go to Compare Results to see the analysis.", icon="✅")

    except Exception as e:
        import traceback
        error_box(f"Training failed: {e}\n{traceback.format_exc()}")
        progress_bar.empty()


def _parse_optimizer_upload(opt_file):
    """Parse uploaded optimizer XML and save to data/optimizer_results.csv."""
    import sys
    import os
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        tmp.write(opt_file.read())
        tmp.close()

        from scripts.parse_optimizer import parse_optimizer_xml, detect_param_columns
        df         = parse_optimizer_xml(tmp.name)
        os.unlink(tmp.name)

        param_cols = detect_param_columns(df)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/exports/optimizer_results.csv", index=False)

        st.success(
            f"Parsed {len(df)} parameter combinations. "
            f"Parameters: {', '.join(param_cols[:4])}{'...' if len(param_cols) > 4 else ''}. "
            f"Estimated training samples: ~{len(df) * 70:,}",
            icon="✅"
        )
        st.rerun()

    except Exception as e:
        st.error(f"Failed to parse optimizer XML: {e}", icon="🚨")


def _render_compare_tab(eas: dict):
    results = st.session_state.get("backtest_results", {})
    if not results:
        empty_state("No backtest results yet.", "📊")
        return

    if not st.session_state.get("model_trained"):
        st.info("Train the AI model first to see comparison results.", icon="ℹ️")
        return

    st.markdown("#### AI vs Baseline — Full Comparison")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        import torch
        import numpy as np
        import config.settings as settings
        from data.storage import DataStorage
        from data.features import compute_features, get_feature_columns
        import datetime

        ckpt          = torch.load(settings.MODEL_SAVE_PATH, map_location="cpu")
        scale_bounds  = ckpt["scale_bounds"]
        feature_cols  = ckpt["feature_cols"]
        stored_symbols = ckpt["symbols"]

        class RegimeModel(torch.nn.Module):
            def __init__(self, in_d, out_d):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_d, 32), torch.nn.Tanh(),
                    torch.nn.Linear(32, 16),   torch.nn.Tanh(),
                    torch.nn.Linear(16, out_d)
                )
            def forward(self, x):
                return torch.sigmoid(self.net(x))

        model = RegimeModel(ckpt["input_dim"], ckpt["output_dim"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        storage  = DataStorage()
        all_data = {}
        for sym in stored_symbols:
            df = storage.load_bars(sym)
            if not df.empty:
                df = compute_features(df)
                all_data[sym] = df

        # Combine monthly data from all results
        all_monthly = []
        for r in results.values():
            df = r.get("monthly_df")
            if df is not None and not df.empty:
                all_monthly.append(df[["sharpe", "total_profit", "win_rate", "num_trades"]].copy())

        if not all_monthly:
            st.warning("No monthly data available.")
            return

        monthly_df = pd.concat(all_monthly).groupby(level=0).mean()

        # Build comparison rows
        rows = []
        for ym, row in monthly_df.iterrows():
            year, month = int(ym[:4]), int(ym[5:7])
            month_dt    = datetime.datetime(year, month, 1)
            prior_start = month_dt - datetime.timedelta(days=60)

            feats_list = []
            for sym, df in all_data.items():
                mask = (df.index >= prior_start) & (df.index < month_dt)
                sl   = df[mask]
                if sl.empty:
                    feats_list.append(np.zeros(len(feature_cols)))
                else:
                    f   = sl[feature_cols].mean().values.astype(np.float32)
                    std = f.std()
                    if std > 1e-8:
                        f = (f - f.mean()) / std
                    feats_list.append(f)

            if not feats_list:
                continue

            x      = torch.tensor(np.concatenate(feats_list), dtype=torch.float32).unsqueeze(0)
            scales = model(x).squeeze(0).detach().numpy()
            lo, hi = scale_bounds["mmLots"][0], scale_bounds["mmLots"][1]
            lot_scale = float(lo + scales[0] * (hi - lo))

            base_profit = float(row["total_profit"])
            ai_profit   = base_profit * lot_scale

            rows.append({
                "month":      ym,
                "base_profit": base_profit,
                "ai_profit":   ai_profit,
                "lot_scale":   round(lot_scale, 3),
                "sharpe":      float(row["sharpe"]),
                "win_rate":    float(row.get("win_rate", 0)),
                "num_trades":  int(row.get("num_trades", 0)),
            })

        if not rows:
            st.warning("Could not compute AI predictions — no overlapping market data.")
            return

        comp = pd.DataFrame(rows).set_index("month")

        # ── Compute full metrics ──────────────────────────────────────────────
        def compute_metrics(profits: np.ndarray, label: str) -> dict:
            equity    = 100_000 + np.cumsum(profits)
            peak      = np.maximum.accumulate(equity)
            dd        = (peak - equity) / peak * 100
            max_dd    = float(dd.max())
            total     = float(profits.sum())
            mean_m    = float(profits.mean())
            std_m     = float(profits.std()) + 1e-8
            sharpe    = float(mean_m / std_m * np.sqrt(12))
            wins      = profits > 0
            losses    = profits < 0
            win_rate  = float(wins.mean()) * 100
            pf        = (profits[wins].sum() / abs(profits[losses].sum())
                         if losses.any() else float('inf'))

            # Recovery factor
            recovery  = total / (max_dd / 100 * 100_000 + 1e-8)

            # Consecutive losses
            max_consec_loss = 0
            cur = 0
            for p in profits:
                if p < 0:
                    cur += 1
                    max_consec_loss = max(max_consec_loss, cur)
                else:
                    cur = 0

            return {
                "label":           label,
                "net_profit":      total,
                "sharpe":          sharpe,
                "max_drawdown":    max_dd,
                "profit_factor":   pf,
                "win_rate":        win_rate,
                "recovery_factor": recovery,
                "max_consec_loss": max_consec_loss,
                "total_trades":    int(comp["num_trades"].sum()),
            }

        base_metrics = compute_metrics(comp["base_profit"].values, "Baseline")
        ai_metrics   = compute_metrics(comp["ai_profit"].values,   "AI-Optimised")

        # ── Metrics table ─────────────────────────────────────────────────────
        st.markdown("##### Summary Metrics")

        def delta_color(key, base_val, ai_val):
            # Higher is better for these
            higher_better = {"net_profit", "sharpe", "profit_factor",
                             "win_rate", "recovery_factor"}
            # Lower is better for these
            lower_better  = {"max_drawdown", "max_consec_loss"}
            if key in higher_better:
                return "normal" if ai_val >= base_val else "inverse"
            if key in lower_better:
                return "inverse" if ai_val >= base_val else "normal"
            return "off"

        def fmt(key, val):
            if key == "net_profit":      return f"${val:,.0f}"
            if key == "max_drawdown":    return f"{val:.2f}%"
            if key == "win_rate":        return f"{val:.1f}%"
            if key == "profit_factor":   return f"{val:.2f}"
            if key == "sharpe":          return f"{val:.2f}"
            if key == "recovery_factor": return f"{val:.2f}"
            if key == "max_consec_loss": return str(int(val))
            if key == "total_trades":    return str(int(val))
            return str(val)

        metric_labels = {
            "net_profit":      "Net Profit",
            "sharpe":          "Sharpe Ratio",
            "max_drawdown":    "Max Drawdown",
            "profit_factor":   "Profit Factor",
            "win_rate":        "Win Rate",
            "recovery_factor": "Recovery Factor",
            "max_consec_loss": "Max Consec. Losses",
        }

        cols = st.columns(len(metric_labels))
        for i, (key, label) in enumerate(metric_labels.items()):
            bv = base_metrics[key]
            av = ai_metrics[key]
            if key == "max_drawdown":
                pct = ((av - bv) / (abs(bv) + 1e-8)) * 100
                delta_str = f"{pct:+.1f}%"
            elif isinstance(bv, float):
                pct = ((av - bv) / (abs(bv) + 1e-8)) * 100
                delta_str = f"{pct:+.1f}%"
            else:
                delta_str = f"{av - bv:+d}"

            with cols[i]:
                st.metric(
                    label   = label,
                    value   = fmt(key, av),
                    delta   = f"AI: {delta_str}  (Base: {fmt(key, bv)})",
                    delta_color = delta_color(key, bv, av),
                    help    = f"Baseline: {fmt(key, bv)}"
                )

        st.markdown("---")

        # ── Equity curve comparison ───────────────────────────────────────────
        st.markdown("##### Equity Curve")
        base_equity = 100_000 + comp["base_profit"].cumsum()
        ai_equity   = 100_000 + comp["ai_profit"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comp.index, y=base_equity,
            mode="lines", name="Baseline",
            line=dict(color="#8B9BB4", width=2, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=comp.index, y=ai_equity,
            mode="lines", name="AI-Optimised",
            line=dict(color="#2E75B6", width=2.5),
            fill="tonexty", fillcolor="rgba(46,117,182,0.08)"
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Equity ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Drawdown comparison ───────────────────────────────────────────────
        st.markdown("##### Drawdown")
        base_peak = np.maximum.accumulate(base_equity.values)
        ai_peak   = np.maximum.accumulate(ai_equity.values)
        base_dd   = (base_peak - base_equity.values) / base_peak * 100
        ai_dd     = (ai_peak   - ai_equity.values)   / ai_peak   * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=comp.index, y=-base_dd,
            mode="lines", name="Baseline DD",
            line=dict(color="#E74C3C", width=1.5, dash="dot"),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.1)"
        ))
        fig_dd.add_trace(go.Scatter(
            x=comp.index, y=-ai_dd,
            mode="lines", name="AI DD",
            line=dict(color="#F39C12", width=1.5),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.1)"
        ))
        fig_dd.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=220,
            margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Monthly profit bars ───────────────────────────────────────────────
        st.markdown("##### Monthly Profit — Baseline vs AI")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=comp.index, y=comp["base_profit"],
            name="Baseline", marker_color="#5A6478", opacity=0.7
        ))
        fig_bar.add_trace(go.Bar(
            x=comp.index, y=comp["ai_profit"],
            name="AI", marker_color="#2E75B6", opacity=0.9
        ))
        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=260, barmode="group",
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Lot scale per month ───────────────────────────────────────────────
        st.markdown("##### AI Lot Scale Per Month")
        fig_scale = go.Figure()
        fig_scale.add_trace(go.Bar(
            x=comp.index, y=comp["lot_scale"],
            marker_color=["#2ECC71" if v > 0.6 else "#E74C3C"
                          for v in comp["lot_scale"]],
        ))
        fig_scale.add_hline(y=1.0, line_dash="dash", line_color="#8B9BB4",
                            annotation_text="Full size")
        fig_scale.add_hline(
            y=scale_bounds["mmLots"][0], line_dash="dot",
            line_color="#E74C3C", annotation_text="Minimum"
        )
        fig_scale.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=220,
            margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Scale (0–1)"
        )
        st.plotly_chart(fig_scale, use_container_width=True)

        # ── Detail table ──────────────────────────────────────────────────────
        with st.expander("Monthly detail table"):
            display = comp.copy()
            display.columns = ["Base Profit", "AI Profit", "Lot Scale",
                                "Sharpe", "Win Rate", "Trades"]
            display["Base Profit"] = display["Base Profit"].map("${:,.0f}".format)
            display["AI Profit"]   = display["AI Profit"].map("${:,.0f}".format)
            display["Win Rate"]    = display["Win Rate"].map("{:.1%}".format)
            st.dataframe(display, use_container_width=True)

    except Exception as e:
        import traceback
        error_box(f"Could not load comparison: {e}\n{traceback.format_exc()}")


def _render_schedule_tab():
    if not st.session_state.get("model_trained"):
        st.info("Train the AI model first to generate a parameter schedule.", icon="ℹ️")
        return

    st.markdown("#### Generate Parameter Schedule")
    st.markdown(
        "<p style='color:#8B9BB4;'>This creates the CSV file that the MT5 Orchestrator EA "
        "reads to apply AI-adjusted parameters each month.</p>",
        unsafe_allow_html=True
    )

    if st.button("⚙️ Generate Schedule CSV", type="primary", use_container_width=True):
        with st.spinner("Generating parameter schedule..."):
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
                import subprocess
                result = subprocess.run(
                    [sys.executable, "scripts/generate_schedule.py"],
                    capture_output=True, text=True,
                    cwd=os.path.join(os.path.dirname(__file__), "..", "..")
                )
                if result.returncode == 0:
                    st.success("Schedule generated!", icon="✅")
                    st.code(result.stdout)
                else:
                    error_box(result.stderr)
            except Exception as e:
                error_box(str(e))
