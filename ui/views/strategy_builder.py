from __future__ import annotations

import copy
import importlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from research.state_store import list_jobs
from services.native_strategy_lab import (
    create_batch_run,
    default_parameter_mutation_space,
    default_promotion_policy_for_template,
    default_structural_mutation_space,
    evaluate_batch_run,
    evaluate_native_strategy,
    generate_batch_candidates,
    list_batch_runs,
    list_generator_runs,
    list_native_strategy_records,
    regenerate_native_strategy_version,
    register_generated_strategy,
    run_batch_generation,
    save_custom_preset,
    strategy_children,
    strategy_lineage_tree,
    strategy_payload_diff,
    update_native_strategy_status,
)
from services.python_strategy_service import (
    available_builder_symbols,
    available_presets,
    available_template_names,
    compile_strategy_spec,
    persist_strategy_spec,
    strategy_spec_from_template,
    template_profile,
)
from ui.components.common import SYMBOLS_COMMON, TIMEFRAMES, empty_state, section_header, success_box
from ui.job_runner import cancel_background_job, enqueue_background_job, job_audit_log, new_job_record, poll_background_jobs
from ui.state import autosave


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def render():
    section_header("Strategy Builder", "Generate native Python strategy families from templates and presets")
    poll_background_jobs(st.session_state, on_success=_integrate_completed_job)

    presets = available_presets()
    if not presets:
        empty_state("No strategy presets are available.", "🧩")
        return

    tab_build, tab_generate, tab_retest, tab_catalog = st.tabs(["Build", "Generate", "Retest", "Catalog"])
    with tab_build:
        _render_builder(presets)
    with tab_generate:
        _render_generate_runner(presets)
    with tab_retest:
        _render_retest_runner()
    with tab_catalog:
        _render_catalog()


def _render_builder(presets: list[dict]):
    builder_symbols = available_builder_symbols()
    default_symbol = st.session_state.get("builder_symbol", builder_symbols[0])
    if default_symbol not in builder_symbols:
        default_symbol = builder_symbols[0]

    st.markdown("#### Choose market")
    selected_symbol = st.selectbox("Instrument", builder_symbols, index=builder_symbols.index(default_symbol), key="builder_symbol")
    template_names = available_template_names(selected_symbol)
    if not template_names:
        empty_state(f"No template families are configured for {selected_symbol}.", "🧩")
        return

    current_template = st.session_state.get("builder_template", template_names[0])
    if current_template not in template_names:
        current_template = template_names[0]
    st.markdown("#### Choose family")
    template_name = st.selectbox("Template", template_names, index=template_names.index(current_template), key="builder_template")
    profile = template_profile(template_name) or {}
    matching_presets = available_presets(symbol=selected_symbol, template_name=template_name)
    preset_options = {preset["id"]: preset for preset in matching_presets}

    preset_ids = [""] + [preset["id"] for preset in matching_presets]
    selected_preset_id = st.selectbox(
        "Preset",
        preset_ids,
        format_func=lambda value: "Start from blank template defaults" if not value else preset_options[value]["label"],
        key="builder_preset",
    )

    if profile:
        supported_tfs = ", ".join(profile.get("supported_timeframes", []))
        st.caption(
            f"Family: {profile.get('family', '')} | Regime: {profile.get('regime', '')} | "
            f"Timeframes: {supported_tfs or 'Any'}"
        )

    col_apply, col_info = st.columns([1, 2])
    with col_apply:
        if st.button("Load Preset", use_container_width=True, type="secondary", disabled=not selected_preset_id):
            preset = preset_options[selected_preset_id]
            st.session_state["builder_payload"] = copy.deepcopy(preset["payload"])
            st.session_state["builder_template"] = preset["template"]
            st.session_state["builder_symbol"] = preset["payload"].get("symbol", selected_symbol)
            st.rerun()
    with col_info:
        if selected_preset_id:
            preset = preset_options[selected_preset_id]
            st.caption(preset.get("description", ""))

    payload = _current_payload(template_name)
    payload["symbol"] = selected_symbol

    with st.form("strategy_builder_form"):
        payload["name"] = st.text_input("Strategy Name", value=payload.get("name", f"{template_name.title()} Strategy"))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_input("Symbol", value=selected_symbol, disabled=True)
        with col2:
            timeframe_options = profile.get("supported_timeframes", TIMEFRAMES) or TIMEFRAMES
            default_tf = payload.get("timeframe", profile.get("preferred_timeframes", ["H1"])[0] if profile.get("preferred_timeframes") else timeframe_options[0])
            if default_tf not in timeframe_options:
                default_tf = timeframe_options[0]
            payload["timeframe"] = st.selectbox("Timeframe", timeframe_options, index=timeframe_options.index(default_tf))
        with col3:
            if template_name in {"trend_pullback", "breakout_confirm"}:
                payload["direction"] = st.selectbox("Direction", ["long", "short"], index=0 if payload.get("direction", "long") == "long" else 1)

        payload["min_bars"] = int(st.number_input("Min Bars", min_value=0, value=int(payload.get("min_bars", 20)), step=1))
        payload["lot_value"] = float(st.number_input("Lot Value", min_value=0.0, value=float(payload.get("lot_value", 1.0)), step=0.1))

        st.markdown("#### Parameters")
        payload["params"] = _render_template_param_inputs(template_name, payload.get("params", {}))

        if template_name == "xau_discovery_grammar":
            st.markdown("#### Discovery Blocks")
            dg1, dg2, dg3 = st.columns(3)
            payload["entry_archetype"] = dg1.selectbox(
                "Entry Archetype",
                ["breakout_stop", "breakout_close", "pullback_trend", "ema_reclaim", "atr_pullback_limit"],
                index=["breakout_stop", "breakout_close", "pullback_trend", "ema_reclaim", "atr_pullback_limit"].index(str(payload.get("entry_archetype", "breakout_stop"))),
            )
            payload["volatility_filter"] = dg2.selectbox(
                "Volatility Filter",
                ["bb_width", "atr_expansion", "none"],
                index=["bb_width", "atr_expansion", "none"].index(str(payload.get("volatility_filter", "bb_width"))),
            )
            payload["session_filter"] = dg3.selectbox(
                "Session Filter",
                ["london_ny", "london_only", "ny_only", "all_day"],
                index=["london_ny", "london_only", "ny_only", "all_day"].index(str(payload.get("session_filter", "london_ny"))),
            )
            dg4, dg5, dg6 = st.columns(3)
            payload["stop_model"] = dg4.selectbox(
                "Stop Model",
                ["atr", "channel", "swing"],
                index=["atr", "channel", "swing"].index(str(payload.get("stop_model", "atr"))),
            )
            payload["target_model"] = dg5.selectbox(
                "Target Model",
                ["fixed_rr", "atr_scaled", "trend_runner"],
                index=["fixed_rr", "atr_scaled", "trend_runner"].index(str(payload.get("target_model", "fixed_rr"))),
            )
            payload["exit_model"] = dg6.selectbox(
                "Exit Model",
                ["session_close", "time_exit", "trailing_atr", "break_even_then_trail", "channel_flip", "atr_time_stop"],
                index=["session_close", "time_exit", "trailing_atr", "break_even_then_trail", "channel_flip", "atr_time_stop"].index(str(payload.get("exit_model", "session_close"))),
            )

        if template_name == "breakout_confirm":
            payload["expiry_bars"] = int(st.number_input("Order Expiry Bars", min_value=1, value=int(payload.get("expiry_bars", 3)), step=1))

        submitted = st.form_submit_button("Generate Strategy", use_container_width=True, type="primary")

    st.session_state["builder_payload"] = payload

    try:
        spec = strategy_spec_from_template(template_name, payload)
        compiled = compile_strategy_spec(spec, strategy_id="preview")
    except Exception as exc:
        st.error(f"Template preview failed: {exc}")
        return

    tab_payload, tab_spec, tab_source = st.tabs(["Payload JSON", "Compiled Spec", "Generated Python"])
    with tab_payload:
        st.code(json.dumps(payload, indent=2), language="json")
    with tab_spec:
        st.code(json.dumps(_spec_to_jsonable(spec), indent=2), language="json")
    with tab_source:
        st.code(compiled.source, language="python")

        if submitted:
            strategy_id = str(uuid.uuid4())[:8]
            record, meta = register_generated_strategy(
                template_name=template_name,
                payload=payload,
                strategy_id=strategy_id,
                origin="builder",
                tags=["builder", template_name],
            )
            _sync_session_strategy(record, meta["compiled"], paths=meta["paths"], template_name=template_name, payload=payload)
            success_box(
                f"Generated **{spec.name}** as a native Python strategy from the `{template_name}` template."
            )
            st.rerun()

    st.markdown("#### Save As Preset")
    preset_label = st.text_input("Preset Label", value=payload.get("name", ""), key=f"preset_label_{template_name}")
    preset_description = st.text_input("Preset Description", value="", key=f"preset_desc_{template_name}")
    if st.button("Save Custom Preset", use_container_width=True, type="secondary", key=f"save_preset_{template_name}"):
        path = save_custom_preset(
            label=preset_label or payload.get("name", "Custom Preset"),
            description=preset_description,
            template_name=template_name,
            payload=payload,
        )
        success_box(f"Saved preset to `{path}`.")
        st.rerun()


def _current_payload(template_name: str) -> dict:
    payload = copy.deepcopy(st.session_state.get("builder_payload", {}))
    if st.session_state.get("builder_template") != template_name:
        payload = {}
        st.session_state["builder_template"] = template_name
    return payload


def _render_template_param_inputs(template_name: str, params: dict) -> dict:
    params = copy.deepcopy(params)

    if template_name == "trend_pullback":
        c1, c2, c3 = st.columns(3)
        params["FastEMA"] = int(c1.number_input("Fast EMA", min_value=1, value=int(params.get("FastEMA", 21)), step=1))
        params["SlowEMA"] = int(c2.number_input("Slow EMA", min_value=2, value=int(params.get("SlowEMA", 55)), step=1))
        params["ATRPeriod"] = int(c3.number_input("ATR Period", min_value=1, value=int(params.get("ATRPeriod", 14)), step=1))
        c4, c5, c6 = st.columns(3)
        params["WaveTrendChannel"] = int(c4.number_input("WaveTrend Channel", min_value=1, value=int(params.get("WaveTrendChannel", 9)), step=1))
        params["WaveTrendAverage"] = int(c5.number_input("WaveTrend Average", min_value=1, value=int(params.get("WaveTrendAverage", 21)), step=1))
        params["ExitAfterBars"] = int(c6.number_input("Exit After Bars", min_value=0, value=int(params.get("ExitAfterBars", 12)), step=1))
    elif template_name == "breakout_confirm":
        c1, c2, c3 = st.columns(3)
        params["HighestPeriod"] = int(c1.number_input("Highest Period", min_value=1, value=int(params.get("HighestPeriod", 20)), step=1))
        params["LowestPeriod"] = int(c2.number_input("Lowest Period", min_value=1, value=int(params.get("LowestPeriod", 20)), step=1))
        params["ATRPeriod"] = int(c3.number_input("ATR Period", min_value=1, value=int(params.get("ATRPeriod", 14)), step=1))
        c4, c5 = st.columns(2)
        params["BBWRPeriod"] = int(c4.number_input("BB Width Period", min_value=1, value=int(params.get("BBWRPeriod", 20)), step=1))
        params["BBWRMin"] = float(c5.number_input("BB Width Min", value=float(params.get("BBWRMin", 0.0)), step=0.1))
        params["MaxDistancePct"] = float(st.number_input("Max Distance %", min_value=0.0, value=float(params.get("MaxDistancePct", 4.0)), step=0.1))
    elif template_name == "sqx_us30_wpr_stoch":
        c1, c2, c3 = st.columns(3)
        params["WPRPeriod"] = int(c1.number_input("WPR Period", min_value=1, value=int(params.get("WPRPeriod", 75)), step=1))
        params["StochK"] = int(c2.number_input("Stoch K", min_value=1, value=int(params.get("StochK", 14)), step=1))
        params["StochD"] = int(c3.number_input("Stoch D", min_value=1, value=int(params.get("StochD", 3)), step=1))
        c4, c5, c6 = st.columns(3)
        params["StochSlow"] = int(c4.number_input("Stoch Slow", min_value=1, value=int(params.get("StochSlow", 3)), step=1))
        params["HighestPeriod"] = int(c5.number_input("Highest Period", min_value=1, value=int(params.get("HighestPeriod", 105)), step=1))
        params["LowestPeriod"] = int(c6.number_input("Lowest Period", min_value=1, value=int(params.get("LowestPeriod", 105)), step=1))
        c7, c8, c9 = st.columns(3)
        params["ATRPeriod1"] = int(c7.number_input("ATR 1", min_value=1, value=int(params.get("ATRPeriod1", 14)), step=1))
        params["ATRPeriod2"] = int(c8.number_input("ATR 2", min_value=1, value=int(params.get("ATRPeriod2", 29)), step=1))
        params["MaxDistancePct"] = float(c9.number_input("Max Distance %", min_value=0.0, value=float(params.get("MaxDistancePct", 6.0)), step=0.1))
        t1, t2 = st.columns(2)
        params["SignalTimeRangeFrom"] = t1.text_input("Signal From", value=str(params.get("SignalTimeRangeFrom", "00:00")))
        params["SignalTimeRangeTo"] = t2.text_input("Signal To", value=str(params.get("SignalTimeRangeTo", "23:59")))
    elif template_name == "sqx_xau_osma_bb":
        c1, c2, c3 = st.columns(3)
        params["StochK"] = int(c1.number_input("Stoch K", min_value=1, value=int(params.get("StochK", 21)), step=1))
        params["StochD"] = int(c2.number_input("Stoch D", min_value=1, value=int(params.get("StochD", 7)), step=1))
        params["StochSlow"] = int(c3.number_input("Stoch Slow", min_value=1, value=int(params.get("StochSlow", 7)), step=1))
        c4, c5, c6 = st.columns(3)
        params["OsmaFast"] = int(c4.number_input("OsMA Fast", min_value=1, value=int(params.get("OsmaFast", 8)), step=1))
        params["OsmaSlow"] = int(c5.number_input("OsMA Slow", min_value=1, value=int(params.get("OsmaSlow", 17)), step=1))
        params["OsmaSignal"] = int(c6.number_input("OsMA Signal", min_value=1, value=int(params.get("OsmaSignal", 9)), step=1))
        c7, c8, c9 = st.columns(3)
        params["BBPeriod"] = int(c7.number_input("BB Period", min_value=1, value=int(params.get("BBPeriod", 167)), step=1))
        params["BBDev"] = float(c8.number_input("BB Dev", min_value=0.1, value=float(params.get("BBDev", 1.8)), step=0.1))
        params["ATRPeriod1"] = int(c9.number_input("ATR 1", min_value=1, value=int(params.get("ATRPeriod1", 200)), step=1))
        t1, t2, t3 = st.columns(3)
        params["ATRPeriod2"] = int(t1.number_input("ATR 2", min_value=1, value=int(params.get("ATRPeriod2", 55)), step=1))
        params["SignalTimeRangeFrom"] = t2.text_input("Signal From", value=str(params.get("SignalTimeRangeFrom", "00:00")))
        params["SignalTimeRangeTo"] = t3.text_input("Signal To", value=str(params.get("SignalTimeRangeTo", "23:59")))
    elif template_name == "xau_breakout_session":
        c1, c2, c3 = st.columns(3)
        params["HighestPeriod"] = int(c1.number_input("Highest Period", min_value=1, value=int(params.get("HighestPeriod", 24)), step=1))
        params["LowestPeriod"] = int(c2.number_input("Lowest Period", min_value=1, value=int(params.get("LowestPeriod", 24)), step=1))
        params["ATRPeriod"] = int(c3.number_input("ATR Period", min_value=1, value=int(params.get("ATRPeriod", 14)), step=1))
        c4, c5, c6 = st.columns(3)
        params["BBWRPeriod"] = int(c4.number_input("BB Width Period", min_value=1, value=int(params.get("BBWRPeriod", 24)), step=1))
        params["BBWRMin"] = float(c5.number_input("BB Width Min", min_value=0.0, value=float(params.get("BBWRMin", 0.015)), step=0.001, format="%.3f"))
        params["EntryBufferATR"] = float(c6.number_input("Entry Buffer ATR", min_value=0.0, value=float(params.get("EntryBufferATR", 0.15)), step=0.05))
        c7, c8, c9 = st.columns(3)
        params["MaxDistancePct"] = float(c7.number_input("Max Distance %", min_value=0.0, value=float(params.get("MaxDistancePct", 3.5)), step=0.1))
        params["ExitAfterBars"] = int(c8.number_input("Exit After Bars", min_value=0, value=int(params.get("ExitAfterBars", 18)), step=1))
        params["SignalTimeRangeFrom"] = c9.text_input("Signal From", value=str(params.get("SignalTimeRangeFrom", "06:00")))
        params["SignalTimeRangeTo"] = st.text_input("Signal To", value=str(params.get("SignalTimeRangeTo", "18:00")))
    elif template_name == "xau_discovery_grammar":
        c1, c2, c3 = st.columns(3)
        params["HighestPeriod"] = int(c1.number_input("Highest Period", min_value=1, value=int(params.get("HighestPeriod", 24)), step=1))
        params["LowestPeriod"] = int(c2.number_input("Lowest Period", min_value=1, value=int(params.get("LowestPeriod", 24)), step=1))
        params["ATRPeriod"] = int(c3.number_input("ATR Period", min_value=1, value=int(params.get("ATRPeriod", 14)), step=1))
        c4, c5, c6 = st.columns(3)
        params["FastEMA"] = int(c4.number_input("Fast EMA", min_value=1, value=int(params.get("FastEMA", 21)), step=1))
        params["SlowEMA"] = int(c5.number_input("Slow EMA", min_value=2, value=int(params.get("SlowEMA", 55)), step=1))
        params["BBWRPeriod"] = int(c6.number_input("BB Width Period", min_value=1, value=int(params.get("BBWRPeriod", 24)), step=1))
        c7, c8, c9 = st.columns(3)
        params["BBWRMin"] = float(c7.number_input("BB Width Min", min_value=0.0, value=float(params.get("BBWRMin", 0.015)), step=0.001, format="%.3f"))
        params["EntryBufferATR"] = float(c8.number_input("Entry Buffer ATR", min_value=0.0, value=float(params.get("EntryBufferATR", 0.15)), step=0.05))
        params["MaxDistancePct"] = float(c9.number_input("Max Distance %", min_value=0.0, value=float(params.get("MaxDistancePct", 3.5)), step=0.1))
        c10, c11, c12 = st.columns(3)
        params["PullbackLookback"] = int(c10.number_input("Pullback Lookback", min_value=1, value=int(params.get("PullbackLookback", 5)), step=1))
        params["TrailATR"] = float(c11.number_input("Trail ATR", min_value=0.1, value=float(params.get("TrailATR", 1.1)), step=0.1))
        params["ExitAfterBars"] = int(c12.number_input("Exit After Bars", min_value=0, value=int(params.get("ExitAfterBars", 18)), step=1))
        c13, c14, c15 = st.columns(3)
        params["PullbackATR"] = float(c13.number_input("Pullback ATR", min_value=0.0, value=float(params.get("PullbackATR", 0.6)), step=0.1))
        params["BreakEvenATR"] = float(c14.number_input("Break-Even ATR", min_value=0.0, value=float(params.get("BreakEvenATR", 0.8)), step=0.1))
        params["TimeStopATR"] = float(c15.number_input("Time Stop ATR", min_value=0.0, value=float(params.get("TimeStopATR", 0.4)), step=0.1))
    elif template_name == "sqx_usdjpy_vwap_wt":
        c1, c2, c3 = st.columns(3)
        params["EMAPeriod1"] = int(c1.number_input("EMA Period", min_value=1, value=int(params.get("EMAPeriod1", 20)), step=1))
        params["VWAPPeriod1"] = int(c2.number_input("VWAP Period", min_value=1, value=int(params.get("VWAPPeriod1", 77)), step=1))
        params["HighestPeriod"] = int(c3.number_input("Highest Period", min_value=1, value=int(params.get("HighestPeriod", 20)), step=1))
        c4, c5, c6 = st.columns(3)
        params["WaveTrendChannel"] = int(c4.number_input("WT Channel", min_value=1, value=int(params.get("WaveTrendChannel", 9)), step=1))
        params["WaveTrendAverage"] = int(c5.number_input("WT Average", min_value=1, value=int(params.get("WaveTrendAverage", 60)), step=1))
        params["PriceEntryMult1"] = float(c6.number_input("Entry Mult", value=float(params.get("PriceEntryMult1", 0.3)), step=0.1))
        c7, c8, c9 = st.columns(3)
        params["IndicatorCrsMAPrd1"] = int(c7.number_input("Exit MA Period", min_value=1, value=int(params.get("IndicatorCrsMAPrd1", 35)), step=1))
        params["ExitAfterBars1"] = int(c8.number_input("Exit After Bars", min_value=0, value=int(params.get("ExitAfterBars1", 14)), step=1))
        params["ATRPeriod1"] = int(c9.number_input("ATR 1", min_value=1, value=int(params.get("ATRPeriod1", 19)), step=1))
        params["ATRPeriod2"] = int(st.number_input("ATR 2", min_value=1, value=int(params.get("ATRPeriod2", 14)), step=1))
    elif template_name == "sqx_hk50_batch_h1":
        c1, c2, c3 = st.columns(3)
        params["LWMAPeriod1"] = int(c1.number_input("LWMA Period", min_value=1, value=int(params.get("LWMAPeriod1", 14)), step=1))
        params["IndicatorCrsMAPrd1"] = int(c2.number_input("Cross MA Period", min_value=1, value=int(params.get("IndicatorCrsMAPrd1", 47)), step=1))
        params["Highest_period"] = int(c3.number_input("Highest Period", min_value=1, value=int(params.get("Highest_period", 50)), step=1))
        c4, c5, c6 = st.columns(3)
        params["Lowest_period"] = int(c4.number_input("Lowest Period", min_value=1, value=int(params.get("Lowest_period", 50)), step=1))
        params["ATR1_period"] = int(c5.number_input("ATR 1", min_value=1, value=int(params.get("ATR1_period", 19)), step=1))
        params["ATR2_period"] = int(c6.number_input("ATR 2", min_value=1, value=int(params.get("ATR2_period", 45)), step=1))
        c7, c8 = st.columns(2)
        params["ATR3_period"] = int(c7.number_input("ATR 3", min_value=1, value=int(params.get("ATR3_period", 14)), step=1))
        params["ATR4_period"] = int(c8.number_input("ATR 4", min_value=1, value=int(params.get("ATR4_period", 100)), step=1))
        t1, t2 = st.columns(2)
        params["SignalTimeRangeFrom"] = t1.text_input("Signal From", value=str(params.get("SignalTimeRangeFrom", "00:13")))
        params["SignalTimeRangeTo"] = t2.text_input("Signal To", value=str(params.get("SignalTimeRangeTo", "00:26")))
        riskx1, riskx2, riskx3 = st.columns(3)
        params["TrailingActCef1"] = float(riskx1.number_input("Trail Activation", value=float(params.get("TrailingActCef1", 1.4)), step=0.1))
        params["TrailingStop1"] = float(riskx2.number_input("Long Trail Stop", value=float(params.get("TrailingStop1", 127.5)), step=0.1))
        params["TrailingStopCoef1"] = float(riskx3.number_input("Short Trail ATR", value=float(params.get("TrailingStopCoef1", 1.0)), step=0.1))
    elif template_name == "sqx_hk50_after_retest_h4":
        c1, c2, c3 = st.columns(3)
        params["DIPeriod1"] = int(c1.number_input("DI Period", min_value=1, value=int(params.get("DIPeriod1", 14)), step=1))
        params["SARStep"] = float(c2.number_input("SAR Step", min_value=0.001, value=float(params.get("SARStep", 0.266)), step=0.01))
        params["SARMax"] = float(c3.number_input("SAR Max", min_value=0.01, value=float(params.get("SARMax", 0.69)), step=0.01))
        c4, c5, c6 = st.columns(3)
        params["ATRPeriod1"] = int(c4.number_input("ATR Period", min_value=1, value=int(params.get("ATRPeriod1", 20)), step=1))
        params["PriceEntryMult1"] = float(c5.number_input("Entry ATR Mult", value=float(params.get("PriceEntryMult1", 0.3)), step=0.1))
        params["SignalTimeRangeFrom"] = c6.text_input("Signal From", value=str(params.get("SignalTimeRangeFrom", "00:13")))
        params["SignalTimeRangeTo"] = st.text_input("Signal To", value=str(params.get("SignalTimeRangeTo", "00:26")))
    elif template_name == "sqx_hk50_before_retest_h4":
        c1, c2, c3 = st.columns(3)
        params["RSIPeriod1"] = int(c1.number_input("RSI Period", min_value=1, value=int(params.get("RSIPeriod1", 14)), step=1))
        params["Period1"] = int(c2.number_input("Range Period", min_value=1, value=int(params.get("Period1", 50)), step=1))
        params["SMAPeriod1"] = int(c3.number_input("SMA Period", min_value=1, value=int(params.get("SMAPeriod1", 20)), step=1))
        c4, c5, c6 = st.columns(3)
        params["ATRPeriod1"] = int(c4.number_input("ATR 1", min_value=1, value=int(params.get("ATRPeriod1", 50)), step=1))
        params["ATRPeriod2"] = int(c5.number_input("ATR 2", min_value=1, value=int(params.get("ATRPeriod2", 19)), step=1))
        params["ATRPeriod3"] = int(c6.number_input("ATR 3", min_value=1, value=int(params.get("ATRPeriod3", 65)), step=1))
        c7, c8, c9 = st.columns(3)
        params["PriceEntryMult1"] = float(c7.number_input("Entry ATR Mult", value=float(params.get("PriceEntryMult1", 0.3)), step=0.1))
        params["ProfitTargetCoef1"] = float(c8.number_input("PT ATR Mult", value=float(params.get("ProfitTargetCoef1", 3.0)), step=0.1))
        params["TrailingActCef1"] = float(c9.number_input("Trail Activation", value=float(params.get("TrailingActCef1", 1.4)), step=0.1))
        t1, t2 = st.columns(2)
        params["SignalTimeRangeFrom"] = t1.text_input("Signal From", value=str(params.get("SignalTimeRangeFrom", "00:13")))
        params["SignalTimeRangeTo"] = t2.text_input("Signal To", value=str(params.get("SignalTimeRangeTo", "00:26")))
    elif template_name == "sqx_uk100_ulcer_keltner_h1":
        c1, c2, c3 = st.columns(3)
        params["UlcerPeriod"] = int(c1.number_input("Ulcer Period", min_value=1, value=int(params.get("UlcerPeriod", 48)), step=1))
        params["KCPeriod"] = int(c2.number_input("Keltner Period", min_value=1, value=int(params.get("KCPeriod", 20)), step=1))
        params["KCMult1"] = float(c3.number_input("KC Mult 1", min_value=0.1, value=float(params.get("KCMult1", 2.25)), step=0.05))
        c4, c5, c6 = st.columns(3)
        params["KCMult2"] = float(c4.number_input("KC Mult 2", min_value=0.1, value=float(params.get("KCMult2", 2.5)), step=0.05))
        params["ATR1_period"] = int(c5.number_input("ATR 1", min_value=1, value=int(params.get("ATR1_period", 50)), step=1))
        params["ATR2_period"] = int(c6.number_input("ATR 2", min_value=1, value=int(params.get("ATR2_period", 19)), step=1))
        c7, c8, c9 = st.columns(3)
        params["ATR3_period"] = int(c7.number_input("ATR 3", min_value=1, value=int(params.get("ATR3_period", 100)), step=1))
        params["ATR4_period"] = int(c8.number_input("ATR 4", min_value=1, value=int(params.get("ATR4_period", 14)), step=1))
        params["SignalTimeRangeFrom"] = c9.text_input("Signal From", value=str(params.get("SignalTimeRangeFrom", "00:13")))
        params["SignalTimeRangeTo"] = st.text_input("Signal To", value=str(params.get("SignalTimeRangeTo", "00:26")))
    else:
        c1, c2, c3 = st.columns(3)
        params["BBPeriod"] = int(c1.number_input("Bollinger Period", min_value=1, value=int(params.get("BBPeriod", 20)), step=1))
        params["BBDev"] = float(c2.number_input("Bollinger Dev", min_value=0.1, value=float(params.get("BBDev", 2.0)), step=0.1))
        params["WPRPeriod"] = int(c3.number_input("WPR Period", min_value=1, value=int(params.get("WPRPeriod", 14)), step=1))
        c4, c5, c6 = st.columns(3)
        params["WPROversold"] = float(c4.number_input("WPR Oversold", value=float(params.get("WPROversold", -80.0)), step=1.0))
        params["WPROverbought"] = float(c5.number_input("WPR Overbought", value=float(params.get("WPROverbought", -20.0)), step=1.0))
        params["ATRPeriod"] = int(c6.number_input("ATR Period", min_value=1, value=int(params.get("ATRPeriod", 14)), step=1))

    risk1, risk2, risk3 = st.columns(3)
    params["mmLots"] = float(risk1.number_input("Lots", min_value=0.0, value=float(params.get("mmLots", 1.0)), step=0.1))
    params["StopLossATR"] = float(risk2.number_input("Stop Loss ATR", min_value=0.0, value=float(params.get("StopLossATR", 1.5)), step=0.1))
    params["ProfitTargetATR"] = float(risk3.number_input("Take Profit ATR", min_value=0.0, value=float(params.get("ProfitTargetATR", 2.0)), step=0.1))
    return params


def _spec_to_jsonable(spec) -> dict:
    return {
        "name": spec.name,
        "symbol": spec.symbol,
        "timeframe": spec.timeframe,
        "params": spec.params,
        "indicators": [vars(item) for item in spec.indicators],
        "series": [vars(item) for item in spec.series],
        "entries": [vars(item) for item in spec.entries],
        "exits": [vars(item) for item in spec.exits],
        "min_bars": spec.min_bars,
        "lot_value": spec.lot_value,
        "description": spec.description,
    }


def _register_generated_strategy(compiled, payload: dict, template_name: str, strategy_id: str, paths: dict[str, str]):
    _sync_session_strategy(
        {
            "strategy_id": strategy_id,
            "name": compiled.spec.name,
            "symbol": compiled.spec.symbol,
            "timeframe": compiled.spec.timeframe,
            "params": compiled.spec.params,
            "strategy_path": paths["strategy_path"],
            "spec_path": paths["spec_path"],
            "strategy_module": compiled.strategy_module,
            "strategy_class": compiled.strategy_class,
        },
        compiled,
        paths=paths,
        template_name=template_name,
        payload=payload,
    )
    autosave()


def _sync_session_strategy(record: dict, compiled, *, paths: dict[str, str], template_name: str, payload: dict):
    module_name = compiled.strategy_module
    module = importlib.reload(sys.modules[module_name]) if module_name in sys.modules else importlib.import_module(module_name)
    getattr(module, compiled.strategy_class)

    st.session_state["ea_counter"] += 1
    st.session_state.setdefault("eas", {})[record["strategy_id"]] = {
        "name": compiled.spec.name,
        "symbol": compiled.spec.symbol,
        "timeframe": compiled.spec.timeframe,
        "params": compiled.spec.params,
        "source": "",
        "review_source": json.dumps(payload, indent=2),
        "engine_source": compiled.source,
        "strategy_path": paths["strategy_path"],
        "review_path": paths["spec_path"],
        "strategy_module": compiled.strategy_module,
        "strategy_class": compiled.strategy_class,
        "strategy_slug": compiled.strategy_slug,
        "conversion_warnings": [f"Generated from `{template_name}` template preset/builder flow."],
        "conversion_functions": [item.kind for item in compiled.spec.indicators],
        "ai_enabled": False,
        "id": record["strategy_id"],
        "origin": "python_template",
        "template_name": template_name,
        "template_payload": copy.deepcopy(payload),
        "spec_path": paths["spec_path"],
    }


def _render_generate_runner(presets: list[dict]):
    st.markdown("#### Generate Candidate Batch")
    builder_symbols = available_builder_symbols()
    default_symbol = st.session_state.get("batch_symbol", builder_symbols[0])
    if default_symbol not in builder_symbols:
        default_symbol = builder_symbols[0]
    selected_symbol = st.selectbox("Instrument", builder_symbols, index=builder_symbols.index(default_symbol), key="batch_symbol")
    filtered_presets = available_presets(symbol=selected_symbol)
    preset_map = {preset["id"]: preset for preset in filtered_presets}
    preset_ids = list(preset_map)
    if not preset_ids:
        empty_state(f"No presets available for {selected_symbol}. Build and save one first, or add a preset.", "🧪")
        return

    selected_preset_id = st.selectbox("Base Preset", preset_ids, format_func=lambda value: preset_map[value]["label"], key="batch_preset")
    preset = preset_map[selected_preset_id]
    profile = template_profile(preset["template"]) or {}
    base_payload = copy.deepcopy(preset["payload"])
    params = base_payload.setdefault("params", {})

    st.caption(preset.get("description", ""))
    if profile:
        st.caption(f"Family: {profile.get('family', '')} | Regime: {profile.get('regime', '')}")
    mutation_space: dict[str, list] = {}
    st.markdown("##### Mutation Space")
    for key, value in list(params.items())[:8]:
        if not isinstance(value, (int, float)):
            continue
        raw = st.text_input(
            f"{key} values",
            value=f"{value}",
            help="Comma-separated values for this parameter.",
            key=f"batch_param_{selected_preset_id}_{key}",
        )
        values: list[Any] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            parsed = float(item)
            if isinstance(value, int):
                parsed = int(round(parsed))
            values.append(parsed)
        if len(values) > 1:
            mutation_space[key] = values

    limit = int(st.number_input("Max Candidates", min_value=1, max_value=100, value=12, step=1, key="batch_limit"))
    search_mode = st.selectbox("Search Mode", ["grid", "random", "progressive"], index=2, key="batch_search_mode")
    include_structural_mutations = st.toggle(
        "Include Structural Mutations",
        value=True,
        help="Add template-aware payload mutations like direction, expiry, min bars, or family-specific exit/session variants.",
        key="batch_structural_mutations",
    )
    default_policy = default_promotion_policy_for_template(preset["template"])
    structural_defaults = default_structural_mutation_space(preset["template"], base_payload)
    if include_structural_mutations and structural_defaults:
        st.caption(f"Auto structural fields: {', '.join(sorted(structural_defaults))}")
    preview_candidates = generate_batch_candidates(
        template_name=preset["template"],
        base_payload=base_payload,
        mutation_space=mutation_space,
        limit=min(limit, 25),
        search_mode=search_mode,
        include_structural_mutations=include_structural_mutations,
    ) if (mutation_space or (include_structural_mutations and structural_defaults)) else None
    if preview_candidates:
        preview_df = pd.DataFrame(
            [
                {
                    "candidate_id": item["candidate_id"],
                    "mutation_distance": item.get("mutation_distance", 0),
                    "structural_changes": json.dumps(item.get("structural_changes", [])),
                    "mutations": json.dumps(item.get("mutations", {})),
                }
                for item in preview_candidates
            ]
        )
        st.markdown("##### Candidate Preview")
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        st.caption(f"{len(preview_df)} candidates generated. Evaluation happens in the Retest tab.")
    generate_now = st.button("Generate Batch", type="primary", use_container_width=True, key="generate_batch_only")
    if generate_now:
        if not mutation_space and not (include_structural_mutations and structural_defaults):
            st.warning("Define at least one mutation or enable structural mutations before generating.")
        else:
            run_record = create_batch_run(
                template_name=preset["template"],
                base_payload=base_payload,
                mutation_space=mutation_space,
                limit=limit,
                search_mode=search_mode,
                promotion_policy=default_policy,
                include_structural_mutations=include_structural_mutations,
            )
            st.session_state["last_batch_id"] = run_record["batch_id"]
            st.session_state["batch_retest_id"] = run_record["batch_id"]
            success_box(f"Generated batch `{run_record['batch_id']}` with {len(run_record['candidates'])} candidates.")

    st.markdown("#### Continuous Generator")
    generator_defaults = default_parameter_mutation_space(preset["template"], base_payload)
    for key, values in mutation_space.items():
        generator_defaults[f"params.{key}" if "." not in key else key] = list(values)
    if not generator_defaults and not (include_structural_mutations and structural_defaults):
        st.info("This preset does not expose generator mutation defaults yet. Add parameter mutations above or enable structural mutations.", icon="ℹ️")
    else:
        st.caption("SQX-style continuous search: keep generating and evaluating candidates until stopped or until the run hits the cap.")
        gen_cfg_1, gen_cfg_2, gen_cfg_3 = st.columns(3)
        max_candidates = int(gen_cfg_1.number_input("Generator Cap", min_value=1, max_value=5000, value=250, step=25, key="generator_cap"))
        discovery_mode = gen_cfg_2.selectbox("Discovery Mode", ["conservative", "balanced", "exploratory"], index=0, key="generator_discovery_mode")
        generator_intrabar = int(gen_cfg_3.number_input("Generator Intrabar", min_value=1, max_value=60, value=1, step=1, key="generator_intrabar"))
        gen_cfg_4, gen_cfg_5, gen_cfg_6, gen_cfg_7 = st.columns(4)
        generator_date_from = str(gen_cfg_4.date_input("Generator From", value=pd.Timestamp("2021-01-01"), key="generator_from"))
        generator_date_to = str(gen_cfg_5.date_input("Generator To", value=pd.Timestamp("2025-01-01"), key="generator_to"))
        generator_run_mc = gen_cfg_6.toggle("Generator MC", value=False, key="generator_mc")
        generator_run_wfo = gen_cfg_7.toggle("Generator WFO", value=False, key="generator_wfo")
        queue_generator = st.button("Start Generator", type="primary", use_container_width=True, key="start_native_generator")
        if queue_generator:
            job = new_job_record(
                kind="native_generator",
                title=f"Native generator | {preset['label']}",
                payload={
                    "template_name": preset["template"],
                    "base_payload": base_payload,
                    "mutation_space": generator_defaults,
                    "date_from": generator_date_from,
                    "date_to": generator_date_to,
                    "max_candidates": max_candidates,
                    "run_mc": generator_run_mc,
                    "run_wfo_checks": generator_run_wfo,
                    "intrabar_steps": generator_intrabar,
                    "include_structural_mutations": include_structural_mutations,
                    "discovery_mode": discovery_mode,
                    "random_seed": 42,
                },
                max_retries=0,
            )
            enqueue_background_job(job)
            st.session_state["generator_job_id"] = job["job_id"]
            autosave()
            success_box(f"Queued generator job `{job['job_id']}`.")

    _render_strategy_builder_jobs()

    generator_runs = list_generator_runs()
    if generator_runs:
        selected_generator_id = st.selectbox(
            "View Generator Run",
            [run["run_id"] for run in reversed(generator_runs)],
            index=0,
            key="generator_view_id",
        )
        generator_run = next(item for item in generator_runs if item["run_id"] == selected_generator_id)
        gm1, gm2, gm3, gm4, gm5 = st.columns(5)
        gm1.metric("Evaluated", str(generator_run.get("evaluated", 0)))
        gm2.metric("Accepted", str(generator_run.get("accepted", 0)))
        gm3.metric("Duplicates", str(generator_run.get("duplicates", 0)))
        gm4.metric("Best Score", f"{float(generator_run.get('best_score', 0.0)):.2f}")
        gm5.metric("Status", str(generator_run.get("status", "unknown")))
        leaderboard = pd.DataFrame(generator_run.get("leaderboard", []))
        if not leaderboard.empty:
            keep = [column for column in ["strategy_id", "name", "score", "accepted", "risk_adjusted"] if column in leaderboard.columns]
            backtest_profit = leaderboard["backtest"].apply(lambda item: item.get("profit_factor", 0.0) if isinstance(item, dict) else 0.0)
            leaderboard = leaderboard.assign(profit_factor=backtest_profit)
            keep.append("profit_factor")
            st.markdown("##### Generator Leaderboard")
            st.dataframe(leaderboard[keep], use_container_width=True, hide_index=True)

    runs = list_batch_runs()
    if runs:
        selected_batch_id = st.selectbox(
            "View Batch",
            [run["batch_id"] for run in reversed(runs)],
            index=0,
            key="batch_view_id",
        )
        run = next(item for item in runs if item["batch_id"] == selected_batch_id)
        if run["candidates"]:
            df = pd.DataFrame(
                [
                    {
                        "strategy_id": item["strategy_id"],
                        "name": item["name"],
                        "status": item.get("status", "generated"),
                        "mutation_distance": item.get("mutation_distance", 0),
                        "structural_changes": json.dumps(item.get("structural_changes", [])),
                        "mutations": json.dumps(item.get("mutations", {})),
                    }
                    for item in run["candidates"]
                ]
            )
            st.markdown("##### Generated Candidates")
            st.dataframe(df, use_container_width=True, hide_index=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Candidates", str(len(df)))
            k2.metric("Template", str(run.get("template_name", "")))
            k3.metric("Stage", str(run.get("stage", "generated")))


def _render_retest_runner():
    st.markdown("#### Retest Generated Batch")
    runs = list_batch_runs()
    if not runs:
        empty_state("No generated batches available yet.", "🧪")
        return
    selected_batch_id = st.selectbox(
        "Batch",
        [run["batch_id"] for run in reversed(runs)],
        index=0,
        key="batch_retest_id",
    )
    run = next(item for item in runs if item["batch_id"] == selected_batch_id)
    candidates = list(run.get("candidates", []))
    if not candidates:
        empty_state("This batch has no candidates.", "🧪")
        return
    left, right = st.columns(2)
    with left:
        date_from = str(st.date_input("From", value=pd.Timestamp("2021-01-01"), key="retest_from"))
    with right:
        date_to = str(st.date_input("To", value=pd.Timestamp("2025-01-01"), key="retest_to"))
    eval_cols = st.columns(4)
    top_n = int(eval_cols[0].number_input("Top N", min_value=1, max_value=max(1, len(candidates)), value=min(5, len(candidates)), step=1, key="retest_top_n"))
    run_mc = eval_cols[1].toggle("Monte Carlo", value=False, key="retest_mc")
    run_wfo = eval_cols[2].toggle("WFO", value=False, key="retest_wfo")
    intrabar_steps = int(eval_cols[3].number_input("Intrabar Steps", min_value=1, max_value=8, value=1, step=1, key="retest_intrabar"))
    actions = st.columns(2)
    if actions[0].button("Retest Now", type="primary", use_container_width=True, key="retest_now"):
        with st.spinner("Evaluating selected candidates..."):
            run = evaluate_batch_run(
                selected_batch_id,
                date_from=date_from,
                date_to=date_to,
                top_n=top_n,
                run_mc=run_mc,
                run_wfo_checks=run_wfo,
                intrabar_steps=intrabar_steps,
            )
        success_box(f"Retested batch `{run['batch_id']}`.")
        st.rerun()
    if actions[1].button("Queue Background Retest", use_container_width=True, key="retest_queue"):
        job = new_job_record(
            kind="native_batch_retest",
            title=f"Native retest | {selected_batch_id}",
            payload={
                "batch_id": selected_batch_id,
                "date_from": date_from,
                "date_to": date_to,
                "top_n": top_n,
                "run_mc": run_mc,
                "run_wfo_checks": run_wfo,
                "intrabar_steps": intrabar_steps,
            },
            max_retries=1,
        )
        enqueue_background_job(job)
        st.session_state["job_registry"] = list_jobs()[:100]
        autosave()
        success_box(f"Queued retest job `{job['job_id']}`.")
    evaluated = [item for item in run.get("candidates", []) if item.get("evaluated")]
    st.markdown("##### Candidates")
    candidate_df = pd.DataFrame(
        [
            {
                "strategy_id": item["strategy_id"],
                "name": item["name"],
                "status": item.get("status", "generated"),
                "score": item.get("score", 0.0),
                "accepted": item.get("accepted", False),
                "trades": item.get("backtest", {}).get("num_trades", 0),
                "profit_factor": item.get("backtest", {}).get("profit_factor", 0.0),
                "drawdown_pct": item.get("backtest", {}).get("max_drawdown_pct", 0.0),
                "sharpe": item.get("backtest", {}).get("sharpe_mean", 0.0),
                "wfo_robustness": item.get("wfo", {}).get("robustness_score", 0.0),
                "stability": item.get("parameter_stability", {}).get("score", 0.0),
                "mutation_distance": item.get("mutation_distance", 0),
                "mutations": json.dumps(item.get("mutations", {})),
            }
            for item in run.get("candidates", [])
        ]
    )
    st.dataframe(candidate_df, use_container_width=True, hide_index=True)
    if evaluated:
        df = pd.DataFrame(
            [
                {
                    "strategy_id": item["strategy_id"],
                    "name": item["name"],
                    "score": item.get("score", 0.0),
                    "accepted": item.get("accepted", False),
                    "trades": item.get("backtest", {}).get("num_trades", 0),
                    "profit_factor": item.get("backtest", {}).get("profit_factor", 0.0),
                    "drawdown_pct": item.get("backtest", {}).get("max_drawdown_pct", 0.0),
                    "wfo_robustness": item.get("wfo", {}).get("robustness_score", 0.0),
                    "stability": item.get("parameter_stability", {}).get("score", 0.0),
                    "risk_adjusted": item.get("risk_adjusted", item.get("score", 0.0)),
                    "bucket": item.get("quality_bucket", "watch"),
                }
                for item in evaluated
            ]
        )
        st.markdown("##### Ranking")
        st.dataframe(df.sort_values("risk_adjusted", ascending=False), use_container_width=True, hide_index=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Evaluated", str(len(df)))
        c2.metric("Accepted", str(int(df["accepted"].sum())))
        c3.metric("Median Score", f"{df['score'].median():.2f}")
        c4.metric("Best Risk-Adjusted", f"{df['risk_adjusted'].max():.2f}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(_scatter_chart(df, "drawdown_pct", "profit_factor", "score", "Drawdown vs Profit Factor"), use_container_width=True)
        with col_b:
            st.plotly_chart(_scatter_chart(df, "trades", "wfo_robustness", "score", "Trades vs WFO Robustness"), use_container_width=True)


def _integrate_completed_job(job: dict, result: dict):
    if not isinstance(job, dict):
        return
    if job.get("kind") not in {"native_batch_generation", "native_batch_retest", "native_generator"}:
        return
    if job.get("kind") == "native_generator":
        run_id = str(result.get("run_id", ""))
        if run_id:
            st.session_state["generator_view_id"] = run_id
        return
    batch_id = str((job.get("payload", {}) or {}).get("batch_id", "")) or str(result.get("run_id", ""))
    if batch_id:
        st.session_state["last_batch_id"] = batch_id
        st.session_state["batch_view_id"] = batch_id
        st.session_state["batch_retest_id"] = batch_id


def _render_strategy_builder_jobs():
    jobs = [job for job in list_jobs() if job.get("kind") in {"native_batch_generation", "native_batch_retest", "native_generator"}]
    if not jobs:
        return
    st.markdown("##### Background Research Jobs")
    listing = pd.DataFrame(jobs)[["job_id", "title", "status", "stage", "progress_pct", "submitted_at", "result_run_id"]]
    st.dataframe(listing, use_container_width=True, hide_index=True)
    selected_job_id = st.selectbox(
        "Batch Job Details",
        [job["job_id"] for job in jobs],
        format_func=lambda value: next((f"{item['title']} | {item['status']} | {value}" for item in jobs if item["job_id"] == value), value),
        key="strategy_builder_job_select",
    )
    selected_job = next(job for job in jobs if job["job_id"] == selected_job_id)
    progress = float(selected_job.get("progress_pct", 0.0))
    st.progress(min(max(progress / 100.0, 0.0), 1.0), text=f"{selected_job.get('stage', '')} ({progress:.0f}%)")
    if selected_job.get("status") in {"queued", "running"}:
        if st.button("Stop Selected Job", use_container_width=True, key=f"cancel_job_{selected_job_id}"):
            cancel_background_job(selected_job_id)
            st.rerun()
    if selected_job.get("result_run_id"):
        st.caption(f"Completed run: {selected_job['result_run_id']}")
    events = pd.DataFrame(list(selected_job.get("events", [])))
    if not events.empty:
        st.dataframe(events.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    audit = pd.DataFrame(job_audit_log(selected_job_id, limit=25))
    if not audit.empty:
        with st.expander("Job Audit Trail", expanded=False):
            st.dataframe(audit, use_container_width=True, hide_index=True)


def _render_catalog():
    st.markdown("#### Native Strategy Catalog")
    records = list_native_strategy_records()
    if not records:
        empty_state("No native strategy records yet.", "📚")
        return

    selected_id = st.selectbox(
        "Strategy Record",
        [record["strategy_id"] for record in records],
        format_func=lambda value: next(
            f"{item['name']} | {item['status']} | {item['template_name']}"
            for item in records if item["strategy_id"] == value
        ),
        key="catalog_strategy_id",
    )
    record = next(item for item in records if item["strategy_id"] == selected_id)

    metrics = record.get("evaluation", {}).get("backtest", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", record.get("status", "draft"))
    c2.metric("Score", f"{record.get('evaluation', {}).get('score', 0.0):.2f}")
    c3.metric("Trades", str(metrics.get("num_trades", 0)))
    c4.metric("PF", f"{metrics.get('profit_factor', 0.0):.2f}")

    actions = st.columns(4)
    if actions[0].button("Evaluate", use_container_width=True, key=f"eval_{selected_id}"):
        with st.spinner("Evaluating strategy with backtest, Monte Carlo, and WFO..."):
            evaluate_native_strategy(
                selected_id,
                date_from="2021-01-01",
                date_to="2025-01-01",
                intrabar_steps=1,
                run_mc=True,
                run_wfo_checks=True,
            )
        st.rerun()
    if actions[1].button("Promote", use_container_width=True, key=f"promote_{selected_id}"):
        update_native_strategy_status(selected_id, status="promoted")
        st.rerun()
    if actions[2].button("Reject", use_container_width=True, key=f"reject_{selected_id}"):
        update_native_strategy_status(selected_id, status="rejected")
        st.rerun()
    if actions[3].button("Archive", use_container_width=True, key=f"archive_{selected_id}"):
        update_native_strategy_status(selected_id, status="archived")
        st.rerun()

    lineage_chain = strategy_lineage_tree(selected_id)
    children = strategy_children(selected_id)
    if lineage_chain:
        st.markdown("##### Lineage Tree")
        lineage_df = pd.DataFrame(
            [
                {
                    "strategy_id": item["strategy_id"],
                    "name": item["name"],
                    "status": item.get("status", ""),
                    "score": item.get("evaluation", {}).get("score", 0.0),
                    "created_at": item.get("created_at", ""),
                }
                for item in lineage_chain + children
            ]
        ).drop_duplicates(subset=["strategy_id"])
        st.dataframe(lineage_df, use_container_width=True, hide_index=True)

    st.markdown("##### Edit / Regenerate")
    edited_payload = st.text_area(
        "Template payload JSON",
        value=json.dumps(record.get("template_payload", {}), indent=2),
        height=260,
        key=f"payload_editor_{selected_id}",
    )
    editor_actions = st.columns(2)
    if editor_actions[0].button("Load Into Builder", use_container_width=True, key=f"load_builder_{selected_id}"):
        st.session_state["builder_template"] = record["template_name"]
        st.session_state["builder_payload"] = json.loads(edited_payload)
        st.session_state["page"] = "Strategy Builder"
        st.rerun()
    if editor_actions[1].button("Regenerate Version", use_container_width=True, key=f"regen_{selected_id}"):
        payload = json.loads(edited_payload)
        child_record, meta = regenerate_native_strategy_version(
            selected_id,
            template_name=record["template_name"],
            payload=payload,
        )
        _sync_session_strategy(child_record, meta["compiled"], paths=meta["paths"], template_name=record["template_name"], payload=payload)
        success_box(f"Generated new version `{child_record['strategy_id']}`.")
        st.rerun()

    compare_pool = [item["strategy_id"] for item in lineage_chain + children if item["strategy_id"] != selected_id]
    if compare_pool:
        diff_target = st.selectbox("Compare Against Version", compare_pool, key=f"diff_target_{selected_id}")
        if st.button("Show Version Diff", use_container_width=True, key=f"show_diff_{selected_id}"):
            diff = strategy_payload_diff(selected_id, diff_target)
            st.markdown("##### Version Diff")
            st.code(json.dumps(diff, indent=2), language="json")

    st.markdown("##### Evaluation")
    st.code(json.dumps(record.get("evaluation", {}), indent=2), language="json")
    st.markdown("##### Lineage")
    st.code(json.dumps(record.get("lineage", {}), indent=2), language="json")
    st.markdown("##### Payload")
    st.code(json.dumps(record.get("template_payload", {}), indent=2), language="json")


def _scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, size_col: str, title: str):
    import plotly.express as px

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        hover_name="name",
        hover_data=["strategy_id", "accepted", "score"],
        color="accepted",
        title=title,
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _bar_rank_chart(df: pd.DataFrame):
    import plotly.express as px

    fig = px.bar(
        df,
        x="name",
        y="risk_adjusted",
        color="bucket",
        hover_data=["strategy_id", "score", "profit_factor", "drawdown_pct"],
        title="Top Candidates By Risk-Adjusted Rank",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="")
    return fig
