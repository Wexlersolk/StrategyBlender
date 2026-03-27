"""
EA import and management.
"""

from pathlib import Path
import re
import uuid
import streamlit as st
from services.conversion_service import convert_ea_source, normalize_symbol, persist_converted_ea
from ui.components.common import (
    section_header, empty_state, success_box, error_box,
    TIMEFRAMES, SYMBOLS_COMMON
)
from ui.state import autosave


def render():
    section_header("EA Manager", "Load Expert Advisors from local files or upload them manually")

    tab_load, tab_manage = st.tabs(["📂 Load EA", "📋 Manage EAs"])

    with tab_load:
        _render_load_tab()

    with tab_manage:
        _render_manage_tab()


def _render_load_tab():
    st.markdown("#### Load an Expert Advisor")
    st.markdown(
        "<p style='color:#8B9BB4;'>Load a `.mq5` source file from `mine/` or upload one manually. "
        "Parameters will be detected automatically.</p>",
        unsafe_allow_html=True
    )

    source_name = None
    source = None

    local_files = sorted(Path("mine").glob("*.mq5"))
    if local_files:
        options = {str(path): path.name for path in local_files}
        selected_path = st.selectbox(
            "Use local file from `mine/`",
            [""] + list(options.keys()),
            format_func=lambda value: "Select a local strategy" if not value else options[value],
            key="mine_strategy_select",
        )
        if selected_path:
            path = Path(selected_path)
            source_name = path.name
            source = path.read_text(encoding="utf-8", errors="replace")
            st.success(f"Loaded local file: **{path.name}**", icon="✅")

    uploaded = st.file_uploader(
        "Or upload a `.mq5` file",
        type=["mq5"],
        help="Upload the MQL5 source file of your Expert Advisor"
    )

    if uploaded is not None:
        source_name = uploaded.name
        source = uploaded.read().decode("utf-8", errors="replace")

    if source is None or source_name is None:
        return

    params = _parse_inputs(source)
    detected = _detect_mql_metadata(source)
    ea_name = source_name.replace(".mq5", "")

    st.success(f"Parsed **{ea_name}** — found {len(params)} configurable parameters", icon="✅")

    with st.form("load_ea_form"):
        st.markdown("#### Assign to Symbol & Timeframe")

        col1, col2 = st.columns(2)
        with col1:
            default_symbol = detected.get("symbol") if detected.get("symbol") in SYMBOLS_COMMON else SYMBOLS_COMMON[0]
            symbol_choice = st.selectbox("Symbol", SYMBOLS_COMMON, index=SYMBOLS_COMMON.index(default_symbol))
            custom_symbol = st.text_input("Or enter custom symbol", placeholder="e.g. ETHUSDT")
        with col2:
            default_tf = detected.get("timeframe") if detected.get("timeframe") in TIMEFRAMES else "H1"
            timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(default_tf))

        symbol = custom_symbol.strip() if custom_symbol.strip() else symbol_choice

        if params:
            st.markdown("#### Detected Parameters")
            st.markdown(
                "<p style='color:#8B9BB4;font-size:0.9rem;'>These parameters were found in the "
                "source file. You can set bounds for AI optimisation later.</p>",
                unsafe_allow_html=True
            )
            cols = st.columns(3)
            for i, (name, default) in enumerate(params.items()):
                with cols[i % 3]:
                    st.text_input(f"{name}", value=str(default), disabled=True,
                                  key=f"param_{name}_{i}")

        submitted = st.form_submit_button("Add EA to Portfolio", use_container_width=True,
                                           type="primary")

        if submitted:
            ea_id = str(uuid.uuid4())[:8]
            conversion = convert_ea_source(
                source=source,
                strategy_name=ea_name,
                symbol=symbol,
                timeframe=timeframe,
                ea_id=ea_id,
            )
            paths = persist_converted_ea(conversion, ea_name)
            st.session_state["ea_counter"] += 1
            st.session_state["eas"][ea_id] = {
                "name":       ea_name,
                "symbol":     symbol,
                "timeframe":  timeframe,
                "params":     conversion.params,
                "source":     source,
                "python_source": conversion.strategytester_source,
                "engine_source": conversion.engine_source,
                "strategy_path": paths["engine_path"],
                "python_path": paths["strategytester_path"],
                "strategy_module": conversion.strategy_module,
                "strategy_class": conversion.strategy_class,
                "conversion_warnings": conversion.warnings,
                "conversion_functions": conversion.functions,
                "ai_enabled": False,
                "id":         ea_id,
            }
            autosave()
            success_box(
                f"Added **{ea_name}** on {symbol} {timeframe} and generated a "
                "local backtest strategy plus a `strategytester5` scaffold."
            )
            st.rerun()


def _render_manage_tab():
    eas = st.session_state.get("eas", {})

    if not eas:
        empty_state("No EAs loaded yet. Use the Load EA tab to add strategies.", "🤖")
        return

    st.markdown(f"#### {len(eas)} strategy/strategies loaded")

    for ea_id, ea in list(eas.items()):
        with st.expander(f"**{ea['name']}**  —  {ea['symbol']} {ea['timeframe']}", expanded=False):
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"**Symbol:** {ea['symbol']}  |  **Timeframe:** {ea['timeframe']}")
                st.markdown(f"**Parameters:** {len(ea.get('params', {}))} detected")
                st.markdown(
                    f"**Converted functions:** {len(ea.get('conversion_functions', []))}"
                )
                if ea.get("params"):
                    param_str = "  |  ".join(
                        f"{k}: {v}" for k, v in list(ea["params"].items())[:6]
                    )
                    st.markdown(f"<small style='color:#8B9BB4;'>{param_str}</small>",
                                unsafe_allow_html=True)
                if ea.get("conversion_warnings"):
                    for warning in ea["conversion_warnings"]:
                        st.warning(warning, icon="⚠️")

            with col2:
                new_symbol = st.selectbox(
                    "Symbol", SYMBOLS_COMMON,
                    index=SYMBOLS_COMMON.index(ea["symbol"]) if ea["symbol"] in SYMBOLS_COMMON else 0,
                    key=f"sym_{ea_id}"
                )
                new_tf = st.selectbox(
                    "Timeframe", TIMEFRAMES,
                    index=TIMEFRAMES.index(ea["timeframe"]) if ea["timeframe"] in TIMEFRAMES else 4,
                    key=f"tf_{ea_id}"
                )
                if st.button("Update", key=f"upd_{ea_id}"):
                    updated = convert_ea_source(
                        source=ea["source"],
                        strategy_name=ea["name"],
                        symbol=new_symbol,
                        timeframe=new_tf,
                        ea_id=ea_id,
                    )
                    paths = persist_converted_ea(updated, ea["name"])
                    st.session_state["eas"][ea_id]["symbol"]    = new_symbol
                    st.session_state["eas"][ea_id]["timeframe"] = new_tf
                    st.session_state["eas"][ea_id]["params"] = updated.params
                    st.session_state["eas"][ea_id]["python_source"] = updated.strategytester_source
                    st.session_state["eas"][ea_id]["engine_source"] = updated.engine_source
                    st.session_state["eas"][ea_id]["strategy_path"] = paths["engine_path"]
                    st.session_state["eas"][ea_id]["python_path"] = paths["strategytester_path"]
                    st.session_state["eas"][ea_id]["strategy_module"] = updated.strategy_module
                    st.session_state["eas"][ea_id]["strategy_class"] = updated.strategy_class
                    st.session_state["eas"][ea_id]["conversion_warnings"] = updated.warnings
                    st.session_state["eas"][ea_id]["conversion_functions"] = updated.functions
                    autosave()
                    st.rerun()

            with col3:
                ai_on = st.toggle("AI", value=ea.get("ai_enabled", False), key=f"ai_{ea_id}")
                st.session_state["eas"][ea_id]["ai_enabled"] = ai_on
                st.markdown(
                    f"<small style='color:{'#2ECC71' if ai_on else '#8B9BB4'};'>"
                    f"{'AI active' if ai_on else 'AI off'}</small>",
                    unsafe_allow_html=True
                )
                if st.button("🗑️ Remove", key=f"del_{ea_id}", type="secondary"):
                    del st.session_state["eas"][ea_id]
                    if ea_id in st.session_state.get("backtest_results", {}):
                        del st.session_state["backtest_results"][ea_id]
                    autosave()
                    st.rerun()


# ── MQL5 source parser ────────────────────────────────────────────────────────
def _parse_inputs(source: str) -> dict:
    """
    Extract `input` parameter names and default values from .mq5 source.
    Handles: input double x = 1.5;  input int y = 14;  input string z = "abc";
    Skips separator strings like "--- Section ---".
    """
    params = {}
    pattern = re.compile(
        r'input\s+(?:const\s+)?(\w+)\s+(\w+)\s*=\s*([^;]+);',
        re.MULTILINE
    )
    skip_types = {"string", "ENUM_ORDER_TYPE_FILLING", "ENUM_SQ_CORNER", "color", "bool"}

    for match in pattern.finditer(source):
        dtype, name, default = match.group(1), match.group(2), match.group(3).strip()

        # Skip section separators and non-numeric types
        if dtype in skip_types:
            continue
        if default.startswith('"'):
            continue
        # Skip names that look like section labels
        if name.startswith("s") and len(name) <= 5:
            continue

        try:
            if dtype in ("double", "float"):
                params[name] = float(default.replace(",", "."))
            elif dtype in ("int", "uint", "long", "ulong"):
                params[name] = int(default)
            else:
                params[name] = default
        except ValueError:
            params[name] = default

    return params


def _detect_mql_metadata(source: str) -> dict:
    meta = {}
    symbol_match = re.search(r"Backtested on\s+(.+?)\s*/", source)
    timeframe_match = re.search(r"/\s*([MHDW]\d(?:,\s*[MHDW]\d)?)", source)
    if symbol_match:
        symbol = normalize_symbol(symbol_match.group(1).strip())
        meta["symbol"] = symbol
    if timeframe_match:
        tf = timeframe_match.group(1).split(",")[0].strip()
        meta["timeframe"] = tf
    return meta
