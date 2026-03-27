from __future__ import annotations

import importlib
import itertools
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.data_loader import load_bars
from research.walk_forward import run_wfo
from services.backtest_service import backtest_result_payload, run_backtest
from services.conversion_service import convert_ea_source, normalize_symbol, persist_converted_ea
from ui.components.common import empty_state, error_box, section_header
from ui.state import autosave


ROOT = Path(__file__).resolve().parent.parent.parent
SCHEDULES_DIR = ROOT / "data" / "exports" / "ai_schedules"

TIMEFRAME_TO_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


MARKET_FEATURE_COLS = [
    "atr_pctile_6m",
    "realized_vol_3m",
    "trend_slope_3m",
    "range_position_3m",
    "breakout_score_3m",
]

PERIOD_CONFIGS = {
    "monthly": {
        "freq": "M",
        "label": "Month",
        "adjective": "monthly",
        "per_year": 12.0,
        "lookback_min": 3,
        "lookback_max": 18,
        "lookback_default": 6,
        "lookback_step": 1,
        "market_short": 3,
        "market_long": 6,
    },
    "weekly": {
        "freq": "W-FRI",
        "label": "Week",
        "adjective": "weekly",
        "per_year": 52.0,
        "lookback_min": 4,
        "lookback_max": 26,
        "lookback_default": 12,
        "lookback_step": 2,
        "market_short": 13,
        "market_long": 26,
    },
}

SOURCE_WEIGHTS = {
    "baseline": 1.0,
    "generated": 0.75,
    "bootstrap": 0.50,
    "wfo_oos": 1.75,
    "wfo_bootstrap": 1.25,
}


def _coerce_metric(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _period_config(period_mode: str) -> dict:
    return PERIOD_CONFIGS.get(str(period_mode or "monthly"), PERIOD_CONFIGS["monthly"])


def _period_key(value, period_mode: str = "monthly") -> str:
    ts = pd.Timestamp(value)
    return str(ts.to_period(_period_config(period_mode)["freq"]).end_time.normalize().date())


def _aggregate_period_stats(payload: dict, period_mode: str = "monthly") -> pd.DataFrame:
    deals = payload.get("deals_df")
    if deals is None or not isinstance(deals, pd.DataFrame) or deals.empty or "profit" not in deals.columns:
        return pd.DataFrame()

    frame = deals.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
            frame = frame.set_index("time")
        else:
            return pd.DataFrame()
    frame = frame.sort_index()
    cfg = _period_config(period_mode)
    grouped = frame.groupby(pd.Grouper(freq=cfg["freq"]))
    periods = grouped["profit"].agg(total_profit="sum", num_trades="count", win_rate=lambda x: (x > 0).mean())
    periods = periods.dropna(how="all")
    periods = periods[periods["num_trades"].fillna(0.0) > 0].copy()
    if periods.empty:
        return pd.DataFrame()
    profits = periods["total_profit"].to_numpy(dtype=float)
    std = profits.std(ddof=0) + 1e-8
    periods["sharpe"] = profits / std * np.sqrt(float(cfg["per_year"]))
    periods["period"] = periods.index.strftime("%Y-%m-%d")
    return periods.set_index("period")


def _filter_active_period_rows(df: pd.DataFrame, min_trades: int, min_abs_profit: float) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "num_trades" not in out.columns:
        out["num_trades"] = 0.0
    out["num_trades"] = pd.to_numeric(out["num_trades"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["total_profit"] = pd.to_numeric(out["total_profit"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask = out["num_trades"] >= max(int(min_trades), 0)
    if float(min_abs_profit) > 0.0:
        mask &= out["total_profit"].abs() >= float(min_abs_profit)
    return out.loc[mask].copy()


def _period_display_name(period_mode: str) -> str:
    return "Monthly" if str(period_mode) == "monthly" else "Weekly"


def _rolling_last_percentile(values: pd.Series, window: int) -> pd.Series:
    def _percentile(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return 0.5
        last = arr[-1]
        return float(np.mean(arr <= last))

    return values.rolling(window, min_periods=max(3, window // 2)).apply(_percentile, raw=True)


def _monthly_market_features(symbol: str, timeframe: str, period_mode: str = "monthly") -> pd.DataFrame:
    cache = st.session_state.setdefault("_market_feature_cache", {})
    cache_key = f"{symbol}|{timeframe}|{period_mode}"
    cached = cache.get(cache_key)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()

    try:
        bars = load_bars(symbol, timeframe)
    except Exception:
        empty = pd.DataFrame(columns=MARKET_FEATURE_COLS)
        cache[cache_key] = empty
        return empty.copy()

    if bars is None or bars.empty:
        empty = pd.DataFrame(columns=MARKET_FEATURE_COLS)
        cache[cache_key] = empty
        return empty.copy()

    bars = bars.copy().sort_index()
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            (bars["high"] - bars["low"]).abs(),
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    returns = bars["close"].pct_change().replace([np.inf, -np.inf], np.nan)

    monthly = pd.DataFrame(index=bars.index)
    monthly["tr"] = tr
    monthly["ret"] = returns
    monthly["close"] = bars["close"]
    monthly["high"] = bars["high"]
    monthly["low"] = bars["low"]
    cfg = _period_config(period_mode)
    short_window = int(cfg["market_short"])
    long_window = int(cfg["market_long"])
    monthly = monthly.groupby(pd.Grouper(freq=cfg["freq"])).agg(
        close_last=("close", "last"),
        high_max=("high", "max"),
        low_min=("low", "min"),
        tr_mean=("tr", "mean"),
        ret_std=("ret", "std"),
    ).dropna(how="all")

    if monthly.empty:
        empty = pd.DataFrame(columns=MARKET_FEATURE_COLS)
        cache[cache_key] = empty
        return empty.copy()

    min_periods = max(2, short_window // 2)
    atr_3m = monthly["tr_mean"].rolling(short_window, min_periods=min_periods).mean()
    atr_pctile_6m = _rolling_last_percentile(atr_3m.fillna(0.0), long_window).fillna(0.5)
    realized_vol_3m = monthly["ret_std"].rolling(short_window, min_periods=min_periods).mean().fillna(0.0)
    trend_slope_3m = monthly["close_last"].pct_change(short_window).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rolling_high = monthly["high_max"].rolling(short_window, min_periods=min_periods).max()
    rolling_low = monthly["low_min"].rolling(short_window, min_periods=min_periods).min()
    range_span = (rolling_high - rolling_low).replace(0.0, np.nan)
    range_position_3m = ((monthly["close_last"] - rolling_low) / range_span).clip(0.0, 1.0).fillna(0.5)
    prev_high = rolling_high.shift(1)
    prev_low = rolling_low.shift(1)
    breakout_up = ((monthly["close_last"] / prev_high.replace(0.0, np.nan)) - 1.0).replace([np.inf, -np.inf], np.nan)
    breakout_down = ((prev_low / monthly["close_last"].replace(0.0, np.nan)) - 1.0).replace([np.inf, -np.inf], np.nan)
    breakout_score_3m = breakout_up.fillna(0.0) - breakout_down.fillna(0.0)

    features = pd.DataFrame(
        {
            "atr_pctile_6m": atr_pctile_6m.astype(float),
            "realized_vol_3m": realized_vol_3m.astype(float),
            "trend_slope_3m": trend_slope_3m.astype(float),
            "range_position_3m": range_position_3m.astype(float),
            "breakout_score_3m": breakout_score_3m.astype(float),
        },
        index=monthly.index.strftime("%Y-%m-%d"),
    )
    features.index.name = "period"
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cache[cache_key] = features
    return features.copy()


def _enrich_with_market_features(df: pd.DataFrame, eas: dict) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for col in MARKET_FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0

    for ea_id, group_idx in out.groupby("ea_id").groups.items():
        ea = eas.get(ea_id)
        if not ea:
            continue
        period_mode = out.loc[group_idx, "period_mode"].iloc[0] if "period_mode" in out.columns and len(group_idx) else "monthly"
        features = _monthly_market_features(ea["symbol"], ea["timeframe"], period_mode=str(period_mode))
        if features.empty:
            continue
        periods = out.loc[group_idx, "month"].astype(str)
        aligned = features.reindex(periods).fillna(0.0)
        for col in MARKET_FEATURE_COLS:
            out.loc[group_idx, col] = aligned[col].to_numpy(dtype=float)

    return out


def _resolve_lot_param_name(params: dict) -> str | None:
    candidates = ["_lots", "mmLots", "lots", "Lots", "lot", "LotSize", "fixed_lots"]
    for name in candidates:
        value = params.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name
    return None


def _resolve_lot_value(params: dict, fallback: float = 1.0) -> float:
    name = _resolve_lot_param_name(params)
    if not name:
        return float(fallback)
    return _coerce_metric(params.get(name), fallback)


def _resolve_lot_step(params: dict) -> float:
    for name in ("mmStep", "lotStep", "LotStep", "volume_step", "VolumeStep"):
        value = params.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            step = abs(float(value))
            if step > 0:
                return step
    return 0.01


def _quantize_lot(value: float, step: float, minimum: float = 0.01) -> float:
    step = max(float(step), 1e-6)
    minimum = max(float(minimum), step)
    quantized = round(round(float(value) / step) * step, 6)
    return max(minimum, quantized)


OBJECTIVE_OPTIONS = [
    "mt5_complex",
    "balanced",
    "net_profit",
    "sharpe",
    "profit_factor",
    "recovery_factor",
]


def _objective_label(name: str) -> str:
    labels = {
        "mt5_complex": "MT5-like Complex",
        "balanced": "Balanced",
        "net_profit": "Net Profit",
        "sharpe": "Sharpe",
        "profit_factor": "Profit Factor",
        "recovery_factor": "Recovery Factor",
    }
    return labels.get(name, name)


def _candidate_objective_scores(month_candidates: pd.DataFrame, history_window: pd.DataFrame, objective: str, period_mode: str = "monthly") -> pd.Series:
    records: list[dict] = []
    history_profits = history_window["total_profit"].astype(float).tolist()
    initial_capital = 100_000.0

    for _, candidate in month_candidates.iterrows():
        profits = np.array(history_profits + [float(candidate["total_profit"])], dtype=float)
        equity = initial_capital + np.cumsum(profits)
        peak = np.maximum.accumulate(np.concatenate([[initial_capital], equity]))
        dd = peak - np.concatenate([[initial_capital], equity])
        dd_abs = float(dd.max()) if len(dd) else 0.0
        dd_pct = float((dd / np.maximum(peak, 1e-9)).max() * 100.0) if len(dd) else 0.0
        pos = profits[profits > 0].sum()
        neg = profits[profits < 0].sum()
        std = profits.std(ddof=0)
        sharpe = 0.0 if std < 1e-9 else float(profits.mean() / std * np.sqrt(float(_period_config(period_mode)["per_year"])))
        net_profit = float(profits.sum())
        profit_factor = float(pos / abs(neg)) if neg < 0 else float(pos if pos > 0 else 0.0)
        recovery_factor = float(net_profit / dd_abs) if dd_abs > 1e-9 else (float(net_profit) if net_profit > 0 else 0.0)
        win_rate = float((profits > 0).mean())
        records.append(
            {
                "net_profit": net_profit,
                "sharpe": sharpe,
                "profit_factor": profit_factor,
                "recovery_factor": recovery_factor,
                "win_rate": win_rate,
                "max_drawdown_pct": dd_pct,
                "lot_scale": float(candidate["lot_scale"]),
            }
        )

    metrics = pd.DataFrame(records, index=month_candidates.index)

    def zscore(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        std = float(clean.std(ddof=0))
        if std < 1e-9:
            return pd.Series(0.0, index=clean.index)
        return (clean - float(clean.mean())) / std

    net_z = zscore(metrics["net_profit"])
    sharpe_z = zscore(metrics["sharpe"])
    pf_z = zscore(metrics["profit_factor"])
    recovery_z = zscore(metrics["recovery_factor"])
    win_z = zscore(metrics["win_rate"])
    dd_z = zscore(-metrics["max_drawdown_pct"])
    scale_penalty = 0.03 * np.abs(np.log(metrics["lot_scale"].clip(lower=1e-6)))

    if objective == "net_profit":
        score = net_z + 0.10 * dd_z - scale_penalty
    elif objective == "sharpe":
        score = sharpe_z + 0.15 * dd_z + 0.10 * recovery_z - scale_penalty
    elif objective == "profit_factor":
        score = pf_z + 0.15 * net_z + 0.10 * dd_z - scale_penalty
    elif objective == "recovery_factor":
        score = recovery_z + 0.20 * net_z + 0.10 * dd_z - scale_penalty
    elif objective == "balanced":
        score = 0.35 * net_z + 0.25 * sharpe_z + 0.15 * recovery_z + 0.15 * pf_z + 0.10 * dd_z - scale_penalty
    else:
        score = 0.30 * net_z + 0.20 * sharpe_z + 0.20 * recovery_z + 0.15 * pf_z + 0.10 * win_z + 0.05 * dd_z - scale_penalty

    return pd.Series(score, index=month_candidates.index)


def _source_weight_for(source_name: str) -> float:
    return float(SOURCE_WEIGHTS.get(str(source_name or "").strip(), 0.75))


def _window_metrics(window: pd.DataFrame) -> dict[str, float]:
    profits = pd.to_numeric(window["total_profit"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    if len(profits) == 0:
        return {
            "drawdown_proxy_lb": 0.0,
            "profit_factor_lb": 0.0,
            "recovery_factor_lb": 0.0,
            "return_vol_lb": 0.0,
            "negative_month_ratio_lb": 0.0,
            "avg_loss_lb": 0.0,
            "worst_month_lb": 0.0,
            "downside_vol_lb": 0.0,
            "win_streak_lb": 0.0,
            "loss_streak_lb": 0.0,
        }

    equity = np.cumsum(profits)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    curve = np.concatenate([[0.0], equity])
    drawdown = peak - curve
    dd_abs = float(drawdown.max()) if len(drawdown) else 0.0
    pos = float(profits[profits > 0].sum())
    neg = float(profits[profits < 0].sum())
    pf = float(pos / abs(neg)) if neg < 0 else float(pos if pos > 0 else 0.0)
    recovery = float(profits.sum() / dd_abs) if dd_abs > 1e-9 else float(profits.sum())
    vol = float(np.std(profits, ddof=0))
    neg_mask = profits < 0
    neg_ratio = float(np.mean(neg_mask))
    avg_loss = float(np.abs(profits[neg_mask]).mean()) if np.any(neg_mask) else 0.0
    worst_month = float(np.min(profits)) if len(profits) else 0.0
    downside_vol = float(np.std(profits[neg_mask], ddof=0)) if np.sum(neg_mask) > 1 else avg_loss

    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    for p in profits:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = 0
            cur_loss = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)

    return {
        "drawdown_proxy_lb": dd_abs,
        "profit_factor_lb": pf,
        "recovery_factor_lb": recovery,
        "return_vol_lb": vol,
        "negative_month_ratio_lb": neg_ratio,
        "avg_loss_lb": avg_loss,
        "worst_month_lb": worst_month,
        "downside_vol_lb": downside_vol,
        "win_streak_lb": float(max_win),
        "loss_streak_lb": float(max_loss),
    }


def _assign_regime(scale: float, neutral_band: float) -> str:
    if scale <= 1.0 - neutral_band:
        return "risk_off"
    if scale >= 1.0 + neutral_band:
        return "risk_on"
    return "normal"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom < 1e-9] = 1.0
    return exp / denom


def render():
    period_mode = st.selectbox(
        "Policy period",
        options=["monthly", "weekly"],
        index=0,
        format_func=lambda x: _period_config(x)["label"],
        key="ai_period_mode",
    )
    section_header(
        "AI Training",
        f"Generate lot-variation data, train a {_period_config(period_mode)['adjective']} lot policy, and compare baseline vs AI",
    )

    eas = st.session_state.get("eas", {})
    results = st.session_state.get("backtest_results", {})
    experiments = st.session_state.get("ai_experiment_results", {})

    tab_dataset, tab_train, tab_compare = st.tabs([
        "🗂️ Dataset",
        "🧠 Train Lot Policy",
        "📊 Compare Policy",
    ])

    with tab_dataset:
        _render_dataset_tab(eas, results, experiments, period_mode)

    with tab_train:
        _render_train_tab(eas, results, experiments, period_mode)

    with tab_compare:
        _render_compare_tab()


def _build_monthly_dataset(
    eas: dict,
    results: dict,
    experiments: dict | None = None,
    selected_ids: list[str] | None = None,
    period_mode: str = "monthly",
) -> pd.DataFrame:
    rows: list[dict] = []
    ids = selected_ids or list(results.keys())

    for ea_id in ids:
        result = results.get(ea_id)
        ea = eas.get(ea_id)
        if not result or not ea:
            continue
        monthly = _aggregate_period_stats(result, period_mode=period_mode)
        if monthly is None or monthly.empty:
            continue

        monthly = monthly.copy()
        base_params = ea.get("params", {})
        lot_param = _resolve_lot_param_name(base_params)
        base_lot = _resolve_lot_value(base_params, 1.0)
        lot_step = _resolve_lot_step(base_params)
        if "total_profit" not in monthly.columns and "profit" in monthly.columns:
            monthly["total_profit"] = monthly["profit"]
        if "num_trades" not in monthly.columns and "trades" in monthly.columns:
            monthly["num_trades"] = monthly["trades"]
        if "win_rate" not in monthly.columns:
            monthly["win_rate"] = np.nan
        if "sharpe" not in monthly.columns:
            monthly["sharpe"] = np.nan

        for month, row in monthly.iterrows():
            rows.append(
                {
                    "month": str(month),
                    "month_dt": pd.Timestamp(str(month)),
                    "ea_id": ea_id,
                    "source_ea_id": ea_id,
                    "run_id": ea_id,
                    "name": ea["name"],
                    "symbol": ea["symbol"],
                    "timeframe": ea["timeframe"],
                    "timeframe_minutes": TIMEFRAME_TO_MINUTES.get(str(ea["timeframe"]).upper(), 0),
                    "lot_param": lot_param or "",
                    "base_lot": base_lot,
                    "lot_step": lot_step,
                    "actual_lot": base_lot,
                    "lot_scale": 1.0,
                    "total_profit": _coerce_metric(row.get("total_profit", 0.0)),
                    "sharpe": _coerce_metric(row.get("sharpe", 0.0)),
                    "win_rate": _coerce_metric(row.get("win_rate", 0.0)),
                    "num_trades": _coerce_metric(row.get("num_trades", 0.0)),
                    "source": "baseline",
                    "period_mode": period_mode,
                }
            )

    for exp_id, result in (experiments or {}).items():
        monthly = _aggregate_period_stats(result, period_mode=period_mode)
        meta = result.get("meta", {})
        if monthly is None or monthly.empty:
            continue
        source_ea_id = meta.get("source_ea_id")
        if selected_ids and source_ea_id not in selected_ids:
            continue
        source_ea = eas.get(source_ea_id, {})
        base_params = source_ea.get("params", {})
        overrides = meta.get("overrides", {}) or {}
        lot_param = _resolve_lot_param_name(overrides) or _resolve_lot_param_name(base_params)
        base_lot = _resolve_lot_value(base_params, 1.0)
        actual_lot = _resolve_lot_value(overrides if lot_param in overrides else {lot_param: base_lot} if lot_param else {}, base_lot)
        lot_step = _resolve_lot_step(base_params)
        monthly = monthly.copy()
        if "total_profit" not in monthly.columns and "profit" in monthly.columns:
            monthly["total_profit"] = monthly["profit"]
        if "num_trades" not in monthly.columns and "trades" in monthly.columns:
            monthly["num_trades"] = monthly["trades"]
        if "win_rate" not in monthly.columns:
            monthly["win_rate"] = np.nan
        if "sharpe" not in monthly.columns:
            monthly["sharpe"] = np.nan
        for month, row in monthly.iterrows():
            rows.append(
                {
                    "month": str(month),
                    "month_dt": pd.Timestamp(str(month)),
                    "ea_id": source_ea_id or meta.get("ea_id", exp_id),
                    "source_ea_id": source_ea_id or meta.get("ea_id", exp_id),
                    "run_id": exp_id,
                    "variant_id": exp_id,
                    "name": meta.get("name", meta.get("source_name", exp_id)),
                    "symbol": meta.get("symbol", ""),
                    "timeframe": meta.get("timeframe", ""),
                    "timeframe_minutes": TIMEFRAME_TO_MINUTES.get(str(meta.get("timeframe", "")).upper(), 0),
                    "lot_param": lot_param or "",
                    "base_lot": base_lot,
                    "lot_step": lot_step,
                    "actual_lot": actual_lot,
                    "lot_scale": actual_lot / base_lot if abs(base_lot) > 1e-9 else 1.0,
                    "total_profit": _coerce_metric(row.get("total_profit", 0.0)),
                    "sharpe": _coerce_metric(row.get("sharpe", 0.0)),
                    "win_rate": _coerce_metric(row.get("win_rate", 0.0)),
                    "num_trades": _coerce_metric(row.get("num_trades", 0.0)),
                    "source": meta.get("dataset_source", "generated"),
                    "period_mode": period_mode,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["ea_id", "run_id", "month_dt"]).reset_index(drop=True)
    df = _enrich_with_market_features(df, eas)
    return df


def _load_generated_strategy_class(ea: dict):
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    normalized_symbol = normalize_symbol(ea["symbol"])
    refreshed = convert_ea_source(
        source=ea["source"],
        strategy_name=ea["name"],
        symbol=normalized_symbol,
        timeframe=ea["timeframe"],
        ea_id=ea["id"],
    )
    paths = persist_converted_ea(refreshed, ea["name"])
    ea["strategy_path"] = paths["engine_path"]
    ea["python_path"] = paths["strategytester_path"]
    ea["strategy_module"] = refreshed.strategy_module
    ea["strategy_class"] = refreshed.strategy_class
    ea["engine_source"] = refreshed.engine_source
    module_name = refreshed.strategy_module
    mod = importlib.reload(sys.modules[module_name]) if module_name in sys.modules else importlib.import_module(module_name)
    return getattr(mod, refreshed.strategy_class)


def _suggest_param_candidates(params: dict) -> list[str]:
    numeric = {k: v for k, v in params.items() if isinstance(v, (int, float))}
    preferred = [
        "_lots",
        "mmLots",
        "StopLossCoef1",
        "ProfitTargetCoef1",
        "StopLossCoef2",
        "ProfitTargetCoef2",
        "TrailingStopCoef1",
        "TrailingActCef1",
        "PriceEntryMult1",
        "IndicatorCrsMAPrd1",
        "IndicatorCrsMAPrd2",
    ]
    ordered = [name for name in preferred if name in numeric]
    for name in numeric:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _parameter_values(
    name: str,
    base_value,
    levels: int,
    width: float,
    lot_param_name: str | None = None,
    aggressive_lot_range: bool = False,
) -> list[float | int]:
    local_levels = int(levels)
    local_width = float(width)
    if aggressive_lot_range and lot_param_name and name == lot_param_name:
        local_levels = max(local_levels, 7)
        local_width = max(local_width, 0.8)
        multipliers = np.array([0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.40, 1.60, 1.80], dtype=float)
        if local_levels < len(multipliers):
            idx = np.linspace(0, len(multipliers) - 1, local_levels).round().astype(int)
            multipliers = multipliers[idx]
    else:
        multipliers = np.linspace(1.0 - local_width, 1.0 + local_width, local_levels)
    vals: list[float | int] = []
    base = float(base_value)
    for mult in multipliers:
        candidate = base * float(mult)
        if isinstance(base_value, int) and not isinstance(base_value, bool):
            candidate = max(1, int(round(candidate)))
        else:
            candidate = round(max(0.01, candidate), 4)
        vals.append(candidate)

    deduped: list[float | int] = []
    for val in vals:
        if val not in deduped:
            deduped.append(val)
    return deduped


def _build_param_variants(
    base_params: dict,
    chosen_params: list[str],
    levels: int,
    width: float,
    lot_param_name: str | None = None,
    aggressive_lot_range: bool = False,
) -> list[dict]:
    if not chosen_params:
        return [base_params.copy()]
    value_lists = []
    for name in chosen_params:
        value_lists.append(
            _parameter_values(
                name,
                base_params[name],
                levels,
                width,
                lot_param_name=lot_param_name,
                aggressive_lot_range=aggressive_lot_range,
            )
        )

    variants = []
    for combo in itertools.product(*value_lists):
        override = base_params.copy()
        label_bits = []
        for name, value in zip(chosen_params, combo):
            override[name] = value
            label_bits.append(f"{name}={value}")
        override["_variant_label"] = " | ".join(label_bits)
        variants.append(override)
    return variants


def _build_param_grid(
    base_params: dict,
    chosen_params: list[str],
    levels: int,
    width: float,
    lot_param_name: str | None = None,
    aggressive_lot_range: bool = False,
) -> dict[str, list[float | int]]:
    if not chosen_params:
        return {}
    grid: dict[str, list[float | int]] = {}
    for name in chosen_params:
        grid[name] = _parameter_values(
            name,
            base_params[name],
            levels,
            width,
            lot_param_name=lot_param_name,
            aggressive_lot_range=aggressive_lot_range,
        )
    return grid


def _bootstrap_monthly_variants(monthly: pd.DataFrame, count: int, seed: int, period_mode: str = "monthly") -> list[pd.DataFrame]:
    if monthly is None or monthly.empty or count <= 0:
        return []
    rng = np.random.default_rng(seed)
    profits = monthly["total_profit"].to_numpy(dtype=float)
    trades = monthly["num_trades"].to_numpy(dtype=float) if "num_trades" in monthly.columns else np.zeros(len(monthly))
    wins = monthly["win_rate"].to_numpy(dtype=float) if "win_rate" in monthly.columns else np.zeros(len(monthly))
    variants = []
    for idx in range(count):
        sample_idx = rng.choice(len(monthly), size=len(monthly), replace=True)
        boot = monthly.copy()
        boot["total_profit"] = profits[sample_idx]
        boot["num_trades"] = trades[sample_idx]
        boot["win_rate"] = wins[sample_idx]
        std = boot["total_profit"].std(ddof=0)
        boot["sharpe"] = 0.0 if std < 1e-9 else boot["total_profit"] / std * np.sqrt(float(_period_config(period_mode)["per_year"]))
        variants.append(boot)
    return variants


def _store_experiment_payload(
    current_results: dict,
    *,
    source_ea_id: str,
    ea_name: str,
    symbol: str,
    timeframe: str,
    payload: dict,
    dataset_source: str,
    name: str,
    overrides: dict | None = None,
    extra_meta: dict | None = None,
) -> dict:
    results = dict(current_results)
    exp_id = f"{source_ea_id}_{uuid.uuid4().hex[:8]}"
    payload["meta"] = {
        "dataset_source": dataset_source,
        "source_ea_id": source_ea_id,
        "source_name": ea_name,
        "name": name,
        "symbol": symbol,
        "timeframe": timeframe,
        "overrides": overrides or {},
        **(extra_meta or {}),
    }
    results[exp_id] = payload
    return results


def _render_dataset_tab(eas: dict, results: dict, experiments: dict, period_mode: str):
    cfg = _period_config(period_mode)
    dataset = _build_monthly_dataset(eas, results, experiments, period_mode=period_mode)
    if dataset.empty:
        empty_state("Run at least one backtest with trade results first.", "🧠")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategies", f"{dataset['ea_id'].nunique()}")
    c2.metric(f"{_period_display_name(period_mode)} Rows", f"{len(dataset)}")
    c3.metric("Generated Runs", f"{len(experiments)}")
    c4.metric("Date Span", f"{dataset['month'].min()} → {dataset['month'].max()}")

    by_strategy = (
        dataset.groupby(["name", "symbol", "timeframe", "source"], as_index=False)
        .agg(
            months=("month", "count"),
            total_profit=("total_profit", "sum"),
            mean_sharpe=("sharpe", "mean"),
            mean_win_rate=("win_rate", "mean"),
        )
        .sort_values("months", ascending=False)
    )
    st.markdown("#### Available Training Data")
    st.dataframe(by_strategy, use_container_width=True, hide_index=True)

    st.markdown(f"#### {_period_display_name(period_mode)} Sample Preview")
    st.dataframe(
        dataset[
            ["month", "name", "symbol", "timeframe", "actual_lot", "lot_scale", "total_profit", "sharpe", "win_rate", "num_trades"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Generate More Training Data")
    if not eas:
        return
    ea_options = {ea_id: f"{ea['name']} — {ea['symbol']} {ea['timeframe']}" for ea_id, ea in eas.items()}
    selected_ea_id = st.selectbox("Strategy for local experiments", list(ea_options.keys()), format_func=lambda x: ea_options[x], key="ai_exp_ea")
    ea = eas[selected_ea_id]
    base_params = {k: v for k, v in ea.get("params", {}).items() if isinstance(v, (int, float))}
    lot_param_name = _resolve_lot_param_name(base_params)
    candidates = _suggest_param_candidates(base_params)
    default_params: list[str] = []
    if lot_param_name and lot_param_name in candidates:
        default_params.append(lot_param_name)
    for candidate in candidates:
        if candidate not in default_params:
            default_params.append(candidate)
        if len(default_params) >= min(2, len(candidates)):
            break
    chosen_params = st.multiselect("Parameters to vary", candidates, default=default_params, key="ai_exp_params")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        levels = st.select_slider("Levels per param", options=[3, 4, 5, 6, 7], value=5, key="ai_exp_levels")
    with col_b:
        width = st.slider("Variation width", min_value=0.05, max_value=1.0, value=0.35, step=0.05, key="ai_exp_width")
    with col_c:
        bootstrap_count = st.select_slider("Bootstrap variants", options=[0, 3, 5, 10], value=3, key="ai_exp_boot")
    with col_d:
        intrabar = st.checkbox("Use M1 intrabar", value=False, key="ai_exp_intrabar")
    with col_e:
        lot_focused = st.checkbox("Lot-focused generation", value=True, key="ai_exp_lot_focus")
    aggressive_lot_range = st.checkbox("Aggressive lot range for AI sizing", value=True, key="ai_exp_lot_aggr")
    if lot_focused and lot_param_name:
        st.caption(f"Lot-focused mode will prioritise `{lot_param_name}` and minimise unrelated parameter noise.")
    effective_params = list(chosen_params)
    if lot_focused and lot_param_name:
        secondary = next((name for name in chosen_params if name != lot_param_name), None)
        effective_params = [lot_param_name] + ([secondary] if secondary else [])
        st.caption(f"Effective varied parameters: {', '.join(effective_params)}")
    date_summary = results.get(selected_ea_id, {}).get("summary", {})
    default_from = pd.Timestamp(date_summary.get("date_from", "2020-01-01"))
    default_to = pd.Timestamp(date_summary.get("date_to", "2025-12-31"))
    col_f, col_g = st.columns(2)
    with col_f:
        date_from = st.date_input("From", value=default_from, key="ai_exp_from")
    with col_g:
        date_to = st.date_input("To", value=default_to, key="ai_exp_to")

    estimated = len(
        _build_param_variants(
            base_params,
            effective_params,
            levels,
            width,
            lot_param_name=lot_param_name,
            aggressive_lot_range=aggressive_lot_range,
        )
    )
    st.caption(f"Estimated local runs: {estimated}")

    col_local, col_wfo = st.columns(2)
    with col_local:
        if st.button("Generate Local Parameter Sweep", type="primary", use_container_width=True, key="ai_exp_run"):
            if not effective_params:
                error_box("Select at least one parameter to vary.")
                return
            strat_cls = _load_generated_strategy_class(ea)
            variants = _build_param_variants(
                base_params,
                effective_params,
                levels,
                width,
                lot_param_name=lot_param_name,
                aggressive_lot_range=aggressive_lot_range,
            )
            progress = st.progress(0, text="Running local experiments...")
            new_results = dict(experiments)
            for idx, variant in enumerate(variants, start=1):
                overrides = {k: v for k, v in variant.items() if not k.startswith("_")}
                result = run_backtest(
                    strat_cls,
                    symbol=ea["symbol"],
                    timeframe=ea["timeframe"],
                    date_from=str(date_from),
                    date_to=str(date_to),
                    overrides=overrides,
                    intrabar_steps=60 if intrabar else 1,
                )
                payload = backtest_result_payload(result)
                new_results = _store_experiment_payload(
                    new_results,
                    source_ea_id=selected_ea_id,
                    ea_name=ea["name"],
                    symbol=ea["symbol"],
                    timeframe=ea["timeframe"],
                    payload=payload,
                    dataset_source="generated",
                    name=f"{ea['name']} [{variant['_variant_label']}]",
                    overrides=overrides,
                )

                bootstrap_seed = idx * 1000
                for boot_idx, boot_df in enumerate(
                    _bootstrap_monthly_variants(payload["monthly_df"], bootstrap_count, seed=bootstrap_seed, period_mode=period_mode)
                ):
                    new_results = _store_experiment_payload(
                        new_results,
                        source_ea_id=selected_ea_id,
                        ea_name=ea["name"],
                        symbol=ea["symbol"],
                        timeframe=ea["timeframe"],
                        payload={
                            "summary": payload["summary"],
                            "monthly_df": boot_df,
                            "deals_df": pd.DataFrame(),
                            "balance_curve_df": pd.DataFrame(),
                        },
                        dataset_source="bootstrap",
                        name=f"{ea['name']} [{variant['_variant_label']}] bootstrap {boot_idx+1}",
                        overrides=overrides,
                    )
                progress.progress(idx / len(variants), text=f"Completed {idx}/{len(variants)} local experiments")

            st.session_state["ai_experiment_results"] = new_results
            autosave()
            st.success(f"Generated {len(variants)} parameter-sweep runs and stored them for training.", icon="✅")
            st.rerun()

    with col_wfo:
        st.markdown("##### Walk-Forward Generation")
        wfo_train_months = st.slider("Train window (months)", min_value=6, max_value=36, value=18, step=3, key="ai_wfo_train")
        wfo_test_months = st.slider("Test window (months)", min_value=1, max_value=12, value=3, step=1, key="ai_wfo_test")
        wfo_metric = st.selectbox("Optimise by", ["sharpe_ratio", "net_profit", "profit_factor", "recovery_factor"], index=0, key="ai_wfo_metric")
        wfo_top_n = st.select_slider("Top sets averaged", options=[1, 2, 3], value=1, key="ai_wfo_topn")
        if st.button("Generate Walk-Forward OOS Data", use_container_width=True, key="ai_wfo_run"):
            if not effective_params:
                error_box("Select at least one parameter to vary for walk-forward generation.")
                return
            param_grid = _build_param_grid(
                base_params,
                effective_params,
                levels,
                width,
                lot_param_name=lot_param_name,
                aggressive_lot_range=aggressive_lot_range,
            )
            if not param_grid:
                error_box("Could not build a parameter grid from the selected parameters.")
                return
            strat_cls = _load_generated_strategy_class(ea)
            df = load_bars(ea["symbol"], ea["timeframe"], date_from=str(date_from), date_to=str(date_to))
            progress = st.progress(0, text="Running walk-forward windows...")
            wfo = run_wfo(
                strategy_class=strat_cls,
                df=df,
                param_grid=param_grid,
                train_months=wfo_train_months,
                test_months=wfo_test_months,
                optimize_by=wfo_metric,
                backtester_kwargs={
                    "initial_capital": 100_000,
                    "lot_value": getattr(strat_cls, "lot_value", 1.0),
                    "intrabar_steps": 60 if intrabar else 1,
                },
                n_top_params=wfo_top_n,
            )
            new_results = dict(experiments)
            total_windows = max(1, len(wfo.windows))
            for idx, window in enumerate(wfo.windows, start=1):
                payload = backtest_result_payload(window.test_results)
                new_results = _store_experiment_payload(
                    new_results,
                    source_ea_id=selected_ea_id,
                    ea_name=ea["name"],
                    symbol=ea["symbol"],
                    timeframe=ea["timeframe"],
                    payload=payload,
                    dataset_source="wfo_oos",
                    name=f"{ea['name']} WFO window {window.window_id}",
                    overrides=window.best_params,
                    extra_meta={
                        "window_id": window.window_id,
                        "train_from": str(window.train_from.date()),
                        "train_to": str(window.train_to.date()),
                        "test_from": str(window.test_from.date()),
                        "test_to": str(window.test_to.date()),
                        "optimize_by": wfo_metric,
                    },
                )
                bootstrap_seed = 50000 + idx * 100
                for boot_idx, boot_df in enumerate(
                    _bootstrap_monthly_variants(payload["monthly_df"], bootstrap_count, seed=bootstrap_seed, period_mode=period_mode)
                ):
                    new_results = _store_experiment_payload(
                        new_results,
                        source_ea_id=selected_ea_id,
                        ea_name=ea["name"],
                        symbol=ea["symbol"],
                        timeframe=ea["timeframe"],
                        payload={
                            "summary": payload["summary"],
                            "monthly_df": boot_df,
                            "deals_df": pd.DataFrame(),
                            "balance_curve_df": pd.DataFrame(),
                        },
                        dataset_source="wfo_bootstrap",
                        name=f"{ea['name']} WFO window {window.window_id} bootstrap {boot_idx+1}",
                        overrides=window.best_params,
                        extra_meta={"window_id": window.window_id},
                    )
                progress.progress(idx / total_windows, text=f"Stored WFO window {idx}/{total_windows}")

            st.session_state["ai_experiment_results"] = new_results
            autosave()
            st.success(f"Generated {len(wfo.windows)} walk-forward out-of-sample windows for training.", icon="✅")
            st.rerun()


def _make_training_examples(monthly_df: pd.DataFrame, lookback: int, objective: str, neutral_band: float, period_mode: str = "monthly") -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    clean = monthly_df.copy()
    metric_cols = ["total_profit", "sharpe", "win_rate", "num_trades", "timeframe_minutes", "lot_scale", "actual_lot", "base_lot", "lot_step"]
    for col in metric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean[metric_cols] = clean[metric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    baseline_only = clean[clean["source"] == "baseline"].copy()
    if baseline_only.empty:
        return pd.DataFrame(), []

    for ea_id, baseline_group in baseline_only.sort_values(["ea_id", "month_dt"]).groupby("ea_id", sort=False):
        baseline_group = baseline_group.reset_index(drop=True)
        if len(baseline_group) <= lookback:
            continue
        candidates_all = clean[clean["ea_id"] == ea_id].copy()

        for idx in range(lookback, len(baseline_group)):
            window = baseline_group.iloc[idx - lookback:idx]
            current = baseline_group.iloc[idx]
            month_candidates = candidates_all[candidates_all["month_dt"] == current["month_dt"]].copy()
            if month_candidates.empty:
                continue

            distinct_scales = sorted({round(float(v), 6) for v in month_candidates["lot_scale"].tolist() if v > 0})
            if len(distinct_scales) < 2:
                continue

            month_candidates["score"] = _candidate_objective_scores(month_candidates, window, objective, period_mode=period_mode)
            month_candidates = month_candidates.sort_values("score", ascending=False).reset_index(drop=True)
            best_n = min(3, len(month_candidates))
            top = month_candidates.head(best_n).copy()
            weights = np.linspace(best_n, 1, best_n)
            target_lot_scale = float(np.average(top["lot_scale"], weights=weights))
            source_weights = np.array([_source_weight_for(src) for src in top.get("source", pd.Series(dtype=object)).tolist()], dtype=float)
            if len(source_weights) != len(weights):
                source_weights = np.ones(len(weights), dtype=float)
            combined_weights = weights * source_weights
            example_weight = float(np.mean(combined_weights)) if len(combined_weights) else 1.0
            top_sources = ",".join(sorted({str(src) for src in top.get("source", pd.Series(dtype=object)).tolist() if src}))
            base_lot = float(current.get("base_lot", 1.0) or 1.0)
            lot_step = float(current.get("lot_step", 0.01) or 0.01)
            target_lot = _quantize_lot(base_lot * target_lot_scale, lot_step)
            target_lot_scale = target_lot / base_lot if abs(base_lot) > 1e-9 else 1.0
            target_regime = _assign_regime(target_lot_scale, neutral_band)
            window_stats = _window_metrics(window)

            row = {
                "ea_id": ea_id,
                "source_ea_id": current.get("source_ea_id", ea_id),
                "run_id": current.get("run_id", ea_id),
                "name": current["name"],
                "symbol": current["symbol"],
                "timeframe": current["timeframe"],
                "month": current["month"],
                "month_dt": current["month_dt"],
                "lot_param": current.get("lot_param", ""),
                "base_lot": base_lot,
                "lot_step": lot_step,
                "actual_profit": float(current["total_profit"]),
                "actual_sharpe": float(current["sharpe"]),
                "actual_win_rate": float(current["win_rate"]),
                "actual_num_trades": float(current["num_trades"]),
                "target_lot": float(target_lot),
                "target_lot_scale": float(target_lot_scale),
                "target_regime": target_regime,
                "example_weight": example_weight,
                "target_sources": top_sources,
                "candidate_count": int(len(month_candidates)),
                "best_candidate_score": float(top["score"].iloc[0]),
                "training_objective": objective,
                "timeframe_minutes": float(current["timeframe_minutes"]),
            }

            row.update(
                {
                    "profit_mean_lb": float(window["total_profit"].mean()),
                    "profit_std_lb": float(window["total_profit"].std(ddof=0) or 0.0),
                    "profit_last": float(window["total_profit"].iloc[-1]),
                    "profit_sum_lb": float(window["total_profit"].sum()),
                    "sharpe_mean_lb": float(window["sharpe"].mean()),
                    "sharpe_last": float(window["sharpe"].iloc[-1]),
                    "win_rate_mean_lb": float(window["win_rate"].mean()),
                    "win_rate_last": float(window["win_rate"].iloc[-1]),
                    "trades_mean_lb": float(window["num_trades"].mean()),
                    "trades_last": float(window["num_trades"].iloc[-1]),
                    "positive_month_ratio_lb": float((window["total_profit"] > 0).mean()),
                    "avg_abs_profit_lb": float(np.abs(window["total_profit"]).mean()),
                    "drawdown_proxy_lb": window_stats["drawdown_proxy_lb"],
                    "profit_factor_lb": window_stats["profit_factor_lb"],
                    "recovery_factor_lb": window_stats["recovery_factor_lb"],
                    "return_vol_lb": window_stats["return_vol_lb"],
                    "negative_month_ratio_lb": window_stats["negative_month_ratio_lb"],
                    "avg_loss_lb": window_stats["avg_loss_lb"],
                    "worst_month_lb": window_stats["worst_month_lb"],
                    "downside_vol_lb": window_stats["downside_vol_lb"],
                    "win_streak_lb": window_stats["win_streak_lb"],
                    "loss_streak_lb": window_stats["loss_streak_lb"],
                    "atr_pctile_6m": float(current.get("atr_pctile_6m", 0.0)),
                    "realized_vol_3m": float(current.get("realized_vol_3m", 0.0)),
                    "trend_slope_3m": float(current.get("trend_slope_3m", 0.0)),
                    "range_position_3m": float(current.get("range_position_3m", 0.5)),
                    "breakout_score_3m": float(current.get("breakout_score_3m", 0.0)),
                    "timeframe_minutes": float(current["timeframe_minutes"]),
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame(), []

    examples = pd.DataFrame(rows).sort_values(["month_dt", "run_id"]).reset_index(drop=True)
    feature_cols = [
        "profit_mean_lb",
        "profit_std_lb",
        "profit_last",
        "profit_sum_lb",
        "sharpe_mean_lb",
        "sharpe_last",
        "win_rate_mean_lb",
        "win_rate_last",
        "trades_mean_lb",
        "trades_last",
        "positive_month_ratio_lb",
        "avg_abs_profit_lb",
        "drawdown_proxy_lb",
        "profit_factor_lb",
        "recovery_factor_lb",
        "return_vol_lb",
        "negative_month_ratio_lb",
        "avg_loss_lb",
        "worst_month_lb",
        "downside_vol_lb",
        "win_streak_lb",
        "loss_streak_lb",
        "atr_pctile_6m",
        "realized_vol_3m",
        "trend_slope_3m",
        "range_position_3m",
        "breakout_score_3m",
        "timeframe_minutes",
    ]
    examples[feature_cols + ["target_lot_scale", "target_lot", "example_weight"]] = (
        examples[feature_cols + ["target_lot_scale", "target_lot", "example_weight"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return examples, feature_cols


def _fit_ridge_regression(examples: pd.DataFrame, feature_cols: list[str], alpha: float, train_ratio: float) -> dict:
    if examples.empty:
        raise ValueError("No training examples available.")

    clean = examples.copy()
    clean[feature_cols + ["target_lot_scale", "example_weight"]] = (
        clean[feature_cols + ["target_lot_scale", "example_weight"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    clean["target_regime"] = clean["target_regime"].fillna("normal")
    examples = clean

    cutoff = max(1, int(len(examples) * train_ratio))
    if cutoff >= len(examples):
        cutoff = len(examples) - 1
    if cutoff < 1:
        raise ValueError("Not enough samples for a train/test split.")

    train = examples.iloc[:cutoff].copy()
    test = examples.iloc[cutoff:].copy()

    X_train_raw = train[feature_cols].to_numpy(dtype=float)
    means = X_train_raw.mean(axis=0)
    stds = X_train_raw.std(axis=0)
    stds[stds < 1e-9] = 1.0

    X_train = (X_train_raw - means) / stds
    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    reg = np.eye(X_train_aug.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0

    class_order = ["risk_off", "normal", "risk_on"]
    y_train_labels = train["target_regime"].astype(str)
    one_hot = np.column_stack([(y_train_labels == label).astype(float).to_numpy() for label in class_order])
    sample_weights = train["example_weight"].astype(float).to_numpy()
    sample_weights = np.clip(sample_weights, 0.25, 5.0)
    W = sample_weights[:, None]
    weights = np.linalg.solve(X_train_aug.T @ (X_train_aug * W) + reg, X_train_aug.T @ (one_hot * W))

    class_scales: dict[str, float] = {}
    for label in class_order:
        mask = train["target_regime"] == label
        if mask.any():
            class_scales[label] = float(np.median(train.loc[mask, "target_lot_scale"].astype(float)))
        else:
            class_scales[label] = 1.0
    class_scales["normal"] = 1.0

    def predict(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = frame[feature_cols].to_numpy(dtype=float)
        scaled = (raw - means) / stds
        aug = np.column_stack([np.ones(len(scaled)), scaled])
        logits = aug @ weights
        probs = _softmax(logits)
        label_idx = np.argmax(probs, axis=1)
        conf = probs[np.arange(len(probs)), label_idx]
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else conf
        scales = np.array([class_scales[class_order[idx]] for idx in label_idx], dtype=float)
        return label_idx, conf, scales, margin

    train_idx, train_conf, train_scale, train_margin = predict(train)
    test_idx, test_conf, test_scale, test_margin = predict(test)
    all_idx, all_conf, all_scale, all_margin = predict(examples)

    train = train.assign(
        pred_regime=[class_order[idx] for idx in train_idx],
        pred_confidence=train_conf,
        pred_lot_scale=train_scale,
        pred_margin=train_margin,
    )
    test = test.assign(
        pred_regime=[class_order[idx] for idx in test_idx],
        pred_confidence=test_conf,
        pred_lot_scale=test_scale,
        pred_margin=test_margin,
    )
    all_examples = examples.assign(
        pred_regime=[class_order[idx] for idx in all_idx],
        pred_confidence=all_conf,
        pred_lot_scale=all_scale,
        pred_margin=all_margin,
    )

    return {
        "weights": weights.tolist(),
        "feature_cols": feature_cols,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "class_order": class_order,
        "class_scales": class_scales,
        "sample_weight_stats": {
            "min": float(sample_weights.min()) if len(sample_weights) else 0.0,
            "mean": float(sample_weights.mean()) if len(sample_weights) else 0.0,
            "max": float(sample_weights.max()) if len(sample_weights) else 0.0,
        },
        "train_df": train,
        "test_df": test,
        "all_df": all_examples,
    }


def _apply_lot_schedule(
    frame: pd.DataFrame,
    scale_min: float,
    scale_max: float,
    confidence_threshold: float = 0.55,
    margin_threshold: float = 0.08,
    risk_on_confidence_threshold: float | None = None,
    risk_on_margin_threshold: float | None = None,
    risk_on_drawdown_limit: float | None = None,
    risk_on_negative_ratio_limit: float | None = None,
    risk_on_downside_vol_limit: float | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["pred_lot_scale"] = pd.to_numeric(out["pred_lot_scale"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    if "pred_confidence" not in out.columns:
        out["pred_confidence"] = 1.0
    out["pred_confidence"] = pd.to_numeric(out["pred_confidence"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    if "pred_margin" not in out.columns:
        out["pred_margin"] = 1.0
    out["pred_margin"] = pd.to_numeric(out["pred_margin"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for col in ("drawdown_proxy_lb", "negative_month_ratio_lb", "downside_vol_lb", "avg_abs_profit_lb"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["actual_profit"] = pd.to_numeric(out["actual_profit"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "base_lot" not in out.columns:
        out["base_lot"] = 1.0
    if "lot_step" not in out.columns:
        out["lot_step"] = 0.01
    out["base_lot"] = pd.to_numeric(out["base_lot"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["lot_step"] = pd.to_numeric(out["lot_step"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.01)
    scale_min = float(scale_min)
    scale_max = float(scale_max)
    threshold = float(confidence_threshold)
    margin_threshold = float(margin_threshold)
    risk_on_conf_threshold = float(risk_on_confidence_threshold if risk_on_confidence_threshold is not None else threshold)
    risk_on_margin_threshold = float(risk_on_margin_threshold if risk_on_margin_threshold is not None else margin_threshold)
    drawdown_limit = float(risk_on_drawdown_limit) if risk_on_drawdown_limit is not None else float("inf")
    negative_ratio_limit = float(risk_on_negative_ratio_limit) if risk_on_negative_ratio_limit is not None else float("inf")
    downside_vol_limit = float(risk_on_downside_vol_limit) if risk_on_downside_vol_limit is not None else float("inf")
    if "pred_regime" not in out.columns:
        out["pred_regime"] = np.where(out["pred_lot_scale"] < 1.0, "risk_off", np.where(out["pred_lot_scale"] > 1.0, "risk_on", "normal"))

    normal_mask = out["pred_regime"] == "normal"
    risk_on_mask = out["pred_regime"] == "risk_on"
    risk_off_mask = out["pred_regime"] == "risk_off"
    gate_mask = pd.Series(False, index=out.index)
    gate_mask.loc[normal_mask] = True
    gate_mask.loc[risk_off_mask] = (
        (out.loc[risk_off_mask, "pred_confidence"] >= threshold)
        & (out.loc[risk_off_mask, "pred_margin"] >= margin_threshold)
    )
    gate_mask.loc[risk_on_mask] = (
        (out.loc[risk_on_mask, "pred_confidence"] >= risk_on_conf_threshold)
        & (out.loc[risk_on_mask, "pred_margin"] >= risk_on_margin_threshold)
        & (out.loc[risk_on_mask, "drawdown_proxy_lb"] <= drawdown_limit)
        & (out.loc[risk_on_mask, "negative_month_ratio_lb"] <= negative_ratio_limit)
        & (out.loc[risk_on_mask, "downside_vol_lb"] <= downside_vol_limit)
    )
    out["lot_scale"] = np.where(gate_mask, out["pred_lot_scale"], 1.0)
    out["lot_scale"] = out["lot_scale"].clip(lower=scale_min, upper=scale_max)
    out["scheduled_lot"] = [
        _quantize_lot(base * scale, step)
        for base, scale, step in zip(out["base_lot"], out["lot_scale"], out["lot_step"])
    ]
    out["lot_scale"] = np.where(out["base_lot"].abs() > 1e-9, out["scheduled_lot"] / out["base_lot"], 1.0)
    out["applied_regime"] = np.where(gate_mask, out["pred_regime"], "normal")
    out["ai_profit"] = out["actual_profit"] * out["lot_scale"]
    return out


def _normalize_schedule_frame(schedule: pd.DataFrame) -> pd.DataFrame:
    out = schedule.copy()
    if out.empty:
        return out

    if "base_lot" not in out.columns:
        out["base_lot"] = 1.0
    if "lot_scale" not in out.columns:
        out["lot_scale"] = 1.0
    if "scheduled_lot" not in out.columns:
        base_lot = pd.to_numeric(out["base_lot"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
        lot_scale = pd.to_numeric(out["lot_scale"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["scheduled_lot"] = base_lot * lot_scale

    out["base_lot"] = pd.to_numeric(out["base_lot"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["lot_scale"] = pd.to_numeric(out["lot_scale"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["scheduled_lot"] = pd.to_numeric(out["scheduled_lot"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(out["base_lot"] * out["lot_scale"])
    return out


def _coerce_payload_df(value, expected_cols: list[str] | None = None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict) and "data" in value:
        try:
            return pd.DataFrame(value["data"], index=value.get("index"))
        except Exception:
            pass
    if expected_cols:
        return pd.DataFrame(columns=expected_cols)
    return pd.DataFrame()


def _normalize_aggregate_payload(payload: dict | None) -> dict:
    payload = dict(payload or {})
    payload["monthly_df"] = _coerce_payload_df(payload.get("monthly_df"), ["total_profit", "num_trades", "win_rate", "sharpe"])
    payload["deals_df"] = _coerce_payload_df(payload.get("deals_df"), ["profit", "direction", "entry_price", "exit_price", "comment"])
    payload["balance_curve_df"] = _coerce_payload_df(payload.get("balance_curve_df"), ["balance"])
    payload["summary"] = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    return payload


def _actual_compare_snapshot(actual_compare: dict | None, eas: dict) -> dict | None:
    if not actual_compare:
        return None

    baseline_payload = actual_compare.get("baseline_aggregate", {})
    ai_payload = actual_compare.get("ai_aggregate", {})
    base_summary = baseline_payload.get("summary", {})
    ai_summary = ai_payload.get("summary", {})

    monthly_rows: list[dict] = []
    baseline_monthly = baseline_payload.get("monthly_df", pd.DataFrame())
    ai_monthly = ai_payload.get("monthly_df", pd.DataFrame())
    if baseline_monthly is not None and ai_monthly is not None:
        base_months = set(baseline_monthly.index.astype(str)) if hasattr(baseline_monthly, "empty") and not baseline_monthly.empty else set()
        ai_months = set(ai_monthly.index.astype(str)) if hasattr(ai_monthly, "empty") and not ai_monthly.empty else set()
        months = sorted(base_months.union(ai_months))
        if months:
            frame = pd.DataFrame(index=months)
            frame.index.name = "month"
            if baseline_monthly is not None and not baseline_monthly.empty:
                frame["baseline_profit"] = baseline_monthly["total_profit"].astype(float)
            if ai_monthly is not None and not ai_monthly.empty:
                frame["ai_profit"] = ai_monthly["total_profit"].astype(float)
            frame = frame.fillna(0.0).reset_index()
            monthly_rows = frame.to_dict(orient="records")

    per_strategy_rows: list[dict] = []
    for ea_id, base_payload in actual_compare.get("baseline_by_ea", {}).items():
        ai_ea_payload = actual_compare.get("ai_by_ea", {}).get(ea_id, {})
        ea = eas.get(ea_id, {})
        base_ea_summary = base_payload.get("summary", {})
        ai_ea_summary = ai_ea_payload.get("summary", {})
        per_strategy_rows.append(
            {
                "name": ea.get("name", ea_id),
                "symbol": ea.get("symbol", ""),
                "timeframe": ea.get("timeframe", ""),
                "baseline_profit": float(base_ea_summary.get("total_profit", 0.0) or 0.0),
                "ai_profit": float(ai_ea_summary.get("total_profit", 0.0) or 0.0),
                "profit_delta": float((ai_ea_summary.get("total_profit", 0.0) or 0.0) - (base_ea_summary.get("total_profit", 0.0) or 0.0)),
                "baseline_sharpe": float(base_ea_summary.get("sharpe_mean", 0.0) or 0.0),
                "ai_sharpe": float(ai_ea_summary.get("sharpe_mean", 0.0) or 0.0),
                "baseline_dd_pct": float(base_ea_summary.get("max_drawdown_pct", 0.0) or 0.0),
                "ai_dd_pct": float(ai_ea_summary.get("max_drawdown_pct", 0.0) or 0.0),
            }
        )

    return {
        "baseline_summary": base_summary,
        "ai_summary": ai_summary,
        "monthly_rows": monthly_rows,
        "per_strategy_rows": per_strategy_rows,
    }


def _build_report_snapshot(artifact: dict, training_log: list[str], eas: dict) -> dict:
    report_id = artifact.get("report_id") or uuid.uuid4().hex[:12]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    schedule_rows = artifact.get("schedule_rows", [])
    compare_snapshot = _actual_compare_snapshot(artifact.get("actual_compare"), eas)
    baseline_test = artifact.get("baseline_test_metrics", artifact.get("baseline_metrics", {}))
    ai_test = artifact.get("ai_test_metrics", artifact.get("ai_metrics", {}))
    return {
        "id": report_id,
        "saved_at": timestamp,
        "scope_label": artifact.get("scope_label", "Unknown"),
        "period_mode": artifact.get("period_mode", "monthly"),
        "period_label": artifact.get("period_label", "Month"),
        "period_adjective": artifact.get("period_adjective", "monthly"),
        "objective_label": artifact.get("objective_label", artifact.get("objective", "")),
        "confidence_threshold": artifact.get("confidence_threshold"),
        "margin_threshold": artifact.get("margin_threshold"),
        "risk_on_confidence_threshold": artifact.get("risk_on_confidence_threshold"),
        "risk_on_margin_threshold": artifact.get("risk_on_margin_threshold"),
        "risk_on_drawdown_limit_mult": artifact.get("risk_on_drawdown_limit_mult"),
        "risk_on_negative_ratio_limit": artifact.get("risk_on_negative_ratio_limit"),
        "risk_on_downside_vol_limit_mult": artifact.get("risk_on_downside_vol_limit_mult"),
        "neutral_band": artifact.get("neutral_band"),
        "period_rows_total": int(artifact.get("period_rows_total", artifact.get("weekly_rows_total", 0)) or 0),
        "period_rows_active": int(artifact.get("period_rows_active", artifact.get("weekly_rows_active", 0)) or 0),
        "min_trades_per_period": artifact.get("min_trades_per_period", artifact.get("min_trades_per_week")),
        "min_abs_profit_per_period": artifact.get("min_abs_profit_per_period", artifact.get("min_abs_profit_per_week")),
        "collapse_threshold": artifact.get("collapse_threshold"),
        "auto_reject_collapsed": artifact.get("auto_reject_collapsed"),
        "train_rows": int(artifact.get("train_rows", 0) or 0),
        "test_rows": int(artifact.get("test_rows", 0) or 0),
        "baseline_test_metrics": baseline_test,
        "ai_test_metrics": ai_test,
        "target_regime_mix": artifact.get("target_regime_mix", {}),
        "predicted_regime_mix": artifact.get("predicted_regime_mix", {}),
        "applied_regime_mix": artifact.get("applied_regime_mix", {}),
        "collapse_warning": artifact.get("collapse_warning", ""),
        "schedule_rows": schedule_rows,
        "schedule_path": artifact.get("schedule_path", ""),
        "training_log": list(training_log),
        "actual_compare_snapshot": compare_snapshot,
    }


def _upsert_saved_report(report: dict):
    reports = list(st.session_state.get("ai_saved_reports", []))
    reports = [existing for existing in reports if existing.get("id") != report.get("id")]
    reports.insert(0, report)
    st.session_state["ai_saved_reports"] = reports[:50]


def _regime_distribution(series: pd.Series | list, labels: list[str] | None = None) -> dict[str, float]:
    ordered = labels or ["risk_off", "normal", "risk_on"]
    values = pd.Series(series, dtype="object").fillna("unknown")
    total = len(values)
    if total == 0:
        return {label: 0.0 for label in ordered}
    counts = values.value_counts(dropna=False)
    out: dict[str, float] = {}
    for label in ordered:
        out[label] = float(counts.get(label, 0)) / total
    for label, count in counts.items():
        if label not in out:
            out[str(label)] = float(count) / total
    return out


def _dominant_regime(distribution: dict[str, float]) -> tuple[str, float]:
    if not distribution:
        return "unknown", 0.0
    label = max(distribution, key=distribution.get)
    return str(label), float(distribution[label])


def _render_saved_reports():
    reports = list(st.session_state.get("ai_saved_reports", []))
    st.markdown("#### Saved AI Reports")
    if not reports:
        st.caption("No saved AI reports yet. Train a policy or run an end-to-end comparison to create one.")
        return

    labels = [
        f"{report.get('saved_at', '')} | {report.get('scope_label', 'Unknown')} | {report.get('objective_label', '')}"
        for report in reports
    ]
    selected_idx = st.selectbox(
        "Saved report",
        options=list(range(len(reports))),
        format_func=lambda idx: labels[idx],
        key="ai_saved_report_select",
    )
    report = reports[selected_idx]

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Load Saved Report", use_container_width=True, key="ai_load_saved_report"):
            st.session_state["training_artifact"] = {
                "report_id": report.get("id"),
                "scope_label": report.get("scope_label", "Saved Report"),
                "period_mode": report.get("period_mode", "monthly"),
                "period_label": report.get("period_label", "Month"),
                "period_adjective": report.get("period_adjective", "monthly"),
                "objective_label": report.get("objective_label"),
                "confidence_threshold": report.get("confidence_threshold"),
                "neutral_band": report.get("neutral_band"),
                "period_rows_total": report.get("period_rows_total", report.get("weekly_rows_total", 0)),
                "period_rows_active": report.get("period_rows_active", report.get("weekly_rows_active", 0)),
                "min_trades_per_period": report.get("min_trades_per_period", report.get("min_trades_per_week")),
                "min_abs_profit_per_period": report.get("min_abs_profit_per_period", report.get("min_abs_profit_per_week")),
                "train_rows": report.get("train_rows", 0),
                "test_rows": report.get("test_rows", 0),
                "baseline_test_metrics": report.get("baseline_test_metrics", {}),
                "ai_test_metrics": report.get("ai_test_metrics", {}),
                "baseline_metrics": report.get("baseline_test_metrics", {}),
                "ai_metrics": report.get("ai_test_metrics", {}),
                "schedule_rows": report.get("schedule_rows", []),
                "schedule_path": report.get("schedule_path", ""),
            }
            st.session_state["training_log"] = report.get("training_log", [])
            st.session_state["model_trained"] = True
            autosave()
            st.rerun()
    with col_b:
        if st.button("Delete Saved Report", use_container_width=True, key="ai_delete_saved_report"):
            st.session_state["ai_saved_reports"] = [r for r in reports if r.get("id") != report.get("id")]
            autosave()
            st.rerun()

    base = report.get("baseline_test_metrics", {})
    ai = report.get("ai_test_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved Test Profit", f"${float(base.get('total_profit', 0.0) or 0.0):,.0f}")
    c2.metric("Saved AI Profit", f"${float(ai.get('total_profit', 0.0) or 0.0):,.0f}")
    c3.metric("Saved Test Sharpe", f"{float(base.get('sharpe', 0.0) or 0.0):.2f}")
    c4.metric("Saved AI Sharpe", f"{float(ai.get('sharpe', 0.0) or 0.0):.2f}")
    period_name = _period_display_name(report.get("period_mode", "monthly"))
    if report.get("period_rows_active") is not None or report.get("weekly_rows_active") is not None:
        st.caption(
            f"Active {period_name.lower()} rows: "
            f"{int(report.get('period_rows_active', report.get('weekly_rows_active', 0)) or 0)}/"
            f"{int(report.get('period_rows_total', report.get('weekly_rows_total', 0)) or 0)} | "
            f"filter trades>={int(report.get('min_trades_per_period', report.get('min_trades_per_week', 1)) or 1)}, "
            f"abs PnL>={float(report.get('min_abs_profit_per_period', report.get('min_abs_profit_per_week', 0.0)) or 0.0):.2f}"
        )

    collapse_warning = report.get("collapse_warning", "")
    if collapse_warning:
        st.warning(collapse_warning, icon="⚠️")

    mixes = []
    for name, key in [
        ("Target", "target_regime_mix"),
        ("Predicted", "predicted_regime_mix"),
        ("Applied", "applied_regime_mix"),
    ]:
        mix = report.get(key) or {}
        if mix:
            mixes.append(
                {
                    "mix": name,
                    "risk_off": round(float(mix.get("risk_off", 0.0)) * 100.0, 1),
                    "normal": round(float(mix.get("normal", 0.0)) * 100.0, 1),
                    "risk_on": round(float(mix.get("risk_on", 0.0)) * 100.0, 1),
                }
            )
    if mixes:
        st.markdown("##### Saved Regime Balance")
        st.dataframe(pd.DataFrame(mixes), use_container_width=True, hide_index=True)

    compare = report.get("actual_compare_snapshot")
    if compare:
        st.markdown("##### Saved End-to-End Summary")
        base_sum = compare.get("baseline_summary", {})
        ai_sum = compare.get("ai_summary", {})
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Baseline Profit", f"${float(base_sum.get('total_profit', 0.0) or 0.0):,.0f}")
        d2.metric("AI Profit", f"${float(ai_sum.get('total_profit', 0.0) or 0.0):,.0f}")
        d3.metric("Baseline DD", f"{float(base_sum.get('max_drawdown_pct', 0.0) or 0.0):.2f}%")
        d4.metric("AI DD", f"{float(ai_sum.get('max_drawdown_pct', 0.0) or 0.0):.2f}%")

        monthly_rows = compare.get("monthly_rows", [])
        if monthly_rows:
            st.dataframe(pd.DataFrame(monthly_rows), use_container_width=True, hide_index=True)


def _equity_metrics(profits: pd.Series, period_mode: str = "monthly") -> dict:
    values = profits.astype(float).to_numpy()
    if len(values) == 0:
        return {
            "total_profit": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "max_drawdown_pct": 0.0,
        }

    equity = 100_000.0 + np.cumsum(values)
    peak = np.maximum.accumulate(equity)
    dd = np.where(peak > 0, (peak - equity) / peak * 100.0, 0.0)
    pos = values[values > 0].sum()
    neg = values[values < 0].sum()
    std = values.std(ddof=0)
    periods_per_year = float(_period_config(period_mode)["per_year"])
    sharpe = 0.0 if std < 1e-9 else float(values.mean() / std * np.sqrt(periods_per_year))
    return {
        "total_profit": float(values.sum()),
        "sharpe": sharpe,
        "profit_factor": float(pos / abs(neg)) if neg < 0 else float("inf"),
        "win_rate": float((values > 0).mean() * 100.0),
        "max_drawdown_pct": float(dd.max()),
    }


def _lot_param_names(params: dict) -> list[str]:
    candidates = ["_lots", "mmLots", "lots", "Lots", "lot", "LotSize", "fixed_lots"]
    return [name for name in candidates if name in params and isinstance(params[name], (int, float))]


def _build_schedule_map(schedule_rows: list[dict], scope: str, eas: dict) -> dict[str, dict[str, float]]:
    schedule_map: dict[str, dict[str, float]] = {}
    single_scope_ea_id = scope if scope != "__all__" and scope in eas else None

    for row in schedule_rows:
        month = str(row.get("month", ""))
        if not month:
            continue
        lot_value = float(row.get("scheduled_lot", 0.0) or 0.0)
        if lot_value <= 0:
            base_lot = float(row.get("base_lot", 1.0) or 1.0)
            scale = float(row.get("lot_scale", 1.0) or 1.0)
            lot_value = base_lot * scale
        ea_id = row.get("ea_id")

        if not ea_id and single_scope_ea_id:
            ea_id = single_scope_ea_id

        if not ea_id:
            name = row.get("name")
            symbol = row.get("symbol")
            timeframe = row.get("timeframe")
            for candidate_id, ea in eas.items():
                if ea["name"] == name and ea["symbol"] == symbol and ea["timeframe"] == timeframe:
                    ea_id = candidate_id
                    break

        if not ea_id:
            continue

        schedule_map.setdefault(ea_id, {})[month] = lot_value

    return schedule_map


def _make_ai_wrapped_strategy(strat_cls: type, month_schedule: dict[str, float], base_params: dict, period_mode: str = "monthly"):
    lot_names = _lot_param_names(getattr(strat_cls, "params", {}))
    if "_lots" not in lot_names and "_lots" in base_params:
        lot_names.append("_lots")

    class AIScheduledStrategy(strat_cls):
        params = dict(getattr(strat_cls, "params", {}))

        def on_bar(self, ctx):
            ym = _period_key(ctx.time, period_mode=period_mode)
            lot_value = float(month_schedule.get(ym, _resolve_lot_value(base_params, 1.0)))
            for name in lot_names:
                if isinstance(self.params.get(name), int) and not isinstance(self.params.get(name), bool):
                    self.params[name] = max(1, int(round(lot_value)))
                else:
                    self.params[name] = round(lot_value, 6)
            super().on_bar(ctx)

    AIScheduledStrategy.__name__ = f"{strat_cls.__name__}AIScheduled"
    return AIScheduledStrategy


def _merge_balance_curves(payloads: list[dict], initial_capital: float = 100_000.0) -> pd.DataFrame:
    curves: list[pd.Series] = []
    for payload in payloads:
        curve = payload.get("balance_curve_df")
        if curve is None or curve.empty or "balance" not in curve.columns:
            continue
        series = curve["balance"].astype(float)
        pnl = series - initial_capital
        curves.append(pnl.rename(None))

    if not curves:
        return pd.DataFrame({"balance": [initial_capital]})

    combined = pd.concat(curves, axis=1).sort_index().ffill().fillna(0.0)
    combined_pnl = combined.sum(axis=1)
    merged = (initial_capital + combined_pnl).rename("balance").to_frame()
    merged.index.name = "time"
    return merged


def _aggregate_payloads(payloads: list[dict], initial_capital: float = 100_000.0, period_mode: str = "monthly") -> dict:
    monthly_frames: list[pd.DataFrame] = []
    deal_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for payload in payloads:
        monthly = payload.get("monthly_df")
        deals = payload.get("deals_df")
        summary = payload.get("summary", {})

        if monthly is not None and not monthly.empty:
            frame = monthly.copy()
            if "total_profit" not in frame.columns and "profit" in frame.columns:
                frame["total_profit"] = frame["profit"]
            if "num_trades" not in frame.columns and "trades" in frame.columns:
                frame["num_trades"] = frame["trades"]
            monthly_frames.append(frame)
        if deals is not None and not deals.empty:
            deal_frames.append(deals.copy())
        summaries.append(summary)

    if monthly_frames:
        monthly = pd.concat(monthly_frames, axis=0)
        monthly = monthly.groupby(monthly.index.astype(str), as_index=True).agg(
            total_profit=("total_profit", "sum"),
            num_trades=("num_trades", "sum"),
            win_rate=("win_rate", "mean"),
            sharpe=("sharpe", "mean"),
        ).sort_index()
    else:
        monthly = pd.DataFrame(columns=["total_profit", "num_trades", "win_rate", "sharpe"])

    if deal_frames:
        deals = pd.concat(deal_frames, axis=0).sort_index()
    else:
        deals = pd.DataFrame(columns=["profit", "direction", "entry_price", "exit_price", "comment"])

    profits = deals["profit"].astype(float) if not deals.empty and "profit" in deals.columns else pd.Series(dtype=float)
    metrics = _equity_metrics(monthly["total_profit"] if not monthly.empty else profits, period_mode=period_mode)
    balance_curve = _merge_balance_curves(payloads, initial_capital=initial_capital)
    if not balance_curve.empty:
        peak = balance_curve["balance"].cummax()
        dd_pct = ((peak - balance_curve["balance"]) / peak.replace(0, np.nan) * 100.0).fillna(0.0)
        dd_abs = (peak - balance_curve["balance"]).fillna(0.0)
        max_dd_pct = float(dd_pct.max()) if not dd_pct.empty else 0.0
        max_dd_abs = float(dd_abs.max()) if not dd_abs.empty else 0.0
    else:
        max_dd_pct = 0.0
        max_dd_abs = 0.0

    gross_profit = float(profits[profits > 0].sum()) if not profits.empty else 0.0
    gross_loss = float(profits[profits < 0].sum()) if not profits.empty else 0.0
    num_trades = int(sum(int(s.get("num_trades", 0) or 0) for s in summaries))
    start_dates = [pd.Timestamp(s["date_from"]) for s in summaries if s.get("date_from")]
    end_dates = [pd.Timestamp(s["date_to"]) for s in summaries if s.get("date_to")]

    summary = {
        "date_from": str(min(start_dates).date()) if start_dates else "",
        "date_to": str(max(end_dates).date()) if end_dates else "",
        "total_profit": float(profits.sum()) if not profits.empty else metrics["total_profit"],
        "sharpe_mean": metrics["sharpe"],
        "win_rate": float((profits > 0).mean()) if not profits.empty else metrics["win_rate"] / 100.0,
        "num_months": int(len(monthly)),
        "num_trades": num_trades,
        "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf"),
        "max_drawdown_pct": max_dd_pct,
        "balance_dd_abs": max_dd_abs,
        "balance_dd_pct": max_dd_pct,
        "equity_dd_abs": max_dd_abs,
        "equity_dd_pct": max_dd_pct,
    }
    return {
        "monthly_df": monthly,
        "deals_df": deals,
        "balance_curve_df": balance_curve,
        "summary": summary,
    }


def _run_end_to_end_compare(artifact: dict, eas: dict, results: dict, progress_bar=None) -> dict:
    scope = artifact["scope"]
    period_mode = str(artifact.get("period_mode", "monthly"))
    selected_ids = list(eas.keys()) if scope == "__all__" else [scope]
    schedule_map = _build_schedule_map(artifact.get("schedule_rows", []), scope, eas)
    baseline_payloads: dict[str, dict] = {}
    ai_payloads: dict[str, dict] = {}
    total = max(1, len(selected_ids))

    for idx, ea_id in enumerate(selected_ids, start=1):
        ea = eas.get(ea_id)
        baseline = results.get(ea_id)
        if not ea or not baseline:
            continue

        summary = baseline.get("summary", {})
        date_from = summary.get("date_from")
        date_to = summary.get("date_to")
        if not date_from or not date_to:
            continue

        base_params = dict(ea.get("params", {}))
        intrabar_steps = 60 if str(ea.get("timeframe", "")).upper() != "M1" else 1

        baseline_payloads[ea_id] = baseline

        if progress_bar is not None:
            progress_bar.progress((idx - 1) / total, text=f"Running AI rerun {idx}/{total}: {ea['name']} {ea['symbol']} {ea['timeframe']}")

        strat_cls = _load_generated_strategy_class(ea)
        ai_cls = _make_ai_wrapped_strategy(strat_cls, schedule_map.get(ea_id, {}), base_params, period_mode=period_mode)
        ai_result = run_backtest(
            ai_cls,
            symbol=ea["symbol"],
            timeframe=ea["timeframe"],
            date_from=str(date_from),
            date_to=str(date_to),
            overrides=base_params,
            intrabar_steps=intrabar_steps,
        )
        ai_payloads[ea_id] = backtest_result_payload(ai_result)

    if progress_bar is not None:
        progress_bar.progress(1.0, text="AI rerun completed.")

    return {
        "scope": scope,
        "baseline_by_ea": baseline_payloads,
        "ai_by_ea": ai_payloads,
        "baseline_aggregate": _aggregate_payloads(list(baseline_payloads.values()), period_mode=period_mode),
        "ai_aggregate": _aggregate_payloads(list(ai_payloads.values()), period_mode=period_mode),
    }


def _render_train_tab(eas: dict, results: dict, experiments: dict, period_mode: str):
    cfg = _period_config(period_mode)
    period_name = _period_display_name(period_mode)
    dataset = _build_monthly_dataset(eas, results, experiments, period_mode=period_mode)
    if dataset.empty:
        empty_state(f"Run at least one backtest first. The AI tab trains from {cfg['adjective']} trade aggregates.", "🧠")
        return

    ea_options = {
        ea_id: f"{eas[ea_id]['name']} — {eas[ea_id]['symbol']} {eas[ea_id]['timeframe']}"
        for ea_id in results
        if ea_id in eas and results.get(ea_id, {}).get("monthly_df") is not None and not results[ea_id]["monthly_df"].empty
    }
    if not ea_options:
        empty_state("No backtests with trade results are available yet.", "🧠")
        return

    option_keys = ["__all__"] + list(ea_options.keys())
    selected = st.selectbox(
        "Training Scope",
        option_keys,
        format_func=lambda x: "All strategies combined" if x == "__all__" else ea_options[x],
    )

    lookback = st.slider(
        f"Lookback {cfg['label'].lower()}s",
        min_value=int(cfg["lookback_min"]),
        max_value=int(cfg["lookback_max"]),
        value=int(cfg["lookback_default"]),
        step=int(cfg["lookback_step"]),
    )
    train_ratio = st.slider("Train split", min_value=0.5, max_value=0.9, value=0.75, step=0.05)
    alpha = st.select_slider("Regularization", options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
    objective = st.selectbox(
        "Training objective",
        OBJECTIVE_OPTIONS,
        index=0,
        format_func=_objective_label,
    )
    neutral_band = st.slider("Neutral band around 1.0x", min_value=0.02, max_value=0.20, value=0.08, step=0.01)
    if period_mode == "weekly":
        active_col_a, active_col_b = st.columns(2)
        with active_col_a:
            min_trades_per_period = st.slider("Minimum trades per week", min_value=1, max_value=5, value=1, step=1)
        with active_col_b:
            min_abs_profit_per_period = st.number_input("Minimum abs weekly PnL", min_value=0.0, value=0.0, step=25.0, help="Ignore near-flat weeks when building weekly AI labels.")
    else:
        min_trades_per_period = 1
        min_abs_profit_per_period = 0.0
    confidence_threshold = st.slider("Confidence threshold", min_value=0.34, max_value=0.90, value=0.55, step=0.01)
    margin_threshold = st.slider("Class margin threshold", min_value=0.00, max_value=0.40, value=0.08, step=0.01)
    risk_on_confidence_threshold = st.slider("Risk-on confidence threshold", min_value=0.34, max_value=0.95, value=max(0.60, float(confidence_threshold)), step=0.01)
    risk_on_margin_threshold = st.slider("Risk-on margin threshold", min_value=0.00, max_value=0.50, value=max(0.15, float(margin_threshold)), step=0.01)
    risk_on_drawdown_limit_mult = st.slider("Risk-on drawdown guard", min_value=0.25, max_value=3.00, value=1.00, step=0.05)
    risk_on_negative_ratio_limit = st.slider("Risk-on negative-month ratio guard", min_value=0.10, max_value=0.80, value=0.45, step=0.05)
    risk_on_downside_vol_limit_mult = st.slider("Risk-on downside-vol guard", min_value=0.25, max_value=3.00, value=1.00, step=0.05)
    collapse_threshold = st.slider("Collapse warning threshold", min_value=0.60, max_value=0.95, value=0.80, step=0.05)
    auto_reject_collapsed = st.checkbox("Auto-reject collapsed models", value=False)
    scale_min, scale_max = st.slider(
        "Lot scaling range",
        min_value=0.25,
        max_value=2.0,
        value=(0.5, 1.5),
        step=0.05,
    )
    if not (scale_min < 1.0 < scale_max):
        st.warning("The lot scaling range should straddle 1.0, otherwise the policy cannot both increase and decrease size.", icon="⚠️")

    selected_ids = list(ea_options.keys()) if selected == "__all__" else [selected]
    filtered_raw = dataset[dataset["ea_id"].isin(selected_ids)].copy()
    filtered = _filter_active_period_rows(filtered_raw, min_trades_per_period, min_abs_profit_per_period)
    examples, feature_cols = _make_training_examples(filtered, lookback, objective, neutral_band, period_mode=period_mode)

    distinct_lot_scales = filtered["lot_scale"].round(6).nunique() if "lot_scale" in filtered.columns else 0
    dropped_rows = max(0, len(filtered_raw) - len(filtered))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Active {period_name} Rows", f"{len(filtered)}")
    c2.metric("Training Examples", f"{len(examples)}")
    c3.metric("Strategies", f"{filtered['ea_id'].nunique()}")
    c4.metric("Distinct Lot Scales", f"{distinct_lot_scales}")
    if dropped_rows > 0:
        st.caption(
            f"Filtered out {dropped_rows} low-information {cfg['label'].lower()}s using trades>={min_trades_per_period}"
            + (f" and abs PnL>={float(min_abs_profit_per_period):.2f}." if float(min_abs_profit_per_period) > 0.0 else ".")
        )

    if examples.empty or len(examples) < 6:
        error_box(f"Not enough active {cfg['adjective']} lot-sizing data. Generate more WFO or lot-focused sweeps, or relax the active-period filters.")
        return

    if distinct_lot_scales < 2:
        error_box("The dataset contains only one lot size, so the AI cannot learn a sizing policy yet. Generate experiments that vary `mmLots` or `_lots` first.")
        return

    st.caption(
        f"This model learns 3 lot regimes from historical {cfg['adjective']} variants with different lot sizes. "
        f"Objective: {_objective_label(objective)}. "
        f"Risk-off uses conf>={confidence_threshold:.2f}, margin>={margin_threshold:.2f}. "
        f"Risk-on uses conf>={risk_on_confidence_threshold:.2f}, margin>={risk_on_margin_threshold:.2f}, "
        f"plus downside guards."
    )

    if st.button(f"Train {period_name} Lot Policy", type="primary", use_container_width=True):
        model = _fit_ridge_regression(examples, feature_cols, alpha=alpha, train_ratio=train_ratio)
        scheduled = _apply_lot_schedule(
            model["all_df"],
            scale_min=scale_min,
            scale_max=scale_max,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            risk_on_confidence_threshold=risk_on_confidence_threshold,
            risk_on_margin_threshold=risk_on_margin_threshold,
            risk_on_drawdown_limit=risk_on_drawdown_limit_mult * float(examples["avg_abs_profit_lb"].median() if len(examples) else 1.0),
            risk_on_negative_ratio_limit=risk_on_negative_ratio_limit,
            risk_on_downside_vol_limit=risk_on_downside_vol_limit_mult * float(examples["avg_abs_profit_lb"].median() if len(examples) else 1.0),
        )
        train_scheduled = _apply_lot_schedule(
            model["train_df"],
            scale_min=scale_min,
            scale_max=scale_max,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            risk_on_confidence_threshold=risk_on_confidence_threshold,
            risk_on_margin_threshold=risk_on_margin_threshold,
            risk_on_drawdown_limit=risk_on_drawdown_limit_mult * float(examples["avg_abs_profit_lb"].median() if len(examples) else 1.0),
            risk_on_negative_ratio_limit=risk_on_negative_ratio_limit,
            risk_on_downside_vol_limit=risk_on_downside_vol_limit_mult * float(examples["avg_abs_profit_lb"].median() if len(examples) else 1.0),
        )
        test_scheduled = _apply_lot_schedule(
            model["test_df"],
            scale_min=scale_min,
            scale_max=scale_max,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            risk_on_confidence_threshold=risk_on_confidence_threshold,
            risk_on_margin_threshold=risk_on_margin_threshold,
            risk_on_drawdown_limit=risk_on_drawdown_limit_mult * float(examples["avg_abs_profit_lb"].median() if len(examples) else 1.0),
            risk_on_negative_ratio_limit=risk_on_negative_ratio_limit,
            risk_on_downside_vol_limit=risk_on_downside_vol_limit_mult * float(examples["avg_abs_profit_lb"].median() if len(examples) else 1.0),
        )

        baseline_metrics = _equity_metrics(scheduled["actual_profit"], period_mode=period_mode)
        ai_metrics = _equity_metrics(scheduled["ai_profit"], period_mode=period_mode)
        baseline_train_metrics = _equity_metrics(train_scheduled["actual_profit"], period_mode=period_mode)
        ai_train_metrics = _equity_metrics(train_scheduled["ai_profit"], period_mode=period_mode)
        baseline_test_metrics = _equity_metrics(test_scheduled["actual_profit"], period_mode=period_mode)
        ai_test_metrics = _equity_metrics(test_scheduled["ai_profit"], period_mode=period_mode)
        target_regime_mix = _regime_distribution(examples["target_regime"])
        predicted_regime_mix = _regime_distribution(scheduled["pred_regime"]) if "pred_regime" in scheduled.columns else {}
        applied_regime_mix = _regime_distribution(scheduled["applied_regime"]) if "applied_regime" in scheduled.columns else {}
        dominant_regime, dominant_share = _dominant_regime(applied_regime_mix)
        collapse_warning = ""
        if dominant_share >= float(collapse_threshold):
            collapse_warning = (
                f"Model collapsed toward `{dominant_regime}` in "
                f"{dominant_share*100:.1f}% of applied predictions."
            )
            if auto_reject_collapsed:
                error_box(
                    f"{collapse_warning} Training was rejected. Increase data diversity, widen the neutral band, "
                    "raise the confidence threshold, or train on a different objective."
                )
                return

        SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)
        scope_name = "all" if selected == "__all__" else eas[selected]["name"]
        schedule_path = SCHEDULES_DIR / f"{scope_name.lower().replace(' ', '_')}_schedule.csv"
        schedule_export = scheduled[
            [
                "ea_id",
                "month",
                "name",
                "symbol",
                "timeframe",
                "actual_profit",
                "actual_sharpe",
                "actual_win_rate",
                "actual_num_trades",
                "pred_regime",
                "pred_confidence",
                "pred_margin",
                "pred_lot_scale",
                "lot_scale",
                "base_lot",
                "scheduled_lot",
                "applied_regime",
                "ai_profit",
            ]
        ].copy()
        schedule_export.to_csv(schedule_path, index=False)

        monthly_summary_rows = schedule_export.groupby("month", as_index=False).agg(
            baseline_profit=("actual_profit", "sum"),
            ai_profit=("ai_profit", "sum"),
            avg_scale=("lot_scale", "mean"),
            avg_lot=("scheduled_lot", "mean"),
        ).to_dict(orient="records")

        artifact = {
            "report_id": uuid.uuid4().hex[:12],
            "scope": selected,
            "scope_label": "All strategies combined" if selected == "__all__" else ea_options[selected],
            "lookback": lookback,
            "period_mode": period_mode,
            "period_label": cfg["label"],
            "period_adjective": cfg["adjective"],
            "alpha": alpha,
            "objective": objective,
            "objective_label": _objective_label(objective),
            "period_rows_total": len(filtered_raw),
            "period_rows_active": len(filtered),
            "min_trades_per_period": int(min_trades_per_period),
            "min_abs_profit_per_period": float(min_abs_profit_per_period),
            "neutral_band": neutral_band,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "risk_on_confidence_threshold": risk_on_confidence_threshold,
            "risk_on_margin_threshold": risk_on_margin_threshold,
            "risk_on_drawdown_limit_mult": risk_on_drawdown_limit_mult,
            "risk_on_negative_ratio_limit": risk_on_negative_ratio_limit,
            "risk_on_downside_vol_limit_mult": risk_on_downside_vol_limit_mult,
            "collapse_threshold": collapse_threshold,
            "auto_reject_collapsed": auto_reject_collapsed,
            "train_ratio": train_ratio,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "feature_cols": feature_cols,
            "train_rows": len(model["train_df"]),
            "test_rows": len(model["test_df"]),
            "class_scales": model["class_scales"],
            "sample_weight_stats": model.get("sample_weight_stats", {}),
            "target_regime_mix": target_regime_mix,
            "predicted_regime_mix": predicted_regime_mix,
            "applied_regime_mix": applied_regime_mix,
            "collapse_warning": collapse_warning,
            "baseline_metrics": baseline_metrics,
            "ai_metrics": ai_metrics,
            "baseline_train_metrics": baseline_train_metrics,
            "ai_train_metrics": ai_train_metrics,
            "baseline_test_metrics": baseline_test_metrics,
            "ai_test_metrics": ai_test_metrics,
            "compare_monthly_rows": monthly_summary_rows,
            "schedule_rows": schedule_export.to_dict(orient="records"),
            "schedule_path": str(schedule_path),
        }

        st.session_state["training_artifact"] = artifact
        st.session_state["model_trained"] = True
        st.session_state["schedule_path"] = str(schedule_path)
        st.session_state["training_log"] = [
            f"Scope: {artifact['scope_label']}",
            f"{period_name} rows total/active: {len(filtered_raw)}/{len(filtered)}",
            f"Training examples: {len(examples)}",
            f"Distinct lot scales: {distinct_lot_scales}",
            f"Active-period filter: trades>={min_trades_per_period}, abs PnL>={float(min_abs_profit_per_period):.2f}",
            f"Lookback: {lookback}",
            f"Objective: {_objective_label(objective)}",
            f"Neutral band: {neutral_band:.2f}",
            f"Confidence threshold: {confidence_threshold:.2f}",
            f"Margin threshold: {margin_threshold:.2f}",
            f"Risk-on confidence threshold: {risk_on_confidence_threshold:.2f}",
            f"Risk-on margin threshold: {risk_on_margin_threshold:.2f}",
            f"Risk-on drawdown guard: {risk_on_drawdown_limit_mult:.2f}x avg abs PnL",
            f"Risk-on negative-month ratio guard: {risk_on_negative_ratio_limit:.2f}",
            f"Risk-on downside-vol guard: {risk_on_downside_vol_limit_mult:.2f}x avg abs PnL",
            f"Sample weights: min={model.get('sample_weight_stats', {}).get('min', 0.0):.2f} mean={model.get('sample_weight_stats', {}).get('mean', 0.0):.2f} max={model.get('sample_weight_stats', {}).get('max', 0.0):.2f}",
            f"Applied regime mix: off={applied_regime_mix.get('risk_off', 0.0)*100:.1f}% normal={applied_regime_mix.get('normal', 0.0)*100:.1f}% on={applied_regime_mix.get('risk_on', 0.0)*100:.1f}%",
            f"Train/test: {len(model['train_df'])}/{len(model['test_df'])}",
            f"Train baseline/AI profit: {baseline_train_metrics['total_profit']:.2f} / {ai_train_metrics['total_profit']:.2f}",
            f"Test baseline/AI profit: {baseline_test_metrics['total_profit']:.2f} / {ai_test_metrics['total_profit']:.2f}",
            f"Saved schedule: {schedule_path}",
        ]
        if collapse_warning:
            st.session_state["training_log"].append(collapse_warning)
        report = _build_report_snapshot(artifact, st.session_state["training_log"], eas)
        _upsert_saved_report(report)
        autosave()
        if collapse_warning:
            st.warning(collapse_warning, icon="⚠️")
        else:
            st.success("AI lot policy trained from lot-variation history.", icon="✅")
        st.rerun()


def _render_compare_tab():
    _render_saved_reports()
    st.markdown("---")
    artifact = st.session_state.get("training_artifact")
    if not artifact:
        if st.session_state.get("model_trained"):
            st.info("A previous training flag exists, but no current in-memory schedule is loaded. Retrain once to rebuild the comparison view.", icon="ℹ️")
        else:
            empty_state("Train a monthly or weekly lot policy first to compare baseline vs AI schedule.", "📊")
        return

    schedule = _normalize_schedule_frame(pd.DataFrame(artifact["schedule_rows"]))
    if schedule.empty:
        empty_state("No schedule rows were produced.", "📊")
        return
    period_mode = str(artifact.get("period_mode", "monthly"))
    period_name = _period_display_name(period_mode)

    st.markdown(f"#### {artifact['scope_label']}")
    st.caption(f"Policy period: {period_name}")
    if artifact.get("objective_label"):
        st.caption(f"Training objective: {artifact['objective_label']}")
    if artifact.get("period_rows_active") is not None or artifact.get("weekly_rows_active") is not None:
        st.caption(
            f"Active {period_name.lower()} rows: "
            f"{int(artifact.get('period_rows_active', artifact.get('weekly_rows_active', 0)) or 0)}/"
            f"{int(artifact.get('period_rows_total', artifact.get('weekly_rows_total', 0)) or 0)} | "
            f"filter trades>={int(artifact.get('min_trades_per_period', artifact.get('min_trades_per_week', 1)) or 1)}, "
            f"abs PnL>={float(artifact.get('min_abs_profit_per_period', artifact.get('min_abs_profit_per_week', 0.0)) or 0.0):.2f}"
        )
    if artifact.get("confidence_threshold") is not None:
        st.caption(f"Confidence threshold: {float(artifact['confidence_threshold']):.2f}")
    if artifact.get("margin_threshold") is not None:
        st.caption(f"Margin threshold: {float(artifact['margin_threshold']):.2f}")
    if artifact.get("risk_on_confidence_threshold") is not None:
        st.caption(f"Risk-on confidence threshold: {float(artifact['risk_on_confidence_threshold']):.2f}")
    if artifact.get("risk_on_margin_threshold") is not None:
        st.caption(f"Risk-on margin threshold: {float(artifact['risk_on_margin_threshold']):.2f}")
    if artifact.get("risk_on_drawdown_limit_mult") is not None:
        st.caption(f"Risk-on drawdown guard: <= {float(artifact['risk_on_drawdown_limit_mult']):.2f}x avg abs {period_name.lower()} PnL")
    if artifact.get("risk_on_negative_ratio_limit") is not None:
        st.caption(f"Risk-on negative-month ratio guard: <= {float(artifact['risk_on_negative_ratio_limit']):.2f}")
    if artifact.get("risk_on_downside_vol_limit_mult") is not None:
        st.caption(f"Risk-on downside-vol guard: <= {float(artifact['risk_on_downside_vol_limit_mult']):.2f}x avg abs {period_name.lower()} PnL")
    if artifact.get("sample_weight_stats"):
        stats = artifact["sample_weight_stats"]
        st.caption(
            f"Training sample weights: min {float(stats.get('min', 0.0)):.2f}, "
            f"mean {float(stats.get('mean', 0.0)):.2f}, "
            f"max {float(stats.get('max', 0.0)):.2f}"
        )
    if artifact.get("collapse_warning"):
        st.warning(str(artifact["collapse_warning"]), icon="⚠️")
    baseline_test = artifact.get("baseline_test_metrics", artifact["baseline_metrics"])
    ai_test = artifact.get("ai_test_metrics", artifact["ai_metrics"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Profit", f"${baseline_test['total_profit']:,.0f}")
    c2.metric("Test AI Profit", f"${ai_test['total_profit']:,.0f}")
    c3.metric("Test Sharpe", f"{baseline_test['sharpe']:.2f}")
    c4.metric("Test AI Sharpe", f"{ai_test['sharpe']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Test DD", f"{baseline_test['max_drawdown_pct']:.2f}%")
    c6.metric("Test AI DD", f"{ai_test['max_drawdown_pct']:.2f}%")
    c7.metric("Train Rows", f"{artifact['train_rows']}")
    c8.metric("Test Rows", f"{artifact['test_rows']}")

    regime_rows = []
    for label, key in [
        ("Target", "target_regime_mix"),
        ("Predicted", "predicted_regime_mix"),
        ("Applied", "applied_regime_mix"),
    ]:
        mix = artifact.get(key) or {}
        if mix:
            regime_rows.append(
                {
                    "mix": label,
                    "risk_off": round(float(mix.get("risk_off", 0.0)) * 100.0, 1),
                    "normal": round(float(mix.get("normal", 0.0)) * 100.0, 1),
                    "risk_on": round(float(mix.get("risk_on", 0.0)) * 100.0, 1),
                }
            )
    if regime_rows:
        st.markdown("#### Regime Balance")
        st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

    monthly_overlay = schedule.groupby("month", as_index=False).agg(
        baseline_profit=("actual_profit", "sum"),
        ai_profit=("ai_profit", "sum"),
        avg_scale=("lot_scale", "mean"),
        avg_lot=("scheduled_lot", "mean"),
    )
    monthly_overlay["baseline_equity"] = 100_000 + monthly_overlay["baseline_profit"].cumsum()
    monthly_overlay["ai_equity"] = 100_000 + monthly_overlay["ai_profit"].cumsum()

    st.markdown("#### End-to-End Rerun")
    eas = st.session_state.get("eas", {})
    results = st.session_state.get("backtest_results", {})
    actual_compare = artifact.get("actual_compare")
    if not actual_compare:
        st.caption(f"The metrics above are derived from the {period_name.lower()} training dataset. Use the rerun below to generate full baseline-vs-AI backtest results in this tab.")
    if st.button("Run End-to-End AI Comparison", type="primary", use_container_width=True, key="ai_compare_rerun"):
        try:
            with st.spinner("Rerunning baseline and AI backtests..."):
                progress = st.progress(0.0, text="Preparing AI rerun...")
                actual_compare = _run_end_to_end_compare(artifact, eas, results, progress_bar=progress)
            artifact["actual_compare"] = actual_compare
            st.session_state["training_artifact"] = artifact
            report = _build_report_snapshot(artifact, st.session_state.get("training_log", []), eas)
            _upsert_saved_report(report)
            autosave()
            st.success("End-to-end AI comparison completed.", icon="✅")
            st.rerun()
        except Exception as exc:
            error_box(f"End-to-end AI comparison failed: {exc}")

    if actual_compare:
        baseline_payload = _normalize_aggregate_payload(actual_compare.get("baseline_aggregate"))
        ai_payload = _normalize_aggregate_payload(actual_compare.get("ai_aggregate"))
        base_summary = baseline_payload["summary"]
        ai_summary = ai_payload["summary"]

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Baseline Profit", f"${base_summary['total_profit']:,.0f}")
        r2.metric("AI Profit", f"${ai_summary['total_profit']:,.0f}")
        r3.metric("Baseline Sharpe", f"{base_summary['sharpe_mean']:.2f}")
        r4.metric("AI Sharpe", f"{ai_summary['sharpe_mean']:.2f}")
        r5.metric("Profit Delta", f"${ai_summary['total_profit'] - base_summary['total_profit']:,.0f}")

        r6, r7, r8, r9, r10 = st.columns(5)
        r6.metric("Baseline DD", f"{base_summary['max_drawdown_pct']:.2f}%")
        r7.metric("AI DD", f"{ai_summary['max_drawdown_pct']:.2f}%")
        r8.metric("Baseline PF", f"{base_summary['profit_factor']:.2f}")
        r9.metric("AI PF", f"{ai_summary['profit_factor']:.2f}")
        r10.metric("Trades", f"{base_summary['num_trades']} / {ai_summary['num_trades']}")

        balance_base = baseline_payload["balance_curve_df"].copy()
        balance_ai = ai_payload["balance_curve_df"].copy()
        eq_fig_actual = go.Figure()
        if not balance_base.empty:
            eq_fig_actual.add_trace(go.Scatter(
                x=balance_base.index,
                y=balance_base["balance"],
                name="Baseline",
                line=dict(color="#9AA4B2", width=2),
            ))
        if not balance_ai.empty:
            eq_fig_actual.add_trace(go.Scatter(
                x=balance_ai.index,
                y=balance_ai["balance"],
                name="AI",
                line=dict(color="#2E75B6", width=3),
            ))
        eq_fig_actual.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=1.08, x=0),
            xaxis_title="Time",
            yaxis_title="Balance",
        )
        st.plotly_chart(eq_fig_actual, use_container_width=True)

        monthly_actual = pd.DataFrame()
        if not baseline_payload["monthly_df"].empty or not ai_payload["monthly_df"].empty:
            monthly_actual = pd.DataFrame(index=sorted(set(baseline_payload["monthly_df"].index.astype(str)).union(set(ai_payload["monthly_df"].index.astype(str)))))
            monthly_actual.index.name = "month"
            if not baseline_payload["monthly_df"].empty:
                monthly_actual["baseline_profit"] = baseline_payload["monthly_df"]["total_profit"].astype(float)
            if not ai_payload["monthly_df"].empty:
                monthly_actual["ai_profit"] = ai_payload["monthly_df"]["total_profit"].astype(float)
            monthly_actual = monthly_actual.fillna(0.0).reset_index()

        if not monthly_actual.empty:
            monthly_actual["baseline_equity"] = 100_000 + monthly_actual["baseline_profit"].cumsum()
            monthly_actual["ai_equity"] = 100_000 + monthly_actual["ai_profit"].cumsum()

            monthly_pnl_actual = go.Figure()
            monthly_pnl_actual.add_trace(go.Bar(x=monthly_actual["month"], y=monthly_actual["baseline_profit"], name="Baseline", marker_color="#9AA4B2"))
            monthly_pnl_actual.add_trace(go.Bar(x=monthly_actual["month"], y=monthly_actual["ai_profit"], name="AI", marker_color="#2E75B6"))
            monthly_pnl_actual.update_layout(
                barmode="group",
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=1.08, x=0),
                xaxis_title=period_name,
                yaxis_title=f"{period_name}ly PnL" if period_name == "Month" else f"{period_name} PnL",
            )
            st.plotly_chart(monthly_pnl_actual, use_container_width=True)

            dd_fig_actual = go.Figure()
            base_peak = np.maximum.accumulate(monthly_actual["baseline_equity"].to_numpy(dtype=float))
            ai_peak = np.maximum.accumulate(monthly_actual["ai_equity"].to_numpy(dtype=float))
            base_dd = np.where(base_peak > 0, (base_peak - monthly_actual["baseline_equity"]) / base_peak * 100.0, 0.0)
            ai_dd = np.where(ai_peak > 0, (ai_peak - monthly_actual["ai_equity"]) / ai_peak * 100.0, 0.0)
            dd_fig_actual.add_trace(go.Scatter(x=monthly_actual["month"], y=-base_dd, name="Baseline DD", line=dict(color="#D26A6A", width=2)))
            dd_fig_actual.add_trace(go.Scatter(x=monthly_actual["month"], y=-ai_dd, name="AI DD", line=dict(color="#F0A33E", width=2)))
            dd_fig_actual.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title=period_name,
                yaxis_title="Drawdown %",
            )
            st.plotly_chart(dd_fig_actual, use_container_width=True)

            st.markdown(f"#### End-to-End {period_name} Comparison")
            st.dataframe(monthly_actual, use_container_width=True, hide_index=True)

        st.markdown("#### Per-Strategy End-to-End Results")
        per_strategy_rows = []
        for ea_id, base_payload in actual_compare["baseline_by_ea"].items():
            ai_ea_payload = actual_compare["ai_by_ea"].get(ea_id, {})
            ea = eas.get(ea_id, {})
            base_ea_summary = base_payload.get("summary", {})
            ai_ea_summary = ai_ea_payload.get("summary", {})
            per_strategy_rows.append(
                {
                    "name": ea.get("name", ea_id),
                    "symbol": ea.get("symbol", ""),
                    "timeframe": ea.get("timeframe", ""),
                    "baseline_profit": float(base_ea_summary.get("total_profit", 0.0) or 0.0),
                    "ai_profit": float(ai_ea_summary.get("total_profit", 0.0) or 0.0),
                    "profit_delta": float((ai_ea_summary.get("total_profit", 0.0) or 0.0) - (base_ea_summary.get("total_profit", 0.0) or 0.0)),
                    "baseline_sharpe": float(base_ea_summary.get("sharpe_mean", 0.0) or 0.0),
                    "ai_sharpe": float(ai_ea_summary.get("sharpe_mean", 0.0) or 0.0),
                    "baseline_dd_pct": float(base_ea_summary.get("max_drawdown_pct", 0.0) or 0.0),
                    "ai_dd_pct": float(ai_ea_summary.get("max_drawdown_pct", 0.0) or 0.0),
                    "baseline_trades": int(base_ea_summary.get("num_trades", 0) or 0),
                    "ai_trades": int(ai_ea_summary.get("num_trades", 0) or 0),
                }
            )
        if per_strategy_rows:
            st.dataframe(pd.DataFrame(per_strategy_rows), use_container_width=True, hide_index=True)

    st.markdown(f"#### {period_name} Policy Comparison")
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=monthly_overlay["month"], y=monthly_overlay["baseline_equity"], name="Baseline", line=dict(color="#9AA4B2", width=2)))
    eq_fig.add_trace(go.Scatter(x=monthly_overlay["month"], y=monthly_overlay["ai_equity"], name="AI Policy", line=dict(color="#2E75B6", width=3)))
    eq_fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis_title=period_name,
        yaxis_title="Equity",
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    monthly_pnl_fig = go.Figure()
    monthly_pnl_fig.add_trace(go.Bar(x=monthly_overlay["month"], y=monthly_overlay["baseline_profit"], name="Baseline", marker_color="#9AA4B2"))
    monthly_pnl_fig.add_trace(go.Bar(x=monthly_overlay["month"], y=monthly_overlay["ai_profit"], name="AI", marker_color="#2E75B6"))
    monthly_pnl_fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis_title=period_name,
        yaxis_title=f"{period_name}ly PnL" if period_name == "Month" else f"{period_name} PnL",
    )
    st.plotly_chart(monthly_pnl_fig, use_container_width=True)

    dd_fig = go.Figure()
    baseline_equity = monthly_overlay["baseline_equity"].to_numpy(dtype=float)
    ai_equity = monthly_overlay["ai_equity"].to_numpy(dtype=float)
    base_peak = np.maximum.accumulate(baseline_equity)
    ai_peak = np.maximum.accumulate(ai_equity)
    base_dd = np.where(base_peak > 0, (base_peak - baseline_equity) / base_peak * 100.0, 0.0)
    ai_dd = np.where(ai_peak > 0, (ai_peak - ai_equity) / ai_peak * 100.0, 0.0)
    dd_fig.add_trace(go.Scatter(x=monthly_overlay["month"], y=-base_dd, name="Baseline DD", line=dict(color="#D26A6A", width=2)))
    dd_fig.add_trace(go.Scatter(x=monthly_overlay["month"], y=-ai_dd, name="AI DD", line=dict(color="#F0A33E", width=2)))
    dd_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title=period_name, yaxis_title="Drawdown %")
    st.plotly_chart(dd_fig, use_container_width=True)

    scale_fig = go.Figure()
    scale_fig.add_trace(go.Bar(x=monthly_overlay["month"], y=monthly_overlay["avg_scale"], name="Avg Lot Scale", marker_color="#2E75B6"))
    scale_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title=period_name, yaxis_title="Scale")
    st.plotly_chart(scale_fig, use_container_width=True)

    lot_fig = go.Figure()
    lot_fig.add_trace(go.Bar(x=monthly_overlay["month"], y=monthly_overlay["avg_lot"], name="Avg Scheduled Lot", marker_color="#F0A33E"))
    lot_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title=period_name, yaxis_title="Lot")
    st.plotly_chart(lot_fig, use_container_width=True)

    st.markdown("#### Schedule Export")
    schedule_path = Path(artifact["schedule_path"])
    st.code(str(schedule_path))
    st.download_button(
        "Download AI Schedule CSV",
        data=schedule.to_csv(index=False).encode("utf-8"),
        file_name=schedule_path.name,
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown(f"#### {period_name} Schedule")
    st.dataframe(schedule, use_container_width=True, hide_index=True)
