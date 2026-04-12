from __future__ import annotations

import copy
import hashlib
import importlib
import itertools
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from engine.backtester import Backtester
from engine.data_loader import load_bars
from research.monte_carlo import run_monte_carlo
from research.walk_forward import run_wfo
from services.backtest_service import backtest_result_payload, run_backtest
from services.backtest_service import default_intrabar_steps, resolve_execution_config, shift_bar_times
from services.python_strategy_service import (
    PRESET_DIR,
    StrategySpec,
    compile_strategy_spec,
    persist_strategy_spec,
    strategy_spec_from_template,
)


ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = ROOT / "data" / "research_state" / "native_strategy_lab"
CATALOG_PATH = LAB_DIR / "catalog.json"
BATCH_RUNS_PATH = LAB_DIR / "batch_runs.json"
GENERATOR_RUNS_PATH = LAB_DIR / "generator_runs.json"
ENGINE_WARMUP_BARS = 60
FAST_FILTER_TAIL_BARS = 400
FAST_FILTER_TRADE_SAMPLE = 8

DEFAULT_PROMOTION_POLICY = {
    "min_trades": 5,
    "min_profit_factor": 1.05,
    "max_drawdown_pct": 35.0,
    "min_wfo_robustness": 0.30,
    "min_mc_prob_profit": 0.45,
    "min_score": 45.0,
}

FAMILY_PROMOTION_POLICIES: dict[str, dict[str, Any]] = {
    "trend_pullback": {
        "min_trades": 10,
        "min_profit_factor": 1.10,
        "max_drawdown_pct": 28.0,
        "min_wfo_robustness": 0.35,
        "min_mc_prob_profit": 0.50,
        "min_score": 50.0,
    },
    "breakout_confirm": {
        "min_trades": 8,
        "min_profit_factor": 1.15,
        "max_drawdown_pct": 25.0,
        "min_wfo_robustness": 0.40,
        "min_mc_prob_profit": 0.52,
        "min_score": 52.0,
    },
    "reversion": {
        "min_trades": 12,
        "min_profit_factor": 1.08,
        "max_drawdown_pct": 22.0,
        "min_wfo_robustness": 0.32,
        "min_mc_prob_profit": 0.48,
        "min_score": 48.0,
    },
    "sqx_us30_wpr_stoch": {
        "min_trades": 10,
        "min_profit_factor": 1.18,
        "max_drawdown_pct": 24.0,
        "min_wfo_robustness": 0.38,
        "min_mc_prob_profit": 0.52,
        "min_score": 54.0,
    },
    "sqx_xau_osma_bb": {
        "min_trades": 9,
        "min_profit_factor": 1.14,
        "max_drawdown_pct": 26.0,
        "min_wfo_robustness": 0.36,
        "min_mc_prob_profit": 0.50,
        "min_score": 52.0,
    },
    "xau_breakout_session": {
        "min_trades": 10,
        "min_profit_factor": 1.16,
        "max_drawdown_pct": 24.0,
        "min_wfo_robustness": 0.40,
        "min_mc_prob_profit": 0.53,
        "min_score": 55.0,
    },
    "sqx_xau_highest_breakout": {
        "min_trades": 8,
        "min_profit_factor": 1.14,
        "max_drawdown_pct": 24.0,
        "min_wfo_robustness": 0.38,
        "min_mc_prob_profit": 0.50,
        "min_score": 53.0,
    },
    "xau_discovery_grammar": {
        "min_trades": 10,
        "min_profit_factor": 1.12,
        "max_drawdown_pct": 25.0,
        "min_wfo_robustness": 0.38,
        "min_mc_prob_profit": 0.50,
        "min_score": 52.0,
    },
    "sqx_usdjpy_vwap_wt": {
        "min_trades": 7,
        "min_profit_factor": 1.16,
        "max_drawdown_pct": 22.0,
        "min_wfo_robustness": 0.40,
        "min_mc_prob_profit": 0.53,
        "min_score": 55.0,
    },
    "sqx_hk50_batch_h1": {
        "min_trades": 8,
        "min_profit_factor": 1.16,
        "max_drawdown_pct": 27.0,
        "min_wfo_robustness": 0.35,
        "min_mc_prob_profit": 0.50,
        "min_score": 52.0,
    },
    "sqx_hk50_after_retest_h4": {
        "min_trades": 6,
        "min_profit_factor": 1.18,
        "max_drawdown_pct": 24.0,
        "min_wfo_robustness": 0.40,
        "min_mc_prob_profit": 0.54,
        "min_score": 55.0,
    },
    "sqx_hk50_before_retest_h4": {
        "min_trades": 6,
        "min_profit_factor": 1.15,
        "max_drawdown_pct": 24.0,
        "min_wfo_robustness": 0.38,
        "min_mc_prob_profit": 0.52,
        "min_score": 53.0,
    },
    "sqx_uk100_ulcer_keltner_h1": {
        "min_trades": 8,
        "min_profit_factor": 1.14,
        "max_drawdown_pct": 23.0,
        "min_wfo_robustness": 0.36,
        "min_mc_prob_profit": 0.50,
        "min_score": 51.0,
    },
}

DEFAULT_SCORING_PROFILE = {
    "sharpe_weight": 30.0,
    "profit_factor_weight": 20.0,
    "drawdown_weight": 20.0,
    "trade_count_weight": 10.0,
    "wfo_weight": 10.0,
    "mc_weight": 5.0,
    "stability_weight": 5.0,
    "sharpe_scale": 2.0,
    "profit_factor_scale": 2.0,
    "drawdown_scale": 30.0,
    "trade_count_scale": 30.0,
}

FAMILY_SCORING_PROFILES: dict[str, dict[str, Any]] = {
    "trend_pullback": {
        "sharpe_weight": 28.0,
        "profit_factor_weight": 18.0,
        "drawdown_weight": 18.0,
        "trade_count_weight": 10.0,
        "wfo_weight": 14.0,
        "mc_weight": 6.0,
        "stability_weight": 6.0,
    },
    "breakout_confirm": {
        "sharpe_weight": 24.0,
        "profit_factor_weight": 24.0,
        "drawdown_weight": 20.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 14.0,
        "mc_weight": 6.0,
        "stability_weight": 4.0,
    },
    "reversion": {
        "sharpe_weight": 22.0,
        "profit_factor_weight": 18.0,
        "drawdown_weight": 24.0,
        "trade_count_weight": 12.0,
        "wfo_weight": 10.0,
        "mc_weight": 6.0,
        "stability_weight": 8.0,
    },
    "sqx_us30_wpr_stoch": {
        "sharpe_weight": 22.0,
        "profit_factor_weight": 24.0,
        "drawdown_weight": 20.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 14.0,
        "mc_weight": 6.0,
        "stability_weight": 6.0,
    },
    "sqx_xau_osma_bb": {
        "sharpe_weight": 24.0,
        "profit_factor_weight": 22.0,
        "drawdown_weight": 18.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 14.0,
        "mc_weight": 8.0,
        "stability_weight": 6.0,
    },
    "xau_breakout_session": {
        "sharpe_weight": 22.0,
        "profit_factor_weight": 24.0,
        "drawdown_weight": 20.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 14.0,
        "mc_weight": 8.0,
        "stability_weight": 4.0,
    },
    "sqx_xau_highest_breakout": {
        "sharpe_weight": 22.0,
        "profit_factor_weight": 24.0,
        "drawdown_weight": 20.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 14.0,
        "mc_weight": 8.0,
        "stability_weight": 4.0,
    },
    "xau_discovery_grammar": {
        "sharpe_weight": 20.0,
        "profit_factor_weight": 22.0,
        "drawdown_weight": 20.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 16.0,
        "mc_weight": 8.0,
        "stability_weight": 6.0,
    },
    "sqx_usdjpy_vwap_wt": {
        "sharpe_weight": 24.0,
        "profit_factor_weight": 22.0,
        "drawdown_weight": 18.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 16.0,
        "mc_weight": 6.0,
        "stability_weight": 6.0,
    },
    "sqx_hk50_batch_h1": {
        "sharpe_weight": 20.0,
        "profit_factor_weight": 22.0,
        "drawdown_weight": 18.0,
        "trade_count_weight": 10.0,
        "wfo_weight": 16.0,
        "mc_weight": 8.0,
        "stability_weight": 6.0,
    },
    "sqx_hk50_after_retest_h4": {
        "sharpe_weight": 24.0,
        "profit_factor_weight": 22.0,
        "drawdown_weight": 18.0,
        "trade_count_weight": 6.0,
        "wfo_weight": 18.0,
        "mc_weight": 6.0,
        "stability_weight": 6.0,
    },
    "sqx_hk50_before_retest_h4": {
        "sharpe_weight": 24.0,
        "profit_factor_weight": 22.0,
        "drawdown_weight": 18.0,
        "trade_count_weight": 6.0,
        "wfo_weight": 18.0,
        "mc_weight": 6.0,
        "stability_weight": 6.0,
    },
    "sqx_uk100_ulcer_keltner_h1": {
        "sharpe_weight": 22.0,
        "profit_factor_weight": 22.0,
        "drawdown_weight": 20.0,
        "trade_count_weight": 8.0,
        "wfo_weight": 16.0,
        "mc_weight": 6.0,
        "stability_weight": 6.0,
    },
}

FAMILY_MUTATION_SPACES: dict[str, dict[str, list[Any]]] = {
    "trend_pullback": {
        "params.FastEMA": [13, 21, 34],
        "params.SlowEMA": [34, 55, 89],
        "params.ATRPeriod": [10, 14, 21],
        "params.ExitAfterBars": [8, 12, 16],
        "direction": ["long", "short"],
    },
    "breakout_confirm": {
        "params.HighestPeriod": [15, 20, 34],
        "params.LowestPeriod": [15, 20, 34],
        "params.ATRPeriod": [10, 14, 21],
        "params.BBWRPeriod": [14, 20, 28],
        "params.MaxDistancePct": [2.5, 4.0, 6.0],
        "direction": ["long", "short"],
        "expiry_bars": [2, 3, 5],
    },
    "reversion": {
        "params.BBPeriod": [14, 20, 26],
        "params.WPRPeriod": [10, 14, 21],
        "params.ATRPeriod": [10, 14, 21],
        "params.WPROversold": [-90.0, -80.0, -70.0],
        "params.WPROverbought": [-30.0, -20.0, -10.0],
    },
    "sqx_usdjpy_vwap_wt": {
        "params.HighestPeriod": [12, 20, 34],
        "params.ExitAfterBars1": [8, 14, 21],
        "params.EMAPeriod1": [13, 20, 34],
        "params.VWAPPeriod1": [55, 77, 120],
        "params.PriceEntryMult1": [0.2, 0.3, 0.5],
    },
    "sqx_xau_osma_bb": {
        "params.BBPeriod": [14, 20, 26],
        "params.OsmaFast": [5, 8, 13],
        "params.OsmaSlow": [13, 17, 26],
        "params.ATRPeriod2": [34, 55, 89],
    },
    "xau_breakout_session": {
        "params.HighestPeriod": [16, 24, 36],
        "params.LowestPeriod": [16, 24, 36],
        "params.ATRPeriod": [10, 14, 21],
        "params.BBWRPeriod": [18, 24, 36],
        "params.BBWRMin": [0.01, 0.015, 0.025],
        "params.EntryBufferATR": [0.1, 0.15, 0.25],
        "params.MaxDistancePct": [2.5, 3.5, 5.0],
        "params.ExitAfterBars": [12, 18, 24],
    },
    "sqx_xau_highest_breakout": {
        "long_signal_mode": ["sma_bias", "smma_pullback", "ha_reclaim"],
        "short_signal_mode": ["lwma_lowest_count", "wt_push", "lwma_hour_quantile"],
        "params.HighestPeriod": [205, 210, 245],
        "params.LowestPeriod": [205, 210, 245],
        "params.LongStopATR": [1.5, 1.7, 2.0],
        "params.LongTargetATR": [3.4, 3.5, 5.7],
        "params.ShortStopATR": [1.5, 1.8, 1.9],
        "params.ShortTargetATR": [1.6, 1.9, 4.8],
        "params.LongExpiryBars": [2, 4, 10],
        "params.ShortExpiryBars": [10, 18],
        "params.LongTrailATR": [0.0, 1.0, 2.4],
        "params.MaxDistancePct": [4.0, 6.0, 8.0],
    },
    "xau_discovery_grammar": {
        "entry_archetype": ["breakout_stop", "breakout_close", "pullback_trend", "ema_reclaim", "atr_pullback_limit"],
        "volatility_filter": ["bb_width", "atr_expansion", "none"],
        "session_filter": ["london_ny", "london_only", "ny_only", "all_day"],
        "stop_model": ["atr", "channel", "swing"],
        "target_model": ["fixed_rr", "atr_scaled", "trend_runner"],
        "exit_model": ["session_close", "time_exit", "trailing_atr", "break_even_then_trail", "channel_flip", "atr_time_stop"],
        "params.HighestPeriod": [16, 24, 36],
        "params.LowestPeriod": [16, 24, 36],
        "params.ATRPeriod": [10, 14, 21],
        "params.FastEMA": [13, 21, 34],
        "params.SlowEMA": [34, 55, 89],
        "params.BBWRPeriod": [18, 24, 36],
        "params.BBWRMin": [0.01, 0.015, 0.025],
        "params.EntryBufferATR": [0.1, 0.15, 0.25],
        "params.StopLossATR": [1.1, 1.4, 1.8],
        "params.ProfitTargetATR": [2.0, 2.8, 3.6],
        "params.TrailATR": [0.8, 1.1, 1.5],
        "params.PullbackLookback": [3, 5, 8],
        "params.PullbackATR": [0.4, 0.6, 0.9],
        "params.BreakEvenATR": [0.6, 0.8, 1.1],
        "params.TimeStopATR": [0.3, 0.4, 0.6],
        "params.ExitAfterBars": [12, 18, 24],
        "params.MaxDistancePct": [2.5, 3.5, 5.0],
    },
    "sqx_us30_wpr_stoch": {
        "params.StochK": [9, 14, 21],
        "params.StochD": [3, 5, 7],
        "params.StochSlow": [3, 5, 7],
        "params.WPRPeriod": [50, 75, 100],
        "params.MaxDistancePct": [4.0, 6.0, 8.0],
    },
    "sqx_hk50_batch_h1": {
        "params.LWMAPeriod1": [10, 14, 21],
        "params.IndicatorCrsMAPrd1": [34, 47, 60],
        "params.Highest_period": [34, 50, 70],
        "params.Lowest_period": [34, 50, 70],
    },
    "sqx_hk50_after_retest_h4": {
        "params.DIPeriod1": [10, 14, 21],
        "params.SARStep": [0.18, 0.266, 0.34],
        "params.PriceEntryMult1": [0.2, 0.3, 0.5],
    },
    "sqx_hk50_before_retest_h4": {
        "params.RSIPeriod1": [10, 14, 21],
        "params.Period1": [34, 50, 70],
        "params.PriceEntryMult1": [0.2, 0.3, 0.5],
    },
    "sqx_uk100_ulcer_keltner_h1": {
        "params.UlcerPeriod": [34, 48, 64],
        "params.KCPeriod": [14, 20, 28],
        "params.KCMult1": [1.75, 2.25, 2.75],
    },
}

FAMILY_WFO_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "trend_pullback": {
        "FastEMA": [13, 21, 34],
        "SlowEMA": [34, 55, 89],
        "ATRPeriod": [10, 14, 21],
        "ExitAfterBars": [8, 12, 16],
    },
    "breakout_confirm": {
        "HighestPeriod": [15, 20, 34],
        "LowestPeriod": [15, 20, 34],
        "ATRPeriod": [10, 14, 21],
        "BBWRPeriod": [14, 20, 28],
        "MaxDistancePct": [2.5, 4.0, 6.0],
    },
    "reversion": {
        "BBPeriod": [14, 20, 26],
        "WPRPeriod": [10, 14, 21],
        "ATRPeriod": [10, 14, 21],
        "WPROversold": [-90.0, -80.0, -70.0],
        "WPROverbought": [-30.0, -20.0, -10.0],
    },
    "sqx_usdjpy_vwap_wt": {
        "HighestPeriod": [12, 20, 34],
        "ExitAfterBars1": [8, 14, 21],
        "EMAPeriod1": [13, 20, 34],
        "VWAPPeriod1": [55, 77, 120],
        "PriceEntryMult1": [0.2, 0.3, 0.5],
    },
    "sqx_xau_osma_bb": {
        "BBPeriod": [120, 167, 220],
        "OsmaFast": [5, 8, 13],
        "OsmaSlow": [13, 17, 26],
        "ATRPeriod2": [34, 55, 89],
    },
    "xau_breakout_session": {
        "HighestPeriod": [16, 24, 36],
        "LowestPeriod": [16, 24, 36],
        "ATRPeriod": [10, 14, 21],
        "BBWRPeriod": [18, 24, 36],
        "BBWRMin": [0.01, 0.015, 0.025],
        "EntryBufferATR": [0.1, 0.15, 0.25],
        "MaxDistancePct": [2.5, 3.5, 5.0],
        "ExitAfterBars": [12, 18, 24],
    },
    "sqx_xau_highest_breakout": {
        "HighestPeriod": [205, 210, 245],
        "LowestPeriod": [205, 210, 245],
        "LongStopATR": [1.5, 1.7, 2.0],
        "LongTargetATR": [3.4, 3.5, 5.7],
        "ShortStopATR": [1.5, 1.8, 1.9],
        "ShortTargetATR": [1.6, 1.9, 4.8],
        "LongExpiryBars": [2, 4, 10],
        "ShortExpiryBars": [10, 18],
        "LongTrailATR": [0.0, 1.0, 2.4],
    },
    "xau_discovery_grammar": {
        "FastEMA": [13, 21, 34],
        "ProfitTargetATR": [2.0, 2.8, 3.6],
        "PullbackLookback": [3, 5, 8],
        "BreakEvenATR": [0.6, 0.8, 1.1],
        "ExitAfterBars": [12, 18, 24],
    },
    "sqx_us30_wpr_stoch": {
        "StochK": [9, 14, 21],
        "StochD": [3, 5, 7],
        "StochSlow": [3, 5, 7],
        "WPRPeriod": [50, 75, 100],
        "MaxDistancePct": [4.0, 6.0, 8.0],
    },
    "sqx_hk50_batch_h1": {
        "LWMAPeriod1": [10, 14, 21],
        "IndicatorCrsMAPrd1": [34, 47, 60],
        "Highest_period": [34, 50, 70],
        "Lowest_period": [34, 50, 70],
    },
    "sqx_hk50_after_retest_h4": {
        "DIPeriod1": [10, 14, 21],
        "SARStep": [0.18, 0.266, 0.34],
        "PriceEntryMult1": [0.2, 0.3, 0.5],
    },
    "sqx_hk50_before_retest_h4": {
        "RSIPeriod1": [10, 14, 21],
        "Period1": [34, 50, 70],
        "PriceEntryMult1": [0.2, 0.3, 0.5],
    },
    "sqx_uk100_ulcer_keltner_h1": {
        "UlcerPeriod": [34, 48, 64],
        "KCPeriod": [14, 20, 28],
        "KCMult1": [1.75, 2.25, 2.75],
    },
}


@dataclass
class NativeStrategyRecord:
    strategy_id: str
    created_at: str
    updated_at: str
    name: str
    template_name: str
    origin: str
    symbol: str
    timeframe: str
    status: str
    params: dict[str, Any]
    template_payload: dict[str, Any]
    lineage: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    strategy_path: str = ""
    spec_path: str = ""
    strategy_module: str = ""
    strategy_class: str = ""
    notes: str = ""
    promotion_policy: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_lab_dir():
    LAB_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return copy.deepcopy(default)


def _save_json(path: Path, payload: Any):
    _ensure_lab_dir()
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _catalog() -> list[dict]:
    return _load_json(CATALOG_PATH, [])


def _save_catalog(records: list[dict]):
    _save_json(CATALOG_PATH, records)


def _batch_runs() -> list[dict]:
    return _load_json(BATCH_RUNS_PATH, [])


def _save_batch_runs(records: list[dict]):
    _save_json(BATCH_RUNS_PATH, records)


def _generator_runs() -> list[dict]:
    return _load_json(GENERATOR_RUNS_PATH, [])


def _save_generator_runs(records: list[dict]):
    _save_json(GENERATOR_RUNS_PATH, records)


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _get_by_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_by_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def save_custom_preset(*, label: str, description: str, template_name: str, payload: dict[str, Any]) -> str:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") or "preset"
    path = PRESET_DIR / f"{slug}.json"
    body = {
        "label": label,
        "description": description,
        "template": template_name,
        "payload": payload,
    }
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return str(path)


def list_native_strategy_records() -> list[dict]:
    return _catalog()


def get_native_strategy_record(strategy_id: str) -> dict | None:
    for record in _catalog():
        if record.get("strategy_id") == strategy_id:
            return record
    return None


def upsert_native_strategy_record(record: NativeStrategyRecord | dict[str, Any]) -> dict:
    payload = asdict(record) if isinstance(record, NativeStrategyRecord) else dict(record)
    records = _catalog()
    for idx, existing in enumerate(records):
        if existing.get("strategy_id") == payload["strategy_id"]:
            records[idx] = payload
            _save_catalog(records)
            return payload
    records.append(payload)
    _save_catalog(records)
    return payload


def update_native_strategy_status(strategy_id: str, *, status: str, notes: str | None = None) -> dict | None:
    records = _catalog()
    for idx, existing in enumerate(records):
        if existing.get("strategy_id") != strategy_id:
            continue
        existing["status"] = status
        existing["updated_at"] = _utc_now()
        if notes is not None:
            existing["notes"] = notes
        records[idx] = existing
        _save_catalog(records)
        return existing
    return None


def update_native_strategy_mt5_validation(
    strategy_id: str,
    *,
    report_path: str,
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    accepted: bool,
) -> dict | None:
    records = _catalog()
    for idx, existing in enumerate(records):
        if existing.get("strategy_id") != strategy_id:
            continue
        existing["mt5_validation"] = {
            "report_path": report_path,
            "metrics": metrics,
            "comparison": comparison,
            "accepted": bool(accepted),
            "validated_at": _utc_now(),
        }
        existing["updated_at"] = _utc_now()
        records[idx] = existing
        _save_catalog(records)
        return existing
    return None


def strategy_children(parent_strategy_id: str) -> list[dict]:
    children = []
    for record in _catalog():
        lineage = record.get("lineage", {})
        if lineage.get("parent_strategy_id") == parent_strategy_id:
            children.append(record)
    return sorted(children, key=lambda item: item.get("created_at", ""))


def strategy_lineage_tree(strategy_id: str) -> list[dict]:
    record = get_native_strategy_record(strategy_id)
    if not record:
        return []
    chain = [record]
    parent_id = record.get("lineage", {}).get("parent_strategy_id")
    while parent_id:
        parent = get_native_strategy_record(parent_id)
        if not parent:
            break
        chain.insert(0, parent)
        parent_id = parent.get("lineage", {}).get("parent_strategy_id")
    return chain


def strategy_payload_diff(left_strategy_id: str, right_strategy_id: str) -> dict[str, Any]:
    left = get_native_strategy_record(left_strategy_id)
    right = get_native_strategy_record(right_strategy_id)
    if not left or not right:
        raise ValueError("Both strategies must exist for diffing")
    left_payload = left.get("template_payload", {})
    right_payload = right.get("template_payload", {})
    left_params = left_payload.get("params", {})
    right_params = right_payload.get("params", {})
    keys = sorted(set(left_params) | set(right_params))
    param_changes = []
    for key in keys:
        left_value = left_params.get(key)
        right_value = right_params.get(key)
        if left_value != right_value:
            param_changes.append(
                {
                    "param": key,
                    "left": left_value,
                    "right": right_value,
                }
            )
    return {
        "left_strategy_id": left_strategy_id,
        "right_strategy_id": right_strategy_id,
        "left_name": left.get("name", ""),
        "right_name": right.get("name", ""),
        "param_changes": param_changes,
        "payload_changed": left_payload != right_payload,
        "status_changed": left.get("status") != right.get("status"),
        "score_left": left.get("evaluation", {}).get("score", 0.0),
        "score_right": right.get("evaluation", {}).get("score", 0.0),
    }


def register_generated_strategy(
    *,
    template_name: str,
    payload: dict[str, Any],
    strategy_id: str,
    origin: str,
    lineage: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    promotion_policy: dict[str, Any] | None = None,
) -> tuple[dict, dict]:
    spec = strategy_spec_from_template(template_name, payload)
    compiled = compile_strategy_spec(spec, strategy_id=strategy_id)
    paths = persist_strategy_spec(compiled)

    record = NativeStrategyRecord(
        strategy_id=strategy_id,
        created_at=_utc_now(),
        updated_at=_utc_now(),
        name=spec.name,
        template_name=template_name,
        origin=origin,
        symbol=spec.symbol,
        timeframe=spec.timeframe,
        status="draft",
        params=spec.params,
        template_payload=copy.deepcopy(payload),
        lineage=dict(lineage or {}),
        tags=list(tags or []),
        strategy_path=paths["strategy_path"],
        spec_path=paths["spec_path"],
        strategy_module=compiled.strategy_module,
        strategy_class=compiled.strategy_class,
        promotion_policy=dict(promotion_policy or {}),
    )
    return upsert_native_strategy_record(record), {
        "compiled": compiled,
        "paths": paths,
        "spec": spec,
    }


def regenerate_native_strategy_version(
    parent_strategy_id: str,
    *,
    template_name: str | None = None,
    payload: dict[str, Any] | None = None,
    origin: str = "catalog_regenerate",
) -> tuple[dict, dict]:
    parent = get_native_strategy_record(parent_strategy_id)
    if not parent:
        raise ValueError(f"Unknown native strategy: {parent_strategy_id}")
    next_template = template_name or parent["template_name"]
    next_payload = copy.deepcopy(payload or parent.get("template_payload", {}))
    child_id = f"{parent_strategy_id}_v{_stable_hash({'template': next_template, 'payload': next_payload, 'ts': _utc_now()})[:6]}"
    return register_generated_strategy(
        template_name=next_template,
        payload=next_payload,
        strategy_id=child_id,
        origin=origin,
        lineage={
            "parent_strategy_id": parent_strategy_id,
            "template_name": next_template,
            "version_of": parent.get("lineage", {}).get("version_of", parent_strategy_id),
        },
        tags=list(set(parent.get("tags", [])) | {"regenerated"}),
    )


def _load_compiled_class(module_name: str, class_name: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    module = importlib.reload(sys.modules[module_name]) if module_name in sys.modules else importlib.import_module(module_name)
    return getattr(module, class_name)


def _load_runtime_strategy_class(record: dict[str, Any]):
    template_name = str(record.get("template_name", "") or "")
    template_payload = record.get("template_payload")
    if template_name and isinstance(template_payload, dict) and template_payload:
        spec = strategy_spec_from_template(template_name, template_payload)
        return _compile_runtime_strategy(spec, str(record.get("strategy_id", "runtime")))
    return _load_compiled_class(record["strategy_module"], record["strategy_class"])


def _default_param_grid(params: dict[str, Any], *, numeric_limit: int = 4) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    numeric_keys = [key for key, value in params.items() if isinstance(value, (int, float))][:numeric_limit]
    for key in numeric_keys:
        value = params[key]
        if isinstance(value, int):
            values = sorted({max(1, int(round(value * 0.8))), int(value), max(1, int(round(value * 1.2)))})
        else:
            values = sorted({round(float(value) * 0.8, 4), round(float(value), 4), round(float(value) * 1.2, 4)})
        if len(values) > 1:
            grid[key] = values
    return grid


def default_wfo_param_grid_for_template(template_name: str, params: dict[str, Any]) -> dict[str, list[Any]]:
    configured = FAMILY_WFO_PARAM_GRIDS.get(template_name, {})
    grid: dict[str, list[Any]] = {}
    for key, values in configured.items():
        if key not in params:
            continue
        current = params.get(key)
        deduped: list[Any] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        if current is not None and current not in deduped:
            deduped.append(current)
        if len(deduped) > 1:
            grid[key] = deduped
    return grid


def _default_promotion_policy() -> dict[str, Any]:
    return dict(DEFAULT_PROMOTION_POLICY)


def default_promotion_policy_for_template(template_name: str) -> dict[str, Any]:
    policy = _default_promotion_policy()
    policy.update(FAMILY_PROMOTION_POLICIES.get(template_name, {}))
    return policy


def default_scoring_profile_for_template(template_name: str) -> dict[str, Any]:
    profile = dict(DEFAULT_SCORING_PROFILE)
    profile.update(FAMILY_SCORING_PROFILES.get(template_name, {}))
    return profile


def _resolve_promotion_policy(record: dict[str, Any]) -> dict[str, Any]:
    policy = default_promotion_policy_for_template(str(record.get("template_name", "")))
    policy.update(record.get("promotion_policy", {}) or {})
    return policy


def _resolve_scoring_profile(record: dict[str, Any]) -> dict[str, Any]:
    profile = default_scoring_profile_for_template(str(record.get("template_name", "")))
    profile.update(record.get("scoring_profile", {}) or {})
    return profile


def _score_candidate(
    *,
    template_name: str,
    backtest: dict[str, Any],
    wfo: dict[str, Any],
    monte_carlo: dict[str, Any],
    stability_score: float,
    scoring_profile: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    profile = dict(scoring_profile or default_scoring_profile_for_template(template_name))
    sharpe_component = min(1.0, max(0.0, backtest.get("sharpe_mean", 0.0) / float(profile["sharpe_scale"]))) * float(profile["sharpe_weight"])
    profit_factor_component = min(1.0, max(0.0, backtest.get("profit_factor", 0.0) / float(profile["profit_factor_scale"]))) * float(profile["profit_factor_weight"])
    drawdown_component = max(0.0, 1.0 - min(1.0, backtest.get("max_drawdown_pct", 100.0) / float(profile["drawdown_scale"]))) * float(profile["drawdown_weight"])
    trade_count_component = min(1.0, backtest.get("num_trades", 0) / float(profile["trade_count_scale"])) * float(profile["trade_count_weight"])
    wfo_component = float(wfo.get("robustness_score", 0.0)) * float(profile["wfo_weight"]) if isinstance(wfo, dict) else 0.0
    mc_component = float(monte_carlo.get("prob_profit", 0.0)) * float(profile["mc_weight"]) if isinstance(monte_carlo, dict) else 0.0
    stability_component = float(stability_score) * float(profile["stability_weight"])
    components = {
        "sharpe": round(sharpe_component, 4),
        "profit_factor": round(profit_factor_component, 4),
        "drawdown": round(drawdown_component, 4),
        "trade_count": round(trade_count_component, 4),
        "wfo": round(wfo_component, 4),
        "monte_carlo": round(mc_component, 4),
        "stability": round(stability_component, 4),
    }
    score = round(sum(components.values()), 2)
    return score, components, profile


def _normalize_mutation_values(path: str, current: Any, values: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    for value in values:
        if path == "min_bars" and int(value) <= ENGINE_WARMUP_BARS:
            continue
        if value not in deduped:
            deduped.append(value)
    if current is not None and not (path == "min_bars" and int(current) <= ENGINE_WARMUP_BARS) and current not in deduped:
        deduped.append(current)
    return deduped


def default_family_mutation_space(template_name: str, base_payload: dict[str, Any]) -> dict[str, list[Any]]:
    configured = FAMILY_MUTATION_SPACES.get(template_name, {})
    mutations: dict[str, list[Any]] = {}
    for path, values in configured.items():
        current = _get_by_path(base_payload, path)
        deduped = _normalize_mutation_values(path, current, list(values))
        if len(deduped) > 1:
            mutations[path] = deduped
    return mutations


def default_structural_mutation_space(template_name: str, base_payload: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        path: values
        for path, values in default_family_mutation_space(template_name, base_payload).items()
        if not path.startswith("params.")
    }


def default_parameter_mutation_space(template_name: str, base_payload: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        path: values
        for path, values in default_family_mutation_space(template_name, base_payload).items()
        if path.startswith("params.")
    }


def _param_stability_score(params_used: list[dict[str, Any]]) -> float:
    if len(params_used) <= 1:
        return 1.0
    numeric_keys = sorted({key for params in params_used for key, value in params.items() if isinstance(value, (int, float))})
    if not numeric_keys:
        return 1.0
    penalties: list[float] = []
    for key in numeric_keys:
        values = [float(params[key]) for params in params_used if key in params]
        if len(values) <= 1:
            continue
        mean = sum(values) / len(values)
        if abs(mean) < 1e-9:
            penalties.append(0.0)
            continue
        std = pd.Series(values).std(ddof=0)
        penalties.append(min(1.0, float(std / abs(mean))))
    if not penalties:
        return 1.0
    return max(0.0, 1.0 - float(sum(penalties) / len(penalties)))


def evaluate_native_strategy(
    strategy_id: str,
    *,
    date_from: str,
    date_to: str,
    intrabar_steps: int = 1,
    run_mc: bool = True,
    run_wfo_checks: bool = True,
    execution_config: dict[str, Any] | None = None,
) -> dict:
    record = get_native_strategy_record(strategy_id)
    if not record:
        raise ValueError(f"Unknown native strategy: {strategy_id}")

    strat_cls = _load_runtime_strategy_class(record)
    execution_config = resolve_execution_config(record["symbol"], execution_config)
    effective_intrabar_steps = default_intrabar_steps(record["symbol"], record["timeframe"], intrabar_steps)
    result = run_backtest(
        strat_cls,
        symbol=record["symbol"],
        timeframe=record["timeframe"],
        date_from=date_from,
        date_to=date_to,
        overrides=record.get("params", {}),
        intrabar_steps=effective_intrabar_steps,
        execution_config=execution_config,
    )
    payload = backtest_result_payload(result)

    evaluation = {
        "backtest": payload["summary"],
        "score_components": {},
        "accepted": False,
        "date_from": date_from,
        "date_to": date_to,
    }

    if run_mc and result.n_trades >= 5:
        mc = run_monte_carlo(result, n_simulations=200)
        evaluation["monte_carlo"] = mc.summary()
    else:
        evaluation["monte_carlo"] = {}

    if run_wfo_checks:
        bar_time_offset_hours = float(getattr(strat_cls, "bar_time_offset_hours", 0.0) or 0.0)
        df = shift_bar_times(
            load_bars(record["symbol"], record["timeframe"], date_from=date_from, date_to=date_to),
            bar_time_offset_hours,
        )
        intrabar_df = None
        if effective_intrabar_steps > 1 and str(record["timeframe"]).upper() != "M1":
            intrabar_df = shift_bar_times(
                load_bars(record["symbol"], "M1", date_from=date_from, date_to=date_to),
                bar_time_offset_hours,
            )
        param_grid = default_wfo_param_grid_for_template(
            str(record.get("template_name", "")),
            record.get("params", {}),
        )
        if not param_grid:
            param_grid = _default_param_grid(record.get("params", {}))
        if param_grid:
            try:
                wfo = run_wfo(
                    strat_cls,
                    df,
                    param_grid=param_grid,
                    train_months=12,
                    test_months=3,
                    optimize_by="sharpe_ratio",
                    backtester_kwargs={
                        "initial_capital": 100_000,
                        "intrabar_steps": int(effective_intrabar_steps),
                        "commission_per_lot": float(execution_config["commission_per_lot"]),
                        "spread_pips": float(execution_config["spread_pips"]),
                        "slippage_pips": float(execution_config["slippage_pips"]),
                        "tick_size": float(execution_config.get("tick_size", 1.0)),
                        "tick_value": float(execution_config.get("tick_value", 0.0)),
                        "contract_size": float(execution_config.get("contract_size", 0.0)),
                        "swap_per_lot_long": float(execution_config.get("swap_per_lot_long", 0.0)),
                        "swap_per_lot_short": float(execution_config.get("swap_per_lot_short", 0.0)),
                        "session_timezone_offset_hours": float(execution_config.get("session_timezone_offset_hours", 0.0)),
                        "use_bar_spread": bool(execution_config.get("use_bar_spread", False)),
                    },
                    intrabar_df=intrabar_df,
                )
                evaluation["wfo"] = wfo.summary()
                evaluation["parameter_stability"] = {
                    "score": round(_param_stability_score(wfo.params_used), 4),
                    "windows": len(wfo.params_used),
                }
                evaluation["wfo_param_grid"] = param_grid
            except Exception as exc:
                evaluation["wfo"] = {"error": str(exc)}
                evaluation["parameter_stability"] = {"score": 0.0, "windows": 0}
                evaluation["wfo_param_grid"] = param_grid
        else:
            evaluation["wfo"] = {}
            evaluation["parameter_stability"] = {"score": 1.0, "windows": 0}
            evaluation["wfo_param_grid"] = {}

    bt = evaluation["backtest"]
    mc = evaluation.get("monte_carlo", {})
    wfo = evaluation.get("wfo", {})
    stability = evaluation.get("parameter_stability", {}).get("score", 0.0)
    scoring_profile = _resolve_scoring_profile(record)
    score, score_components, scoring_profile = _score_candidate(
        template_name=str(record.get("template_name", "")),
        backtest=bt,
        wfo=wfo,
        monte_carlo=mc,
        stability_score=float(stability),
        scoring_profile=scoring_profile,
    )

    policy = _resolve_promotion_policy(record)
    mc_prob_profit = float(mc.get("prob_profit", 0.0)) if isinstance(mc, dict) else 0.0
    wfo_robustness = float(wfo.get("robustness_score", 0.0)) if isinstance(wfo, dict) else 0.0
    mt5_validation = record.get("mt5_validation", {}) if isinstance(record.get("mt5_validation", {}), dict) else {}
    requires_mt5_gate = str(record.get("symbol", "")).strip().upper() == "XAUUSD"
    mt5_ok = bool(mt5_validation.get("accepted", False)) if requires_mt5_gate else True
    accepted = (
        bt.get("num_trades", 0) >= int(policy["min_trades"])
        and bt.get("profit_factor", 0.0) >= float(policy["min_profit_factor"])
        and bt.get("max_drawdown_pct", 100.0) <= float(policy["max_drawdown_pct"])
        and wfo_robustness >= float(policy["min_wfo_robustness"])
        and mc_prob_profit >= float(policy["min_mc_prob_profit"])
        and score >= float(policy["min_score"])
        and mt5_ok
    )

    evaluation["score_components"] = score_components
    evaluation["score"] = round(score, 2)
    evaluation["scoring_profile"] = scoring_profile
    evaluation["accepted"] = bool(accepted)
    evaluation["promotion_policy"] = policy
    evaluation["promotion_checks"] = {
        "trades_ok": bt.get("num_trades", 0) >= int(policy["min_trades"]),
        "profit_factor_ok": bt.get("profit_factor", 0.0) >= float(policy["min_profit_factor"]),
        "drawdown_ok": bt.get("max_drawdown_pct", 100.0) <= float(policy["max_drawdown_pct"]),
        "wfo_ok": wfo_robustness >= float(policy["min_wfo_robustness"]),
        "mc_ok": mc_prob_profit >= float(policy["min_mc_prob_profit"]),
        "score_ok": score >= float(policy["min_score"]),
        "mt5_ok": mt5_ok,
    }
    evaluation["intrabar_steps"] = int(effective_intrabar_steps)
    evaluation["mt5_gate_required"] = bool(requires_mt5_gate)

    record["evaluation"] = evaluation
    record["updated_at"] = _utc_now()
    if accepted and record.get("status") == "draft":
        record["status"] = "candidate"
    upsert_native_strategy_record(record)
    return evaluation


def _effective_mutation_space(
    template_name: str,
    base_payload: dict[str, Any],
    mutation_space: dict[str, list[Any]],
    *,
    include_structural_mutations: bool,
) -> dict[str, list[Any]]:
    family_space = default_family_mutation_space(template_name, base_payload)
    combined: dict[str, list[Any]] = {}
    for key, values in mutation_space.items():
        target_path = key if "." in key else f"params.{key}"
        if _get_by_path(base_payload, target_path) is None and "." not in key:
            target_path = key
        current = _get_by_path(base_payload, target_path)
        deduped = _normalize_mutation_values(target_path, current, list(values))
        if len(deduped) > 1:
            combined[target_path] = deduped
    if include_structural_mutations:
        for key, values in family_space.items():
            if key.startswith("params."):
                continue
            combined.setdefault(key, list(values))
    return combined


def list_generator_runs() -> list[dict]:
    return _generator_runs()


def get_generator_run(run_id: str) -> dict | None:
    for record in _generator_runs():
        if record.get("run_id") == run_id:
            return record
    return None


def upsert_generator_run(record: dict[str, Any]) -> dict[str, Any]:
    records = _generator_runs()
    payload = dict(record)
    for idx, existing in enumerate(records):
        if existing.get("run_id") == payload.get("run_id"):
            records[idx] = payload
            _save_generator_runs(records)
            return payload
    records.append(payload)
    _save_generator_runs(records)
    return payload


def _top_generator_candidates(candidates: list[dict[str, Any]], *, limit: int = 25) -> list[dict[str, Any]]:
    ranked = rank_batch_candidates(candidates)
    return ranked[: max(1, int(limit))]


def _generator_execution_config(symbol: str) -> dict[str, Any]:
    config = resolve_execution_config(symbol)
    normalized = str(symbol or "").strip().upper()
    if normalized == "XAUUSD":
        config["spread_pips"] = round(float(config["spread_pips"]) * 1.35, 5)
        config["slippage_pips"] = round(max(float(config["slippage_pips"]) * 1.75, 0.02), 5)
    return config


def _sample_generator_candidate(
    *,
    template_name: str,
    base_payload: dict[str, Any],
    mutation_space: dict[str, list[Any]],
    include_structural_mutations: bool,
    rng: random.Random,
    discovery_mode: str,
    seen_hashes: set[str],
) -> dict[str, Any] | None:
    effective = _effective_mutation_space(
        template_name,
        base_payload,
        mutation_space,
        include_structural_mutations=include_structural_mutations,
    )
    keys = sorted(effective)
    if not keys:
        return None

    max_changes = min(len(keys), 2 if discovery_mode == "conservative" else 3 if discovery_mode == "balanced" else 5)
    min_changes = 1
    if max_changes < 1:
        return None

    for _ in range(50):
        change_count = rng.randint(min_changes, max_changes)
        chosen_keys = rng.sample(keys, k=change_count)
        payload = copy.deepcopy(base_payload)
        lineage_mutations: dict[str, Any] = {}
        changed_paths: list[str] = []
        for key in chosen_keys:
            current = _get_by_path(payload, key)
            options = [value for value in effective[key] if value != current]
            if not options:
                continue
            value = rng.choice(options)
            _set_by_path(payload, key, value)
            lineage_mutations[key] = value
            changed_paths.append(key)
        if not changed_paths:
            continue
        candidate_hash = _stable_hash({"template": template_name, "payload": payload})
        if candidate_hash in seen_hashes:
            continue
        seen_hashes.add(candidate_hash)
        return {
            "candidate_id": f"cand_{candidate_hash}",
            "payload": payload,
            "mutations": lineage_mutations,
            "mutation_distance": len(changed_paths),
            "structural_changes": [path for path in changed_paths if not path.startswith("params.")],
        }
    return None


def run_generator_session(
    *,
    template_name: str,
    base_payload: dict[str, Any],
    mutation_space: dict[str, list[Any]],
    date_from: str,
    date_to: str,
    max_candidates: int = 500,
    run_mc: bool = False,
    run_wfo_checks: bool = False,
    intrabar_steps: int = 1,
    include_structural_mutations: bool = True,
    discovery_mode: str = "conservative",
    random_seed: int = 42,
    progress_callback=None,
) -> dict[str, Any]:
    if not template_name:
        raise ValueError("Generator session requires template_name.")
    effective_space = _effective_mutation_space(
        template_name,
        base_payload,
        mutation_space,
        include_structural_mutations=include_structural_mutations,
    )
    if not effective_space:
        raise ValueError("Generator session requires a non-empty mutation space.")

    max_candidates = max(1, int(max_candidates))
    rng = random.Random(int(random_seed))
    run_id = f"generator_{_stable_hash({'template': template_name, 'payload': base_payload, 'max': max_candidates, 'seed': random_seed, 'ts': _utc_now()})}"
    run_record = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "template_name": template_name,
        "base_payload": copy.deepcopy(base_payload),
        "mutation_space": copy.deepcopy(mutation_space),
        "effective_mutation_space": copy.deepcopy(effective_space),
        "date_from": date_from,
        "date_to": date_to,
        "max_candidates": max_candidates,
        "intrabar_steps": int(intrabar_steps),
        "run_mc": bool(run_mc),
        "run_wfo_checks": bool(run_wfo_checks),
        "include_structural_mutations": bool(include_structural_mutations),
        "discovery_mode": discovery_mode,
        "status": "running",
        "stage": "running",
        "attempted": 0,
        "duplicates": 0,
        "evaluated": 0,
        "accepted": 0,
        "best_score": 0.0,
        "best_strategy_id": "",
        "candidates": [],
        "leaderboard": [],
    }
    upsert_generator_run(run_record)

    seen_hashes: set[str] = set()
    seen_fingerprints: set[str] = set()
    df = _load_fast_filter_df(str(base_payload.get("symbol", "")), str(base_payload.get("timeframe", "")))
    generation_execution_config = _generator_execution_config(str(base_payload.get("symbol", "")))
    attempt_budget = max(max_candidates * 12, 100)

    for attempt in range(1, attempt_budget + 1):
        run_record["attempted"] = attempt
        if progress_callback:
            progress_callback(
                run_record["evaluated"] / max_candidates,
                "generating",
                f"Generated {run_record['evaluated']}/{max_candidates} unique candidates after {attempt} attempts.",
            )
        candidate = _sample_generator_candidate(
            template_name=template_name,
            base_payload=base_payload,
            mutation_space=mutation_space,
            include_structural_mutations=include_structural_mutations,
            rng=rng,
            discovery_mode=discovery_mode,
            seen_hashes=seen_hashes,
        )
        if candidate is None:
            continue
        fingerprint = _behavioral_fingerprint(candidate["payload"], template_name, df)
        if fingerprint in seen_fingerprints:
            run_record["duplicates"] += 1
            continue
        seen_fingerprints.add(fingerprint)

        strategy_id = f"{candidate['candidate_id']}_{run_record['evaluated']:04d}"
        payload = copy.deepcopy(candidate["payload"])
        payload["name"] = f"{payload.get('name', template_name)} [G{run_record['evaluated'] + 1:04d}]"
        register_generated_strategy(
            template_name=template_name,
            payload=payload,
            strategy_id=strategy_id,
            origin="generator_run",
            lineage={
                "generator_run_id": run_id,
                "mutations": candidate["mutations"],
                "mutation_distance": candidate["mutation_distance"],
                "structural_changes": candidate.get("structural_changes", []),
            },
            tags=["generator", template_name],
            promotion_policy=default_promotion_policy_for_template(template_name),
        )
        evaluation = evaluate_native_strategy(
            strategy_id,
            date_from=date_from,
            date_to=date_to,
            intrabar_steps=intrabar_steps,
            run_mc=run_mc,
            run_wfo_checks=run_wfo_checks,
            execution_config=generation_execution_config,
        )
        candidate_record = {
            "strategy_id": strategy_id,
            "name": payload.get("name", strategy_id),
            "status": "evaluated",
            "evaluated": True,
            "template_name": template_name,
            "mutations": candidate["mutations"],
            "mutation_distance": candidate["mutation_distance"],
            "structural_changes": candidate.get("structural_changes", []),
            "behavioral_fingerprint": fingerprint,
            "score": evaluation.get("score", 0.0),
            "accepted": evaluation.get("accepted", False),
            "backtest": evaluation.get("backtest", {}),
            "wfo": evaluation.get("wfo", {}),
            "parameter_stability": evaluation.get("parameter_stability", {}),
            "monte_carlo": evaluation.get("monte_carlo", {}),
        }
        run_record["candidates"].append(candidate_record)
        run_record["evaluated"] += 1
        if candidate_record["accepted"]:
            run_record["accepted"] += 1
        if float(candidate_record.get("score", 0.0)) >= float(run_record.get("best_score", 0.0)):
            run_record["best_score"] = float(candidate_record.get("score", 0.0))
            run_record["best_strategy_id"] = strategy_id
        run_record["leaderboard"] = _top_generator_candidates(run_record["candidates"])
        run_record["updated_at"] = _utc_now()
        upsert_generator_run(run_record)
        if run_record["evaluated"] >= max_candidates:
            break

    run_record["updated_at"] = _utc_now()
    run_record["status"] = "completed" if run_record["evaluated"] >= max_candidates else "stopped"
    run_record["stage"] = run_record["status"]
    upsert_generator_run(run_record)
    return run_record


def _load_fast_filter_df(symbol: str, timeframe: str) -> pd.DataFrame | None:
    try:
        df = load_bars(symbol, timeframe)
    except Exception:
        return None
    if df.empty:
        return None
    return df.tail(FAST_FILTER_TAIL_BARS).copy()


def _compile_runtime_strategy(spec: StrategySpec, strategy_id: str):
    compiled = compile_strategy_spec(spec, strategy_id=f"{strategy_id}_fingerprint")
    namespace: dict[str, Any] = {"__name__": f"_native_strategy_lab_{compiled.strategy_slug}"}
    exec(compiled.source, namespace)
    return namespace[compiled.strategy_class]


def _behavioral_fingerprint(payload: dict[str, Any], template_name: str, df: pd.DataFrame | None) -> str:
    if df is None or len(df) <= ENGINE_WARMUP_BARS + 5:
        return f"payload:{_stable_hash({'template': template_name, 'payload': payload})}"
    spec = strategy_spec_from_template(template_name, payload)
    strategy_id = _stable_hash({"template": template_name, "payload": payload})
    strat_cls = _compile_runtime_strategy(spec, strategy_id)
    execution_config = resolve_execution_config(str(payload.get("symbol", "")))
    result = Backtester(
        initial_capital=100_000,
        lot_value=getattr(strat_cls, "lot_value", 1.0),
        commission_per_lot=float(execution_config["commission_per_lot"]),
        spread_pips=float(execution_config["spread_pips"]),
        slippage_pips=float(execution_config["slippage_pips"]),
        tick_size=float(execution_config.get("tick_size", 1.0)),
        tick_value=float(execution_config.get("tick_value", 0.0)),
        contract_size=float(execution_config.get("contract_size", 0.0)),
        swap_per_lot_long=float(execution_config.get("swap_per_lot_long", 0.0)),
        swap_per_lot_short=float(execution_config.get("swap_per_lot_short", 0.0)),
        session_timezone_offset_hours=float(execution_config.get("session_timezone_offset_hours", 0.0)),
        use_bar_spread=bool(execution_config.get("use_bar_spread", False)),
    ).run(strat_cls(), df)
    trade_signature = [
        (
            int(trade.opened_bar),
            int(trade.closed_bar),
            trade.direction.value,
            str(trade.comment),
            round(float(trade.entry_price), 6),
            round(float(trade.exit_price), 6),
        )
        for trade in result.trades[:FAST_FILTER_TRADE_SAMPLE]
    ]
    payload = {
        "trades": int(result.n_trades),
        "profit_factor": round(float(result.profit_factor), 6),
        "drawdown_pct": round(float(result.max_drawdown[1]), 6),
        "net_profit": round(float(result.net_profit), 2),
        "trade_signature": trade_signature,
    }
    return _stable_hash(payload)


def _fast_filter_behavioral_duplicates(
    *,
    template_name: str,
    base_payload: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = _load_fast_filter_df(str(base_payload.get("symbol", "")), str(base_payload.get("timeframe", "")))
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_fingerprints: dict[str, str] = {}
    for candidate in candidates:
        fingerprint = _behavioral_fingerprint(candidate["payload"], template_name, df)
        enriched = {
            **candidate,
            "behavioral_fingerprint": fingerprint,
        }
        if fingerprint in seen_fingerprints:
            enriched["duplicate_of"] = seen_fingerprints[fingerprint]
            duplicates.append(enriched)
            continue
        seen_fingerprints[fingerprint] = candidate["candidate_id"]
        kept.append(enriched)
    return kept, duplicates


def generate_batch_candidates(
    *,
    template_name: str,
    base_payload: dict[str, Any],
    mutation_space: dict[str, list[Any]],
    limit: int = 50,
    search_mode: str = "grid",
    random_seed: int = 42,
    include_structural_mutations: bool = False,
) -> list[dict[str, Any]]:
    mutation_space = _effective_mutation_space(
        template_name,
        base_payload,
        mutation_space,
        include_structural_mutations=include_structural_mutations,
    )
    keys = sorted(mutation_space)
    if not keys:
        return []
    candidates: list[dict[str, Any]] = []
    combos = list(itertools.product(*(mutation_space[key] for key in keys)))
    if search_mode == "random":
        combos.sort(key=lambda combo: _stable_hash({"combo": combo, "seed": int(random_seed)}))
    elif search_mode == "progressive":
        base_combo = tuple(
            _get_by_path(base_payload, key)
            for key in keys
        )
        combos.sort(key=lambda combo: sum(0 if combo[idx] == base_combo[idx] else 1 for idx in range(len(keys))))
    seen_hashes: set[str] = set()
    for combo in combos[: int(limit)]:
        payload = copy.deepcopy(base_payload)
        lineage_mutations: dict[str, Any] = {}
        mutation_distance = 0
        changed_paths: list[str] = []
        for key, value in zip(keys, combo):
            target_path = key
            _set_by_path(payload, target_path, value)
            lineage_mutations[key] = value
            base_value = _get_by_path(base_payload, target_path)
            if value != base_value:
                mutation_distance += 1
                changed_paths.append(key)
        candidate_hash = _stable_hash({"template": template_name, "payload": payload})
        if candidate_hash in seen_hashes:
            continue
        seen_hashes.add(candidate_hash)
        candidates.append(
            {
                "candidate_id": f"cand_{candidate_hash}",
                "template_name": template_name,
                "payload": payload,
                "mutations": lineage_mutations,
                "mutation_distance": mutation_distance,
                "structural_changes": [path for path in changed_paths if "." in path or path in {"direction", "expiry_bars", "min_bars"}],
            }
        )
    return candidates


def create_batch_run(
    *,
    template_name: str,
    base_payload: dict[str, Any],
    mutation_space: dict[str, list[Any]],
    limit: int = 25,
    search_mode: str = "grid",
    promotion_policy: dict[str, Any] | None = None,
    include_structural_mutations: bool = False,
) -> dict[str, Any]:
    batch_id = f"batch_{_stable_hash({'template': template_name, 'payload': base_payload, 'mutations': mutation_space, 'limit': limit, 'mode': search_mode, 'generated_only': True, 'include_structural_mutations': include_structural_mutations})}"
    candidates = generate_batch_candidates(
        template_name=template_name,
        base_payload=base_payload,
        mutation_space=mutation_space,
        limit=limit,
        search_mode=search_mode,
        include_structural_mutations=include_structural_mutations,
    )
    unique_candidates, duplicate_candidates = _fast_filter_behavioral_duplicates(
        template_name=template_name,
        base_payload=base_payload,
        candidates=candidates,
    )
    effective_policy = default_promotion_policy_for_template(template_name)
    effective_policy.update(promotion_policy or {})
    generated_candidates: list[dict[str, Any]] = []
    base_hash = _stable_hash({"template": template_name, "payload": base_payload})
    for idx, candidate in enumerate(unique_candidates):
        strategy_id = f"{candidate['candidate_id']}_{idx:02d}"
        payload = copy.deepcopy(candidate["payload"])
        payload["name"] = f"{payload.get('name', template_name)} [{idx + 1:02d}]"
        record, _ = register_generated_strategy(
            template_name=template_name,
            payload=payload,
            strategy_id=strategy_id,
            origin="batch_generation",
            lineage={
                "batch_id": batch_id,
                "mutations": candidate["mutations"],
                "base_hash": base_hash,
                "mutation_distance": candidate.get("mutation_distance", 0),
                "structural_changes": candidate.get("structural_changes", []),
            },
            tags=["batch", template_name],
            promotion_policy=effective_policy,
        )
        generated_candidates.append(
            {
                "strategy_id": strategy_id,
                "name": record["name"],
                "template_name": template_name,
                "mutations": candidate["mutations"],
                "mutation_distance": candidate.get("mutation_distance", 0),
                "structural_changes": candidate.get("structural_changes", []),
                "behavioral_fingerprint": candidate.get("behavioral_fingerprint", ""),
                "status": "generated",
                "evaluated": False,
            }
        )

    run_record = {
        "batch_id": batch_id,
        "created_at": _utc_now(),
        "template_name": template_name,
        "base_payload": base_payload,
        "mutation_space": mutation_space,
        "search_mode": search_mode,
        "promotion_policy": effective_policy,
        "include_structural_mutations": bool(include_structural_mutations),
        "stage": "filtered",
        "requested_candidates": len(candidates),
        "generated_candidates": len(unique_candidates),
        "duplicate_candidates": [
            {
                "candidate_id": item["candidate_id"],
                "duplicate_of": item.get("duplicate_of", ""),
                "mutations": item.get("mutations", {}),
                "mutation_distance": item.get("mutation_distance", 0),
                "behavioral_fingerprint": item.get("behavioral_fingerprint", ""),
            }
            for item in duplicate_candidates
        ],
        "candidates": generated_candidates,
    }
    batch_runs = _batch_runs()
    batch_runs = [record for record in batch_runs if record.get("batch_id") != batch_id]
    batch_runs.append(run_record)
    _save_batch_runs(batch_runs)
    return run_record


def evaluate_batch_run(
    batch_id: str,
    *,
    date_from: str,
    date_to: str,
    candidate_ids: list[str] | None = None,
    top_n: int | None = None,
    run_mc: bool = False,
    run_wfo_checks: bool = False,
    intrabar_steps: int = 1,
    progress_callback=None,
) -> dict[str, Any]:
    runs = _batch_runs()
    run_record = next((record for record in runs if record.get("batch_id") == batch_id), None)
    if not run_record:
        raise ValueError(f"Unknown batch: {batch_id}")
    candidates = list(run_record.get("candidates", []))
    if candidate_ids:
        selected = [item for item in candidates if item.get("strategy_id") in set(candidate_ids)]
    elif top_n:
        selected = sorted(candidates, key=lambda item: (item.get("mutation_distance", 0), item.get("strategy_id", "")))[: int(top_n)]
    else:
        selected = candidates
    total = max(len(selected), 1)
    evaluated_map: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(selected):
        if progress_callback:
            progress_callback(idx / total, "evaluating", f"Retesting {idx + 1}/{len(selected)} candidates.")
        evaluation = evaluate_native_strategy(
            item["strategy_id"],
            date_from=date_from,
            date_to=date_to,
            intrabar_steps=intrabar_steps,
            run_mc=run_mc,
            run_wfo_checks=run_wfo_checks,
        )
        evaluated_map[item["strategy_id"]] = {
            **item,
            "score": evaluation["score"],
            "accepted": evaluation["accepted"],
            "backtest": evaluation.get("backtest", {}),
            "wfo": evaluation.get("wfo", {}),
            "parameter_stability": evaluation.get("parameter_stability", {}),
            "monte_carlo": evaluation.get("monte_carlo", {}),
            "quality_bucket": "promote" if evaluation["accepted"] else ("watch" if evaluation["score"] >= 40 else "reject"),
            "evaluated": True,
            "status": "evaluated",
        }
    merged: list[dict[str, Any]] = []
    for item in candidates:
        merged.append(evaluated_map.get(item["strategy_id"], item))
    ranked = rank_batch_candidates([item for item in merged if item.get("evaluated")])
    ranked_map = {item["strategy_id"]: item for item in ranked}
    final_candidates = [ranked_map.get(item["strategy_id"], item) for item in merged]
    run_record["date_from"] = date_from
    run_record["date_to"] = date_to
    run_record["stage"] = "evaluated"
    run_record["evaluation_mode"] = {
        "run_mc": bool(run_mc),
        "run_wfo_checks": bool(run_wfo_checks),
        "intrabar_steps": intrabar_steps,
        "top_n": top_n,
    }
    run_record["candidates"] = final_candidates
    run_record["updated_at"] = _utc_now()
    batch_runs = [record for record in runs if record.get("batch_id") != batch_id]
    batch_runs.append(run_record)
    _save_batch_runs(batch_runs)
    if progress_callback:
        progress_callback(1.0, "saved", f"Saved retest results for {batch_id}.")
    return run_record


def rank_batch_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = copy.deepcopy(candidates)
    if not ranked:
        return ranked
    for item in ranked:
        bt = item.get("backtest", {})
        item["quality_bucket"] = "promote" if item.get("accepted") else ("watch" if item.get("score", 0.0) >= 40 else "reject")
        item["risk_adjusted"] = round(
            float(item.get("score", 0.0))
            + float(item.get("wfo", {}).get("robustness_score", 0.0)) * 10.0
            - min(20.0, float(bt.get("max_drawdown_pct", 0.0)) * 0.5),
            2,
        )
    ranked.sort(key=lambda item: (item.get("accepted", False), item.get("risk_adjusted", 0.0), item.get("score", 0.0)), reverse=True)
    if ranked:
        metrics = {
            "score": [float(item.get("score", 0.0)) for item in ranked],
            "profit_factor": [float(item.get("backtest", {}).get("profit_factor", 0.0)) for item in ranked],
            "drawdown": [float(item.get("backtest", {}).get("max_drawdown_pct", 0.0)) for item in ranked],
        }
        score_sorted = sorted(metrics["score"], reverse=True)
        pf_sorted = sorted(metrics["profit_factor"], reverse=True)
        dd_sorted = sorted(metrics["drawdown"])
        for item in ranked:
            item["rank_score"] = score_sorted.index(float(item.get("score", 0.0))) + 1
            item["rank_profit_factor"] = pf_sorted.index(float(item.get("backtest", {}).get("profit_factor", 0.0))) + 1
            item["rank_drawdown"] = dd_sorted.index(float(item.get("backtest", {}).get("max_drawdown_pct", 0.0))) + 1
    return ranked


def run_batch_generation(
    *,
    template_name: str,
    base_payload: dict[str, Any],
    mutation_space: dict[str, list[Any]],
    date_from: str,
    date_to: str,
    limit: int = 25,
    search_mode: str = "grid",
    promotion_policy: dict[str, Any] | None = None,
    include_structural_mutations: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    generated = create_batch_run(
        template_name=template_name,
        base_payload=base_payload,
        mutation_space=mutation_space,
        limit=limit,
        search_mode=search_mode,
        promotion_policy=promotion_policy,
        include_structural_mutations=include_structural_mutations,
    )
    if progress_callback:
        progress_callback(
            1.0,
            "filtered",
            f"Generated {len(generated['candidates'])} unique candidates for {template_name}. Retest separately for backtest/WFO/MC.",
        )
    return generated


def list_batch_runs() -> list[dict]:
    return _batch_runs()
