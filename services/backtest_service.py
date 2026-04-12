from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
import pandas as pd

from config import settings
from engine.base_strategy import BaseStrategy
from engine.backtester import Backtester
from engine.data_loader import available_symbols, load_bars
from engine.policy import OverlayPolicy


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def default_execution_config(symbol: str | None = None) -> dict:
    return dict(settings.symbol_execution_defaults(symbol))


def resolve_execution_config(symbol: str | None = None, execution_config: dict | None = None) -> dict:
    resolved = default_execution_config(symbol)
    for key, value in (execution_config or {}).items():
        if value is None:
            continue
        resolved[key] = value
    return resolved


def default_intrabar_steps(symbol: str | None, timeframe: str | None, requested_steps: int = 1) -> int:
    symbol_norm = str(symbol or "").strip().upper()
    timeframe_norm = str(timeframe or "").strip().upper()
    if int(requested_steps) > 1:
        return int(requested_steps)
    if symbol_norm == "XAUUSD" and timeframe_norm == "H1":
        return 60
    return int(requested_steps)


def preload_date_from(date_from: str, timeframe: str, symbol: str | None = None) -> str:
    start = pd.Timestamp(date_from)
    tf = str(timeframe or "").upper()
    if str(symbol or "").strip().upper() == "XAUUSD" and tf == "H1":
        return str((start - pd.Timedelta(days=30)).date())
    if tf == "H1":
        return str((start - pd.Timedelta(days=14)).date())
    if tf == "H4":
        return str((start - pd.Timedelta(days=30)).date())
    return str((start - pd.Timedelta(days=7)).date())


def shift_bar_times(df: pd.DataFrame | None, offset_hours: float = 0.0) -> pd.DataFrame | None:
    if df is None or abs(float(offset_hours)) < 1e-9:
        return df
    shifted = df.copy()
    shifted.index = shifted.index + pd.to_timedelta(float(offset_hours), unit="h")
    shifted = shifted[~shifted.index.duplicated(keep="last")].sort_index()
    return shifted


def discover_strategies() -> dict[str, type]:
    strategies: dict[str, type] = {}
    strategy_root = ROOT / "strategies"
    if not strategy_root.exists():
        return strategies

    for py_file in sorted(strategy_root.rglob("*.py")):
        if py_file.name.startswith("_"):
            continue
        relative = py_file.relative_to(ROOT).with_suffix("")
        module_name = ".".join(relative.parts)
        try:
            if module_name in sys.modules:
                mod = importlib.reload(sys.modules[module_name])
            else:
                mod = importlib.import_module(module_name)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(cls, BaseStrategy)
                    and cls is not BaseStrategy
                    and cls.__module__ == module_name
                ):
                    display = getattr(cls, "name", cls.__name__)
                    strategies[display] = cls
        except Exception:
            continue

    return strategies


def run_backtest(
    strat_cls: type,
    *,
    symbol: str,
    timeframe: str,
    date_from: str,
    date_to: str,
    overrides: dict | None = None,
    intrabar_steps: int = 1,
    overlay_policy: OverlayPolicy | None = None,
    execution_config: dict | None = None,
):
    intrabar_steps = default_intrabar_steps(symbol, timeframe, intrabar_steps)
    load_from = preload_date_from(date_from, timeframe, symbol) if date_from else None
    df = load_bars(symbol, timeframe, date_from=load_from, date_to=date_to)
    intrabar_df = None
    if intrabar_steps > 1 and timeframe.upper() != "M1":
        intrabar_df = load_bars(symbol, "M1", date_from=load_from or date_from, date_to=date_to)
    execution = resolve_execution_config(symbol, execution_config)
    bt = Backtester(
        initial_capital=100_000,
        lot_value=getattr(strat_cls, "lot_value", 1.0),
        intrabar_steps=intrabar_steps,
        overlay_policy=overlay_policy,
        commission_per_lot=float(execution["commission_per_lot"]),
        spread_pips=float(execution["spread_pips"]),
        slippage_pips=float(execution["slippage_pips"]),
        tick_size=float(execution.get("tick_size", settings.DEFAULT_TICK_SIZE)),
        tick_value=float(execution.get("tick_value", settings.DEFAULT_TICK_VALUE)),
        contract_size=float(execution.get("contract_size", settings.DEFAULT_CONTRACT_SIZE)),
        swap_per_lot_long=float(execution.get("swap_per_lot_long", settings.DEFAULT_SWAP_PER_LOT_LONG)),
        swap_per_lot_short=float(execution.get("swap_per_lot_short", settings.DEFAULT_SWAP_PER_LOT_SHORT)),
        session_timezone_offset_hours=float(execution.get("session_timezone_offset_hours", settings.DEFAULT_SESSION_TIMEZONE_OFFSET_HOURS)),
        use_bar_spread=bool(execution.get("use_bar_spread", False)),
    )
    overrides = overrides or {}
    strategy = strat_cls(**{k: v for k, v in overrides.items() if k in strat_cls.params})
    bar_time_offset_hours = float(getattr(strat_cls, "bar_time_offset_hours", 0.0) or 0.0)
    df = shift_bar_times(df, bar_time_offset_hours)
    intrabar_df = shift_bar_times(intrabar_df, bar_time_offset_hours)
    return bt.run(
        strategy,
        df,
        intrabar_df=intrabar_df,
        date_from=date_from,
        date_to=date_to,
    )


def backtest_result_payload(result) -> dict:
    monthly = result.monthly_stats()
    if not monthly.empty:
        monthly = monthly.copy()
        if "profit" in monthly.columns and "total_profit" not in monthly.columns:
            monthly["total_profit"] = monthly["profit"]
        if "trades" in monthly.columns and "num_trades" not in monthly.columns:
            monthly["num_trades"] = monthly["trades"]

    deals = pd.DataFrame(
        [
            {
                "time": trade.closed_time,
                "profit": trade.net_profit,
                "direction": trade.direction.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "comment": trade.comment,
            }
            for trade in result.trades
        ]
    )
    if not deals.empty:
        deals = deals.set_index("time").sort_index()

    balance_curve = result.equity_curve.rename("balance").to_frame()
    if balance_curve.empty:
        balance_curve = pd.DataFrame({"balance": [result.initial_capital]})
    balance_curve.index.name = "time"

    summary = {
        "date_from": str(result.date_from.date()),
        "date_to": str(result.date_to.date()),
        "total_profit": float(result.net_profit),
        "sharpe_mean": float(result.sharpe_ratio),
        "win_rate": float(result.win_rate),
        "num_months": int(len(monthly)),
        "num_trades": int(result.n_trades),
        "num_decisions": int(len(result.decision_records)),
        "profit_factor": float(result.profit_factor),
        "max_drawdown_pct": float(result.max_drawdown[1]),
        "balance_dd_abs": float(result.balance_drawdown_maximal[0]),
        "balance_dd_pct": float(result.balance_drawdown_maximal[1]),
        "equity_dd_abs": float(result.equity_drawdown_maximal[0]),
        "equity_dd_pct": float(result.equity_drawdown_maximal[1]),
    }
    return {
        "monthly_df": monthly,
        "deals_df": deals,
        "balance_curve_df": balance_curve,
        "summary": summary,
    }


def available_backtest_symbols(timeframe: str = "H1") -> list[str]:
    symbols = available_symbols(timeframe)
    return symbols if symbols else ["HK50.cash", "XAUUSD", "US30.cash", "USDJPY"]
