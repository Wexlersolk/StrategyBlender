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
    df = load_bars(symbol, timeframe)
    intrabar_df = None
    if intrabar_steps > 1 and timeframe.upper() != "M1":
        intrabar_df = load_bars(symbol, "M1", date_from=date_from, date_to=date_to)
    execution = execution_config or {}
    bt = Backtester(
        initial_capital=100_000,
        lot_value=getattr(strat_cls, "lot_value", 1.0),
        intrabar_steps=intrabar_steps,
        overlay_policy=overlay_policy,
        commission_per_lot=float(execution.get("commission_per_lot", settings.DEFAULT_COMMISSION_PER_LOT)),
        spread_pips=float(execution.get("spread_pips", settings.DEFAULT_SPREAD_PIPS)),
        slippage_pips=float(execution.get("slippage_pips", settings.DEFAULT_SLIPPAGE_PIPS)),
    )
    overrides = overrides or {}
    strategy = strat_cls(**{k: v for k, v in overrides.items() if k in strat_cls.params})
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
