from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.backtest_service import default_intrabar_steps, run_backtest
from services.mt5_report_service import parse_mt5_order_rows, parse_mt5_trade_sequence
from services.native_strategy_lab import _load_runtime_strategy_class, get_native_strategy_record


def _first_mt5_order(report_path: str, strategy_tag: str) -> dict:
    orders = parse_mt5_order_rows(report_path)
    for row in orders:
        if row.get("state") != "filled":
            continue
        comment = str(row.get("comment", ""))
        if strategy_tag not in comment:
            continue
        return {
            "open_time": row.get("open_time"),
            "filled_time": row.get("time"),
            "type": row.get("type"),
            "price": row.get("price"),
            "stop_loss": row.get("stop_loss"),
            "take_profit": row.get("take_profit"),
            "comment": comment,
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug native pending-order lifecycle against MT5 first order.")
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--strategy-tag")
    parser.add_argument("--date-from", default="2020-01-01")
    parser.add_argument("--date-to", default="2020-03-01")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    record = get_native_strategy_record(args.strategy_id)
    if not record:
        raise SystemExit(f"Strategy not found in catalog: {args.strategy_id}")

    strat_cls = _load_runtime_strategy_class(record)
    result = run_backtest(
        strat_cls,
        symbol=record["symbol"],
        timeframe=record["timeframe"],
        date_from=args.date_from,
        date_to=args.date_to,
        overrides=record.get("params", {}),
        intrabar_steps=default_intrabar_steps(record["symbol"], record["timeframe"], 1),
    )

    events = result.debug_events or []
    native_pending = [event for event in events if event.get("kind", "").startswith("pending_")]
    mt5_trade = (parse_mt5_trade_sequence(args.report_path, strategy_tag=args.strategy_tag or args.strategy_id) or [{}])[0]
    mt5_order = _first_mt5_order(args.report_path, args.strategy_tag or args.strategy_id)

    payload = {
        "strategy_id": args.strategy_id,
        "native_events": native_pending[: args.limit],
        "first_native_trade": (
            {
                "entry_time": result.trades[0].opened_time.strftime("%Y.%m.%d %H:%M:%S"),
                "exit_time": result.trades[0].closed_time.strftime("%Y.%m.%d %H:%M:%S"),
                "entry_price": round(float(result.trades[0].entry_price), 5),
                "exit_price": round(float(result.trades[0].exit_price), 5),
                "reason": result.trades[0].close_reason.value,
                "comment": result.trades[0].comment,
            }
            if result.trades
            else {}
        ),
        "first_mt5_order": mt5_order,
        "first_mt5_trade": mt5_trade,
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
