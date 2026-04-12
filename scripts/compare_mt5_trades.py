from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.backtest_service import default_intrabar_steps, run_backtest
from services.mt5_report_service import compare_trade_sequences, parse_mt5_trade_sequence
from services.native_strategy_lab import _load_runtime_strategy_class, get_native_strategy_record


def _native_trade_sequence(strategy_id: str, *, date_from: str, date_to: str) -> list[dict]:
    record = get_native_strategy_record(strategy_id)
    if not record:
        raise SystemExit(f"Strategy not found in catalog: {strategy_id}")
    strat_cls = _load_runtime_strategy_class(record)
    result = run_backtest(
        strat_cls,
        symbol=record["symbol"],
        timeframe=record["timeframe"],
        date_from=date_from,
        date_to=date_to,
        overrides=record.get("params", {}),
        intrabar_steps=default_intrabar_steps(record["symbol"], record["timeframe"], 1),
    )
    sequence = []
    for trade in result.trades:
        sequence.append(
            {
                "entry_time": str(trade.opened_time.strftime("%Y.%m.%d %H:%M:%S")),
                "exit_time": str(trade.closed_time.strftime("%Y.%m.%d %H:%M:%S")),
                "entry_side": trade.direction.value,
                "exit_reason": trade.close_reason.value,
                "entry_price": round(float(trade.entry_price), 5),
                "exit_price": round(float(trade.exit_price), 5),
                "comment": trade.comment,
            }
        )
    return sequence


def main():
    parser = argparse.ArgumentParser(description="Compare MT5 order sequence to native trade sequence.")
    parser.add_argument("report_path")
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--date-from", default="2020-01-01")
    parser.add_argument("--date-to", default="2026-03-23")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--strategy-tag")
    args = parser.parse_args()

    tag = args.strategy_tag or args.strategy_id
    native_trades = _native_trade_sequence(args.strategy_id, date_from=args.date_from, date_to=args.date_to)
    mt5_trades = parse_mt5_trade_sequence(args.report_path, strategy_tag=tag)
    comparison = compare_trade_sequences(native_trades, mt5_trades, limit=args.limit)
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
