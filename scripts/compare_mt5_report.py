from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.mt5_report_service import compare_mt5_to_native, mt5_correlation_acceptance, mt5_validation_acceptance, parse_mt5_report
from services.native_strategy_lab import get_native_strategy_record, update_native_strategy_mt5_validation


def main():
    parser = argparse.ArgumentParser(description="Compare an MT5 tester report to a native strategy evaluation.")
    parser.add_argument("report_path")
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    report = parse_mt5_report(args.report_path)
    record = get_native_strategy_record(args.strategy_id)
    if not record:
        raise SystemExit(f"Strategy not found in catalog: {args.strategy_id}")
    native = dict(record.get("evaluation", {}).get("backtest", {}))
    if not native:
        raise SystemExit(f"Strategy has no native backtest evaluation: {args.strategy_id}")

    comparison = compare_mt5_to_native(report, native)
    payload = {"mt5": report, "native": native, "comparison": comparison}
    if args.persist:
        policy = dict(record.get("evaluation", {}).get("promotion_policy", record.get("promotion_policy", {})))
        thresholds_ok = mt5_validation_acceptance(
            report,
            min_trades=int(policy.get("min_trades", 0)),
            min_profit_factor=float(policy.get("min_profit_factor", 0.0)),
            max_drawdown_pct=float(policy.get("max_drawdown_pct", 100.0)),
        )
        correlation_ok = mt5_correlation_acceptance(comparison)
        accepted = thresholds_ok and correlation_ok
        update_native_strategy_mt5_validation(
            args.strategy_id,
            report_path=args.report_path,
            metrics=report,
            comparison=comparison,
            accepted=accepted,
        )
        payload["mt5_validation"] = {
            "accepted": accepted,
            "thresholds_ok": thresholds_ok,
            "correlation_ok": correlation_ok,
        }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
