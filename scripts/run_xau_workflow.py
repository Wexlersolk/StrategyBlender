from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.native_strategy_lab import (
    create_batch_run,
    default_parameter_mutation_space,
    default_promotion_policy_for_template,
    evaluate_batch_run,
    evaluate_native_strategy,
    register_generated_strategy,
)
from services.python_strategy_service import preset_by_id


def _strategy_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{stamp}"


def _print_json(title: str, payload: dict):
    print(f"\n{title}")
    print(json.dumps(payload, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Register, evaluate, and batch-generate XAU breakout strategies from the native preset."
    )
    parser.add_argument("--preset-id", default="xau_breakout_session", help="Preset id from config/strategy_presets")
    parser.add_argument("--date-from", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--date-to", default="2025-12-31", help="Backtest end date")
    parser.add_argument("--intrabar-steps", type=int, default=60, help="Intrabar replay steps; use 60 for M1-backed runs")
    parser.add_argument("--batch-limit", type=int, default=18, help="Number of mutation combinations to generate")
    parser.add_argument("--search-mode", choices=["grid", "progressive", "random"], default="progressive")
    parser.add_argument("--skip-base-mc", action="store_true", help="Skip Monte Carlo on the base strategy")
    parser.add_argument("--skip-base-wfo", action="store_true", help="Skip walk-forward checks on the base strategy")
    parser.add_argument("--evaluate-batch", action="store_true", help="Evaluate generated batch candidates after creation")
    parser.add_argument("--top-n", type=int, default=6, help="How many generated candidates to evaluate when --evaluate-batch is set")
    parser.add_argument("--run-mc", action="store_true", help="Run Monte Carlo during batch candidate evaluation")
    parser.add_argument("--run-wfo", action="store_true", help="Run walk-forward checks during batch candidate evaluation")
    args = parser.parse_args()

    preset = preset_by_id(args.preset_id)
    if not preset:
        raise SystemExit(f"Unknown preset id: {args.preset_id}")

    template_name = str(preset["template"])
    payload = dict(preset["payload"])
    strategy_id = _strategy_id(args.preset_id)

    record, _ = register_generated_strategy(
        template_name=template_name,
        payload=payload,
        strategy_id=strategy_id,
        origin="cli_xau_workflow",
        tags=["cli", "xau", template_name],
        promotion_policy=default_promotion_policy_for_template(template_name),
    )
    print(f"Registered base strategy: {record['strategy_id']} ({record['name']})")

    evaluation = evaluate_native_strategy(
        record["strategy_id"],
        date_from=args.date_from,
        date_to=args.date_to,
        intrabar_steps=args.intrabar_steps,
        run_mc=not args.skip_base_mc,
        run_wfo_checks=not args.skip_base_wfo,
    )
    _print_json("Base strategy evaluation", evaluation)

    mutation_space = default_parameter_mutation_space(template_name, payload)
    batch = create_batch_run(
        template_name=template_name,
        base_payload=payload,
        mutation_space=mutation_space,
        limit=args.batch_limit,
        search_mode=args.search_mode,
        promotion_policy=default_promotion_policy_for_template(template_name),
        include_structural_mutations=False,
    )
    print(f"\nCreated batch: {batch['batch_id']}")
    print(f"Generated candidates: {batch['generated_candidates']} unique / {batch['requested_candidates']} requested")

    if not args.evaluate_batch:
        print("\nBatch evaluation skipped. Re-run with --evaluate-batch to score candidates.")
        return

    evaluated = evaluate_batch_run(
        batch["batch_id"],
        date_from=args.date_from,
        date_to=args.date_to,
        top_n=args.top_n,
        run_mc=bool(args.run_mc),
        run_wfo_checks=bool(args.run_wfo),
        intrabar_steps=args.intrabar_steps,
    )
    ranked = [item for item in evaluated.get("candidates", []) if item.get("evaluated")]
    ranked = sorted(
        ranked,
        key=lambda item: (bool(item.get("accepted")), float(item.get("score", 0.0))),
        reverse=True,
    )
    _print_json("Top evaluated candidates", ranked[: min(5, len(ranked))])


if __name__ == "__main__":
    main()
