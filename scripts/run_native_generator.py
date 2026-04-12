from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.native_strategy_lab import default_parameter_mutation_space, run_generator_session
from services.python_strategy_service import preset_by_id


def main():
    parser = argparse.ArgumentParser(description="Run a continuous native strategy generator session from a preset.")
    parser.add_argument("--preset-id", default="xau_breakout_session")
    parser.add_argument("--date-from", default="2021-01-01")
    parser.add_argument("--date-to", default="2025-01-01")
    parser.add_argument("--max-candidates", type=int, default=250)
    parser.add_argument("--intrabar-steps", type=int, default=1)
    parser.add_argument("--discovery-mode", choices=["conservative", "balanced", "exploratory"], default="conservative")
    parser.add_argument("--include-structural-mutations", action="store_true")
    parser.add_argument("--run-mc", action="store_true")
    parser.add_argument("--run-wfo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preset = preset_by_id(args.preset_id)
    if not preset:
        raise SystemExit(f"Unknown preset id: {args.preset_id}")

    payload = dict(preset["payload"])
    mutation_space = default_parameter_mutation_space(str(preset["template"]), payload)
    run = run_generator_session(
        template_name=str(preset["template"]),
        base_payload=payload,
        mutation_space=mutation_space,
        date_from=args.date_from,
        date_to=args.date_to,
        max_candidates=args.max_candidates,
        run_mc=bool(args.run_mc),
        run_wfo_checks=bool(args.run_wfo),
        intrabar_steps=args.intrabar_steps,
        include_structural_mutations=bool(args.include_structural_mutations),
        discovery_mode=str(args.discovery_mode),
        random_seed=int(args.seed),
    )
    print(json.dumps(run, indent=2, default=str))


if __name__ == "__main__":
    main()
