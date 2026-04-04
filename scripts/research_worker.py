from __future__ import annotations

import argparse
import sys

from services.research_worker import run_forever, run_once


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the StrategyBlender research worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Idle polling interval in seconds.")
    parser.add_argument("--owner-id", default="", help="Explicit authenticated owner to process jobs for.")
    parser.add_argument("--worker-id", default="", help="Stable worker identifier.")
    args = parser.parse_args(argv)
    if args.once:
        run_once(worker_id=args.worker_id or None, owner_id=args.owner_id or None)
    else:
        run_forever(
            poll_interval_seconds=float(args.poll_interval),
            worker_id=args.worker_id or None,
            owner_id=args.owner_id or None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
