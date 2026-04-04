from __future__ import annotations

import argparse
import sys

from research.experiment_registry import export_experiment_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a saved experiment bundle as a zip archive.")
    parser.add_argument("--run-id", required=True, help="Saved experiment run_id to export.")
    parser.add_argument("--output", default="", help="Optional archive base path without extension.")
    args = parser.parse_args(argv)
    archive = export_experiment_report(args.run_id, output_path=args.output or None)
    print(str(archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
