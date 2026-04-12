from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.mt5_export_service import export_native_strategy_to_mt5


def main():
    parser = argparse.ArgumentParser(description="Export a native strategy to an MT5 EA from its stored payload.")
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--output-name")
    args = parser.parse_args()

    payload = export_native_strategy_to_mt5(args.strategy_id, output_name=args.output_name)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
