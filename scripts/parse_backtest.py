"""
Batch MQL5 -> strategytester5 converter.

Usage:
    python scripts/parse_backtest.py
    python scripts/parse_backtest.py path/to/MyEA.mq5 --symbol XAUUSD --timeframe M15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert.mt5_to_python import convert_file


DEFAULT_INPUT_DIR = Path("strategies")
DEFAULT_OUTPUT_DIR = Path("convert/generated")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    return parser.parse_args()


def _iter_mq5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.mq5"))


def main():
    args = _parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    mq5_files = _iter_mq5_files(input_path)
    if not mq5_files:
        print(f"No .mq5 files found in {input_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(mq5_files)} file(s) into {output_dir}\n")

    for mq5_file in mq5_files:
        output_path = output_dir / f"{mq5_file.stem}.py"
        result = convert_file(
            input_path=mq5_file,
            output_path=output_path,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        print(f"{mq5_file} -> {output_path}")
        print(f"  inputs    : {len(result.inputs)}")
        print(f"  functions : {len(result.functions)}")
        if result.warnings:
            for warning in result.warnings:
                print(f"  warning   : {warning}")
        print("")


if __name__ == "__main__":
    main()
