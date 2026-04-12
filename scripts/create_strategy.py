from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.python_strategy_service import (
    available_template_names,
    compile_strategy_spec,
    persist_strategy_spec,
    strategy_spec_from_dict,
    strategy_spec_from_template,
)


def main():
    parser = argparse.ArgumentParser(description="Compile a StrategySpec JSON file or template payload into a Python strategy module.")
    parser.add_argument("spec_path", help="Path to a StrategySpec JSON file or template payload JSON file")
    parser.add_argument("--template", choices=available_template_names(), help="Interpret the JSON input as a template payload")
    parser.add_argument("--strategy-id", default=None, help="Optional suffix to make the generated slug unique")
    args = parser.parse_args()

    payload = json.loads(Path(args.spec_path).read_text(encoding="utf-8"))
    if args.template:
        spec = strategy_spec_from_template(args.template, payload)
    else:
        spec = strategy_spec_from_dict(payload)
    compiled = compile_strategy_spec(spec, strategy_id=args.strategy_id)
    paths = persist_strategy_spec(compiled)

    print(f"Generated strategy module: {compiled.strategy_module}")
    print(f"Python file: {paths['strategy_path']}")
    print(f"Spec file: {paths['spec_path']}")


if __name__ == "__main__":
    main()
