"""
Heuristic MQL5 -> local Python review scaffold converter.

This is intentionally conservative: it extracts parameters and function bodies,
then emits a local Python review scaffold with translated stubs.
Complex MQL5 constructs still need manual review after conversion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionResult:
    strategy_name: str
    review_source: str
    inputs: dict
    functions: list[str]
    warnings: list[str]


def parse_mql5_inputs(source: str) -> dict:
    params = {}
    pattern = re.compile(
        r"input\s+(?:const\s+)?(\w+)\s+(\w+)\s*=\s*([^;]+);",
        re.MULTILINE,
    )
    skip_types = {"string", "bool", "color", "datetime"}

    for match in pattern.finditer(source):
        dtype, name, default = match.groups()
        if dtype in skip_types or dtype.startswith("ENUM_"):
            continue
        value = default.strip().replace(",", ".")
        try:
            if dtype in {"double", "float"}:
                params[name] = float(value)
            elif dtype in {"int", "uint", "long", "ulong"}:
                params[name] = int(value)
            else:
                params[name] = value
        except ValueError:
            params[name] = value
    return params


def _extract_function_blocks(source: str) -> list[tuple[str, str, str]]:
    signature_re = re.compile(
        r"(?P<return_type>\w[\w\s\*&<>:]*)\s+"
        r"(?P<name>\w+)\s*"
        r"\((?P<args>[^)]*)\)\s*\{",
        re.MULTILINE,
    )
    blocks: list[tuple[str, str, str]] = []
    idx = 0

    while True:
        match = signature_re.search(source, idx)
        if not match:
            break

        start = match.end() - 1
        depth = 0
        end = start
        while end < len(source):
            char = source[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1

        body = source[start + 1:end - 1]
        blocks.append((match.group("name"), match.group("args").strip(), body.strip()))
        idx = end

    return blocks


def _translate_condition(expr: str) -> str:
    translated = expr.strip()
    replacements = [
        ("&&", " and "),
        ("||", " or "),
        ("true", "True"),
        ("false", "False"),
        ("NULL", "None"),
    ]
    for old, new in replacements:
        translated = translated.replace(old, new)
    translated = re.sub(r"\bMathAbs\(", "abs(", translated)
    translated = re.sub(r"\bNormalizeDouble\(", "round(", translated)
    return translated


def _translate_body(body: str) -> list[str]:
    lines: list[str] = []
    indent = 1
    unsupported_hits: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = line.replace("\t", "    ")
        line = re.sub(r"//(.*)$", r"# \1", line)

        if line.startswith("}"):
            indent = max(1, indent - line.count("}"))
            line = line.lstrip("}").strip()
            if not line:
                continue

        if line.endswith("{"):
            head = line[:-1].strip()
            python_head = _translate_control_line(head)
            lines.append(("    " * indent) + python_head)
            indent += 1
            continue

        if line == "{":
            indent += 1
            continue

        if line == "}":
            indent = max(1, indent - 1)
            continue

        translated = _translate_statement(line.rstrip(";"))
        if translated.startswith("# UNSUPPORTED:"):
            unsupported_hits.append(translated)
        lines.append(("    " * indent) + translated)

    if not lines:
        return ["    pass"]

    if unsupported_hits:
        lines.insert(0, "    # Manual review required for unsupported MQL5 constructs below.")

    return lines


def _translate_control_line(line: str) -> str:
    if line.startswith("if"):
        expr = line[line.find("(") + 1: line.rfind(")")]
        return f"if {_translate_condition(expr)}:"
    if line.startswith("else if"):
        expr = line[line.find("(") + 1: line.rfind(")")]
        return f"elif {_translate_condition(expr)}:"
    if line == "else":
        return "else:"
    if line.startswith("while"):
        expr = line[line.find("(") + 1: line.rfind(")")]
        return f"while {_translate_condition(expr)}:"
    return f"# UNSUPPORTED BLOCK: {line}"


def _translate_statement(line: str) -> str:
    if line.startswith("if ") or line.startswith("if("):
        expr = line[line.find("(") + 1: line.rfind(")")]
        return f"if {_translate_condition(expr)}:"
    if line.startswith("else if"):
        expr = line[line.find("(") + 1: line.rfind(")")]
        return f"elif {_translate_condition(expr)}:"
    if line == "else":
        return "else:"
    if line.startswith("return"):
        return _translate_condition(line)
    if line.startswith("Print("):
        return f"print({_translate_condition(line[6:-1])})"
    if line.startswith("for(") or line.startswith("for "):
        return f"# UNSUPPORTED: {line}"
    if line.startswith("switch("):
        return f"# UNSUPPORTED: {line}"

    typed = re.match(
        r"^(?:int|double|float|bool|long|ulong|string|datetime)\s+([A-Za-z_]\w*)\s*=\s*(.+)$",
        line,
    )
    if typed:
        name, expr = typed.groups()
        return f"{name} = {_translate_condition(expr)}"

    assigned = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", line)
    if assigned:
        name, expr = assigned.groups()
        return f"{name} = {_translate_condition(expr)}"

    if line.startswith("#"):
        return line

    return _translate_condition(line)


def convert_mql5_to_python(
    source: str,
    strategy_name: str,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
) -> ConversionResult:
    params = parse_mql5_inputs(source)
    blocks = _extract_function_blocks(source)
    warnings: list[str] = []

    if not blocks:
        warnings.append("No MQL5 functions detected; emitted template-only scaffold.")

    if "OnTick" not in {name for name, _, _ in blocks}:
        warnings.append("OnTick() was not found; generated local review entrypoint is a placeholder.")

    python_lines = [
        '"""',
        "Auto-generated local review scaffold from MQL5 source.",
        "",
        "This file is for inspection and manual adaptation inside StrategyBlender.",
        "It is not a live-trading bridge and it is not intended to talk to MT5.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"STRATEGY_NAME = {strategy_name!r}",
        f"SYMBOL = {symbol!r}",
        f"TIMEFRAME = {timeframe!r}",
        "",
        f"PARAMS = {params!r}",
        "",
        "def pos_exists(*_args, **_kwargs) -> bool:",
        '    """Compatibility stub retained for manual review of converted control flow."""',
        "    return False",
        "",
    ]

    function_names: list[str] = []
    for name, args, body in blocks:
        function_names.append(name)
        py_name = {
            "OnInit": "on_init",
            "OnTick": "on_tick",
            "OnDeinit": "on_deinit",
        }.get(name, name)
        py_args = "reason=None" if name == "OnDeinit" else ""
        python_lines.append(f"def {py_name}({py_args}):")
        python_lines.append(f"    \"\"\"Converted from MQL5 {name}({args})\"\"\"")
        python_lines.extend(_translate_body(body))
        python_lines.append("")

    if "OnTick" not in function_names:
        python_lines.extend([
            "def on_tick_review():",
            "    # Manual implementation required: no OnTick() was found in the source EA.",
            "    pass",
            "",
        ])

    python_lines.extend([
        "def main():",
        "    # Manual review entrypoint only; StrategyBlender executes generated engine strategies instead.",
        "    if 'on_init' in globals():",
        "        on_init()",
        "    if 'on_tick' in globals():",
        "        on_tick()",
        "    elif 'on_tick_review' in globals():",
        "        on_tick_review()",
        "",
        "if __name__ == '__main__':",
        "    main()",
        "",
    ])

    if any(
        token in source
        for token in ["iCustom(", "CopyBuffer(", "SetIndexBuffer(", "IndicatorCreate(", "switch("]
    ):
        warnings.append(
            "Complex indicator/buffer logic was detected. Review the generated Python before running it."
        )

    return ConversionResult(
        strategy_name=strategy_name,
        review_source="\n".join(python_lines),
        inputs=params,
        functions=function_names,
        warnings=warnings,
    )


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
) -> ConversionResult:
    src_path = Path(input_path)
    out_path = Path(output_path)
    source = src_path.read_text(encoding="utf-8", errors="replace")
    result = convert_mql5_to_python(
        source=source,
        strategy_name=src_path.stem,
        symbol=symbol,
        timeframe=timeframe,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.review_source, encoding="utf-8")
    return result
