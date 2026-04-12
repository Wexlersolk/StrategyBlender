from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / "strategies" / "specs"
ENGINE_STRATEGY_DIR = ROOT / "strategies" / "generated"
PRESET_DIR = ROOT / "config" / "strategy_presets"


SUPPORTED_INDICATORS = {
    "ema": "ema",
    "sma": "sma",
    "wma": "wma",
    "smma": "smma",
    "osma": "osma",
    "bollinger_bands": "bollinger_bands",
    "select_price": "select_price",
    "sq_adx": "sq_adx",
    "sq_atr": "sq_atr",
    "sq_bb_width_ratio": "sq_bb_width_ratio",
    "sq_fibo": "sq_fibo",
    "sq_heiken_ashi": "sq_heiken_ashi",
    "sq_highest": "sq_highest",
    "sq_hull_moving_average": "sq_hull_moving_average",
    "sq_ichimoku": "sq_ichimoku",
    "sq_keltner_channel": "sq_keltner_channel",
    "sq_lowest": "sq_lowest",
    "sq_parabolic_sar": "sq_parabolic_sar",
    "sq_session_ohlc": "sq_session_ohlc",
    "sq_stochastic": "sq_stochastic",
    "sq_ulcer_index": "sq_ulcer_index",
    "sq_vwap": "sq_vwap",
    "sq_wave_trend": "sq_wave_trend",
    "sq_wpr": "sq_wpr",
    "SqADX": "sq_adx",
    "SqATR": "sq_atr",
    "SqBBWidthRatio": "sq_bb_width_ratio",
    "SqFibo": "sq_fibo",
    "SqHeikenAshi": "sq_heiken_ashi",
    "SqHighest": "sq_highest",
    "SqHullMovingAverage": "sq_hull_moving_average",
    "SqIchimoku": "sq_ichimoku",
    "SqKeltnerChannel": "sq_keltner_channel",
    "SqLowest": "sq_lowest",
    "SqParabolicSAR": "sq_parabolic_sar",
    "SqSessionOHLC": "sq_session_ohlc",
    "SqStochastic": "sq_stochastic",
    "SqUlcerIndex": "sq_ulcer_index",
    "SqVWAP": "sq_vwap",
    "SqWaveTrend": "sq_wave_trend",
    "SqWPR": "sq_wpr",
}

SERIES_ARG_RENDERERS = {
    "series": lambda value: f'df[{value!r}]',
    "open_": lambda value: f'df[{value!r}]',
    "high": lambda value: f'df[{value!r}]',
    "low": lambda value: f'df[{value!r}]',
    "close": lambda value: f'df[{value!r}]',
    "volume": lambda value: f'df[{value!r}]',
    "df": lambda _value: "df",
}

COMPUTE_ALLOWED_NAMES = {
    "df",
    "self",
    "pd",
    "np",
    "abs",
    "min",
    "max",
    "int",
    "float",
    "bool",
    "str",
    "sma",
    "_count_compare",
    "_cross_above",
    "_cross_below",
    "_falling",
    "_rising",
    "_compare_window",
    "_sq_count_compare",
    "_hhmm",
    "_in_time_window",
}

ON_BAR_ALLOWED_NAMES = COMPUTE_ALLOWED_NAMES | {
    "ctx",
    "i",
}


@dataclass
class IndicatorSpec:
    name: str
    kind: str
    args: dict[str, Any]
    outputs: dict[str, str] | None = None


@dataclass
class SeriesSpec:
    name: str
    expression: str


@dataclass
class EntryRuleSpec:
    name: str
    side: str
    order_type: str
    when: str
    lots: str
    stop_loss: str
    take_profit: str
    price: str | None = None
    expiry_bars: int = 1
    comment: str = ""
    trail_dist: str | None = None
    trail_activation: str | None = None
    exit_after_bars: int | str | None = None


@dataclass
class ExitRuleSpec:
    name: str
    when: str
    action: str = "close_all"


@dataclass
class StrategySpec:
    name: str
    symbol: str
    timeframe: str
    params: dict[str, Any] = field(default_factory=dict)
    indicators: list[IndicatorSpec] = field(default_factory=list)
    series: list[SeriesSpec] = field(default_factory=list)
    entries: list[EntryRuleSpec] = field(default_factory=list)
    exits: list[ExitRuleSpec] = field(default_factory=list)
    min_bars: int = 0
    lot_value: float = 1.0
    bar_time_offset_hours: float = 0.0
    allow_entries_when_position_open: bool = False
    description: str = ""


@dataclass
class CompiledStrategy:
    spec: StrategySpec
    strategy_slug: str
    strategy_class: str
    strategy_module: str
    strategy_path: str
    spec_path: str
    source: str


@dataclass
class StrategyTemplate:
    name: str
    description: str
    build: Any
    family: str = "general"
    regime: str = ""
    supported_symbols: tuple[str, ...] = ()
    supported_timeframes: tuple[str, ...] = ()
    preferred_timeframes: tuple[str, ...] = ()
    sort_order: int = 100


TARGET_BUILD_UNIVERSE: tuple[str, ...] = (
    "USDJPY",
    "EURUSD",
    "GBPJPY",
    "XAUUSD",
    "US30.cash",
    "US500.cash",
    "NAS100.cash",
    "HK50.cash",
    "GER40.cash",
    "UK100.cash",
)

SYMBOL_ALIASES = {
    "US30": "US30.cash",
    "US500": "US500.cash",
    "US100": "NAS100.cash",
    "NAS100": "NAS100.cash",
    "HK50": "HK50.cash",
    "GER40": "GER40.cash",
    "UK100": "UK100.cash",
}


def canonical_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip()
    return SYMBOL_ALIASES.get(raw.upper(), raw)


def _template_matches_symbol(template: StrategyTemplate, symbol: str) -> bool:
    canonical = canonical_symbol(symbol)
    return not template.supported_symbols or canonical in template.supported_symbols


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "strategy"


def class_name_from_slug(slug: str) -> str:
    base = "".join(part.capitalize() for part in slug.split("_")) or "GeneratedStrategy"
    if not base[0].isalpha():
        base = f"Generated{base}"
    return base


class ExpressionValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: set[str]):
        self.allowed_names = allowed_names

    def visit_Name(self, node: ast.Name):
        if node.id not in self.allowed_names:
            raise ValueError(f"Unsupported name in expression: {node.id}")

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed in strategy expressions")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id not in self.allowed_names:
            raise ValueError(f"Unsupported function in expression: {node.func.id}")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST):
        allowed_nodes = (
            ast.Expression,
            ast.BoolOp,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Attribute,
            ast.Subscript,
            ast.Slice,
            ast.Constant,
            ast.Tuple,
            ast.List,
            ast.Dict,
            ast.keyword,
            ast.And,
            ast.Or,
            ast.Not,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
            ast.Invert,
            ast.USub,
            ast.UAdd,
        )
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        super().generic_visit(node)


def _validate_expression(expr: str, allowed_names: set[str]):
    parsed = ast.parse(expr, mode="eval")
    ExpressionValidator(allowed_names).visit(parsed)


def _normalize_indicator_kind(kind: str) -> str:
    if kind not in SUPPORTED_INDICATORS:
        supported = ", ".join(sorted(SUPPORTED_INDICATORS))
        raise ValueError(f"Unsupported indicator kind: {kind}. Supported values: {supported}")
    return SUPPORTED_INDICATORS[kind]


def _render_code_value(value: Any) -> str:
    if isinstance(value, dict) and set(value) == {"param"}:
        return f"self.p({value['param']!r})"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _render_indicator_arg(arg_name: str, value: Any) -> str:
    if isinstance(value, dict) and set(value) == {"param"}:
        return f"self.p({value['param']!r})"
    if arg_name in SERIES_ARG_RENDERERS:
        if not isinstance(value, str):
            raise ValueError(f"Indicator argument {arg_name} must reference a dataframe column name")
        return SERIES_ARG_RENDERERS[arg_name](value)
    return _render_code_value(value)


def _render_indicator(spec: IndicatorSpec) -> tuple[list[str], str]:
    func_name = _normalize_indicator_kind(spec.kind)
    args = ", ".join(
        _render_indicator_arg(arg_name, arg_value)
        for arg_name, arg_value in spec.args.items()
    )
    lines: list[str] = []

    if spec.outputs:
        temp_name = f"_{spec.name}_result"
        lines.append(f"        {temp_name} = {func_name}({args})")
        for source_key, column_name in spec.outputs.items():
            lines.append(f'        df[{column_name!r}] = {temp_name}[{source_key!r}]')
    else:
        lines.append(f'        df[{spec.name!r}] = {func_name}({args})')

    return lines, func_name


def _render_optional_numeric(value: int | str | None) -> str:
    if value is None:
        return "0"
    if isinstance(value, int):
        return str(value)
    _validate_expression(value, ON_BAR_ALLOWED_NAMES)
    return value


def _serialize_spec(spec: StrategySpec) -> dict[str, Any]:
    data = asdict(spec)
    return data


def strategy_spec_from_dict(payload: dict[str, Any]) -> StrategySpec:
    return StrategySpec(
        name=payload["name"],
        symbol=payload["symbol"],
        timeframe=payload["timeframe"],
        params=payload.get("params", {}),
        indicators=[IndicatorSpec(**item) for item in payload.get("indicators", [])],
        series=[SeriesSpec(**item) for item in payload.get("series", [])],
        entries=[EntryRuleSpec(**item) for item in payload.get("entries", [])],
        exits=[ExitRuleSpec(**item) for item in payload.get("exits", [])],
        min_bars=int(payload.get("min_bars", 0)),
        lot_value=float(payload.get("lot_value", 1.0)),
        allow_entries_when_position_open=bool(payload.get("allow_entries_when_position_open", False)),
        description=payload.get("description", ""),
    )


def _required(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in payload:
        return payload[key]
    if default is not None:
        return default
    raise ValueError(f"Missing required template field: {key}")


def _param_payload(_payload: dict[str, Any], key: str) -> dict[str, str]:
    return {"param": str(key)}


def build_trend_pullback_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    direction = payload.get("direction", "long")
    if direction not in {"long", "short"}:
        raise ValueError("trend_pullback.direction must be 'long' or 'short'")

    fast_ema = str(payload.get("fast_ema_param", "FastEMA"))
    slow_ema = str(payload.get("slow_ema_param", "SlowEMA"))
    atr_period = str(payload.get("atr_period_param", "ATRPeriod"))
    lots = str(payload.get("lots_param", "mmLots"))
    stop_loss = str(payload.get("stop_loss_atr_param", "StopLossATR"))
    take_profit = str(payload.get("take_profit_atr_param", "ProfitTargetATR"))
    wt_channel = str(payload.get("wt_channel_param", "WaveTrendChannel"))
    wt_average = str(payload.get("wt_average_param", "WaveTrendAverage"))
    exit_after = str(payload.get("exit_after_bars_param", "ExitAfterBars"))

    params = dict(payload.get("params", {}))
    params.setdefault(fast_ema, 20)
    params.setdefault(slow_ema, 50)
    params.setdefault(atr_period, 14)
    params.setdefault(lots, 1.0)
    params.setdefault(stop_loss, 2.0)
    params.setdefault(take_profit, 3.0)
    params.setdefault(wt_channel, 9)
    params.setdefault(wt_average, 21)
    params.setdefault(exit_after, 12)

    long_side = direction == "long"
    trend_expr = 'df["ema_fast"] > df["ema_slow"]' if long_side else 'df["ema_fast"] < df["ema_slow"]'
    wave_expr = 'df["wt_main"] > df["wt_signal"]' if long_side else 'df["wt_main"] < df["wt_signal"]'
    pullback_expr = 'df["close"] <= df["ema_fast"]' if long_side else 'df["close"] >= df["ema_fast"]'
    stop_loss_expr = (
        f'ctx.open - float(df["atr"].iloc[i - 1]) * float(self.p("{stop_loss}"))'
        if long_side
        else f'ctx.open + float(df["atr"].iloc[i - 1]) * float(self.p("{stop_loss}"))'
    )
    take_profit_expr = (
        f'ctx.open + float(df["atr"].iloc[i - 1]) * float(self.p("{take_profit}"))'
        if long_side
        else f'ctx.open - float(df["atr"].iloc[i - 1]) * float(self.p("{take_profit}"))'
    )

    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 20)),
        lot_value=float(payload.get("lot_value", 1.0)),
        description=payload.get("description", "Trend-pullback template generated from the SQ-like rule engine."),
        indicators=[
            IndicatorSpec("ema_fast", "ema", {"series": "close", "period": _param_payload(params, fast_ema)}),
            IndicatorSpec("ema_slow", "ema", {"series": "close", "period": _param_payload(params, slow_ema)}),
            IndicatorSpec(
                "wavetrend",
                "sq_wave_trend",
                {"high": "high", "low": "low", "close": "close", "channel_length": _param_payload(params, wt_channel), "average_length": _param_payload(params, wt_average)},
                outputs={"main": "wt_main", "signal": "wt_signal"},
            ),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": _param_payload(params, atr_period)}),
        ],
        series=[
            SeriesSpec(
                "entry_signal",
                f'({trend_expr}) & ({pullback_expr}) & ({wave_expr})',
            ),
        ],
        entries=[
            EntryRuleSpec(
                name=f"{direction}_entry",
                side=direction,
                order_type="market",
                when='bool(df["entry_signal"].iloc[i])',
                lots=f'float(self.p("{lots}"))',
                stop_loss=stop_loss_expr,
                take_profit=take_profit_expr,
                comment=f"template_{direction}",
                exit_after_bars=f'int(self.p("{exit_after}"))',
            )
        ],
    )


def build_breakout_confirm_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    direction = payload.get("direction", "long")
    if direction not in {"long", "short"}:
        raise ValueError("breakout_confirm.direction must be 'long' or 'short'")

    highest_period = str(payload.get("highest_period_param", "HighestPeriod"))
    lowest_period = str(payload.get("lowest_period_param", "LowestPeriod"))
    atr_period = str(payload.get("atr_period_param", "ATRPeriod"))
    lots = str(payload.get("lots_param", "mmLots"))
    stop_loss = str(payload.get("stop_loss_atr_param", "StopLossATR"))
    take_profit = str(payload.get("take_profit_atr_param", "ProfitTargetATR"))
    max_distance = str(payload.get("max_distance_pct_param", "MaxDistancePct"))
    bbwr_period = str(payload.get("bbwr_period_param", "BBWRPeriod"))
    bbwr_min = str(payload.get("bbwr_min_param", "BBWRMin"))

    params = dict(payload.get("params", {}))
    params.setdefault(highest_period, 20)
    params.setdefault(lowest_period, 20)
    params.setdefault(atr_period, 14)
    params.setdefault(lots, 1.0)
    params.setdefault(stop_loss, 2.0)
    params.setdefault(take_profit, 3.0)
    params.setdefault(max_distance, 3.0)
    params.setdefault(bbwr_period, 20)
    params.setdefault(bbwr_min, 0.0)

    long_side = direction == "long"
    level_col = "highest_level" if long_side else "lowest_level"
    price_expr = f'float(df["{level_col}"].iloc[i - 1])'
    distance_check = f'abs({price_expr} - ctx.open) / max(abs(ctx.open), 1e-9) * 100.0 <= float(self.p("{max_distance}"))'
    signal_expr = (
        f'(df["close"] > df["{level_col}"].shift(1)) & (df["bbwr"] >= float(self.p("{bbwr_min}")))'
        if long_side
        else f'(df["close"] < df["{level_col}"].shift(1)) & (df["bbwr"] >= float(self.p("{bbwr_min}")))'
    )
    stop_loss_expr = (
        f'{price_expr} - float(df["atr"].iloc[i - 1]) * float(self.p("{stop_loss}"))'
        if long_side
        else f'{price_expr} + float(df["atr"].iloc[i - 1]) * float(self.p("{stop_loss}"))'
    )
    take_profit_expr = (
        f'{price_expr} + float(df["atr"].iloc[i - 1]) * float(self.p("{take_profit}"))'
        if long_side
        else f'{price_expr} - float(df["atr"].iloc[i - 1]) * float(self.p("{take_profit}"))'
    )

    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 25)),
        lot_value=float(payload.get("lot_value", 1.0)),
        description=payload.get("description", "Breakout confirmation template generated from the SQ-like rule engine."),
        indicators=[
            IndicatorSpec("highest_level", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": _param_payload(params, highest_period), "mode": 2}),
            IndicatorSpec("lowest_level", "sq_lowest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": _param_payload(params, lowest_period), "mode": 3}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": _param_payload(params, atr_period)}),
            IndicatorSpec("bbwr", "sq_bb_width_ratio", {"series": "close", "period": _param_payload(params, bbwr_period), "deviations": 2.0}),
        ],
        series=[
            SeriesSpec("entry_signal", signal_expr),
        ],
        entries=[
            EntryRuleSpec(
                name=f"{direction}_breakout",
                side=direction,
                order_type="buy_stop" if long_side else "sell_stop",
                when=f'bool(df["entry_signal"].iloc[i]) and ({distance_check})',
                price=price_expr,
                lots=f'float(self.p("{lots}"))',
                stop_loss=stop_loss_expr,
                take_profit=take_profit_expr,
                comment=f"template_breakout_{direction}",
                expiry_bars=int(payload.get("expiry_bars", 3)),
            )
        ],
    )


def build_reversion_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")

    bb_period = str(payload.get("bb_period_param", "BBPeriod"))
    bb_dev = str(payload.get("bb_dev_param", "BBDev"))
    wpr_period = str(payload.get("wpr_period_param", "WPRPeriod"))
    atr_period = str(payload.get("atr_period_param", "ATRPeriod"))
    lots = str(payload.get("lots_param", "mmLots"))
    stop_loss = str(payload.get("stop_loss_atr_param", "StopLossATR"))
    take_profit = str(payload.get("take_profit_atr_param", "ProfitTargetATR"))
    oversold = str(payload.get("oversold_param", "WPROversold"))
    overbought = str(payload.get("overbought_param", "WPROverbought"))

    params = dict(payload.get("params", {}))
    params.setdefault(bb_period, 20)
    params.setdefault(bb_dev, 2.0)
    params.setdefault(wpr_period, 14)
    params.setdefault(atr_period, 14)
    params.setdefault(lots, 1.0)
    params.setdefault(stop_loss, 1.5)
    params.setdefault(take_profit, 2.0)
    params.setdefault(oversold, -80.0)
    params.setdefault(overbought, -20.0)

    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 20)),
        lot_value=float(payload.get("lot_value", 1.0)),
        description=payload.get("description", "Mean-reversion template generated from the SQ-like rule engine."),
        indicators=[
            IndicatorSpec(
                "bollinger",
                "bollinger_bands",
                {"open_": "open", "high": "high", "low": "low", "close": "close", "period": _param_payload(params, bb_period), "deviations": _param_payload(params, bb_dev), "mode": 0},
                outputs={"upper": "bb_upper", "lower": "bb_lower", "middle": "bb_middle"},
            ),
            IndicatorSpec("wpr", "sq_wpr", {"high": "high", "low": "low", "close": "close", "period": _param_payload(params, wpr_period)}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": _param_payload(params, atr_period)}),
        ],
        series=[
            SeriesSpec("long_signal", f'(df["close"] < df["bb_lower"]) & (df["wpr"] <= float(self.p("{oversold}")))'),
            SeriesSpec("short_signal", f'(df["close"] > df["bb_upper"]) & (df["wpr"] >= float(self.p("{overbought}")))'),
        ],
        entries=[
            EntryRuleSpec(
                name="long_reversion",
                side="long",
                order_type="market",
                when='bool(df["long_signal"].iloc[i])',
                lots=f'float(self.p("{lots}"))',
                stop_loss=f'ctx.open - float(df["atr"].iloc[i - 1]) * float(self.p("{stop_loss}"))',
                take_profit=f'ctx.open + float(df["atr"].iloc[i - 1]) * float(self.p("{take_profit}"))',
                comment="template_reversion_long",
            ),
            EntryRuleSpec(
                name="short_reversion",
                side="short",
                order_type="market",
                when='bool(df["short_signal"].iloc[i])',
                lots=f'float(self.p("{lots}"))',
                stop_loss=f'ctx.open + float(df["atr"].iloc[i - 1]) * float(self.p("{stop_loss}"))',
                take_profit=f'ctx.open - float(df["atr"].iloc[i - 1]) * float(self.p("{take_profit}"))',
                comment="template_reversion_short",
            ),
        ],
    )


def build_us30_sqx_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "WPRPeriod": 75,
        "StochK": 14,
        "StochD": 3,
        "StochSlow": 3,
        "HighestPeriod": 105,
        "LowestPeriod": 105,
        "ATRPeriod1": 14,
        "ATRPeriod2": 29,
        "mmLots": 1.0,
        "StopLossCoef1": 2.0,
        "ProfitTargetCoef1": 2.5,
        "StopLossCoef2": 2.0,
        "ProfitTargetCoef2": 2.5,
        "TrailingStopCoef1": 1.0,
        "MaxDistancePct": 6.0,
        "SignalTimeRangeFrom": "00:00",
        "SignalTimeRangeTo": "23:59",
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    long_window = int(payload.get("wpr_quantile_window", 250))
    long_shift = int(payload.get("wpr_shift", 10))
    short_bars = int(payload.get("short_falling_bars", 12))
    short_shift = int(payload.get("short_falling_shift", 3))
    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=max(40, long_window),
        lot_value=float(payload.get("lot_value", 1.0)),
        description=payload.get("description", "US30 SQX WPR/Stochastic breakout family."),
        indicators=[
            IndicatorSpec("stoch", "sq_stochastic", {"high": "high", "low": "low", "close": "close", "k_period": {"param": "StochK"}, "d_period": {"param": "StochD"}, "slowing": {"param": "StochSlow"}, "ma_method": "wma", "price_mode": "lowhigh"}, outputs={"signal": "stoch_signal"}),
            IndicatorSpec("wpr", "sq_wpr", {"high": "high", "low": "low", "close": "close", "period": {"param": "WPRPeriod"}}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod1"}}),
            IndicatorSpec("atr_2", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod2"}}),
            IndicatorSpec("highest", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "HighestPeriod"}, "mode": 0}),
            IndicatorSpec("lowest", "sq_lowest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "LowestPeriod"}, "mode": 0}),
        ],
        series=[
            SeriesSpec("wpr_ref", f'df["wpr"].shift({long_shift})'),
            SeriesSpec("long_signal", f'df["wpr_ref"] <= df["wpr_ref"].rolling({long_window}, min_periods={long_window}).quantile(0.669)'),
            SeriesSpec("short_signal", f'_falling(df["stoch_signal"], {short_bars}, {short_shift})'),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
        ],
        entries=[
            EntryRuleSpec(
                name="us30_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["long_signal"].iloc[i]) and not bool(df["short_signal"].iloc[i])',
                price='float(df["highest"].iloc[i - 1])',
                lots='float(self.p("mmLots"))',
                stop_loss='float(df["highest"].iloc[i - 1]) - float(self.p("StopLossCoef1")) * float(df["atr"].iloc[i - 1])',
                take_profit='float(df["highest"].iloc[i - 1]) + float(self.p("ProfitTargetCoef1")) * float(df["atr"].iloc[i - 1])',
                comment="sqx_us30_long",
                expiry_bars=25,
            ),
            EntryRuleSpec(
                name="us30_short",
                side="short",
                order_type="sell_stop",
                when='bool(df["short_signal"].iloc[i]) and not bool(df["long_signal"].iloc[i]) and (abs(float(df["lowest"].iloc[i - 1]) - ctx.open) / max(abs(ctx.open), 1e-9) * 100.0 <= float(self.p("MaxDistancePct")))',
                price='float(df["lowest"].iloc[i - 1])',
                lots='float(self.p("mmLots"))',
                stop_loss='float(df["lowest"].iloc[i - 1]) + float(self.p("StopLossCoef2")) * float(df["atr"].iloc[i - 1])',
                take_profit='float(df["lowest"].iloc[i - 1]) - float(self.p("ProfitTargetCoef2")) * float(df["atr"].iloc[i - 1])',
                trail_dist='float(self.p("TrailingStopCoef1")) * float(df["atr_2"].iloc[i - 1])',
                comment="sqx_us30_short",
                expiry_bars=18,
            ),
        ],
    )


def build_xau_sqx_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "StochK": 21,
        "StochD": 7,
        "StochSlow": 7,
        "OsmaFast": 8,
        "OsmaSlow": 17,
        "OsmaSignal": 9,
        "BBPeriod": 167,
        "BBDev": 1.8,
        "ATRPeriod1": 200,
        "ATRPeriod2": 55,
        "mmLots": 1.0,
        "SignalTimeRangeFrom": "00:00",
        "SignalTimeRangeTo": "23:59",
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 20)),
        lot_value=float(payload.get("lot_value", 100.0)),
        description=payload.get("description", "XAU SQX Stochastic/OsMA/Bollinger family."),
        indicators=[
            IndicatorSpec("stoch", "sq_stochastic", {"high": "high", "low": "low", "close": "close", "k_period": {"param": "StochK"}, "d_period": {"param": "StochD"}, "slowing": {"param": "StochSlow"}, "ma_method": "sma", "price_mode": "lowhigh"}, outputs={"main": "stoch_main"}),
            IndicatorSpec("osma", "osma", {"open_": "open", "high": "high", "low": "low", "close": "close", "fast_period": {"param": "OsmaFast"}, "slow_period": {"param": "OsmaSlow"}, "signal_period": {"param": "OsmaSignal"}, "mode": 0}),
            IndicatorSpec("bb", "bollinger_bands", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "BBPeriod"}, "deviations": {"param": "BBDev"}, "mode": 1}, outputs={"upper": "bb_upper", "lower": "bb_lower"}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod1"}}),
            IndicatorSpec("atr_2", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod2"}}),
        ],
        series=[
            SeriesSpec("long_signal", '_compare_window(df["stoch_main"], df["osma"], 6, 5, True, True) & (df["open"].shift(3) < df["bb_upper"].shift(3)) & (df["open"].shift(2) > df["bb_upper"].shift(3))'),
            SeriesSpec("short_signal", '_compare_window(df["stoch_main"], df["osma"], 6, 5, True, False) & (df["open"].shift(3) > df["bb_lower"].shift(3)) & (df["open"] < df["bb_lower"].shift(3))'),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
        ],
        entries=[
            EntryRuleSpec(
                name="xau_long",
                side="long",
                order_type="market",
                when='bool(df["long_signal"].iloc[i]) and _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))',
                lots='float(self.p("mmLots"))',
                stop_loss='ctx.open * (1.0 - 0.008)',
                take_profit='ctx.open * (1.0 + 0.015)',
                trail_dist='70.0',
                trail_activation='4.0 * float(df["atr"].iloc[i - 1])',
                exit_after_bars=36,
                comment="sqx_xau_long",
            ),
            EntryRuleSpec(
                name="xau_short",
                side="short",
                order_type="market",
                when='bool(df["short_signal"].iloc[i]) and _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))',
                lots='float(self.p("mmLots"))',
                stop_loss='ctx.open * (1.0 + 0.010)',
                take_profit='ctx.open * (1.0 - 0.046)',
                trail_dist='2.7 * float(df["atr_2"].iloc[i - 1])',
                trail_activation='70.0',
                comment="sqx_xau_short",
            ),
        ],
    )


def build_xau_breakout_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "HighestPeriod": 24,
        "LowestPeriod": 24,
        "ATRPeriod": 14,
        "BBWRPeriod": 24,
        "BBWRMin": 0.015,
        "EntryBufferATR": 0.15,
        "StopLossATR": 1.4,
        "ProfitTargetATR": 2.8,
        "MaxDistancePct": 3.5,
        "mmLots": 1.0,
        "SignalTimeRangeFrom": "06:00",
        "SignalTimeRangeTo": "18:00",
        "ExitAfterBars": 18,
    }
    for key, value in defaults.items():
        params.setdefault(key, value)

    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 30)),
        lot_value=float(payload.get("lot_value", 100.0)),
        description=payload.get("description", "XAU volatility breakout with ATR buffer and session gating."),
        indicators=[
            IndicatorSpec("highest_level", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "HighestPeriod"}, "mode": 2}),
            IndicatorSpec("lowest_level", "sq_lowest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "LowestPeriod"}, "mode": 3}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod"}}),
            IndicatorSpec("bbwr", "sq_bb_width_ratio", {"series": "close", "period": {"param": "BBWRPeriod"}, "deviations": 2.0}),
        ],
        series=[
            SeriesSpec("volatility_ok", 'df["bbwr"] >= float(self.p("BBWRMin"))'),
            SeriesSpec("long_signal", 'df["close"] > df["highest_level"].shift(1)'),
            SeriesSpec("short_signal", 'df["close"] < df["lowest_level"].shift(1)'),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
        ],
        entries=[
            EntryRuleSpec(
                name="xau_breakout_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["volatility_ok"].shift(1).eq(True).iloc[i]) and _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo"))) and bool(df["long_signal"].shift(1).eq(True).iloc[i]) and (abs(float(ctx.open) - float(df["highest_level"].shift(1).iloc[i])) / max(float(ctx.open), 1e-9) * 100.0 <= float(self.p("MaxDistancePct")))',
                price='float(df["highest_level"].shift(1).iloc[i]) + (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))',
                lots='float(self.p("mmLots"))',
                stop_loss='(float(df["highest_level"].shift(1).iloc[i]) + (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))) - (float(df["atr"].iloc[i - 1]) * float(self.p("StopLossATR")))',
                take_profit='(float(df["highest_level"].shift(1).iloc[i]) + (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))) + (float(df["atr"].iloc[i - 1]) * float(self.p("ProfitTargetATR")))',
                expiry_bars=3,
                exit_after_bars='int(self.p("ExitAfterBars"))',
                comment="xau_breakout_long",
            ),
            EntryRuleSpec(
                name="xau_breakout_short",
                side="short",
                order_type="sell_stop",
                when='bool(df["volatility_ok"].shift(1).eq(True).iloc[i]) and _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo"))) and bool(df["short_signal"].shift(1).eq(True).iloc[i]) and (abs(float(ctx.open) - float(df["lowest_level"].shift(1).iloc[i])) / max(float(ctx.open), 1e-9) * 100.0 <= float(self.p("MaxDistancePct")))',
                price='float(df["lowest_level"].shift(1).iloc[i]) - (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))',
                lots='float(self.p("mmLots"))',
                stop_loss='(float(df["lowest_level"].shift(1).iloc[i]) - (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))) + (float(df["atr"].iloc[i - 1]) * float(self.p("StopLossATR")))',
                take_profit='(float(df["lowest_level"].shift(1).iloc[i]) - (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))) - (float(df["atr"].iloc[i - 1]) * float(self.p("ProfitTargetATR")))',
                expiry_bars=3,
                exit_after_bars='int(self.p("ExitAfterBars"))',
                comment="xau_breakout_short",
            ),
        ],
    )


def build_sqx_xau_highest_breakout_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "HighestPeriod": 245,
        "LowestPeriod": 245,
        "PullbackSignalPeriod": 10,
        "FastSMAPeriod": 10,
        "SignalMAPeriod": 57,
        "LWMAPeriod": 54,
        "ShortLWMAPeriod": 14,
        "SMMAPeriod": 30,
        "HAFloorMAPeriod": 67,
        "WaveTrendChannel": 9,
        "WaveTrendAverage": 21,
        "ATRLongStopPeriod": 14,
        "ATRLongTargetPeriod": 14,
        "ATRLongTrailPeriod": 40,
        "ATRShortStopPeriod": 14,
        "ATRShortTargetPeriod": 19,
        "LongStopATR": 2.0,
        "LongTargetATR": 3.5,
        "ShortStopATR": 1.5,
        "ShortTargetATR": 4.8,
        "LongExpiryBars": 10,
        "ShortExpiryBars": 18,
        "LongTrailATR": 1.0,
        "LongTrailActivationATR": 0.0,
        "ShortTrailATR": 0.0,
        "ShortTrailActivationATR": 0.0,
        "HourQuantileLookback": 809,
        "HourQuantileThreshold": 66.5,
        "MaxDistancePct": 6.0,
        "mmLots": 1.0,
        "SignalTimeRangeFrom": "00:00",
        "SignalTimeRangeTo": "23:59",
    }
    for key, value in defaults.items():
        params.setdefault(key, value)

    long_signal_mode = str(payload.get("long_signal_mode", "sma_bias"))
    short_signal_mode = str(payload.get("short_signal_mode", "lwma_lowest_count"))

    long_signal_map = {
        "sma_bias": 'df["sma_fast"].shift(3) > df["sma_fast"].shift(3).rolling(int(self.p("SignalMAPeriod")), min_periods=5).mean()',
        "smma_pullback": 'df["low"].rolling(int(self.p("PullbackSignalPeriod")), min_periods=2).min().shift(3) <= df["smma_mid"].shift(2)',
        "ha_reclaim": 'df["ha_floor"].shift(1) > df["ha_floor"].shift(1).rolling(int(self.p("HAFloorMAPeriod")), min_periods=5).mean()',
    }
    short_signal_map = {
        "lwma_lowest_count": '_sq_count_compare(df["lwma_fast"], df["lowest_typical"].shift(2), 2, 1, greater=False, not_strict=True)',
        "wt_push": '(df["close"].shift(1) > df["smma_mid"].shift(1)) & _rising(df["wt_main"], 2, 1)',
        "lwma_hour_quantile": '(df["lwma_short"].shift(1) > df["low"].shift(2)) & (df["hour_ref"].shift(3) <= df["hour_ref"].shift(3).rolling(int(self.p("HourQuantileLookback")), min_periods=25).quantile(float(self.p("HourQuantileThreshold")) / 100.0))',
    }
    long_signal = long_signal_map.get(long_signal_mode, long_signal_map["sma_bias"])
    short_signal = short_signal_map.get(short_signal_mode, short_signal_map["lwma_lowest_count"])

    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 80)),
        lot_value=float(payload.get("lot_value", 100.0)),
        bar_time_offset_hours=float(payload.get("bar_time_offset_hours", 6.0)),
        description=payload.get(
            "description",
            "SQX-inspired XAU highest-breakout family mined from Batch12.04, with stop entries and ATR-managed exits.",
        ),
        indicators=[
            IndicatorSpec("highest_level", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "HighestPeriod"}, "mode": 2}),
            IndicatorSpec("lowest_level", "sq_lowest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "LowestPeriod"}, "mode": 3}),
            IndicatorSpec("typical_price", "select_price", {"open_": "open", "high": "high", "low": "low", "close": "close", "mode": 5}),
            IndicatorSpec("sma_fast", "sma", {"series": "close", "period": {"param": "FastSMAPeriod"}}),
            IndicatorSpec("lwma_fast", "wma", {"series": "close", "period": {"param": "LWMAPeriod"}}),
            IndicatorSpec("lwma_short", "wma", {"series": "close", "period": {"param": "ShortLWMAPeriod"}}),
            IndicatorSpec("smma_mid", "smma", {"series": "typical_price", "period": {"param": "SMMAPeriod"}}),
            IndicatorSpec("ha", "sq_heiken_ashi", {"df": "df"}, outputs={"open": "ha_open", "close": "ha_close"}),
            IndicatorSpec("wt", "sq_wave_trend", {"high": "high", "low": "low", "close": "close", "channel_length": {"param": "WaveTrendChannel"}, "average_length": {"param": "WaveTrendAverage"}}, outputs={"main": "wt_main"}),
            IndicatorSpec("atr_long_stop", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRLongStopPeriod"}}),
            IndicatorSpec("atr_long_target", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRLongTargetPeriod"}}),
            IndicatorSpec("atr_long_trail", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRLongTrailPeriod"}}),
            IndicatorSpec("atr_short_stop", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRShortStopPeriod"}}),
            IndicatorSpec("atr_short_target", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRShortTargetPeriod"}}),
        ],
        series=[
            SeriesSpec("lowest_typical", 'df["typical_price"].rolling(int(self.p("PullbackSignalPeriod")), min_periods=2).min()'),
            SeriesSpec("ha_floor", 'df[["low", "ha_open", "ha_close"]].min(axis=1)'),
            SeriesSpec("hour_ref", 'df.index.to_series().dt.hour.astype(float)'),
            SeriesSpec("long_signal", long_signal),
            SeriesSpec("short_signal", short_signal),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
        ],
        entries=[
            EntryRuleSpec(
                name="sqx_xau_highest_breakout_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["long_signal"].shift(1).eq(True).iloc[i]) and _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo"))) and (abs(float(ctx.open) - float(df["highest_level"].shift(1).iloc[i])) / max(float(ctx.open), 1e-9) * 100.0 <= float(self.p("MaxDistancePct")))',
                price='float(df["highest_level"].shift(1).iloc[i])',
                lots='float(self.p("mmLots"))',
                stop_loss='float(df["highest_level"].shift(1).iloc[i]) - float(self.p("LongStopATR")) * float(df["atr_long_stop"].iloc[i - 1])',
                take_profit='float(df["highest_level"].shift(1).iloc[i]) + float(self.p("LongTargetATR")) * float(df["atr_long_target"].iloc[i - 1])',
                trail_dist='float(self.p("LongTrailATR")) * float(df["atr_long_trail"].iloc[i - 1])',
                trail_activation='float(self.p("LongTrailActivationATR")) * float(df["atr_long_trail"].iloc[i - 1])',
                expiry_bars=int(params.get("LongExpiryBars", 10)),
                comment="sqx_xau_highest_breakout_long",
            ),
            EntryRuleSpec(
                name="sqx_xau_highest_breakout_short",
                side="short",
                order_type="sell_stop",
                when='bool(df["short_signal"].shift(1).eq(True).iloc[i]) and not bool(df["long_signal"].shift(1).eq(True).iloc[i]) and _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo"))) and (abs(float(ctx.open) - float(df["lowest_level"].shift(1).iloc[i])) / max(float(ctx.open), 1e-9) * 100.0 <= float(self.p("MaxDistancePct")))',
                price='float(df["lowest_level"].shift(1).iloc[i])',
                lots='float(self.p("mmLots"))',
                stop_loss='float(df["lowest_level"].shift(1).iloc[i]) + float(self.p("ShortStopATR")) * float(df["atr_short_stop"].iloc[i - 1])',
                take_profit='float(df["lowest_level"].shift(1).iloc[i]) - float(self.p("ShortTargetATR")) * float(df["atr_short_target"].iloc[i - 1])',
                trail_dist='float(self.p("ShortTrailATR")) * float(df["atr_short_target"].iloc[i - 1])',
                trail_activation='float(self.p("ShortTrailActivationATR")) * float(df["atr_short_target"].iloc[i - 1])',
                expiry_bars=int(params.get("ShortExpiryBars", 18)),
                comment="sqx_xau_highest_breakout_short",
            ),
        ],
    )


def build_xau_discovery_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "HighestPeriod": 24,
        "LowestPeriod": 24,
        "ATRPeriod": 14,
        "FastEMA": 21,
        "SlowEMA": 55,
        "BBWRPeriod": 24,
        "BBWRMin": 0.015,
        "EntryBufferATR": 0.15,
        "StopLossATR": 1.4,
        "ProfitTargetATR": 2.8,
        "TrailATR": 1.1,
        "ExitAfterBars": 18,
        "PullbackLookback": 5,
        "PullbackATR": 0.6,
        "BreakEvenATR": 0.8,
        "TimeStopATR": 0.4,
        "MaxDistancePct": 3.5,
        "mmLots": 1.0,
    }
    for key, value in defaults.items():
        params.setdefault(key, value)

    entry_archetype = str(payload.get("entry_archetype", "breakout_stop"))
    volatility_filter = str(payload.get("volatility_filter", "bb_width"))
    session_filter = str(payload.get("session_filter", "london_ny"))
    stop_model = str(payload.get("stop_model", "atr"))
    target_model = str(payload.get("target_model", "fixed_rr"))
    exit_model = str(payload.get("exit_model", "session_close"))

    session_windows = {
        "all_day": ("00:00", "23:59"),
        "london_only": ("06:00", "11:59"),
        "ny_only": ("12:00", "18:00"),
        "london_ny": ("06:00", "18:00"),
    }
    session_from, session_to = session_windows.get(session_filter, ("06:00", "18:00"))
    params.setdefault("SignalTimeRangeFrom", session_from)
    params.setdefault("SignalTimeRangeTo", session_to)

    if volatility_filter == "none":
        volatility_ok = "True"
    elif volatility_filter == "atr_expansion":
        volatility_ok = 'df["atr"].shift(1) >= (df["atr"].shift(1).rolling(48, min_periods=12).mean() * 1.05)'
    else:
        volatility_ok = 'df["bbwr"] >= float(self.p("BBWRMin"))'

    if entry_archetype == "ema_reclaim":
        long_signal = (
            '(df["close"] > df["ema_slow"]) & '
            '(df["low"].rolling(int(self.p("PullbackLookback")), min_periods=2).min() < df["ema_fast"]) & '
            '(df["close"] > df["ema_fast"]) & '
            '(df["close"].shift(1) <= df["ema_fast"].shift(1))'
        )
        short_signal = (
            '(df["close"] < df["ema_slow"]) & '
            '(df["high"].rolling(int(self.p("PullbackLookback")), min_periods=2).max() > df["ema_fast"]) & '
            '(df["close"] < df["ema_fast"]) & '
            '(df["close"].shift(1) >= df["ema_fast"].shift(1))'
        )
        long_order_type = "market"
        short_order_type = "market"
        long_price = None
        short_price = None
        long_entry_base = "ctx.open"
        short_entry_base = "ctx.open"
    elif entry_archetype == "atr_pullback_limit":
        long_signal = '(df["close"] > df["ema_slow"]) & (df["close"] > df["ema_fast"])'
        short_signal = '(df["close"] < df["ema_slow"]) & (df["close"] < df["ema_fast"])'
        long_order_type = "buy_limit"
        short_order_type = "sell_limit"
        long_price = 'float(df["ema_fast"].shift(1).iloc[i]) - (float(df["atr"].iloc[i - 1]) * float(self.p("PullbackATR")))'
        short_price = 'float(df["ema_fast"].shift(1).iloc[i]) + (float(df["atr"].iloc[i - 1]) * float(self.p("PullbackATR")))'
        long_entry_base = f"({long_price})"
        short_entry_base = f"({short_price})"
    elif entry_archetype == "pullback_trend":
        long_signal = (
            '(df["close"] > df["ema_slow"]) & '
            '(df["low"].rolling(int(self.p("PullbackLookback")), min_periods=2).min() <= df["ema_fast"]) & '
            '(df["close"] > df["ema_fast"])'
        )
        short_signal = (
            '(df["close"] < df["ema_slow"]) & '
            '(df["high"].rolling(int(self.p("PullbackLookback")), min_periods=2).max() >= df["ema_fast"]) & '
            '(df["close"] < df["ema_fast"])'
        )
        long_order_type = "market"
        short_order_type = "market"
        long_price = None
        short_price = None
        long_entry_base = "ctx.open"
        short_entry_base = "ctx.open"
    elif entry_archetype == "breakout_close":
        long_signal = 'df["close"] > df["highest_level"].shift(1)'
        short_signal = 'df["close"] < df["lowest_level"].shift(1)'
        long_order_type = "market"
        short_order_type = "market"
        long_price = None
        short_price = None
        long_entry_base = "ctx.open"
        short_entry_base = "ctx.open"
    else:
        long_signal = 'df["close"] > df["highest_level"].shift(1)'
        short_signal = 'df["close"] < df["lowest_level"].shift(1)'
        long_order_type = "buy_stop"
        short_order_type = "sell_stop"
        long_price = 'float(df["highest_level"].shift(1).iloc[i]) + (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))'
        short_price = 'float(df["lowest_level"].shift(1).iloc[i]) - (float(df["atr"].iloc[i - 1]) * float(self.p("EntryBufferATR")))'
        long_entry_base = f"({long_price})"
        short_entry_base = f"({short_price})"

    if stop_model == "channel":
        long_stop = 'min(float(df["lowest_level"].shift(1).iloc[i]), float(ctx.open) - float(df["atr"].iloc[i - 1]) * 0.5)'
        short_stop = 'max(float(df["highest_level"].shift(1).iloc[i]), float(ctx.open) + float(df["atr"].iloc[i - 1]) * 0.5)'
    elif stop_model == "swing":
        long_stop = 'float(df["low"].rolling(int(self.p("PullbackLookback")) + 2, min_periods=2).min().shift(1).iloc[i])'
        short_stop = 'float(df["high"].rolling(int(self.p("PullbackLookback")) + 2, min_periods=2).max().shift(1).iloc[i])'
    else:
        long_stop = f"{long_entry_base} - (float(df['atr'].iloc[i - 1]) * float(self.p('StopLossATR')))"
        short_stop = f"{short_entry_base} + (float(df['atr'].iloc[i - 1]) * float(self.p('StopLossATR')))"

    if target_model == "trend_runner":
        long_target = f"{long_entry_base} + (float(df['atr'].iloc[i - 1]) * float(self.p('ProfitTargetATR')) * 1.6)"
        short_target = f"{short_entry_base} - (float(df['atr'].iloc[i - 1]) * float(self.p('ProfitTargetATR')) * 1.6)"
    elif target_model == "atr_scaled":
        long_target = f"{long_entry_base} + (float(df['atr'].iloc[i - 1]) * float(self.p('ProfitTargetATR')) * 1.2)"
        short_target = f"{short_entry_base} - (float(df['atr'].iloc[i - 1]) * float(self.p('ProfitTargetATR')) * 1.2)"
    else:
        long_target = f"{long_entry_base} + (float(df['atr'].iloc[i - 1]) * float(self.p('ProfitTargetATR')))"
        short_target = f"{short_entry_base} - (float(df['atr'].iloc[i - 1]) * float(self.p('ProfitTargetATR')))"

    exit_after_bars = 'int(self.p("ExitAfterBars"))' if exit_model in {"time_exit", "atr_time_stop"} else None
    trail_dist = 'float(df["atr"].iloc[i - 1]) * float(self.p("TrailATR"))' if exit_model in {"trailing_atr", "break_even_then_trail"} else None
    trail_activation = 'float(df["atr"].iloc[i - 1])' if exit_model == "trailing_atr" else 'float(df["atr"].iloc[i - 1]) * float(self.p("BreakEvenATR"))' if exit_model == "break_even_then_trail" else None

    exits: list[ExitRuleSpec] = []
    if session_filter != "all_day" or exit_model == "session_close":
        exits.append(
            ExitRuleSpec(
                "session_exit",
                'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))',
            )
        )
    if exit_model == "channel_flip":
        exits.extend(
            [
                ExitRuleSpec("channel_flip_long", 'ctx.has_long and bool(df["close"].iloc[i] < df["ema_fast"].iloc[i])'),
                ExitRuleSpec("channel_flip_short", 'ctx.has_short and bool(df["close"].iloc[i] > df["ema_fast"].iloc[i])'),
            ]
        )
    if exit_model == "atr_time_stop":
        exits.extend(
            [
                ExitRuleSpec("atr_time_stop_long", 'ctx.has_long and (ctx.long_entry_price - ctx.open) > float(df["atr"].iloc[i - 1]) * float(self.p("TimeStopATR"))'),
                ExitRuleSpec("atr_time_stop_short", 'ctx.has_short and (ctx.open - ctx.short_entry_price) > float(df["atr"].iloc[i - 1]) * float(self.p("TimeStopATR"))'),
            ]
        )

    entry_gate = '_in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'
    # Market-entry families must only consume closed-bar state; otherwise the native engine
    # can implicitly use the same bar's close/low/high while still entering at that bar's open.
    long_signal_ref = 'bool(df["long_signal"].shift(1).eq(True).iloc[i])'
    short_signal_ref = 'bool(df["short_signal"].shift(1).eq(True).iloc[i])'
    volatility_ref = 'bool(df["volatility_ok"].eq(True).iloc[i])' if volatility_filter == "atr_expansion" else 'bool(df["volatility_ok"].shift(1).eq(True).iloc[i])'
    distance_price_ref = 'float(df["close"].shift(1).iloc[i])'
    distance_gate = f'(abs(float(ctx.open) - {distance_price_ref}) / max(float(ctx.open), 1e-9) * 100.0 <= float(self.p("MaxDistancePct")))'

    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 40)),
        lot_value=float(payload.get("lot_value", 100.0)),
        description=payload.get("description", "Constrained XAU discovery template with mixable approved blocks."),
        indicators=[
            IndicatorSpec("highest_level", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "HighestPeriod"}, "mode": 2}),
            IndicatorSpec("lowest_level", "sq_lowest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "LowestPeriod"}, "mode": 3}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod"}}),
            IndicatorSpec("bbwr", "sq_bb_width_ratio", {"series": "close", "period": {"param": "BBWRPeriod"}, "deviations": 2.0}),
            IndicatorSpec("ema_fast", "ema", {"series": "close", "period": {"param": "FastEMA"}}),
            IndicatorSpec("ema_slow", "ema", {"series": "close", "period": {"param": "SlowEMA"}}),
        ],
        series=[
            SeriesSpec("volatility_ok", volatility_ok),
            SeriesSpec("long_signal", long_signal),
            SeriesSpec("short_signal", short_signal),
        ],
        exits=exits,
        entries=[
            EntryRuleSpec(
                name="xau_discovery_long",
                side="long",
                order_type=long_order_type,
                when=f'{volatility_ref} and {entry_gate} and {long_signal_ref} and {distance_gate}',
                price=long_price,
                lots='float(self.p("mmLots"))',
                stop_loss=long_stop,
                take_profit=long_target,
                trail_dist=trail_dist,
                trail_activation=trail_activation,
                exit_after_bars=exit_after_bars,
                expiry_bars=3 if long_order_type != "market" else 1,
                comment="xau_discovery_long",
            ),
            EntryRuleSpec(
                name="xau_discovery_short",
                side="short",
                order_type=short_order_type,
                when=f'{volatility_ref} and {entry_gate} and {short_signal_ref} and {distance_gate}',
                price=short_price,
                lots='float(self.p("mmLots"))',
                stop_loss=short_stop,
                take_profit=short_target,
                trail_dist=trail_dist,
                trail_activation=trail_activation,
                exit_after_bars=exit_after_bars,
                expiry_bars=3 if short_order_type != "market" else 1,
                comment="xau_discovery_short",
            ),
        ],
    )


def build_usdjpy_sqx_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "EMAPeriod1": 20,
        "VWAPPeriod1": 77,
        "WaveTrendChannel": 9,
        "WaveTrendAverage": 60,
        "IndicatorCrsMAPrd1": 35,
        "IndicatorCrsMAPrd2": 31,
        "PriceEntryMult1": 0.3,
        "ExitAfterBars1": 14,
        "ProfitTargetCoef1": 3.0,
        "StopLossCoef1": 2.8,
        "ATRPeriod1": 19,
        "ATRPeriod2": 14,
        "HighestPeriod": 20,
        "mmLots": 1.0,
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 12)),
        lot_value=float(payload.get("lot_value", 1.0)),
        description=payload.get("description", "USDJPY SQX EMA/VWAP/WaveTrend breakout family."),
        indicators=[
            IndicatorSpec("ema_1", "ema", {"series": "close", "period": {"param": "EMAPeriod1"}}),
            IndicatorSpec("wt", "sq_wave_trend", {"high": "high", "low": "low", "close": "close", "channel_length": {"param": "WaveTrendChannel"}, "average_length": {"param": "WaveTrendAverage"}}, outputs={"main": "wavetrend"}),
            IndicatorSpec("vwap", "sq_vwap", {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "period": {"param": "VWAPPeriod1"}}),
            IndicatorSpec("highest", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "HighestPeriod"}, "mode": 3}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod1"}}),
            IndicatorSpec("atr_2", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod2"}}),
        ],
        series=[
            SeriesSpec("long_entry_signal", '_count_compare(df["open"], df["ema_1"].shift(2), 5, 3, greater=False, allow_equal=False)'),
            SeriesSpec("long_exit_signal", '_cross_below(df["vwap"].shift(3), sma(df["vwap"].shift(3), int(self.p("IndicatorCrsMAPrd1")))) & (df["wavetrend"].shift(1) > 0)'),
        ],
        exits=[
            ExitRuleSpec("long_exit", 'ctx.has_long and bool(df["long_exit_signal"].iloc[i])'),
        ],
        entries=[
            EntryRuleSpec(
                name="usdjpy_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["long_entry_signal"].iloc[i])',
                price='float(df["highest"].iloc[i - 2]) + float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 1])',
                lots='float(self.p("mmLots"))',
                stop_loss='(float(df["highest"].iloc[i - 2]) + float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 1])) - float(self.p("StopLossCoef1")) * float(df["atr_2"].iloc[i - 1])',
                take_profit='(float(df["highest"].iloc[i - 2]) + float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 1])) + float(self.p("ProfitTargetCoef1")) * float(df["atr"].iloc[i - 1])',
                exit_after_bars='int(self.p("ExitAfterBars1"))',
                comment="sqx_usdjpy_long",
                expiry_bars=17,
            ),
        ],
    )


def build_hk50_batch_h1_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "mmLots": 70.0,
        "StopLossCoef1": 2.5,
        "ProfitTargetCoef1": 2.8,
        "TrailingActCef1": 1.4,
        "TrailingStop1": 127.5,
        "StopLossCoef2": 3.0,
        "ProfitTargetCoef2": 2.2,
        "TrailingStopCoef1": 1.0,
        "LWMAPeriod1": 14,
        "IndicatorCrsMAPrd1": 47,
        "ATR1_period": 19,
        "ATR2_period": 45,
        "ATR3_period": 14,
        "ATR4_period": 100,
        "Highest_period": 50,
        "Lowest_period": 50,
        "SignalTimeRangeFrom": "00:13",
        "SignalTimeRangeTo": "00:26",
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 55)),
        lot_value=float(payload.get("lot_value", 0.1285)),
        description=payload.get("description", "HK50 H1 Batch family using AO plus LWMA/EMA asymmetry."),
        indicators=[
            IndicatorSpec("lwma", "wma", {"series": "close", "period": {"param": "LWMAPeriod1"}}),
            IndicatorSpec("lwma_ma", "ema", {"series": "lwma", "period": {"param": "IndicatorCrsMAPrd1"}}),
            IndicatorSpec("atr_19", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR1_period"}}),
            IndicatorSpec("atr_45", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR2_period"}}),
            IndicatorSpec("atr_14", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR3_period"}}),
            IndicatorSpec("atr_100", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR4_period"}}),
            IndicatorSpec("highest", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "Highest_period"}, "mode": 0}),
            IndicatorSpec("lowest", "sq_lowest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "Lowest_period"}, "mode": 0}),
        ],
        series=[
            SeriesSpec("mid", '(df["high"] + df["low"]) / 2.0'),
            SeriesSpec("ao", 'df["mid"].rolling(5).mean() - df["mid"].rolling(34).mean()'),
            SeriesSpec("long_signal", '_cross_above(df["ao"], 0.0)'),
            SeriesSpec("short_signal", '_cross_below(df["lwma"], df["lwma_ma"])'),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
        ],
        entries=[
            EntryRuleSpec(
                name="hk50_batch_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["long_signal"].iloc[i]) and not bool(df["short_signal"].iloc[i])',
                price='float(df["highest"].iloc[i - 1])',
                lots='float(self.p("mmLots"))',
                stop_loss='float(df["highest"].iloc[i - 1]) - float(self.p("StopLossCoef1")) * float(df["atr_19"].iloc[i - 1])',
                take_profit='float(df["highest"].iloc[i - 1]) + float(self.p("ProfitTargetCoef1")) * float(df["atr_19"].iloc[i - 1])',
                trail_dist='float(self.p("TrailingStop1"))',
                trail_activation='float(self.p("TrailingActCef1")) * float(df["atr_45"].iloc[i - 1])',
                expiry_bars=1,
                comment="hk50_batch_long",
            ),
            EntryRuleSpec(
                name="hk50_batch_short",
                side="short",
                order_type="sell_stop",
                when='bool(df["short_signal"].iloc[i]) and not bool(df["long_signal"].iloc[i])',
                price='float(df["lowest"].iloc[i - 1])',
                lots='float(self.p("mmLots"))',
                stop_loss='float(df["lowest"].iloc[i - 1]) + float(self.p("StopLossCoef2")) * float(df["atr_14"].iloc[i - 1])',
                take_profit='float(df["lowest"].iloc[i - 1]) - float(self.p("ProfitTargetCoef2")) * float(df["atr_19"].iloc[i - 1])',
                trail_dist='float(self.p("TrailingStopCoef1")) * float(df["atr_100"].iloc[i - 1])',
                expiry_bars=1,
                comment="hk50_batch_short",
            ),
        ],
    )


def build_hk50_after_retest_h4_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "DIPeriod1": 14,
        "SARStep": 0.266,
        "SARMax": 0.69,
        "ATRPeriod1": 20,
        "PriceEntryMult1": 0.3,
        "StopLossATR": 2.0,
        "ProfitTargetATR": 3.0,
        "mmLots": 1.0,
        "SignalTimeRangeFrom": "00:13",
        "SignalTimeRangeTo": "00:26",
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 40)),
        lot_value=float(payload.get("lot_value", 0.1285)),
        description=payload.get("description", "HK50 H4 AfterRetest family using ADX, Ichimoku, and SAR."),
        indicators=[
            IndicatorSpec("adx", "sq_adx", {"high": "high", "low": "low", "close": "close", "period": {"param": "DIPeriod1"}}, outputs={"plus_di": "adx_plus", "minus_di": "adx_minus"}),
            IndicatorSpec("sar", "sq_parabolic_sar", {"high": "high", "low": "low", "step": {"param": "SARStep"}, "maximum": {"param": "SARMax"}}),
            IndicatorSpec("ichi", "sq_ichimoku", {"high": "high", "low": "low", "close": "close", "tenkan": 9, "kijun": 26, "senkou": 51}, outputs={"kijun": "ichi_kijun"}),
            IndicatorSpec("atr", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod1"}}),
        ],
        series=[
            SeriesSpec("long_signal", '_rising(df["adx_plus"], 2, 3) & (df["adx_plus"] > df["adx_minus"]) & (df["close"] > df["sar"])'),
            SeriesSpec("short_signal", '_rising(df["adx_minus"], 2, 3) & (df["adx_minus"] > df["adx_plus"]) & (df["close"] < df["sar"])'),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
        ],
        entries=[
            EntryRuleSpec(
                name="hk50_after_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["long_signal"].iloc[i])',
                price='float(df["ichi_kijun"].iloc[i - 3]) + float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 3])',
                lots='float(self.p("mmLots"))',
                stop_loss='(float(df["ichi_kijun"].iloc[i - 3]) + float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 3])) - float(self.p("StopLossATR")) * float(df["atr"].iloc[i - 1])',
                take_profit='(float(df["ichi_kijun"].iloc[i - 3]) + float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 3])) + float(self.p("ProfitTargetATR")) * float(df["atr"].iloc[i - 1])',
                comment="hk50_after_long",
                expiry_bars=8,
            ),
            EntryRuleSpec(
                name="hk50_after_short",
                side="short",
                order_type="sell_stop",
                when='bool(df["short_signal"].iloc[i])',
                price='float(df["ichi_kijun"].iloc[i - 3]) - float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 3])',
                lots='float(self.p("mmLots"))',
                stop_loss='(float(df["ichi_kijun"].iloc[i - 3]) - float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 3])) + float(self.p("StopLossATR")) * float(df["atr"].iloc[i - 1])',
                take_profit='(float(df["ichi_kijun"].iloc[i - 3]) - float(self.p("PriceEntryMult1")) * float(df["atr"].iloc[i - 3])) - float(self.p("ProfitTargetATR")) * float(df["atr"].iloc[i - 1])',
                comment="hk50_after_short",
                expiry_bars=8,
            ),
        ],
    )


def build_hk50_before_retest_h4_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "RSIPeriod1": 14,
        "Period1": 50,
        "SMAPeriod1": 20,
        "ATRPeriod1": 50,
        "ATRPeriod2": 19,
        "ATRPeriod3": 65,
        "PriceEntryMult1": 0.3,
        "ProfitTargetCoef1": 3.0,
        "TrailingActCef1": 1.4,
        "mmLots": 1.0,
        "SignalTimeRangeFrom": "00:13",
        "SignalTimeRangeTo": "00:26",
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 70)),
        lot_value=float(payload.get("lot_value", 0.1285)),
        description=payload.get("description", "HK50 H4 BeforeRetest family using RSI, Fibo, SMA, and ATR."),
        indicators=[
            IndicatorSpec("typical_price", "select_price", {"open_": "open", "high": "high", "low": "low", "close": "close", "mode": 5}),
            IndicatorSpec("lowest", "sq_lowest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "Period1"}, "mode": 1}),
            IndicatorSpec("highest", "sq_highest", {"open_": "open", "high": "high", "low": "low", "close": "close", "period": {"param": "Period1"}, "mode": 1}),
            IndicatorSpec("fibo", "sq_fibo", {"df": "df", "fibo_range": 2, "x": 0, "fibo_level": -9999999.0, "custom_fibo_level": -61.8, "start_date": None}),
            IndicatorSpec("sma_entry", "sma", {"series": "typical_price", "period": {"param": "SMAPeriod1"}}),
            IndicatorSpec("atr_1", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod1"}}),
            IndicatorSpec("atr_2", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod2"}}),
            IndicatorSpec("atr_3", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATRPeriod3"}}),
        ],
        series=[
            SeriesSpec("rsi_1", '100.0 - (100.0 / (1.0 + (df["typical_price"].diff().clip(lower=0).rolling(int(self.p("RSIPeriod1")), min_periods=1).mean() / df["typical_price"].diff().clip(upper=0).abs().rolling(int(self.p("RSIPeriod1")), min_periods=1).mean().replace(0.0, np.nan)))).fillna(50.0)'),
            SeriesSpec("long_signal", '_falling(df["rsi_1"], 2, 1)'),
            SeriesSpec("short_signal", '_rising(df["rsi_1"], 2, 1)'),
            SeriesSpec("long_exit_signal", 'df["lowest"].shift(2) != df["fibo"]'),
            SeriesSpec("short_exit_signal", 'df["highest"].shift(2) == df["fibo"]'),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
            ExitRuleSpec("before_retest_long_exit", 'ctx.has_long and bool(df["long_exit_signal"].iloc[i])'),
            ExitRuleSpec("before_retest_short_exit", 'ctx.has_short and bool(df["short_exit_signal"].iloc[i])'),
        ],
        entries=[
            EntryRuleSpec(
                name="hk50_before_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["long_signal"].iloc[i])',
                price='float(df["sma_entry"].iloc[i - 3]) + float(self.p("PriceEntryMult1")) * float(df["atr_1"].iloc[i - 1])',
                lots='float(self.p("mmLots"))',
                stop_loss='(float(df["sma_entry"].iloc[i - 3]) + float(self.p("PriceEntryMult1")) * float(df["atr_1"].iloc[i - 1])) - float(self.p("ProfitTargetCoef1")) * float(df["atr_2"].iloc[i - 1])',
                take_profit='(float(df["sma_entry"].iloc[i - 3]) + float(self.p("PriceEntryMult1")) * float(df["atr_1"].iloc[i - 1])) + float(self.p("ProfitTargetCoef1")) * float(df["atr_2"].iloc[i - 1])',
                trail_activation='float(self.p("TrailingActCef1")) * float(df["atr_3"].iloc[i - 1])',
                comment="hk50_before_long",
                expiry_bars=8,
            ),
            EntryRuleSpec(
                name="hk50_before_short",
                side="short",
                order_type="sell_stop",
                when='bool(df["short_signal"].iloc[i])',
                price='float(df["sma_entry"].iloc[i - 3]) - float(self.p("PriceEntryMult1")) * float(df["atr_1"].iloc[i - 1])',
                lots='float(self.p("mmLots"))',
                stop_loss='(float(df["sma_entry"].iloc[i - 3]) - float(self.p("PriceEntryMult1")) * float(df["atr_1"].iloc[i - 1])) + float(self.p("ProfitTargetCoef1")) * float(df["atr_2"].iloc[i - 1])',
                take_profit='(float(df["sma_entry"].iloc[i - 3]) - float(self.p("PriceEntryMult1")) * float(df["atr_1"].iloc[i - 1])) - float(self.p("ProfitTargetCoef1")) * float(df["atr_2"].iloc[i - 1])',
                trail_activation='float(self.p("TrailingActCef1")) * float(df["atr_3"].iloc[i - 1])',
                comment="hk50_before_short",
                expiry_bars=8,
            ),
        ],
    )


def build_uk100_ulcer_keltner_h1_template(payload: dict[str, Any]) -> StrategySpec:
    name = _required(payload, "name")
    symbol = _required(payload, "symbol")
    timeframe = _required(payload, "timeframe")
    params = dict(payload.get("params", {}))
    defaults = {
        "UlcerPeriod": 48,
        "KCPeriod": 20,
        "KCMult1": 2.25,
        "KCMult2": 2.5,
        "ATR1_period": 50,
        "ATR2_period": 19,
        "ATR3_period": 100,
        "ATR4_period": 14,
        "mmLots": 1.0,
        "SignalTimeRangeFrom": "00:13",
        "SignalTimeRangeTo": "00:26",
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    return StrategySpec(
        name=name,
        symbol=symbol,
        timeframe=timeframe,
        params=params,
        min_bars=int(payload.get("min_bars", 60)),
        lot_value=float(payload.get("lot_value", 1.0)),
        description=payload.get("description", "UK100 H1 Ulcer/Keltner/Heiken Ashi family."),
        indicators=[
            IndicatorSpec("ulcer", "sq_ulcer_index", {"close": "close", "period": {"param": "UlcerPeriod"}}),
            IndicatorSpec("kc_1", "sq_keltner_channel", {"high": "high", "low": "low", "close": "close", "period": {"param": "KCPeriod"}, "multiplier": {"param": "KCMult1"}}, outputs={"upper": "kc1_upper", "lower": "kc1_lower"}),
            IndicatorSpec("kc_2", "sq_keltner_channel", {"high": "high", "low": "low", "close": "close", "period": {"param": "KCPeriod"}, "multiplier": {"param": "KCMult2"}}, outputs={"upper": "kc2_upper", "lower": "kc2_lower"}),
            IndicatorSpec("ha", "sq_heiken_ashi", {"df": "df"}, outputs={"open": "ha_open", "close": "ha_close"}),
            IndicatorSpec("atr_1", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR1_period"}}),
            IndicatorSpec("atr_2", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR2_period"}}),
            IndicatorSpec("atr_3", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR3_period"}}),
            IndicatorSpec("atr_4", "sq_atr", {"high": "high", "low": "low", "close": "close", "period": {"param": "ATR4_period"}}),
        ],
        series=[
            SeriesSpec("long_signal", '_rising(df["ulcer"], 2, 1) & (df["close"] < df["kc1_lower"])'),
            SeriesSpec("short_signal", '_rising(df["ulcer"], 2, 1) & (df["close"] > df["kc1_upper"])'),
            SeriesSpec("short_exit_signal", 'df["kc2_lower"].shift(2) < df["kc2_lower"].rolling(558, min_periods=50).quantile(0.532)'),
        ],
        exits=[
            ExitRuleSpec("session_exit", 'ctx.has_position and not _in_time_window(ctx.time, str(self.p("SignalTimeRangeFrom")), str(self.p("SignalTimeRangeTo")))'),
            ExitRuleSpec("uk100_short_exit", 'ctx.has_short and bool(df["short_exit_signal"].iloc[i])'),
        ],
        entries=[
            EntryRuleSpec(
                name="uk100_long",
                side="long",
                order_type="buy_stop",
                when='bool(df["long_signal"].iloc[i])',
                price='max(float(ctx._df["high"].iloc[i - 3]), float(df["ha_open"].iloc[i - 3]), float(df["ha_close"].iloc[i - 3])) - (2.30 * float(df["atr_1"].iloc[i - 3]))',
                lots='float(self.p("mmLots"))',
                stop_loss='(max(float(ctx._df["high"].iloc[i - 3]), float(df["ha_open"].iloc[i - 3]), float(df["ha_close"].iloc[i - 3])) - (2.30 * float(df["atr_1"].iloc[i - 3]))) - (1.5 * float(df["atr_2"].iloc[i - 1]))',
                take_profit='(max(float(ctx._df["high"].iloc[i - 3]), float(df["ha_open"].iloc[i - 3]), float(df["ha_close"].iloc[i - 3])) - (2.30 * float(df["atr_1"].iloc[i - 3]))) + (3.5 * float(df["atr_2"].iloc[i - 1]))',
                comment="uk100_long",
                expiry_bars=6,
            ),
            EntryRuleSpec(
                name="uk100_short",
                side="short",
                order_type="market",
                when='bool(df["short_signal"].iloc[i])',
                lots='float(self.p("mmLots"))',
                stop_loss='ctx.open + (2.0 * float(df["atr_4"].iloc[i - 1]))',
                take_profit='ctx.open - (3.4 * float(df["atr_4"].iloc[i - 1]))',
                trail_activation='1.2 * float(df["atr_3"].iloc[i - 1])',
                comment="uk100_short",
            ),
        ],
    )


TEMPLATE_REGISTRY: dict[str, StrategyTemplate] = {
    "trend_pullback": StrategyTemplate(
        name="trend_pullback",
        description="EMA trend filter with WaveTrend confirmation and pullback entry.",
        build=build_trend_pullback_template,
        family="fx_trend_pullback",
        regime="Trend continuation after pullback",
        supported_symbols=("USDJPY", "EURUSD", "GBPJPY", "XAUUSD"),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H1", "H4"),
        sort_order=10,
    ),
    "breakout_confirm": StrategyTemplate(
        name="breakout_confirm",
        description="Highest/lowest breakout with BB-width confirmation and ATR risk.",
        build=build_breakout_confirm_template,
        family="breakout_confirm",
        regime="Volatility expansion breakout",
        supported_symbols=("US30.cash", "US500.cash", "NAS100.cash", "HK50.cash", "GER40.cash", "UK100.cash", "XAUUSD", "USDJPY", "EURUSD", "GBPJPY"),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H1",),
        sort_order=20,
    ),
    "reversion": StrategyTemplate(
        name="reversion",
        description="Bollinger plus WPR mean-reversion entries with ATR risk.",
        build=build_reversion_template,
        family="mean_reversion",
        regime="Stretch and snap-back reversion",
        supported_symbols=("XAUUSD", "EURUSD", "USDJPY", "GBPJPY", "GER40.cash"),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H1",),
        sort_order=30,
    ),
    "sqx_us30_wpr_stoch": StrategyTemplate(
        name="sqx_us30_wpr_stoch",
        description="Native first-class version of the US30 WPR/Stochastic conversion family.",
        build=build_us30_sqx_template,
        family="index_momentum",
        regime="Index breakout and momentum continuation",
        supported_symbols=("US30.cash", "US500.cash", "NAS100.cash", "GER40.cash"),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H1",),
        sort_order=40,
    ),
    "sqx_xau_osma_bb": StrategyTemplate(
        name="sqx_xau_osma_bb",
        description="Native first-class version of the XAU Stochastic/OsMA/Bollinger conversion family.",
        build=build_xau_sqx_template,
        family="gold_hybrid",
        regime="Gold trend and reversal hybrid",
        supported_symbols=("XAUUSD",),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H1",),
        sort_order=50,
    ),
    "xau_breakout_session": StrategyTemplate(
        name="xau_breakout_session",
        description="Native XAU breakout family with ATR-buffered stop entries and session gating.",
        build=build_xau_breakout_template,
        family="gold_breakout",
        regime="Gold volatility expansion breakout",
        supported_symbols=("XAUUSD",),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H1",),
        sort_order=55,
    ),
    "sqx_xau_highest_breakout": StrategyTemplate(
        name="sqx_xau_highest_breakout",
        description="SQX-inspired XAU highest-breakout family mined from Batch12.04 reports and MT5 source.",
        build=build_sqx_xau_highest_breakout_template,
        family="gold_sqx_breakout",
        regime="Gold breakout continuation with stop entries and ATR exits",
        supported_symbols=("XAUUSD",),
        supported_timeframes=("H1",),
        preferred_timeframes=("H1",),
        sort_order=56,
    ),
    "xau_discovery_grammar": StrategyTemplate(
        name="xau_discovery_grammar",
        description="Constrained XAU discovery family that mixes approved rule blocks.",
        build=build_xau_discovery_template,
        family="gold_discovery",
        regime="Constrained XAU structural discovery",
        supported_symbols=("XAUUSD",),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H1",),
        sort_order=57,
    ),
    "sqx_usdjpy_vwap_wt": StrategyTemplate(
        name="sqx_usdjpy_vwap_wt",
        description="Native first-class version of the USDJPY EMA/VWAP/WaveTrend conversion family.",
        build=build_usdjpy_sqx_template,
        family="fx_breakout_continuation",
        regime="FX breakout continuation",
        supported_symbols=("USDJPY", "EURUSD", "GBPJPY"),
        supported_timeframes=("H1", "H4"),
        preferred_timeframes=("H4",),
        sort_order=60,
    ),
    "sqx_hk50_batch_h1": StrategyTemplate(
        name="sqx_hk50_batch_h1",
        description="Native HK50 H1 batch family with AO and LWMA asymmetry.",
        build=build_hk50_batch_h1_template,
        family="index_open_drive",
        regime="Index open-drive breakout",
        supported_symbols=("HK50.cash", "GER40.cash", "US30.cash", "NAS100.cash"),
        supported_timeframes=("H1",),
        preferred_timeframes=("H1",),
        sort_order=70,
    ),
    "sqx_hk50_after_retest_h4": StrategyTemplate(
        name="sqx_hk50_after_retest_h4",
        description="Native HK50 H4 after-retest family using ADX, SAR, and Ichimoku.",
        build=build_hk50_after_retest_h4_template,
        family="index_retest_breakout",
        regime="Retest then continuation breakout",
        supported_symbols=("HK50.cash", "GER40.cash", "US30.cash"),
        supported_timeframes=("H4",),
        preferred_timeframes=("H4",),
        sort_order=80,
    ),
    "sqx_hk50_before_retest_h4": StrategyTemplate(
        name="sqx_hk50_before_retest_h4",
        description="Native HK50 H4 before-retest family using RSI, Fibo, and SMA.",
        build=build_hk50_before_retest_h4_template,
        family="index_prebreak_setup",
        regime="Pre-break tension before retest",
        supported_symbols=("HK50.cash", "GER40.cash", "US30.cash"),
        supported_timeframes=("H4",),
        preferred_timeframes=("H4",),
        sort_order=90,
    ),
    "sqx_uk100_ulcer_keltner_h1": StrategyTemplate(
        name="sqx_uk100_ulcer_keltner_h1",
        description="Native UK100 H1 ulcer-index and keltner-channel family.",
        build=build_uk100_ulcer_keltner_h1_template,
        family="index_volatility_reversion",
        regime="Volatility stretch and reversal",
        supported_symbols=("UK100.cash", "GER40.cash", "US500.cash"),
        supported_timeframes=("H1",),
        preferred_timeframes=("H1",),
        sort_order=100,
    ),
}


def template_profile(template_name: str) -> dict[str, Any] | None:
    template = TEMPLATE_REGISTRY.get(template_name)
    if template is None:
        return None
    return {
        "name": template.name,
        "description": template.description,
        "family": template.family,
        "regime": template.regime,
        "supported_symbols": list(template.supported_symbols),
        "supported_timeframes": list(template.supported_timeframes),
        "preferred_timeframes": list(template.preferred_timeframes),
        "sort_order": int(template.sort_order),
    }


def template_profiles(symbol: str | None = None) -> list[dict[str, Any]]:
    templates = list(TEMPLATE_REGISTRY.values())
    if symbol:
        templates = [template for template in templates if _template_matches_symbol(template, symbol)]
    templates.sort(key=lambda item: (item.sort_order, item.name))
    return [template_profile(template.name) for template in templates if template_profile(template.name)]


def available_builder_symbols() -> list[str]:
    return list(TARGET_BUILD_UNIVERSE)


def available_template_names(symbol: str | None = None) -> list[str]:
    names = [
        template.name
        for template in TEMPLATE_REGISTRY.values()
        if symbol is None or _template_matches_symbol(template, symbol)
    ]
    return [item for item in sorted(names, key=lambda name: (TEMPLATE_REGISTRY[name].sort_order, name))]


def strategy_spec_from_template(template_name: str, payload: dict[str, Any]) -> StrategySpec:
    template = TEMPLATE_REGISTRY.get(template_name)
    if template is None:
        supported = ", ".join(available_template_names())
        raise ValueError(f"Unknown strategy template: {template_name}. Supported templates: {supported}")
    symbol = canonical_symbol(str(payload.get("symbol", "")))
    timeframe = str(payload.get("timeframe", "")).upper()
    if template.supported_symbols and symbol and symbol not in template.supported_symbols:
        raise ValueError(
            f"Template `{template_name}` is not configured for symbol `{symbol}`. "
            f"Supported symbols: {', '.join(template.supported_symbols)}"
        )
    if template.supported_timeframes and timeframe and timeframe not in template.supported_timeframes:
        raise ValueError(
            f"Template `{template_name}` is not configured for timeframe `{timeframe}`. "
            f"Supported timeframes: {', '.join(template.supported_timeframes)}"
        )
    payload = {**payload, "symbol": symbol or payload.get("symbol", "")}
    return template.build(payload)


def available_presets(symbol: str | None = None, template_name: str | None = None) -> list[dict[str, Any]]:
    presets: list[dict[str, Any]] = []
    if not PRESET_DIR.exists():
        return presets

    for path in sorted(PRESET_DIR.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        preset_template = payload.get("template")
        if preset_template not in TEMPLATE_REGISTRY:
            continue
        preset_payload = payload.get("payload", {})
        preset_symbol = canonical_symbol(str(preset_payload.get("symbol", "")))
        if symbol and preset_symbol != canonical_symbol(symbol):
            continue
        if template_name and preset_template != template_name:
            continue
        presets.append(
            {
                "id": path.stem,
                "path": str(path),
                "label": payload.get("label", path.stem.replace("_", " ").title()),
                "description": payload.get("description", ""),
                "template": preset_template,
                "payload": {**preset_payload, "symbol": preset_symbol or preset_payload.get("symbol", "")},
            }
        )
    return presets


def preset_by_id(preset_id: str) -> dict[str, Any] | None:
    for preset in available_presets():
        if preset["id"] == preset_id:
            return preset
    return None


def _load_spec_payload(spec: StrategySpec | dict[str, Any]) -> StrategySpec:
    if isinstance(spec, StrategySpec):
        return spec
    return strategy_spec_from_dict(spec)


def compile_strategy_spec(
    spec: StrategySpec | dict[str, Any],
    *,
    strategy_id: str | None = None,
) -> CompiledStrategy:
    spec_obj = _load_spec_payload(spec)

    for item in spec_obj.series:
        _validate_expression(item.expression, COMPUTE_ALLOWED_NAMES)

    for item in spec_obj.entries:
        _validate_expression(item.when, ON_BAR_ALLOWED_NAMES)
        _validate_expression(item.lots, ON_BAR_ALLOWED_NAMES)
        _validate_expression(item.stop_loss, ON_BAR_ALLOWED_NAMES)
        _validate_expression(item.take_profit, ON_BAR_ALLOWED_NAMES)
        if item.price:
            _validate_expression(item.price, ON_BAR_ALLOWED_NAMES)
        if item.trail_dist:
            _validate_expression(item.trail_dist, ON_BAR_ALLOWED_NAMES)
        if item.trail_activation:
            _validate_expression(item.trail_activation, ON_BAR_ALLOWED_NAMES)
        if item.side not in {"long", "short"}:
            raise ValueError(f"Unsupported entry side: {item.side}")
        if item.order_type not in {"market", "buy_stop", "sell_stop", "buy_limit", "sell_limit"}:
            raise ValueError(f"Unsupported order type: {item.order_type}")

    for item in spec_obj.exits:
        _validate_expression(item.when, ON_BAR_ALLOWED_NAMES)
        if item.action not in {"close_all", "cancel_pending"}:
            raise ValueError(f"Unsupported exit action: {item.action}")

    slug_base = slugify(f"{spec_obj.name}_{strategy_id or spec_obj.symbol}_{spec_obj.timeframe}")
    class_name = class_name_from_slug(slug_base)
    strategy_path = ENGINE_STRATEGY_DIR / f"{slug_base}.py"
    spec_path = SPEC_DIR / f"{slug_base}.json"

    indicator_lines: list[str] = []
    indicator_imports: set[str] = {"sma"}
    for indicator in spec_obj.indicators:
        lines, func_name = _render_indicator(indicator)
        indicator_lines.extend(lines)
        indicator_imports.add(func_name)

    series_lines: list[str] = []
    for item in spec_obj.series:
        series_lines.append(f'        df[{item.name!r}] = {item.expression}')

    if not indicator_lines and not series_lines:
        compute_body = ["        return df"]
    else:
        compute_body = [*indicator_lines, *series_lines, "        return df"]

    exit_lines: list[str] = []
    for item in spec_obj.exits:
        action_line = "ctx.close_all()" if item.action == "close_all" else "ctx.cancel_pending()"
        exit_lines.extend(
            [
                f"        if {item.when}:",
                f"            {action_line}",
                "            return",
            ]
        )

    entry_lines: list[str] = []
    position_guard = "" if spec_obj.allow_entries_when_position_open else "        if ctx.has_position or ctx.has_pending:\n            return\n"
    for item in spec_obj.entries:
        order_key = (item.side, item.order_type)
        supported_orders = {
            ("long", "market"),
            ("short", "market"),
            ("long", "buy_stop"),
            ("short", "sell_stop"),
            ("long", "buy_limit"),
            ("short", "sell_limit"),
        }
        if order_key not in supported_orders:
            raise ValueError(f"Invalid side/order_type combination: {item.side}/{item.order_type}")

        method_name = {
            ("long", "market"): "buy_market",
            ("short", "market"): "sell_market",
            ("long", "buy_stop"): "buy_stop",
            ("short", "sell_stop"): "sell_stop",
            ("long", "buy_limit"): "buy_limit",
            ("short", "sell_limit"): "sell_limit",
        }[order_key]

        call_args = [
            f"                sl={item.stop_loss},",
            f"                tp={item.take_profit},",
            f"                lots={item.lots},",
        ]
        if item.price:
            call_args.append(f"                price={item.price},")
        if item.order_type != "market":
            call_args.append(f"                expiry_bars={int(item.expiry_bars)},")
        call_args.append(f"                comment={item.comment!r},")

        entry_lines.extend(
            [
                f"        if {item.when}:",
                f"            self._next_trail_dist = {item.trail_dist or '0.0'}",
                f"            self._next_trail_activation = {item.trail_activation or '0.0'}",
                f"            self._next_exit_after_bars = {_render_optional_numeric(item.exit_after_bars)}",
                f"            ctx.{method_name}(",
                *call_args,
                "            )",
                "            return",
            ]
        )

    indicator_import_block = ", ".join(sorted(indicator_imports))
    if indicator_import_block:
        indicator_import_block = f"from engine.indicators import {indicator_import_block}"

    description_line = spec_obj.description or "Generated from StrategySpec."
    source_lines = [
        '"""',
        "Auto-generated Python strategy.",
        description_line,
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        "from engine.base_strategy import BaseStrategy, BarContext",
    ]
    if indicator_import_block:
        source_lines.append(indicator_import_block)
    source_lines.extend(
        [
            "",
            "",
            "def _count_compare(left: pd.Series, right: pd.Series, bars: int, shift: int = 0, *, greater: bool, allow_equal: bool) -> pd.Series:",
            "    lhs = left.shift(shift)",
            "    rhs = right.shift(shift)",
            "    if greater:",
            "        ok = lhs >= rhs if allow_equal else lhs > rhs",
            "    else:",
            "        ok = lhs <= rhs if allow_equal else lhs < rhs",
            "    return ok.rolling(int(bars), min_periods=int(bars)).sum().eq(int(bars))",
            "",
            "",
            "def _sq_count_compare(left: pd.Series, right: pd.Series, bars: int, shift: int = 0, *, greater: bool, not_strict: bool, rounding: int = 5) -> pd.Series:",
            "    comparisons = []",
            "    strict_hits = []",
            "    for offset in range(int(bars)):",
            "        lhs = left.shift(int(shift) + offset).round(int(rounding))",
            "        rhs = right.shift(int(shift) + offset).round(int(rounding))",
            "        if greater:",
            "            comparisons.append(lhs >= rhs if not_strict else lhs > rhs)",
            "            strict_hits.append(lhs > rhs)",
            "        else:",
            "            comparisons.append(lhs <= rhs if not_strict else lhs < rhs)",
            "            strict_hits.append(lhs < rhs)",
            "    all_ok = comparisons[0]",
            "    any_strict = strict_hits[0]",
            "    for item in comparisons[1:]:",
            "        all_ok = all_ok & item",
            "    for item in strict_hits[1:]:",
            "        any_strict = any_strict | item",
            "    return all_ok & any_strict",
            "",
            "",
            "def _cross_above(left: pd.Series, right: pd.Series | float) -> pd.Series:",
            "    rhs = right if isinstance(right, pd.Series) else pd.Series(float(right), index=left.index)",
            "    return (left.shift(1) <= rhs.shift(1)) & (left > rhs)",
            "",
            "",
            "def _cross_below(left: pd.Series, right: pd.Series | float) -> pd.Series:",
            "    rhs = right if isinstance(right, pd.Series) else pd.Series(float(right), index=left.index)",
            "    return (left.shift(1) >= rhs.shift(1)) & (left < rhs)",
            "",
            "",
            "def _compare_window(left: pd.Series, right: pd.Series, bars: int, shift: int, allow_same: bool, greater: bool) -> pd.Series:",
            "    lhs = left.shift(shift)",
            "    rhs = right.shift(shift)",
            "    if greater:",
            "        ok = lhs >= rhs if allow_same else lhs > rhs",
            "        strict = lhs > rhs",
            "    else:",
            "        ok = lhs <= rhs if allow_same else lhs < rhs",
            "        strict = lhs < rhs",
            "    return ok.rolling(int(bars), min_periods=int(bars)).sum().eq(int(bars)) & strict.rolling(int(bars), min_periods=int(bars)).sum().gt(0)",
            "",
            "",
            "def _falling(series: pd.Series, bars: int, shift: int = 0) -> pd.Series:",
            "    ref = series.shift(shift)",
            "    delta = ref.diff()",
            "    return delta.rolling(int(bars), min_periods=int(bars)).apply(lambda values: float((values < 0).all()), raw=True).fillna(0.0) > 0.5",
            "",
            "",
            "def _rising(series: pd.Series, bars: int, shift: int = 0) -> pd.Series:",
            "    ref = series.shift(shift)",
            "    delta = ref.diff()",
            "    return delta.rolling(int(bars), min_periods=int(bars)).apply(lambda values: float((values > 0).all()), raw=True).fillna(0.0) > 0.5",
            "",
            "",
            "def _hhmm(ts: pd.Timestamp) -> int:",
            "    return ts.hour * 100 + ts.minute",
            "",
            "",
            "def _in_time_window(ts: pd.Timestamp, start: str, end: str) -> bool:",
            "    start_hhmm = int(start[:2]) * 100 + int(start[3:])",
            "    end_hhmm = int(end[:2]) * 100 + int(end[3:])",
            "    current = _hhmm(ts)",
            "    if start_hhmm <= end_hhmm:",
            "        return start_hhmm <= current <= end_hhmm",
            "    return current >= start_hhmm or current <= end_hhmm",
            "",
            "",
            f"class {class_name}(BaseStrategy):",
            f"    name = {spec_obj.name!r}",
            f"    symbol = {spec_obj.symbol!r}",
            f"    timeframe = {spec_obj.timeframe!r}",
            f"    lot_value = {float(spec_obj.lot_value)!r}",
            f"    bar_time_offset_hours = {float(spec_obj.bar_time_offset_hours)!r}",
            f"    params = {spec_obj.params!r}",
            "",
            "    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:",
            *compute_body,
            "",
            "    def on_start(self, df: pd.DataFrame):",
            "        self._next_trail_dist = 0.0",
            "        self._next_trail_activation = 0.0",
            "        self._next_exit_after_bars = 0",
            "",
            "    def on_bar(self, ctx: BarContext):",
            "        i = ctx.bar_index",
            "        df = ctx._df",
            f"        if i < {int(spec_obj.min_bars)}:",
            "            return",
            *exit_lines,
        ]
    )
    if position_guard:
        source_lines.extend(position_guard.rstrip("\n").splitlines())
    source_lines.extend(entry_lines or ["        return"])
    source_lines.append("")

    return CompiledStrategy(
        spec=spec_obj,
        strategy_slug=slug_base,
        strategy_class=class_name,
        strategy_module=f"strategies.generated.{slug_base}",
        strategy_path=str(strategy_path),
        spec_path=str(spec_path),
        source="\n".join(source_lines),
    )


def persist_strategy_spec(compiled: CompiledStrategy) -> dict[str, str]:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    ENGINE_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    Path(compiled.strategy_path).write_text(compiled.source, encoding="utf-8")
    Path(compiled.spec_path).write_text(
        json.dumps(_serialize_spec(compiled.spec), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "strategy_path": compiled.strategy_path,
        "spec_path": compiled.spec_path,
    }
