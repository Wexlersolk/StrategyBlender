# Conversion Support Matrix

StrategyBlender is a local research tool. Conversion means:

- generating a local engine strategy that can run inside the Python backtester
- generating a Python review scaffold that mirrors the source structure for inspection

It does not mean a one-to-one MT5 execution bridge.

## Supported

- StrategyQuant US30-style WPR/Stochastic pattern
  Detection:
  `sqGetIndicatorValue(WILLIAMSPR_1, 0, 4)` and `sqIsFalling(STOCHASTIC_1, 35`
  Result:
  dedicated engine builder with WPR, Stochastic, ATR, Highest, Lowest, session-window handling

- StrategyQuant XAU-style OsMA/Bollinger pattern
  Detection:
  `sqGetIndicatorValue(OSMA_1, 0, 3)` and `BOLLINGERBANDS_1`
  Result:
  dedicated engine builder with Stochastic, OsMA, Bollinger Bands, ATR, timed exits

- StrategyQuant USDJPY-style EMA/VWAP/WaveTrend pattern
  Detection:
  `sqGetIndicatorValue(VWAP_1, 0, 3)` and `sqGetIndicatorValue(BBWIDTHRATIO_1, 0, 3)`
  Result:
  dedicated engine builder with EMA, VWAP, WaveTrend, ATR, BB width logic

- Generic StrategyQuant-style fallback
  Result:
  local engine strategy using extracted params and a heuristic MA/AO/ATR/Highest/Lowest template

- Input parsing for common numeric MQL5 `input` declarations
  Supported types:
  `double`, `float`, `int`, `uint`, `long`, `ulong`

- Review-scaffold translation of simple control flow
  Supported forms:
  `if`, `else if`, `else`, `while`, simple typed assignments, simple assignments, `Print`, `return`

## Partially Supported

- `iCustom(...)` argument extraction
  Used to recover indicator parameters when the local engine has a Python port, but not to reproduce arbitrary indicator runtime behavior.

- Custom indicators under `mine/Indicators/`
  The converter detects references and warns when a dedicated Python port is missing.

- Time-window logic, Friday exits, and session constraints
  Ported for the dedicated builders, but still approximated by the local backtester.

- Complex StrategyQuant EAs outside the dedicated builders
  The generic fallback tries to preserve parameter intent and basic risk structure, but entry logic is heuristic.

## Unsupported

- MT5 live execution or broker orchestration
- `strategytester5` bridges
- One-to-one MQL5 trade semantics
- Arbitrary `CopyBuffer`, `IndicatorCreate`, and buffer-management logic
- `for` loops and `switch` statements in the review scaffold
- Automatic ports for custom indicators without a Python implementation in `engine/indicators.py`

## Warnings You Should Expect

- `Complex indicator/buffer logic was detected`
  The review scaffold saw `iCustom`, `CopyBuffer`, `IndicatorCreate`, `SetIndexBuffer`, or `switch`.

- `... no dedicated Python port exists yet`
  The EA references a custom indicator that is not implemented in the local engine.

- `Generated local strategy uses a fallback ...`
  The source did not match a dedicated builder, so the engine strategy is heuristic.

## Canonical Workflow

1. Load `.mq5` in `EA Manager`.
2. Inspect conversion warnings.
3. Use the generated engine strategy for local backtests.
4. Treat the review scaffold as a debugging aid, not an executable bridge.
