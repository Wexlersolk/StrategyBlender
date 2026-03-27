# StrategyBlender

Local-first platform for:
- loading MT5 `.mq5` strategies
- converting them into Python strategy artifacts
- backtesting them on exported MT5 market data
- training and comparing AI workflows on the resulting backtests

## Current Flow

1. Put MT5 strategy source files in `mine/` or load them through the UI.
2. Start the app with:
   `streamlit run ui/app.py`
3. In `EA Manager`, load a strategy and generate:
   - a local engine strategy for research/backtesting
   - a `strategytester5` scaffold for future MT5-side orchestration
4. In `Backtests`, run the generated strategy on data from:
   `data/exports/MT5 data export/`

## Data

The app now reads MT5 tab-delimited export files directly from:
`data/exports/MT5 data export/`

Expected format:
- `<DATE>`
- `<TIME>`
- `<OPEN>`
- `<HIGH>`
- `<LOW>`
- `<CLOSE>`
- `<TICKVOL>`

Minute data is resampled in code for `H1`, `H4`, and `D1` backtests.

## Project Structure

- `ui/` contains the Streamlit app shell and views
- `services/` contains reusable conversion and backtest orchestration logic
- `engine/` contains the local strategy/backtest engine
- `strategies/` contains hand-written and generated local Python strategies
- `convert/` contains MT5-to-Python conversion utilities
- `data/` contains exported market data and compatibility loaders
- `mine/` is the local drop folder for raw `.mq5` strategy files
- `scripts/` contains console utilities

## Notes

- MT5 login is no longer part of the UI.
- MT5 is only relevant for future console-side data download workflows.
- Conversion of large StrategyQuant EAs is heuristic and may still require manual review.
