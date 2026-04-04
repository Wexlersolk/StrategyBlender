# StrategyBlender

Local-first platform for:
- loading MT5 `.mq5` strategies
- converting them into local Python strategy artifacts
- backtesting them on exported market data
- training and comparing AI workflows on the resulting backtests

## Current Flow

1. Put MT5 strategy source files in `mine/` or load them through the UI.
2. Start the app with:
   `streamlit run ui/app.py`
3. In `EA Manager`, load a strategy and generate:
   - a local engine strategy for research/backtesting
   - a Python review scaffold mirroring the original MQL structure
4. In `Backtests`, run the generated strategy on data from:
   `data/exports/MT5 data export/`

## Research Runtime

- Persisted research jobs, datasets, artifacts, and experiment manifests are stored under `data/research_state/`.
- The research worker now runs as a separate local process pool, not inside the Streamlit request loop.
- The UI requires sign-in before accessing persisted research state.
- Default bootstrap credentials are controlled by:
  `STRATEGYBLENDER_BOOTSTRAP_USER`
  `STRATEGYBLENDER_BOOTSTRAP_PASSWORD`

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
- `docs/conversion_support.md` documents supported conversion patterns

## Notes

- This is a local research tool; MT5 bridge/orchestration code has been retired.
- Conversion of large StrategyQuant EAs is heuristic and may still require manual review.
