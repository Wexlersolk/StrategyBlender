from __future__ import annotations
import pandas as pd
import sqlalchemy as sa
from pathlib import Path


def _get_engine():
    root    = Path(__file__).parent.parent
    db_path = root / "market_data.db"
    return sa.create_engine(f"sqlite:///{db_path}")


def load_bars(
    symbol:    str,
    timeframe: str = "H1",
    date_from: str = None,
    date_to:   str = None,
) -> pd.DataFrame:
    engine   = _get_engine()
    safe_sym = symbol.replace('.', '_').replace(' ', '_')

    candidates = [
        f"{safe_sym}_{timeframe}",
        f"{safe_sym}_{timeframe.lower()}",
        f"{safe_sym}_daily",
        safe_sym,
    ]

    available  = sa.inspect(engine).get_table_names()
    table_name = next((c for c in candidates if c in available), None)

    if table_name is None:
        raise ValueError(
            f"No data for {symbol} {timeframe}.\n"
            f"Available: {available}"
        )

    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)

    time_col = next(
        (c for c in df.columns if c.lower() in ("time","index","date","datetime")),
        None
    )
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df.index   = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]
    df         = df.rename(columns={"tick_volume": "volume", "vol": "volume"})
    df         = df.sort_index()

    if date_from:
        df = df[df.index >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df.index <= pd.Timestamp(date_to)]

    return df


def available_symbols(timeframe: str = "H1") -> list[str]:
    engine  = _get_engine()
    tables  = sa.inspect(engine).get_table_names()
    suffix  = f"_{timeframe}"
    return [
        t[:-len(suffix)].replace("_", ".", 1)
        for t in tables if t.endswith(suffix)
    ]
