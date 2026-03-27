from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "data" / "exports" / "MT5 data export"

RESAMPLE_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().lower()


def _canonical_symbol_from_filename(path: Path) -> str:
    name = path.name
    match = re.match(r"(.+?)_M\d+_", name)
    if match:
        return match.group(1)
    match = re.match(r"(.+?)_(?:H\d+|D\d+|W\d+|MN\d+)_", name)
    if match:
        return match.group(1)
    return path.stem.split("_")[0]


def _candidate_files(symbol: str) -> list[Path]:
    if not EXPORT_DIR.exists():
        return []
    target = _normalize_symbol(symbol)
    candidates = []
    for path in EXPORT_DIR.iterdir():
        if not path.is_file():
            continue
        if _normalize_symbol(_canonical_symbol_from_filename(path)) == target:
            candidates.append(path)
    candidates.sort(key=lambda p: (p.stat().st_size, p.name), reverse=True)
    return candidates


def _parse_mt5_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    rename_map = {
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "volume",
        "<VOL>": "real_volume",
        "<SPREAD>": "spread",
    }
    missing = {"<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"} - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected MT5 export format in {path.name}. Missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(
        df["<DATE>"].astype(str) + " " + df["<TIME>"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )
    out = df.rename(columns=rename_map)
    out.index = timestamps
    out.index.name = "time"
    out = out[~out.index.isna()]
    keep_cols = [c for c in ["open", "high", "low", "close", "volume", "real_volume", "spread"] if c in out.columns]
    out = out[keep_cols].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


@lru_cache(maxsize=64)
def _load_minute_bars(symbol: str) -> pd.DataFrame:
    files = _candidate_files(symbol)
    if not files:
        raise ValueError(f"No MT5 export file found for {symbol} in {EXPORT_DIR}")
    return _parse_mt5_export(files[0])


def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = timeframe.upper()
    if tf == "M1":
        return df.copy()
    if tf not in RESAMPLE_RULES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "real_volume" in df.columns:
        agg["real_volume"] = "sum"
    if "spread" in df.columns:
        agg["spread"] = "mean"

    out = df.resample(RESAMPLE_RULES[tf]).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def load_bars(
    symbol: str,
    timeframe: str = "H1",
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    df = _load_minute_bars(symbol)
    if date_from:
        df = df[df.index >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df.index <= pd.Timestamp(date_to)]
    df = _resample(df, timeframe)
    return df


def available_symbols(timeframe: str = "H1") -> list[str]:
    if not EXPORT_DIR.exists():
        return []
    symbols = {
        _canonical_symbol_from_filename(path)
        for path in EXPORT_DIR.iterdir()
        if path.is_file()
    }
    return sorted(symbols)
