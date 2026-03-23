"""
scripts/update_data.py

Fetches OHLCV data from MT5 and stores in SQLite.
Fetches both D1 (for AI training features) and H1 (for backtesting).

Run with:
    python scripts/update_data.py
    python scripts/update_data.py --timeframe H1
    python scripts/update_data.py --timeframe D1
    python scripts/update_data.py --timeframe all
"""

import sys
import argparse
sys.path.append('.')

import sqlalchemy as sa
import yaml
from datetime import datetime, timedelta
import config.settings as settings
from data.fetcher import MT5Fetcher


def table_name(symbol: str, timeframe: str) -> str:
    return symbol.replace('.', '_').replace(' ', '_') + f'_{timeframe}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', default='all',
                        help='H1, D1, or all (default: all)')
    args = parser.parse_args()

    # Load symbols
    with open(settings.SYMBOLS_CONFIG) as f:
        cfg = yaml.safe_load(f)
    symbols = [s['name'] for s in cfg['symbols']]

    # Timeframes to fetch
    if args.timeframe.lower() == 'all':
        timeframes = ['H1', 'D1']
    else:
        timeframes = [args.timeframe.upper()]

    print(f"Fetching {timeframes} data for: {symbols}")

    fetcher = MT5Fetcher()
    engine  = sa.create_engine(f"sqlite:///{settings.DB_PATH.replace('sqlite:///', '')}")

    end   = datetime.now()
    start = end - timedelta(days=365 * 6)

    for sym in symbols:
        for tf in timeframes:
            print(f"\nFetching {sym} {tf}...")
            df = fetcher.fetch_historical(sym, timeframe=tf,
                                          start_date=start, end_date=end)
            if df.empty:
                print(f"  No data returned")
                continue

            tbl = table_name(sym, tf)
            df.to_sql(tbl, engine, if_exists='replace')
            print(f"  Saved {len(df)} bars → {tbl}")

    fetcher.shutdown()

    # Summary
    print(f"\n{'='*40}")
    print("Database contents:")
    with engine.connect() as conn:
        tables = sa.inspect(engine).get_table_names()
        for t in sorted(tables):
            count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {t}")).scalar()
            print(f"  {t}: {count:,} rows")


if __name__ == '__main__':
    main()
