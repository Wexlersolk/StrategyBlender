import pandas as pd
from sqlalchemy import create_engine, text
import config.settings as settings


def _table_name(symbol):
    """Sanitize symbol name for use as a SQL table name (dots and dashes are invalid)."""
    return symbol.replace('.', '_').replace('-', '_') + "_daily"


class DataStorage:
    """Handles saving and loading market data to/from SQL database."""

    def __init__(self):
        self.engine = create_engine(settings.DB_PATH)

    def save_bars(self, symbol, df):
        """Store DataFrame in table named '{symbol}_daily'. Appends, ignoring duplicates."""
        table = _table_name(symbol)
        df.to_sql(table, self.engine, if_exists='append', index=True)

    def load_bars(self, symbol, start_date=None, end_date=None):
        """
        Load OHLCV bars for a symbol between optional start/end dates.
        Uses parameterized queries to prevent SQL injection.
        """
        table = _table_name(symbol)

        if start_date and end_date:
            query = text(f"SELECT * FROM {table} WHERE time >= :start_date AND time <= :end_date")
            params = {"start_date": start_date, "end_date": end_date}
        elif start_date:
            query = text(f"SELECT * FROM {table} WHERE time >= :start_date")
            params = {"start_date": start_date}
        elif end_date:
            query = text(f"SELECT * FROM {table} WHERE time <= :end_date")
            params = {"end_date": end_date}
        else:
            query = text(f"SELECT * FROM {table}")
            params = {}

        try:
            df = pd.read_sql(
                query,
                self.engine,
                params=params,
                index_col='time',
                parse_dates=['time']
            )
        except Exception as e:
            print(f"Warning: could not load data for {symbol}: {e}")
            return pd.DataFrame()

        return df

    def get_all_symbols_data(self, symbols, start_date, end_date):
        """Load data for multiple symbols and return a dict of symbol -> DataFrame."""
        data = {}
        for sym in symbols:
            df = self.load_bars(sym, start_date, end_date)
            if not df.empty:
                data[sym] = df
            else:
                print(f"Warning: no data found for {sym}")
        return data
