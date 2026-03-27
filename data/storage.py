import pandas as pd

from engine.data_loader import load_bars as load_exported_bars

class DataStorage:
    """Compatibility wrapper around the exported MT5 data files."""

    def __init__(self):
        self.engine = None

    def save_bars(self, symbol, df):
        raise NotImplementedError("Saving bars is no longer handled through DataStorage.")

    def load_bars(self, symbol, start_date=None, end_date=None):
        try:
            return load_exported_bars(symbol, "D1", date_from=start_date, date_to=end_date)
        except Exception as e:
            print(f"Warning: could not load data for {symbol}: {e}")
            return pd.DataFrame()

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
