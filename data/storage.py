import pandas as pd
from sqlalchemy import create_engine
import config.settings as settings

class DataStorage:
    """Handles saving and loading market data to/from SQL database."""
    
    def __init__(self):
        self.engine = create_engine(settings.DB_PATH)
    
    def save_bars(self, symbol, df):
        """Store DataFrame in table named f"{symbol}_daily"."""
        table_name = f"{symbol}_daily"
        df.to_sql(table_name, self.engine, if_exists='append', index=True)
    
    def load_bars(self, symbol, start_date=None, end_date=None):
        """Load bars for a symbol between dates."""
        table_name = f"{symbol}_daily"
        query = f"SELECT * FROM {table_name}"
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append(f"time >= '{start_date}'")
            if end_date:
                conditions.append(f"time <= '{end_date}'")
            query += " WHERE " + " AND ".join(conditions)
        df = pd.read_sql(query, self.engine, index_col='time', parse_dates=['time'])
        return df
    
    def get_all_symbols_data(self, symbols, start_date, end_date):
        """Load data for multiple symbols and return a dict symbol->DataFrame."""
        data = {}
        for sym in symbols:
            df = self.load_bars(sym, start_date, end_date)
            if not df.empty:
                data[sym] = df
        return data
