# Script to fetch new daily data from MT5 and store in database.
import sys
sys.path.append('.')

from data.fetcher import MT5Fetcher
from data.storage import DataStorage
import yaml
import config.settings as settings
from datetime import datetime, timedelta

def main():
    # Load symbols
    with open(settings.SYMBOLS_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    symbols = [s['name'] for s in config['symbols']]
    
    fetcher = MT5Fetcher()
    storage = DataStorage()
    
    # For each symbol, fetch last N days (e.g., 100) and store if not already present.
    # To avoid duplicates, you might fetch from last stored date to now.
    # Simplified: fetch last 365 days and store (append mode will handle duplicates if index is time)
    end = datetime.now()
    start = end - timedelta(days=365*6)
    
    for sym in symbols:
        print(f"Fetching {sym}...")
        df = fetcher.fetch_historical(sym, start_date=start, end_date=end)
        if not df.empty:
            storage.save_bars(sym, df)
            print(f"Saved {len(df)} bars for {sym}")
        else:
            print(f"No data for {sym}")
    
    fetcher.shutdown()

if __name__ == "__main__":
    main()
