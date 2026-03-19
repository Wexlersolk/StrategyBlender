import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import config.settings as settings

class MT5Fetcher:
    """Fetch OHLCV data from MetaTrader 5."""
    
    def __init__(self):
        if not mt5.initialize(path=settings.MT5_PATH):
            raise Exception("MT5 initialization failed")
        
        # Login if credentials are provided
        if settings.MT5_LOGIN != 0 and settings.MT5_PASSWORD and settings.MT5_SERVER:
            authorized = mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER
            )
            if not authorized:
                error = mt5.last_error()
                raise Exception(f"MT5 login failed: {error}")

    def fetch_historical(self, symbol, timeframe=mt5.TIMEFRAME_D1, 
                         start_date=None, end_date=None):
        """
        Fetch historical bars.
        :param symbol: str, e.g. 'XAUUSD'
        :param timeframe: mt5 timeframe constant
        :param start_date: datetime
        :param end_date: datetime
        :return: DataFrame with columns: time, open, high, low, close, tick_volume
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365*3)  # 3 years default

        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df[['open','high','low','close','tick_volume']]

    def fetch_live(self, symbol, timeframe=mt5.TIMEFRAME_D1, count=100):
        """Fetch recent bars (live operation)."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df[['open','high','low','close','tick_volume']]

    def shutdown(self):
        mt5.shutdown()
