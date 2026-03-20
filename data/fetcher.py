import pandas as pd
from mt5linux import MetaTrader5
from datetime import datetime, timedelta
import config.settings as settings

# Shared mt5linux instance — bridge server must be running on port 18812
_mt5 = MetaTrader5(host='localhost', port=18812)


class MT5Fetcher:
    """Fetch OHLCV data from MetaTrader 5."""

    def __init__(self):
        if not _mt5.initialize(path=settings.MT5_PATH):
            raise Exception("MT5 initialization failed")

        if settings.MT5_LOGIN != 0 and settings.MT5_PASSWORD and settings.MT5_SERVER:
            authorized = _mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER
            )
            if not authorized:
                raise Exception(f"MT5 login failed: {_mt5.last_error()}")

    def _clean_df(self, rates):
        """Convert raw MT5 rates array to a clean DataFrame."""
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]

        # MT5 returns tick_volume as uint64 — SQLAlchemy/SQLite don't support it
        df['tick_volume'] = df['tick_volume'].astype('int64')

        return df

    def fetch_historical(self, symbol, timeframe=None, start_date=None, end_date=None):
        """
        Fetch historical bars.
        :param symbol: str, e.g. 'XAUUSD'
        :param timeframe: mt5 timeframe constant (defaults to TIMEFRAME_D1)
        :param start_date: datetime
        :param end_date: datetime
        :return: DataFrame with columns: open, high, low, close, tick_volume
        """
        if timeframe is None:
            timeframe = _mt5.TIMEFRAME_D1
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 3)

        rates = _mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        return self._clean_df(rates)

    def fetch_live(self, symbol, timeframe=None, count=100):
        """Fetch the most recent bars."""
        if timeframe is None:
            timeframe = _mt5.TIMEFRAME_D1

        rates = _mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        return self._clean_df(rates)

    def shutdown(self):
        _mt5.shutdown()
