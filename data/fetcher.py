import pandas as pd
from mt5linux import MetaTrader5
from datetime import datetime, timedelta
import config.settings as settings

_mt5 = MetaTrader5(host='localhost', port=18812)

# Timeframe constants — resolved at import time
def _tf(name):
    try:
        return getattr(_mt5, f'TIMEFRAME_{name}')
    except AttributeError:
        defaults = {'M1':1,'M5':5,'M15':15,'M30':30,
                    'H1':16385,'H4':16388,'D1':16408}
        return defaults.get(name, 16408)

TIMEFRAMES = {tf: _tf(tf) for tf in ['M1','M5','M15','M30','H1','H4','D1']}


class MT5Fetcher:
    """Fetch OHLCV data from MetaTrader 5 for any timeframe."""

    def __init__(self):
        if not _mt5.initialize():
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
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        df['tick_volume'] = df['tick_volume'].astype('int64')
        return df

    def fetch_historical(self, symbol, timeframe='D1',
                         start_date=None, end_date=None):
        """
        Fetch historical OHLCV bars.
        timeframe: 'M1','M5','M15','M30','H1','H4','D1'
        """
        tf_const = TIMEFRAMES.get(timeframe.upper(), TIMEFRAMES['D1'])

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 6)
        if end_date is None:
            end_date = datetime.now()

        rates = _mt5.copy_rates_range(symbol, tf_const, start_date, end_date)
        if rates is None or len(rates) == 0:
            print(f"  No data for {symbol} {timeframe}")
            return pd.DataFrame()

        return self._clean_df(rates)

    def shutdown(self):
        _mt5.shutdown()
