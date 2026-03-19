import pandas as pd
import numpy as np
import talib

def compute_features(df):
    """
    Compute technical indicators from OHLCV data.
    Input: DataFrame with columns: open, high, low, close, volume (tick_volume)
    Output: DataFrame with added feature columns.
    """
    df = df.copy()
    
    # Price-based
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_width'] = (df['bbands_upper'] - df['bbands_lower']) / df['bbands_middle']
    
    # Momentum
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    
    # Trend
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Volume
    df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
    
    # Rolling statistics
    df['close_ma20'] = df['close'].rolling(20).mean()
    df['close_ma50'] = df['close'].rolling(50).mean()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Drop NaN rows (first periods with insufficient data)
    df.dropna(inplace=True)
    
    return df

def get_feature_columns():
    """Return list of feature column names (excluding price columns)."""
    return ['returns', 'log_returns', 'atr', 'bb_width', 'rsi', 'macd', 
            'macd_signal', 'macd_hist', 'adx', 'volume_ratio', 
            'close_ma20', 'close_ma50', 'volatility_20']
