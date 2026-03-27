from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


PRICE_CLOSE = 0
PRICE_OPEN = 1
PRICE_HIGH = 2
PRICE_LOW = 3
PRICE_MEDIAN = 4
PRICE_TYPICAL = 5
PRICE_WEIGHTED = 6


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=max(int(period), 1), adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    weights = pd.Series(range(1, period + 1), dtype=float)
    return series.rolling(period).apply(
        lambda values: float((pd.Series(values) * weights).sum() / weights.sum()),
        raw=False,
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(max(int(period), 1)).mean()


def _price_series(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    mode: int,
) -> pd.Series:
    if mode == PRICE_OPEN:
        return open_
    if mode == PRICE_HIGH:
        return high
    if mode == PRICE_LOW:
        return low
    if mode == PRICE_MEDIAN:
        return (high + low) / 2.0
    if mode == PRICE_TYPICAL:
        return (high + low + close) / 3.0
    if mode == PRICE_WEIGHTED:
        return (high + low + (2.0 * close)) / 4.0
    return close


def select_price(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    mode: int,
) -> pd.Series:
    return _price_series(open_, high, low, close, mode)


def bollinger_bands(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    deviations: float = 2.0,
    mode: int = PRICE_CLOSE,
) -> pd.DataFrame:
    period = max(int(period), 1)
    source = _price_series(open_, high, low, close, int(mode))
    middle = source.rolling(period, min_periods=period).mean()
    std = source.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + float(deviations) * std
    lower = middle - float(deviations) * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower}, index=source.index)


def osma(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    mode: int = PRICE_CLOSE,
) -> pd.Series:
    source = _price_series(open_, high, low, close, int(mode))
    macd = ema(source, fast_period) - ema(source, slow_period)
    signal = ema(macd, signal_period)
    return macd - signal


def sq_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (low - prev_close).abs(),
            (high - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    values: list[float] = []
    prev_atr = 0.0
    for idx, tr in enumerate(true_range.astype(float)):
        if idx == 0:
            atr_value = float(high.iloc[0] - low.iloc[0])
        else:
            lookback = min(idx + 1, period)
            multiplier = prev_atr if pd.notna(prev_atr) else 0.0
            atr_value = (((lookback - 1) * multiplier) + float(tr)) / lookback
        values.append(atr_value)
        prev_atr = atr_value

    return pd.Series(values, index=close.index, dtype=float)


def sq_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (low - prev_close).abs(),
            (high - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def sq_highest(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    mode: int = PRICE_HIGH,
) -> pd.Series:
    period = max(int(period), 1)
    source = _price_series(open_, high, low, close, int(mode))
    result = source.rolling(period, min_periods=period).max()
    if len(result) > 0:
        result.iloc[: period - 1] = 0.0
    return result.fillna(0.0)


def sq_lowest(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    mode: int = PRICE_LOW,
) -> pd.Series:
    period = max(int(period), 1)
    source = _price_series(open_, high, low, close, int(mode))
    result = source.rolling(period, min_periods=period).min()
    if len(result) > 0:
        result.iloc[: period - 1] = 0.0
    return result.fillna(0.0)


def sq_sr_percent_rank(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    mode: int,
    length: int,
    atr_period: int,
) -> pd.Series:
    length = max(int(length), 2)
    mode = 2 if int(mode) not in (1, 2) else int(mode)
    atr_values = sq_atr(high, low, close, max(int(atr_period), 2))

    values = [0.0] * len(close)
    for idx in range(len(close)):
        if idx < length:
            continue
        count = 0
        current_close = float(close.iloc[idx])
        current_atr = float(atr_values.iloc[idx])
        for lookback in range(1, length + 1):
            prev_idx = idx - lookback
            hi = float(high.iloc[prev_idx])
            lo = float(low.iloc[prev_idx])
            if mode == 1:
                in_range = current_close > lo and current_close < hi
            else:
                in_range = current_close > (lo - current_atr) and current_close < (hi + current_atr)
            if in_range:
                count += 1
        values[idx] = (count / length) * 100.0

    return pd.Series(values, index=close.index, dtype=float)


def _calculate_session_times(
    current_time: pd.Timestamp,
    start_hours: int,
    start_minutes: int,
    end_hours: int,
    end_minutes: int,
    start_hhmm: int,
    end_hhmm: int,
    previous_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    day_start = current_time.normalize()
    start_time = day_start + timedelta(hours=start_hours, minutes=start_minutes)
    end_time = day_start + timedelta(hours=end_hours, minutes=end_minutes)

    if start_hhmm >= end_hhmm:
        end_time += timedelta(days=1)

    if current_time < (end_time - timedelta(days=1)):
        start_time -= timedelta(days=1)
        end_time -= timedelta(days=1)
    elif previous_end is not None and (current_time > end_time or end_time == previous_end):
        start_time += timedelta(days=1)
        end_time += timedelta(days=1)

    return start_time, end_time


def sq_session_ohlc(
    df: pd.DataFrame,
    type_: int,
    start_hours: int,
    start_minutes: int,
    end_hours: int,
    end_minutes: int,
    days_ago: int = 0,
) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("sq_session_ohlc requires a DatetimeIndex")

    days_ago = max(int(days_ago), 0)
    type_ = int(type_)
    start_hhmm = int(start_hours) * 100 + int(start_minutes)
    end_hhmm = int(end_hours) * 100 + int(end_minutes)

    session_open = [-1.0] * (days_ago + 1)
    session_high = [-1.0] * (days_ago + 1)
    session_low = [-1.0] * (days_ago + 1)
    session_close = [-1.0] * (days_ago + 1)

    values: list[float] = []
    session_start: pd.Timestamp | None = None
    session_end: pd.Timestamp | None = None
    waiting_for_session = True

    for idx, current_time in enumerate(df.index):
        if session_end is None or current_time >= session_end:
            session_start, session_end = _calculate_session_times(
                current_time=current_time,
                start_hours=int(start_hours),
                start_minutes=int(start_minutes),
                end_hours=int(end_hours),
                end_minutes=int(end_minutes),
                start_hhmm=start_hhmm,
                end_hhmm=end_hhmm,
                previous_end=session_end,
            )
            waiting_for_session = True

        if session_start is not None and current_time >= session_start:
            if waiting_for_session:
                for shift_idx in range(days_ago, 0, -1):
                    session_open[shift_idx] = session_open[shift_idx - 1]
                    session_high[shift_idx] = session_high[shift_idx - 1]
                    session_low[shift_idx] = session_low[shift_idx - 1]
                    session_close[shift_idx] = session_close[shift_idx - 1]

                session_open[0] = float(df["open"].iloc[idx])
                session_high[0] = float(df["high"].iloc[idx])
                session_low[0] = float(df["low"].iloc[idx])
                session_close[0] = float(df["close"].iloc[idx])
                waiting_for_session = False
            elif current_time < session_end:
                session_high[0] = max(session_high[0], float(df["high"].iloc[idx]))
                session_low[0] = min(session_low[0], float(df["low"].iloc[idx]))
                session_close[0] = float(df["close"].iloc[idx])

        if type_ == 1:
            values.append(session_open[days_ago])
        elif type_ == 2:
            values.append(session_high[days_ago])
        elif type_ == 3:
            values.append(session_low[days_ago])
        elif type_ == 4:
            values.append(session_close[days_ago])
        else:
            values.append(0.0)

    return pd.Series(values, index=df.index, dtype=float)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(max(int(period), 1), min_periods=1).mean()


def smma(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _ma(series: pd.Series, period: int, method: str = "sma") -> pd.Series:
    method = (method or "sma").lower()
    if method == "ema":
        return ema(series, period)
    if method == "smma":
        return smma(series, period)
    if method == "wma":
        return wma(series, period)
    return sma(series, period)


def sq_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    period = max(int(period), 1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
        dtype=float,
    )
    tr = sq_true_range(high, low, close)
    tr_smooth = smma(tr, period).replace(0.0, np.nan)
    plus_di = (100.0 * smma(plus_dm, period) / tr_smooth).fillna(0.0)
    minus_di = (100.0 * smma(minus_dm, period) / tr_smooth).fillna(0.0)
    di_sum = plus_di + minus_di
    dx = pd.Series(
        np.where(di_sum.abs() < 1e-12, 50.0, 100.0 * (plus_di - minus_di).abs() / di_sum),
        index=high.index,
        dtype=float,
    )
    adx = smma(dx, period)
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=high.index)


def sq_aroon(high: pd.Series, low: pd.Series, period: int = 9) -> pd.DataFrame:
    period = max(int(period), 2)

    def _highest_idx(values: np.ndarray) -> float:
        return float(period - 1 - int(np.argmax(values)))

    def _lowest_idx(values: np.ndarray) -> float:
        return float(period - 1 - int(np.argmin(values)))

    up = 100.0 - high.rolling(period, min_periods=period).apply(_highest_idx, raw=True) * 100.0 / period
    down = 100.0 - low.rolling(period, min_periods=period).apply(_lowest_idx, raw=True) * 100.0 / period
    return pd.DataFrame({"bulls": up.fillna(0.0), "bears": down.fillna(0.0)}, index=high.index)


def sq_avg_volume(volume: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period), 1)
    return volume.rolling(period, min_periods=1).mean()


def sq_bb_width_ratio(series: pd.Series, period: int = 20, deviations: float = 2.0) -> pd.Series:
    period = max(int(period), 1)
    mid = sma(series, period)
    std = series.rolling(period, min_periods=1).std(ddof=0)
    upper = mid + deviations * std
    lower = mid - deviations * std
    return ((upper - lower) / mid.replace(0.0, np.nan)).fillna(0.0) * 100.0


def sq_bears_power(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 13,
    mode: int = PRICE_CLOSE,
) -> pd.Series:
    base = _price_series(open_, high, low, close, mode)
    return low - ema(base, max(int(period), 1))


def sq_bulls_power(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 13,
    mode: int = PRICE_CLOSE,
) -> pd.Series:
    base = _price_series(open_, high, low, close, mode)
    return high - ema(base, max(int(period), 1))


def sq_cci(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    mode: int = PRICE_TYPICAL,
) -> pd.Series:
    period = max(int(period), 1)
    src = _price_series(open_, high, low, close, mode)
    ma = sma(src, period)
    md = src.rolling(period, min_periods=1).apply(lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
    return ((src - ma) / (0.015 * md.replace(0.0, np.nan))).fillna(0.0)


def sq_efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    period = max(int(period), 1)
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period, min_periods=1).sum()
    return (direction / volatility.replace(0.0, np.nan)).fillna(0.0)


def sq_fractal(high: pd.Series, low: pd.Series, bars: int = 5) -> pd.DataFrame:
    bars = max(int(bars), 3)
    if bars % 2 == 0:
        bars += 1
    wing = (bars - 1) // 2
    up = pd.Series(0.0, index=high.index)
    down = pd.Series(0.0, index=high.index)
    for idx in range(wing, len(high) - wing):
        h = high.iloc[idx]
        l = low.iloc[idx]
        if h >= high.iloc[idx - wing: idx + wing + 1].max():
            up.iloc[idx] = float(h)
        if l <= low.iloc[idx - wing: idx + wing + 1].min():
            down.iloc[idx] = float(l)
    return pd.DataFrame({"up": up, "down": down}, index=high.index)


def sq_gann_hilo(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    period = max(int(period), 1)
    avg_high = sma(high, period)
    avg_low = sma(low, period)
    trend = pd.Series(index=close.index, dtype=float)
    trend[:] = np.where(close >= avg_high, avg_low, np.where(close <= avg_low, avg_high, np.nan))
    return trend.ffill().fillna(avg_low)


def sq_heiken_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = pd.Series(index=df.index, dtype=float)
    if len(df):
        ha_open.iloc[0] = float((df["open"].iloc[0] + df["close"].iloc[0]) / 2.0)
    for idx in range(1, len(df)):
        ha_open.iloc[idx] = float((ha_open.iloc[idx - 1] + ha_close.iloc[idx - 1]) / 2.0)
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close}, index=df.index)


def sq_highest_index(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    mode: int = PRICE_HIGH,
) -> pd.Series:
    period = max(int(period), 1)
    source = _price_series(open_, high, low, close, mode)
    return source.rolling(period, min_periods=period).apply(
        lambda values: float(period - 1 - int(np.argmax(values))),
        raw=True,
    ).fillna(0.0)


def sq_lowest_index(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    mode: int = PRICE_LOW,
) -> pd.Series:
    period = max(int(period), 1)
    source = _price_series(open_, high, low, close, mode)
    return source.rolling(period, min_periods=period).apply(
        lambda values: float(period - 1 - int(np.argmin(values))),
        raw=True,
    ).fillna(0.0)


def sq_hull_moving_average(series: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period), 1)
    half = max(period // 2, 1)
    root = max(int(np.sqrt(period)), 1)
    hull = (2.0 * wma(series, half)) - wma(series, period)
    return wma(hull, root)


def sq_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    tenkan = max(int(tenkan), 1)
    kijun = max(int(kijun), 1)
    senkou = max(int(senkou), 1)
    tenkan_sen = (high.rolling(tenkan, min_periods=1).max() + low.rolling(tenkan, min_periods=1).min()) / 2.0
    kijun_sen = (high.rolling(kijun, min_periods=1).max() + low.rolling(kijun, min_periods=1).min()) / 2.0
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_span_b = ((high.rolling(senkou, min_periods=1).max() + low.rolling(senkou, min_periods=1).min()) / 2.0).shift(kijun)
    chikou_span = close.shift(-kijun)
    return pd.DataFrame(
        {
            "tenkan": tenkan_sen,
            "kijun": kijun_sen,
            "senkou_a": senkou_span_a,
            "senkou_b": senkou_span_b,
            "chikou": chikou_span,
        },
        index=high.index,
    )


def sq_kama(close: pd.Series, er_period: int = 10, fast_period: int = 2, slow_period: int = 30) -> pd.Series:
    er = sq_efficiency_ratio(close, er_period)
    fast = 2.0 / (max(int(fast_period), 1) + 1.0)
    slow = 2.0 / (max(int(slow_period), 1) + 1.0)
    sc = (er * (fast - slow) + slow) ** 2
    out = pd.Series(index=close.index, dtype=float)
    if len(close):
        out.iloc[0] = float(close.iloc[0])
    for idx in range(1, len(close)):
        prev = out.iloc[idx - 1]
        out.iloc[idx] = prev + sc.iloc[idx] * (close.iloc[idx] - prev)
    return out


def sq_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    mid = ema(close, max(int(period), 1))
    rng = sq_atr(high, low, close, max(int(period), 1))
    return pd.DataFrame({"middle": mid, "upper": mid + multiplier * rng, "lower": mid - multiplier * rng}, index=close.index)


def sq_mt_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, multiplier: float = 2.0) -> pd.DataFrame:
    typical = (high + low + close) / 3.0
    mid = ema(typical, max(int(period), 1))
    rng = sq_atr(high, low, close, max(int(period), 1))
    return pd.DataFrame({"middle": mid, "upper": mid + multiplier * rng, "lower": mid - multiplier * rng}, index=close.index)


def sq_laguerre_rsi(close: pd.Series, gamma: float = 0.5) -> pd.Series:
    gamma = float(np.clip(gamma, 0.0, 0.99))
    l0 = pd.Series(index=close.index, dtype=float)
    l1 = pd.Series(index=close.index, dtype=float)
    l2 = pd.Series(index=close.index, dtype=float)
    l3 = pd.Series(index=close.index, dtype=float)
    out = pd.Series(index=close.index, dtype=float)
    if len(close) == 0:
        return out
    l0.iloc[0] = l1.iloc[0] = l2.iloc[0] = l3.iloc[0] = float(close.iloc[0])
    out.iloc[0] = 0.5
    for idx in range(1, len(close)):
        price = float(close.iloc[idx])
        l0.iloc[idx] = (1.0 - gamma) * price + gamma * l0.iloc[idx - 1]
        l1.iloc[idx] = -gamma * l0.iloc[idx] + l0.iloc[idx - 1] + gamma * l1.iloc[idx - 1]
        l2.iloc[idx] = -gamma * l1.iloc[idx] + l1.iloc[idx - 1] + gamma * l2.iloc[idx - 1]
        l3.iloc[idx] = -gamma * l2.iloc[idx] + l2.iloc[idx - 1] + gamma * l3.iloc[idx - 1]
        cu = sum(max(a - b, 0.0) for a, b in [(l0.iloc[idx], l1.iloc[idx]), (l1.iloc[idx], l2.iloc[idx]), (l2.iloc[idx], l3.iloc[idx])])
        cd = sum(max(b - a, 0.0) for a, b in [(l0.iloc[idx], l1.iloc[idx]), (l1.iloc[idx], l2.iloc[idx]), (l2.iloc[idx], l3.iloc[idx])])
        out.iloc[idx] = cu / (cu + cd) if (cu + cd) else out.iloc[idx - 1]
    return out * 100.0


def sq_linreg(series: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period), 1)
    x = np.arange(period, dtype=float)

    def _linreg(values: np.ndarray) -> float:
        slope, intercept = np.polyfit(x, values, 1)
        return float(intercept + slope * (period - 1))

    return series.rolling(period, min_periods=period).apply(_linreg, raw=True).fillna(0.0)


def sq_parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, maximum: float = 0.2) -> pd.Series:
    if len(high) == 0:
        return pd.Series(dtype=float)
    step = float(step)
    maximum = float(maximum)
    sar = pd.Series(index=high.index, dtype=float)
    uptrend = True
    ep = float(high.iloc[0])
    af = step
    sar.iloc[0] = float(low.iloc[0])
    for idx in range(1, len(high)):
        prev_sar = sar.iloc[idx - 1]
        sar.iloc[idx] = prev_sar + af * (ep - prev_sar)
        if uptrend:
            sar.iloc[idx] = min(sar.iloc[idx], float(low.iloc[idx - 1]), float(low.iloc[idx]))
            if float(low.iloc[idx]) < sar.iloc[idx]:
                uptrend = False
                sar.iloc[idx] = ep
                ep = float(low.iloc[idx])
                af = step
            else:
                if float(high.iloc[idx]) > ep:
                    ep = float(high.iloc[idx])
                    af = min(af + step, maximum)
        else:
            sar.iloc[idx] = max(sar.iloc[idx], float(high.iloc[idx - 1]), float(high.iloc[idx]))
            if float(high.iloc[idx]) > sar.iloc[idx]:
                uptrend = True
                sar.iloc[idx] = ep
                ep = float(high.iloc[idx])
                af = step
            else:
                if float(low.iloc[idx]) < ep:
                    ep = float(low.iloc[idx])
                    af = min(af + step, maximum)
    return sar


def sq_qqe(close: pd.Series, rsi_period: int = 14, smooth_period: int = 5, factor: float = 4.236) -> pd.DataFrame:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = smma(gain, max(int(rsi_period), 1))
    avg_loss = smma(loss, max(int(rsi_period), 1))
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    rsi_ma = ema(rsi, max(int(smooth_period), 1))
    atr_rsi = ema((rsi_ma - rsi_ma.shift(1)).abs().fillna(0.0), max(int(rsi_period), 1))
    upper = rsi_ma + factor * atr_rsi
    lower = rsi_ma - factor * atr_rsi
    return pd.DataFrame({"qqe": rsi_ma, "upper": upper, "lower": lower}, index=close.index)


def sq_roc(close: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period), 1)
    prev = close.shift(period)
    return (((close - prev) / prev.replace(0.0, np.nan)) * 100.0).fillna(0.0)


def sq_schaff_trend_cycle(close: pd.Series, fast: int = 23, slow: int = 50, cycle: int = 10) -> pd.Series:
    fast_ema = ema(close, max(int(fast), 1))
    slow_ema = ema(close, max(int(slow), 1))
    macd = fast_ema - slow_ema
    low_macd = macd.rolling(max(int(cycle), 1), min_periods=1).min()
    high_macd = macd.rolling(max(int(cycle), 1), min_periods=1).max()
    stoch1 = ((macd - low_macd) / (high_macd - low_macd).replace(0.0, np.nan) * 100.0).fillna(0.0)
    stoch1 = ema(stoch1, 3)
    low_stoch = stoch1.rolling(max(int(cycle), 1), min_periods=1).min()
    high_stoch = stoch1.rolling(max(int(cycle), 1), min_periods=1).max()
    stc = ((stoch1 - low_stoch) / (high_stoch - low_stoch).replace(0.0, np.nan) * 100.0).fillna(0.0)
    return ema(stc, 3)


def sq_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 5,
    d_period: int = 3,
    slowing: int = 3,
    ma_method: str = "sma",
    price_mode: str = "lowhigh",
) -> pd.DataFrame:
    k_period = max(int(k_period), 1)
    d_period = max(int(d_period), 1)
    slowing = max(int(slowing), 1)
    if str(price_mode).lower() == "closeclose":
        low_src = close.rolling(k_period, min_periods=1).min()
        high_src = close.rolling(k_period, min_periods=1).max()
    else:
        low_src = low.rolling(k_period, min_periods=1).min()
        high_src = high.rolling(k_period, min_periods=1).max()
    raw_k = ((close - low_src) / (high_src - low_src).replace(0.0, np.nan) * 100.0).fillna(50.0).clip(0.0, 100.0)
    main = _ma(raw_k, slowing, ma_method)
    signal = _ma(main, d_period, ma_method)
    return pd.DataFrame({"main": main, "signal": signal}, index=close.index)


def sq_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    atr_values = sq_atr(high, low, close, max(int(period), 1))
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr_values
    lower_basic = hl2 - multiplier * atr_values
    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    if len(close) == 0:
        return pd.DataFrame({"supertrend": supertrend, "direction": direction})
    supertrend.iloc[0] = lower_band.iloc[0]
    direction.iloc[0] = 1
    for idx in range(1, len(close)):
        upper_band.iloc[idx] = (
            upper_basic.iloc[idx]
            if close.iloc[idx - 1] > upper_band.iloc[idx - 1]
            else min(upper_basic.iloc[idx], upper_band.iloc[idx - 1])
        )
        lower_band.iloc[idx] = (
            lower_basic.iloc[idx]
            if close.iloc[idx - 1] < lower_band.iloc[idx - 1]
            else max(lower_basic.iloc[idx], lower_band.iloc[idx - 1])
        )
        if supertrend.iloc[idx - 1] == upper_band.iloc[idx - 1]:
            direction.iloc[idx] = 1 if close.iloc[idx] > upper_band.iloc[idx] else -1
        else:
            direction.iloc[idx] = -1 if close.iloc[idx] < lower_band.iloc[idx] else 1
        supertrend.iloc[idx] = lower_band.iloc[idx] if direction.iloc[idx] == 1 else upper_band.iloc[idx]
    return pd.DataFrame({"supertrend": supertrend, "direction": direction}, index=close.index)


def sq_ulcer_index(close: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period), 1)
    rolling_max = close.rolling(period, min_periods=1).max()
    drawdown_pct = ((close - rolling_max) / rolling_max.replace(0.0, np.nan) * 100.0).fillna(0.0)
    return np.sqrt((drawdown_pct.pow(2)).rolling(period, min_periods=1).mean())


def sq_vwap(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
    period = max(int(period), 1)
    price = (open_ + high + low + close) / 4.0
    num = (price * volume).rolling(period, min_periods=1).sum()
    den = volume.rolling(period, min_periods=1).sum()
    return (num / den.replace(0.0, np.nan)).fillna(0.0)


def sq_vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    period = max(int(period), 1)
    tr = sq_true_range(high, low, close).rolling(period, min_periods=1).sum()
    vm_plus = (high - low.shift(1)).abs().rolling(period, min_periods=1).sum()
    vm_minus = (low - high.shift(1)).abs().rolling(period, min_periods=1).sum()
    return pd.DataFrame(
        {"plus_vi": (vm_plus / tr.replace(0.0, np.nan)).fillna(0.0), "minus_vi": (vm_minus / tr.replace(0.0, np.nan)).fillna(0.0)},
        index=close.index,
    )


def sq_wpr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    period = max(int(period), 1)
    max_high = high.rolling(period, min_periods=1).max()
    min_low = low.rolling(period, min_periods=1).min()
    out = -((max_high - close) * 100.0 / (max_high - min_low).replace(0.0, np.nan))
    return out.fillna(0.0)


def sq_wave_trend(high: pd.Series, low: pd.Series, close: pd.Series, channel_length: int = 10, average_length: int = 21, signal_length: int = 4) -> pd.DataFrame:
    hlc3 = (high + low + close) / 3.0
    esa = ema(hlc3, max(int(channel_length), 1))
    d = ema((hlc3 - esa).abs(), max(int(channel_length), 1))
    ci = ((hlc3 - esa) / (0.015 * d.replace(0.0, np.nan))).fillna(0.0)
    wt1 = ema(ci, max(int(average_length), 1))
    wt2 = sma(wt1, max(int(signal_length), 1))
    return pd.DataFrame({"main": wt1, "signal": wt2}, index=close.index)


def _session_boundaries(index: pd.DatetimeIndex, time_from: str, time_to: str) -> tuple[pd.Series, pd.Series]:
    start_h, start_m = map(int, time_from.split(":"))
    end_h, end_m = map(int, time_to.split(":"))
    start = pd.Series(index.normalize() + pd.to_timedelta(start_h, unit="h") + pd.to_timedelta(start_m, unit="m"), index=index)
    end = pd.Series(index.normalize() + pd.to_timedelta(end_h, unit="h") + pd.to_timedelta(end_m, unit="m"), index=index)
    overnight = end <= start
    end.loc[overnight] = end.loc[overnight] + pd.Timedelta(days=1)
    before_start = index < start
    start.loc[before_start] = start.loc[before_start] - pd.Timedelta(days=1)
    end.loc[before_start] = end.loc[before_start] - pd.Timedelta(days=1)
    return start, end


def sq_highest_in_range(df: pd.DataFrame, time_from: str = "00:00", time_to: str = "00:00") -> pd.Series:
    start, end = _session_boundaries(df.index, time_from, time_to)
    current_high = 0.0
    last_value = 0.0
    last_usable = 0.0
    prev_window = None
    out = []
    for idx, ts in enumerate(df.index):
        window = (start.iloc[idx], end.iloc[idx])
        if prev_window is None or window != prev_window:
            if current_high > 0:
                last_value = current_high
            current_high = 0.0
            prev_window = window
        if start.iloc[idx] <= ts < end.iloc[idx]:
            current_high = max(current_high, float(df["high"].iloc[idx]))
        if last_value > 0:
            last_usable = last_value
            out.append(last_value)
        else:
            out.append(last_usable)
    return pd.Series(out, index=df.index, dtype=float)


def sq_lowest_in_range(df: pd.DataFrame, time_from: str = "00:00", time_to: str = "00:00") -> pd.Series:
    start, end = _session_boundaries(df.index, time_from, time_to)
    inf = float(0x6FFFFFFF)
    current_low = inf
    last_value = inf
    last_usable = 0.0
    prev_window = None
    out = []
    for idx, ts in enumerate(df.index):
        window = (start.iloc[idx], end.iloc[idx])
        if prev_window is None or window != prev_window:
            if current_low < inf:
                last_value = current_low
            current_low = inf
            prev_window = window
        if start.iloc[idx] <= ts < end.iloc[idx]:
            current_low = min(current_low, float(df["low"].iloc[idx]))
        if last_value < inf:
            last_usable = last_value
            out.append(last_value)
        else:
            out.append(last_usable)
    return pd.Series(out, index=df.index, dtype=float)


def sq_pivots(df: pd.DataFrame, start_hour: int = 8, start_minute: int = 20) -> pd.DataFrame:
    session_shift = pd.to_timedelta(int(start_hour), unit="h") + pd.to_timedelta(int(start_minute), unit="m")
    session_key = (df.index - session_shift).normalize()
    out = pd.DataFrame(0.0, index=df.index, columns=["pp", "r1", "r2", "r3", "s1", "s2", "s3"])
    grouped = df.groupby(session_key)
    session_stats = grouped.agg({"high": "max", "low": "min", "close": "last"})
    prev_stats = session_stats.shift(1)
    for key, rows in grouped.groups.items():
        if key not in prev_stats.index or prev_stats.loc[key].isna().any():
            continue
        prev_high = float(prev_stats.loc[key, "high"])
        prev_low = float(prev_stats.loc[key, "low"])
        prev_close = float(prev_stats.loc[key, "close"])
        pp = (prev_high + prev_low + prev_close) / 3.0
        r1 = (2 * pp) - prev_low
        s1 = (2 * pp) - prev_high
        r2 = pp + (prev_high - prev_low)
        s2 = pp - (prev_high - prev_low)
        r3 = pp + 2 * (prev_high - prev_low)
        s3 = pp - 2 * (prev_high - prev_low)
        out.loc[rows, :] = [pp, r1, r2, r3, s1, s2, s3]
    return out


def sq_reflex(close: pd.Series, period: int = 24) -> pd.Series:
    period = max(int(period), 1)
    a1 = float(np.exp(-1.414 * np.pi / (period * 0.5)))
    b1 = float(2.0 * a1 * np.cos(np.deg2rad((1.414 * 180.0) / (period * 0.5))))
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3
    ssm = pd.Series(index=close.index, dtype=float)
    ms = pd.Series(index=close.index, dtype=float)
    out = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        value = float(close.iloc[i])
        if i > 1:
            ssm.iloc[i] = c1 * (value + float(close.iloc[i - 1])) / 2.0 + c2 * ssm.iloc[i - 1] + c3 * ssm.iloc[i - 2]
        else:
            ssm.iloc[i] = value
        tslope = (ssm.iloc[i - period] - ssm.iloc[i]) / period if i >= period else 0.0
        reflex_sum = 0.0
        if i > period:
            for a in range(1, period + 1):
                reflex_sum += ssm.iloc[i] + a * tslope - ssm.iloc[i - a]
        reflex_sum /= period
        ms.iloc[i] = (0.04 * reflex_sum * reflex_sum) + (0.96 * ms.iloc[i - 1] if i > 0 else 0.0)
        out.iloc[i] = reflex_sum / np.sqrt(ms.iloc[i]) if ms.iloc[i] != 0 else 0.0
    return out


def sq_commercials_index(
    open_: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period_index: int = 200,
    cot_period: int = 40,
    smoothing: int = 3,
) -> pd.Series:
    cot_period = max(int(cot_period), 1)
    period_index = max(int(period_index), 1)
    oc = open_ - close
    atr_vals = sq_atr(high, low, close, cot_period).replace(0.0, np.nan)
    value1 = ((oc.rolling(cot_period + 1, min_periods=1).sum() / cot_period) / atr_vals * 100.0).fillna(0.0)
    min_v = value1.rolling(period_index, min_periods=1).min()
    max_v = value1.rolling(period_index, min_periods=1).max()
    value2 = ((value1 - min_v) / (max_v - min_v).replace(0.0, np.nan) * 100.0).fillna(0.0)
    return sma(value2, max(int(smoothing), 1))


def sq_fibo(
    df: pd.DataFrame,
    fibo_range: int = 1,
    x: int = 0,
    fibo_level: float = 61.8,
    custom_fibo_level: float = 0.0,
    start_date: pd.Timestamp | None = None,
) -> pd.Series:
    level = custom_fibo_level if float(fibo_level) == -9999999 else float(fibo_level)
    fibo_range = int(fibo_range)
    idx = df.index
    out = pd.Series(0.0, index=idx, dtype=float)

    if fibo_range in {1, 5}:
        key = idx.normalize()
    elif fibo_range in {2, 6}:
        key = (idx - pd.to_timedelta(idx.weekday, unit="D")).normalize()
    elif fibo_range in {3, 7}:
        key = pd.to_datetime(idx.to_period("M").astype(str))
    elif fibo_range in {4, 8}:
        days = max(int(x), 1)
        key = ((idx.normalize() - idx.normalize().min()) // pd.Timedelta(days=days)).astype(str)
    else:
        key = None

    if fibo_range in {9, 10}:
        bars = max(int(x), 1)
        if fibo_range == 9:
            upper = df["high"].rolling(bars, min_periods=1).max()
            lower = df["low"].rolling(bars, min_periods=1).min()
        else:
            open_close = pd.concat([df["open"], df["close"]], axis=1)
            upper = open_close.max(axis=1).rolling(bars, min_periods=1).max()
            lower = open_close.min(axis=1).rolling(bars, min_periods=1).min()
        bullish = df["close"].shift(1) > df["open"].shift(1)
        delta = (upper - lower) * level / 100.0
        out[:] = np.where(bullish, upper - delta, lower + delta)
        return out.ffill().fillna(0.0)

    grouped = df.groupby(key)
    prev = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"}).shift(1)
    for grp, rows in grouped.groups.items():
        if grp not in prev.index or prev.loc[grp].isna().any():
            continue
        upper = float(prev.loc[grp, "high"])
        lower = float(prev.loc[grp, "low"])
        if fibo_range in {5, 6, 7, 8}:
            upper = max(float(prev.loc[grp, "open"]), float(prev.loc[grp, "close"]))
            lower = min(float(prev.loc[grp, "open"]), float(prev.loc[grp, "close"]))
        bullish = float(prev.loc[grp, "close"]) > float(prev.loc[grp, "open"])
        delta = (upper - lower) * level / 100.0
        out.loc[rows] = (upper - delta) if bullish else (lower + delta)
    if start_date is not None:
        out.loc[idx < pd.Timestamp(start_date)] = 0.0
    return out.ffill().fillna(0.0)


INDICATOR_REGISTRY = {
    "SqADX": sq_adx,
    "SqATR": sq_atr,
    "SqAroon": sq_aroon,
    "SqAvgVolume": sq_avg_volume,
    "SqBBWidthRatio": sq_bb_width_ratio,
    "SqBearsPower": sq_bears_power,
    "SqBullsPower": sq_bulls_power,
    "SqCCI": sq_cci,
    "SqEfficiencyRatio": sq_efficiency_ratio,
    "SqFractal": sq_fractal,
    "SqGannHiLo": sq_gann_hilo,
    "SqHeikenAshi": sq_heiken_ashi,
    "SqHighest": sq_highest,
    "SqHighestIndex": sq_highest_index,
    "SqHullMovingAverage": sq_hull_moving_average,
    "SqIchimoku": sq_ichimoku,
    "SqKAMA": sq_kama,
    "SqKeltnerChannel": sq_keltner_channel,
    "SqLaguerreRSI": sq_laguerre_rsi,
    "SqLinReg": sq_linreg,
    "SqLowest": sq_lowest,
    "SqLowestIndex": sq_lowest_index,
    "SqMTKeltnerChannel": sq_mt_keltner_channel,
    "SqParabolicSAR": sq_parabolic_sar,
    "SqQQE": sq_qqe,
    "SqROC": sq_roc,
    "SqSRPercentRank": sq_sr_percent_rank,
    "SqSchaffTrendCycle": sq_schaff_trend_cycle,
    "SqSessionOHLC": sq_session_ohlc,
    "SqStochastic": sq_stochastic,
    "SqSuperTrend": sq_supertrend,
    "SqTrueRange": sq_true_range,
    "SqUlcerIndex": sq_ulcer_index,
    "SqVWAP": sq_vwap,
    "SqVortex": sq_vortex,
    "SqWPR": sq_wpr,
    "SqWaveTrend": sq_wave_trend,
    "CommercialsIndex": sq_commercials_index,
    "SqFibo": sq_fibo,
    "SqHighestInRange": sq_highest_in_range,
    "SqLowestInRange": sq_lowest_in_range,
    "SqPivots": sq_pivots,
    "SqReflex": sq_reflex,
}


def is_indicator_supported(name: str) -> bool:
    func = INDICATOR_REGISTRY.get(name)
    return func is not None and getattr(func, "__name__", "") != "_raise"


def supported_indicator_names() -> list[str]:
    return sorted(name for name in INDICATOR_REGISTRY if is_indicator_supported(name))


def indicator_function(name: str):
    if name not in INDICATOR_REGISTRY:
        raise KeyError(f"Unknown indicator: {name}")
    return INDICATOR_REGISTRY[name]
