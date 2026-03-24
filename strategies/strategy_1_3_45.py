"""
strategies/strategy_1_3_45.py

Python translation of: 28.02.HK50.Batch1Strategy 1.3.45

Note: This version prioritises research usability over exact MT5 replication.
  - 96 trades vs MT5's 105 (8% difference — acceptable)
  - Win rate 51% vs MT5's 58% (OHLC trailing stop limitation)
  - Relative comparisons between parameter sets are valid
  - Good enough for WFO, Monte Carlo, parameter search, AI training
"""

from __future__ import annotations
import pandas as pd
import talib
from engine.base_strategy import BaseStrategy, BarContext


class Strategy_1_3_45(BaseStrategy):

    name = "Strategy 1.3.45 (HK50 H1)"

    params = {
        "mmLots":              70.0,
        "StopLossCoef1":        2.5,
        "ProfitTargetCoef1":    2.8,
        "TrailingActCef1":      1.4,
        "TrailingStop1":      127.5,
        "StopLossCoef2":        3.0,
        "ProfitTargetCoef2":    2.2,
        "TrailingStopCoef1":    1.0,
        "LWMAPeriod1":           14,
        "IndicatorCrsMAPrd1":    47,
        "ATR1_period":           19,
        "ATR2_period":           45,
        "ATR3_period":           14,
        "ATR4_period":          100,
        "Highest_period":        50,
        "Lowest_period":         50,
    }

    lot_value: float = 0.1285   # HK50: 1 HKD per point / USDHKD ~7.78

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        mid           = (df["high"] + df["low"]) / 2.0
        df["ao"]      = mid.rolling(5).mean() - mid.rolling(34).mean()
        df["lwma"]    = talib.WMA(c, self.p("LWMAPeriod1"))
        df["lwma_ma"] = talib.EMA(df["lwma"].values, self.p("IndicatorCrsMAPrd1"))
        df["atr_19"]  = talib.ATR(h, l, c, self.p("ATR1_period"))
        df["atr_45"]  = talib.ATR(h, l, c, self.p("ATR2_period"))
        df["atr_14"]  = talib.ATR(h, l, c, self.p("ATR3_period"))
        df["atr_100"] = talib.ATR(h, l, c, self.p("ATR4_period"))
        df["highest"] = df["high"].rolling(self.p("Highest_period")).max()
        df["lowest"]  = df["low"].rolling(self.p("Lowest_period")).min()
        return df

    def on_bar(self, ctx: BarContext):
        i = ctx.bar_index

        if ctx.has_position or ctx.has_pending:
            return

        df = ctx._df

        # Long: AO crosses from negative to positive
        long_signal  = self.crosses_above(df["ao"], 0.0, i)

        # Short: LWMA crosses below its EMA
        short_signal = self.crosses_below_series(df["lwma"], df["lwma_ma"], i)

        if long_signal and not short_signal:
            entry = float(df["highest"].iloc[i - 1])
            atr19 = float(df["atr_19"].iloc[i - 1])
            atr45 = float(df["atr_45"].iloc[i - 1])

            if any(v != v or v == 0 for v in [entry, atr19, atr45]):
                return

            sl = entry - self.p("StopLossCoef1")     * atr19
            tp = entry + self.p("ProfitTargetCoef1") * atr19

            self._next_trail_dist       = self.p("TrailingStop1")
            self._next_trail_activation = self.p("TrailingActCef1") * atr45

            ctx.buy_stop(
                price=entry, sl=sl, tp=tp,
                lots=self.p("mmLots"),
                expiry_bars=1,
                comment="long_ao_cross",
            )

        elif short_signal and not long_signal:
            entry  = float(df["lowest"].iloc[i - 1])
            atr14  = float(df["atr_14"].iloc[i - 1])
            atr19  = float(df["atr_19"].iloc[i - 1])
            atr100 = float(df["atr_100"].iloc[i - 1])

            if any(v != v or v == 0 for v in [entry, atr14, atr19, atr100]):
                return

            sl = entry + self.p("StopLossCoef2")     * atr14
            tp = entry - self.p("ProfitTargetCoef2") * atr19

            self._next_trail_dist       = self.p("TrailingStopCoef1") * atr100
            self._next_trail_activation = 0.0

            ctx.sell_stop(
                price=entry, sl=sl, tp=tp,
                lots=self.p("mmLots"),
                expiry_bars=1,
                comment="short_lwma_cross",
            )

    def on_start(self, df: pd.DataFrame):
        self._next_trail_dist       = 0.0
        self._next_trail_activation = 0.0
