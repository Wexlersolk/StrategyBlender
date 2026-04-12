from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.native_strategy_lab import get_native_strategy_record


ROOT = Path(__file__).resolve().parent.parent
MT5_GENERATED_DIR = ROOT / "mt5" / "generated"


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _session_hours(session_filter: str) -> tuple[int, int]:
    mapping = {
        "all_day": (0, 23),
        "london_only": (6, 11),
        "ny_only": (12, 18),
        "london_ny": (6, 18),
    }
    return mapping.get(session_filter, (6, 18))


def _magic_number(strategy_id: str) -> int:
    digits = "".join(ch for ch in strategy_id if ch.isdigit())
    if not digits:
        return 1100001
    return int(digits[-7:])


def _render_xau_discovery_ea(strategy_id: str, payload: dict[str, Any]) -> str:
    params = dict(payload.get("params", {}))
    entry = str(payload.get("entry_archetype", "breakout_stop"))
    volatility = str(payload.get("volatility_filter", "bb_width"))
    session_filter = str(payload.get("session_filter", "london_ny"))
    stop_model = str(payload.get("stop_model", "atr"))
    target_model = str(payload.get("target_model", "fixed_rr"))
    exit_model = str(payload.get("exit_model", "session_close"))
    from_hour, to_hour = _session_hours(session_filter)

    return f"""#property strict

#include <Trade/Trade.mqh>

CTrade trade;

input group "Strategy"
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;
input double InpLots = { _float(params.get("mmLots", 1.0), 1.0) };
input int InpMagicNumber = {_magic_number(strategy_id)};

input group "Discovery Blocks"
input string InpEntryArchetype = "{entry}";
input string InpVolatilityFilter = "{volatility}";
input string InpSessionFilter = "{session_filter}";
input string InpStopModel = "{stop_model}";
input string InpTargetModel = "{target_model}";
input string InpExitModel = "{exit_model}";

input group "Parameters"
input int HighestPeriod = {_int(params.get("HighestPeriod", 24), 24)};
input int LowestPeriod = {_int(params.get("LowestPeriod", 24), 24)};
input int ATRPeriod = {_int(params.get("ATRPeriod", 14), 14)};
input int FastEMA = {_int(params.get("FastEMA", 21), 21)};
input int SlowEMA = {_int(params.get("SlowEMA", 55), 55)};
input int BBWRPeriod = {_int(params.get("BBWRPeriod", 24), 24)};
input double BBWRMin = {_float(params.get("BBWRMin", 0.015), 0.015)};
input double EntryBufferATR = {_float(params.get("EntryBufferATR", 0.15), 0.15)};
input double StopLossATR = {_float(params.get("StopLossATR", 1.4), 1.4)};
input double ProfitTargetATR = {_float(params.get("ProfitTargetATR", 2.8), 2.8)};
input double TrailATR = {_float(params.get("TrailATR", 1.1), 1.1)};
input int ExitAfterBars = {_int(params.get("ExitAfterBars", 18), 18)};
input int PullbackLookback = {_int(params.get("PullbackLookback", 5), 5)};
input double PullbackATR = {_float(params.get("PullbackATR", 0.6), 0.6)};
input double BreakEvenATR = {_float(params.get("BreakEvenATR", 0.8), 0.8)};
input double TimeStopATR = {_float(params.get("TimeStopATR", 0.4), 0.4)};
input double MaxDistancePct = {_float(params.get("MaxDistancePct", 3.5), 3.5)};

input group "Session"
input int SignalFromHour = {from_hour};
input int SignalFromMinute = 0;
input int SignalToHour = {to_hour};
input int SignalToMinute = 0;

datetime g_last_bar_time = 0;
int g_ema_fast_handle = INVALID_HANDLE;
int g_ema_slow_handle = INVALID_HANDLE;
int g_atr_handle = INVALID_HANDLE;
int g_bands_handle = INVALID_HANDLE;
datetime g_long_expiry = 0;
datetime g_short_expiry = 0;


bool CopyValue(const int handle, const int buffer, const int shift, double &value)
{{
   double tmp[];
   ArraySetAsSeries(tmp, true);
   if(CopyBuffer(handle, buffer, shift, 1, tmp) < 1)
      return false;
   value = tmp[0];
   return true;
}}


bool IsNewBar()
{{
   datetime current_bar = iTime(_Symbol, InpTimeframe, 0);
   if(current_bar == 0)
      return false;
   if(current_bar != g_last_bar_time)
   {{
      g_last_bar_time = current_bar;
      return true;
   }}
   return false;
}}


bool InTimeWindow(const datetime when)
{{
   MqlDateTime dt;
   TimeToStruct(when, dt);
   int hhmm = dt.hour * 100 + dt.min;
   int from_hhmm = SignalFromHour * 100 + SignalFromMinute;
   int to_hhmm = SignalToHour * 100 + SignalToMinute;
   if(from_hhmm <= to_hhmm)
      return (hhmm >= from_hhmm && hhmm <= to_hhmm);
   return (hhmm >= from_hhmm || hhmm <= to_hhmm);
}}


bool HasOpenPosition()
{{
   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = PositionGetTicket(idx);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      return true;
   }}
   return false;
}}


bool HasPendingOrders()
{{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(type == ORDER_TYPE_BUY_STOP || type == ORDER_TYPE_SELL_STOP || type == ORDER_TYPE_BUY_LIMIT || type == ORDER_TYPE_SELL_LIMIT)
         return true;
   }}
   return false;
}}


void CloseManagedPositions()
{{
   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = PositionGetTicket(idx);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      trade.PositionClose(ticket);
   }}
}}


void CancelManagedOrders()
{{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;
      trade.OrderDelete(ticket);
   }}
}}


void CancelExpiredOrders(const datetime bar_time)
{{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;

      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      string comment = OrderGetString(ORDER_COMMENT);
      if((type == ORDER_TYPE_BUY_STOP || type == ORDER_TYPE_BUY_LIMIT) && comment == "{strategy_id}_long" && g_long_expiry != 0 && bar_time >= g_long_expiry)
         trade.OrderDelete(ticket);
      if((type == ORDER_TYPE_SELL_STOP || type == ORDER_TYPE_SELL_LIMIT) && comment == "{strategy_id}_short" && g_short_expiry != 0 && bar_time >= g_short_expiry)
         trade.OrderDelete(ticket);
   }}
}}


double LowestLow(const int lookback, const int start_shift)
{{
   double value = DBL_MAX;
   for(int shift = start_shift; shift < start_shift + lookback; ++shift)
   {{
      double low = iLow(_Symbol, InpTimeframe, shift);
      if(low == 0.0)
         continue;
      value = MathMin(value, low);
   }}
   return (value == DBL_MAX) ? 0.0 : value;
}}


double HighestHigh(const int lookback, const int start_shift)
{{
   double value = -DBL_MAX;
   for(int shift = start_shift; shift < start_shift + lookback; ++shift)
   {{
      double high = iHigh(_Symbol, InpTimeframe, shift);
      if(high == 0.0)
         continue;
      value = MathMax(value, high);
   }}
   return (value == -DBL_MAX) ? 0.0 : value;
}}


double BollingerWidthRatio(const int shift)
{{
   double middle = 0.0;
   double upper = 0.0;
   double lower = 0.0;
   if(!CopyValue(g_bands_handle, 0, shift, middle))
      return 0.0;
   if(!CopyValue(g_bands_handle, 1, shift, upper))
      return 0.0;
   if(!CopyValue(g_bands_handle, 2, shift, lower))
      return 0.0;
   if(MathAbs(middle) < 1e-9)
      return 0.0;
   return ((upper - lower) / middle) * 100.0;
}}


bool AtrExpansionOk()
{{
   double atr_values[];
   ArraySetAsSeries(atr_values, true);
   int copied = CopyBuffer(g_atr_handle, 0, 1, 48, atr_values);
   if(copied < 12)
      return false;
   double atr_now = atr_values[0];
   double sum = 0.0;
   for(int i = 0; i < copied; ++i)
      sum += atr_values[i];
   double atr_mean = sum / copied;
   return atr_now >= (atr_mean * 1.05);
}}


bool VolatilityOk()
{{
   if(InpVolatilityFilter == "none")
      return true;
   if(InpVolatilityFilter == "atr_expansion")
      return AtrExpansionOk();
   return BollingerWidthRatio(1) >= BBWRMin;
}}


double TargetMultiple()
{{
   if(InpTargetModel == "trend_runner")
      return ProfitTargetATR * 1.6;
   if(InpTargetModel == "atr_scaled")
      return ProfitTargetATR * 1.2;
   return ProfitTargetATR;
}}


double LongStopPrice(const double entry_price, const double atr_value)
{{
   if(InpStopModel == "channel")
      return MathMin(LowestLow(LowestPeriod, 2), entry_price - atr_value * 0.5);
   if(InpStopModel == "swing")
      return LowestLow(PullbackLookback + 2, 2);
   return entry_price - atr_value * StopLossATR;
}}


double ShortStopPrice(const double entry_price, const double atr_value)
{{
   if(InpStopModel == "channel")
      return MathMax(HighestHigh(HighestPeriod, 2), entry_price + atr_value * 0.5);
   if(InpStopModel == "swing")
      return HighestHigh(PullbackLookback + 2, 2);
   return entry_price + atr_value * StopLossATR;
}}


double LongTargetPrice(const double entry_price, const double atr_value)
{{
   return entry_price + atr_value * TargetMultiple();
}}


double ShortTargetPrice(const double entry_price, const double atr_value)
{{
   return entry_price - atr_value * TargetMultiple();
}}


bool DistanceOk(const double current_open, const double ref_price)
{{
   if(current_open <= 0.0)
      return false;
   return (MathAbs(current_open - ref_price) / current_open * 100.0) <= MaxDistancePct;
}}


void ApplyManagedExits()
{{
   double current_close = iClose(_Symbol, InpTimeframe, 0);
   double ema_fast = 0.0;
   double atr_value = 0.0;
   CopyValue(g_ema_fast_handle, 0, 0, ema_fast);
   CopyValue(g_atr_handle, 0, 1, atr_value);

   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = PositionGetTicket(idx);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);

      if(InpExitModel == "channel_flip")
      {{
         if(type == POSITION_TYPE_BUY && current_close < ema_fast)
            trade.PositionClose(ticket);
         if(type == POSITION_TYPE_SELL && current_close > ema_fast)
            trade.PositionClose(ticket);
      }}
      if(InpExitModel == "atr_time_stop")
      {{
         if(type == POSITION_TYPE_BUY && (entry - current_close) > atr_value * TimeStopATR)
            trade.PositionClose(ticket);
         if(type == POSITION_TYPE_SELL && (current_close - entry) > atr_value * TimeStopATR)
            trade.PositionClose(ticket);
      }}
   }}
}}


bool LongSignalMarket(double &atr_value)
{{
   double ema_fast_1 = 0.0, ema_fast_2 = 0.0, ema_slow_1 = 0.0;
   if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1))
      return false;
   if(!CopyValue(g_ema_fast_handle, 0, 2, ema_fast_2))
      return false;
   if(!CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
      return false;
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;
   if(!VolatilityOk())
      return false;

   double close_1 = iClose(_Symbol, InpTimeframe, 1);
   double close_2 = iClose(_Symbol, InpTimeframe, 2);
   double open_1 = iOpen(_Symbol, InpTimeframe, 1);
   if(close_1 <= 0.0 || open_1 <= 0.0)
      return false;
   if(!DistanceOk(iOpen(_Symbol, InpTimeframe, 0), close_1))
      return false;

   if(InpEntryArchetype == "ema_reclaim")
   {{
      return (
         close_1 > ema_slow_1 &&
         LowestLow(PullbackLookback, 1) < ema_fast_1 &&
         close_1 > ema_fast_1 &&
         close_2 <= ema_fast_2
      );
   }}
   if(InpEntryArchetype == "pullback_trend")
   {{
      return (
         close_1 > ema_slow_1 &&
         LowestLow(PullbackLookback, 1) <= ema_fast_1 &&
         close_1 > ema_fast_1
      );
   }}
   if(InpEntryArchetype == "breakout_close")
      return close_1 > HighestHigh(HighestPeriod, 2);
   return false;
}}


bool ShortSignalMarket(double &atr_value)
{{
   double ema_fast_1 = 0.0, ema_fast_2 = 0.0, ema_slow_1 = 0.0;
   if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1))
      return false;
   if(!CopyValue(g_ema_fast_handle, 0, 2, ema_fast_2))
      return false;
   if(!CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
      return false;
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;
   if(!VolatilityOk())
      return false;

   double close_1 = iClose(_Symbol, InpTimeframe, 1);
   double close_2 = iClose(_Symbol, InpTimeframe, 2);
   double open_1 = iOpen(_Symbol, InpTimeframe, 1);
   if(close_1 <= 0.0 || open_1 <= 0.0)
      return false;
   if(!DistanceOk(iOpen(_Symbol, InpTimeframe, 0), close_1))
      return false;

   if(InpEntryArchetype == "ema_reclaim")
   {{
      return (
         close_1 < ema_slow_1 &&
         HighestHigh(PullbackLookback, 1) > ema_fast_1 &&
         close_1 < ema_fast_1 &&
         close_2 >= ema_fast_2
      );
   }}
   if(InpEntryArchetype == "pullback_trend")
   {{
      return (
         close_1 < ema_slow_1 &&
         HighestHigh(PullbackLookback, 1) >= ema_fast_1 &&
         close_1 < ema_fast_1
      );
   }}
   if(InpEntryArchetype == "breakout_close")
      return close_1 < LowestLow(LowestPeriod, 2);
   return false;
}}


bool LongSignalPending(double &atr_value, double &entry_price, double &sl, double &tp, ENUM_ORDER_TYPE &order_type)
{{
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;
   if(!VolatilityOk())
      return false;
   double open_0 = iOpen(_Symbol, InpTimeframe, 0);
   if(open_0 <= 0.0)
      return false;

   if(InpEntryArchetype == "atr_pullback_limit")
   {{
      double ema_fast_1 = 0.0, ema_slow_1 = 0.0, close_1 = iClose(_Symbol, InpTimeframe, 1);
      if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1) || !CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
         return false;
      if(!(close_1 > ema_slow_1 && close_1 > ema_fast_1))
         return false;
      if(!DistanceOk(open_0, close_1))
         return false;
      entry_price = ema_fast_1 - (atr_value * PullbackATR);
      sl = LongStopPrice(entry_price, atr_value);
      tp = LongTargetPrice(entry_price, atr_value);
      order_type = ORDER_TYPE_BUY_LIMIT;
      return true;
   }}

   if(InpEntryArchetype == "breakout_stop")
   {{
      double highest_level = HighestHigh(HighestPeriod, 2);
      double close_1 = iClose(_Symbol, InpTimeframe, 1);
      if(!(close_1 > highest_level) || !DistanceOk(open_0, highest_level))
         return false;
      entry_price = highest_level + (atr_value * EntryBufferATR);
      sl = LongStopPrice(entry_price, atr_value);
      tp = LongTargetPrice(entry_price, atr_value);
      order_type = ORDER_TYPE_BUY_STOP;
      return true;
   }}

   return false;
}}


bool ShortSignalPending(double &atr_value, double &entry_price, double &sl, double &tp, ENUM_ORDER_TYPE &order_type)
{{
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;
   if(!VolatilityOk())
      return false;
   double open_0 = iOpen(_Symbol, InpTimeframe, 0);
   if(open_0 <= 0.0)
      return false;

   if(InpEntryArchetype == "atr_pullback_limit")
   {{
      double ema_fast_1 = 0.0, ema_slow_1 = 0.0, close_1 = iClose(_Symbol, InpTimeframe, 1);
      if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1) || !CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
         return false;
      if(!(close_1 < ema_slow_1 && close_1 < ema_fast_1))
         return false;
      if(!DistanceOk(open_0, close_1))
         return false;
      entry_price = ema_fast_1 + (atr_value * PullbackATR);
      sl = ShortStopPrice(entry_price, atr_value);
      tp = ShortTargetPrice(entry_price, atr_value);
      order_type = ORDER_TYPE_SELL_LIMIT;
      return true;
   }}

   if(InpEntryArchetype == "breakout_stop")
   {{
      double lowest_level = LowestLow(LowestPeriod, 2);
      double close_1 = iClose(_Symbol, InpTimeframe, 1);
      if(!(close_1 < lowest_level) || !DistanceOk(open_0, lowest_level))
         return false;
      entry_price = lowest_level - (atr_value * EntryBufferATR);
      sl = ShortStopPrice(entry_price, atr_value);
      tp = ShortTargetPrice(entry_price, atr_value);
      order_type = ORDER_TYPE_SELL_STOP;
      return true;
   }}

   return false;
}}


int OnInit()
{{
   trade.SetExpertMagicNumber(InpMagicNumber);
   g_ema_fast_handle = iMA(_Symbol, InpTimeframe, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_ema_slow_handle = iMA(_Symbol, InpTimeframe, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_atr_handle = iATR(_Symbol, InpTimeframe, ATRPeriod);
   g_bands_handle = iBands(_Symbol, InpTimeframe, BBWRPeriod, 0, 2.0, PRICE_CLOSE);
   if(g_ema_fast_handle == INVALID_HANDLE || g_ema_slow_handle == INVALID_HANDLE || g_atr_handle == INVALID_HANDLE || g_bands_handle == INVALID_HANDLE)
      return INIT_FAILED;
   return INIT_SUCCEEDED;
}}


void OnDeinit(const int reason)
{{
   if(g_ema_fast_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_fast_handle);
   if(g_ema_slow_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_slow_handle);
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
   if(g_bands_handle != INVALID_HANDLE)
      IndicatorRelease(g_bands_handle);
}}


void OnTick()
{{
   if(!IsNewBar())
      return;

   datetime bar_time = iTime(_Symbol, InpTimeframe, 0);
   bool session_ok = InTimeWindow(bar_time);

   CancelExpiredOrders(bar_time);

   if(!session_ok)
   {{
      CloseManagedPositions();
      CancelManagedOrders();
      return;
   }}

   ApplyManagedExits();

   if(HasOpenPosition() || HasPendingOrders())
      return;

   double atr_value = 0.0;
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return;

   if(InpEntryArchetype == "ema_reclaim" || InpEntryArchetype == "pullback_trend" || InpEntryArchetype == "breakout_close")
   {{
      if(LongSignalMarket(atr_value))
      {{
         double sl = LongStopPrice(ask, atr_value);
         double tp = LongTargetPrice(ask, atr_value);
         trade.Buy(InpLots, _Symbol, 0.0, sl, tp, "{strategy_id}_long");
         return;
      }}
      if(ShortSignalMarket(atr_value))
      {{
         double sl = ShortStopPrice(bid, atr_value);
         double tp = ShortTargetPrice(bid, atr_value);
         trade.Sell(InpLots, _Symbol, 0.0, sl, tp, "{strategy_id}_short");
         return;
      }}
      return;
   }}

   double entry_price = 0.0, sl = 0.0, tp = 0.0;
   ENUM_ORDER_TYPE order_type;
   datetime expiry = bar_time + PeriodSeconds(InpTimeframe) * 3;
   if(LongSignalPending(atr_value, entry_price, sl, tp, order_type))
   {{
      if(order_type == ORDER_TYPE_BUY_STOP && trade.BuyStop(InpLots, entry_price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "{strategy_id}_long"))
         g_long_expiry = expiry;
      if(order_type == ORDER_TYPE_BUY_LIMIT && trade.BuyLimit(InpLots, entry_price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "{strategy_id}_long"))
         g_long_expiry = expiry;
      return;
   }}

   if(ShortSignalPending(atr_value, entry_price, sl, tp, order_type))
   {{
      if(order_type == ORDER_TYPE_SELL_STOP && trade.SellStop(InpLots, entry_price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "{strategy_id}_short"))
         g_short_expiry = expiry;
      if(order_type == ORDER_TYPE_SELL_LIMIT && trade.SellLimit(InpLots, entry_price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "{strategy_id}_short"))
         g_short_expiry = expiry;
   }}
}}
"""


def _render_sqx_xau_highest_breakout_ea(strategy_id: str, payload: dict[str, Any]) -> str:
    params = dict(payload.get("params", {}))
    long_mode = str(payload.get("long_signal_mode", "sma_bias"))
    short_mode = str(payload.get("short_signal_mode", "lwma_lowest_count"))

    return f"""#property strict

#include <Trade/Trade.mqh>

CTrade trade;

input group "Strategy"
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;
input double InpLots = {_float(params.get("mmLots", 1.0), 1.0)};
input int InpMagicNumber = {_magic_number(strategy_id)};

input group "SQX Blocks"
input string InpLongSignalMode = "{long_mode}";
input string InpShortSignalMode = "{short_mode}";

input group "Parameters"
input int HighestPeriod = {_int(params.get("HighestPeriod", 245), 245)};
input int LowestPeriod = {_int(params.get("LowestPeriod", 245), 245)};
input int PullbackSignalPeriod = {_int(params.get("PullbackSignalPeriod", 10), 10)};
input int FastSMAPeriod = {_int(params.get("FastSMAPeriod", 10), 10)};
input int SignalMAPeriod = {_int(params.get("SignalMAPeriod", 57), 57)};
input int LWMAPeriod = {_int(params.get("LWMAPeriod", 54), 54)};
input int ShortLWMAPeriod = {_int(params.get("ShortLWMAPeriod", 14), 14)};
input int SMMAPeriod = {_int(params.get("SMMAPeriod", 30), 30)};
input int HAFloorMAPeriod = {_int(params.get("HAFloorMAPeriod", 67), 67)};
input int WaveTrendChannel = {_int(params.get("WaveTrendChannel", 9), 9)};
input int WaveTrendAverage = {_int(params.get("WaveTrendAverage", 21), 21)};
input int ATRLongStopPeriod = {_int(params.get("ATRLongStopPeriod", 14), 14)};
input int ATRLongTargetPeriod = {_int(params.get("ATRLongTargetPeriod", 14), 14)};
input int ATRLongTrailPeriod = {_int(params.get("ATRLongTrailPeriod", 40), 40)};
input int ATRShortStopPeriod = {_int(params.get("ATRShortStopPeriod", 14), 14)};
input int ATRShortTargetPeriod = {_int(params.get("ATRShortTargetPeriod", 19), 19)};
input double LongStopATR = {_float(params.get("LongStopATR", 2.0), 2.0)};
input double LongTargetATR = {_float(params.get("LongTargetATR", 3.5), 3.5)};
input double ShortStopATR = {_float(params.get("ShortStopATR", 1.5), 1.5)};
input double ShortTargetATR = {_float(params.get("ShortTargetATR", 4.8), 4.8)};
input int LongExpiryBars = {_int(params.get("LongExpiryBars", 10), 10)};
input int ShortExpiryBars = {_int(params.get("ShortExpiryBars", 18), 18)};
input double LongTrailATR = {_float(params.get("LongTrailATR", 1.0), 1.0)};
input double LongTrailActivationATR = {_float(params.get("LongTrailActivationATR", 0.0), 0.0)};
input double ShortTrailATR = {_float(params.get("ShortTrailATR", 0.0), 0.0)};
input double ShortTrailActivationATR = {_float(params.get("ShortTrailActivationATR", 0.0), 0.0)};
input int HourQuantileLookback = {_int(params.get("HourQuantileLookback", 809), 809)};
input double HourQuantileThreshold = {_float(params.get("HourQuantileThreshold", 66.5), 66.5)};
input double MaxDistancePct = {_float(params.get("MaxDistancePct", 6.0), 6.0)};

input group "Session"
input int SignalFromHour = {_int(str(params.get("SignalTimeRangeFrom", "00:00"))[:2], 0)};
input int SignalFromMinute = {_int(str(params.get("SignalTimeRangeFrom", "00:00"))[3:], 0)};
input int SignalToHour = {_int(str(params.get("SignalTimeRangeTo", "23:59"))[:2], 23)};
input int SignalToMinute = {_int(str(params.get("SignalTimeRangeTo", "23:59"))[3:], 59)};

datetime g_last_bar_time = 0;
datetime g_long_expiry = 0;
datetime g_short_expiry = 0;
int g_sma_fast_handle = INVALID_HANDLE;
int g_lwma_fast_handle = INVALID_HANDLE;
int g_lwma_short_handle = INVALID_HANDLE;
int g_smma_mid_handle = INVALID_HANDLE;
int g_wt_handle = INVALID_HANDLE;
int g_atr_long_stop_handle = INVALID_HANDLE;
int g_atr_long_target_handle = INVALID_HANDLE;
int g_atr_long_trail_handle = INVALID_HANDLE;
int g_atr_short_stop_handle = INVALID_HANDLE;
int g_atr_short_target_handle = INVALID_HANDLE;
int g_ha_handle = INVALID_HANDLE;

bool CopyValue(const int handle, const int buffer, const int shift, double &value)
{{
   double tmp[];
   ArraySetAsSeries(tmp, true);
   if(CopyBuffer(handle, buffer, shift, 1, tmp) < 1)
      return false;
   value = tmp[0];
   return true;
}}

bool IsNewBar()
{{
   datetime current_bar = iTime(_Symbol, InpTimeframe, 0);
   if(current_bar == 0)
      return false;
   if(current_bar != g_last_bar_time)
   {{
      g_last_bar_time = current_bar;
      return true;
   }}
   return false;
}}

bool InTimeWindow(const datetime when)
{{
   MqlDateTime dt;
   TimeToStruct(when, dt);
   int hhmm = dt.hour * 100 + dt.min;
   int from_hhmm = SignalFromHour * 100 + SignalFromMinute;
   int to_hhmm = SignalToHour * 100 + SignalToMinute;
   if(from_hhmm <= to_hhmm)
      return (hhmm >= from_hhmm && hhmm <= to_hhmm);
   return (hhmm >= from_hhmm || hhmm <= to_hhmm);
}}

bool HasOpenPosition()
{{
   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = PositionGetTicket(idx);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      return true;
   }}
   return false;
}}

bool HasPendingOrders()
{{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;
      return true;
   }}
   return false;
}}

void CloseManagedPositions()
{{
   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = PositionGetTicket(idx);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      trade.PositionClose(ticket);
   }}
}}

void CancelManagedOrders()
{{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;
      trade.OrderDelete(ticket);
   }}
}}

void CancelExpiredOrders(const datetime bar_time)
{{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {{
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      string comment = OrderGetString(ORDER_COMMENT);
      if(type == ORDER_TYPE_BUY_STOP && comment == "{strategy_id}_long" && g_long_expiry != 0 && bar_time >= g_long_expiry)
         trade.OrderDelete(ticket);
      if(type == ORDER_TYPE_SELL_STOP && comment == "{strategy_id}_short" && g_short_expiry != 0 && bar_time >= g_short_expiry)
         trade.OrderDelete(ticket);
   }}
}}

double HighestHigh(const int lookback, const int start_shift)
{{
   double value = -DBL_MAX;
   for(int shift = start_shift; shift < start_shift + lookback; ++shift)
   {{
      double high = iHigh(_Symbol, InpTimeframe, shift);
      if(high == 0.0)
         continue;
      value = MathMax(value, high);
   }}
   return (value == -DBL_MAX) ? 0.0 : value;
}}

double LowestLow(const int lookback, const int start_shift)
{{
   double value = DBL_MAX;
   for(int shift = start_shift; shift < start_shift + lookback; ++shift)
   {{
      double low = iLow(_Symbol, InpTimeframe, shift);
      if(low == 0.0)
         continue;
      value = MathMin(value, low);
   }}
   return (value == DBL_MAX) ? 0.0 : value;
}}

double TypicalPrice(const int shift)
{{
   return (iHigh(_Symbol, InpTimeframe, shift) + iLow(_Symbol, InpTimeframe, shift) + iClose(_Symbol, InpTimeframe, shift)) / 3.0;
}}

double LowestTypical(const int lookback, const int start_shift)
{{
   double value = DBL_MAX;
   for(int shift = start_shift; shift < start_shift + lookback; ++shift)
   {{
      double tp = TypicalPrice(shift);
      if(tp == 0.0)
         continue;
      value = MathMin(value, tp);
   }}
   return (value == DBL_MAX) ? 0.0 : value;
}}

double HeikenAshiFloor(const int shift)
{{
   double ha_open = 0.0;
   double ha_close = 0.0;
   if(!CopyValue(g_ha_handle, 0, shift, ha_open))
      return 0.0;
   if(!CopyValue(g_ha_handle, 3, shift, ha_close))
      return 0.0;
   return MathMin(iLow(_Symbol, InpTimeframe, shift), MathMin(ha_open, ha_close));
}}

double RollingMean(const int handle, const int shift, const int count)
{{
   double values[];
   ArraySetAsSeries(values, true);
   if(CopyBuffer(handle, 0, shift, count, values) < count)
      return 0.0;
   double sum = 0.0;
   for(int i = 0; i < count; ++i)
      sum += values[i];
   return sum / count;
}}

bool DistanceOk(const double current_open, const double ref_price)
{{
   if(current_open <= 0.0 || ref_price <= 0.0)
      return false;
   return (MathAbs(current_open - ref_price) / current_open * 100.0) <= MaxDistancePct;
}}

bool LongSignal()
{{
   double open_0 = iOpen(_Symbol, InpTimeframe, 0);
   double highest_1 = HighestHigh(HighestPeriod, 2);
   if(!DistanceOk(open_0, highest_1))
      return false;
   if(InpLongSignalMode == "smma_pullback")
   {{
      double smma_2 = 0.0;
      if(!CopyValue(g_smma_mid_handle, 0, 2, smma_2))
         return false;
      return LowestLow(PullbackSignalPeriod, 3) <= smma_2;
   }}
   if(InpLongSignalMode == "ha_reclaim")
   {{
      double floor_1 = HeikenAshiFloor(1);
      if(floor_1 <= 0.0)
         return false;
      double sum = 0.0;
      int count = 0;
      for(int shift = 1; shift <= HAFloorMAPeriod; ++shift)
      {{
         double floor_value = HeikenAshiFloor(shift);
         if(floor_value <= 0.0)
            continue;
         sum += floor_value;
         count++;
      }}
      if(count < 5)
         return false;
      return floor_1 > (sum / count);
   }}
   double sma_3 = 0.0;
   if(!CopyValue(g_sma_fast_handle, 0, 3, sma_3))
      return false;
   double sma_mean = RollingMean(g_sma_fast_handle, 3, SignalMAPeriod);
   if(sma_mean <= 0.0)
      return false;
   return sma_3 > sma_mean;
}}

bool ShortSignal()
{{
   double open_0 = iOpen(_Symbol, InpTimeframe, 0);
   double lowest_1 = LowestLow(LowestPeriod, 2);
   if(!DistanceOk(open_0, lowest_1))
      return false;
   if(InpShortSignalMode == "wt_push")
   {{
      double wt_1 = 0.0, wt_2 = 0.0;
      double smma_1 = 0.0;
      if(!CopyValue(g_wt_handle, 0, 1, wt_1) || !CopyValue(g_wt_handle, 0, 2, wt_2) || !CopyValue(g_smma_mid_handle, 0, 1, smma_1))
         return false;
      return (iClose(_Symbol, InpTimeframe, 1) > smma_1) && (wt_1 > wt_2);
   }}
   if(InpShortSignalMode == "lwma_hour_quantile")
   {{
      double lwma_short_1 = 0.0;
      if(!CopyValue(g_lwma_short_handle, 0, 1, lwma_short_1))
         return false;
      if(!(lwma_short_1 > iLow(_Symbol, InpTimeframe, 2)))
         return false;
      MqlDateTime dt;
      TimeToStruct(iTime(_Symbol, InpTimeframe, 3), dt);
      double cutoff = (HourQuantileThreshold / 100.0) * 23.0;
      return dt.hour <= cutoff;
   }}
   double lwma_1 = 0.0;
   if(!CopyValue(g_lwma_fast_handle, 0, 1, lwma_1))
      return false;
   return lwma_1 < LowestTypical(PullbackSignalPeriod, 3);
}}

int OnInit()
{{
   trade.SetExpertMagicNumber(InpMagicNumber);
   g_sma_fast_handle = iMA(_Symbol, InpTimeframe, FastSMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   g_lwma_fast_handle = iMA(_Symbol, InpTimeframe, LWMAPeriod, 0, MODE_LWMA, PRICE_CLOSE);
   g_lwma_short_handle = iMA(_Symbol, InpTimeframe, ShortLWMAPeriod, 0, MODE_LWMA, PRICE_CLOSE);
   g_smma_mid_handle = iMA(_Symbol, InpTimeframe, SMMAPeriod, 0, MODE_SMMA, PRICE_MEDIAN);
   g_wt_handle = iCustom(_Symbol, InpTimeframe, "SqWaveTrend", WaveTrendChannel, WaveTrendAverage);
   g_atr_long_stop_handle = iATR(_Symbol, InpTimeframe, ATRLongStopPeriod);
   g_atr_long_target_handle = iATR(_Symbol, InpTimeframe, ATRLongTargetPeriod);
   g_atr_long_trail_handle = iATR(_Symbol, InpTimeframe, ATRLongTrailPeriod);
   g_atr_short_stop_handle = iATR(_Symbol, InpTimeframe, ATRShortStopPeriod);
   g_atr_short_target_handle = iATR(_Symbol, InpTimeframe, ATRShortTargetPeriod);
   g_ha_handle = iCustom(_Symbol, InpTimeframe, "SqHeikenAshi");
   if(g_sma_fast_handle == INVALID_HANDLE || g_lwma_fast_handle == INVALID_HANDLE || g_lwma_short_handle == INVALID_HANDLE || g_smma_mid_handle == INVALID_HANDLE || g_wt_handle == INVALID_HANDLE || g_atr_long_stop_handle == INVALID_HANDLE || g_atr_long_target_handle == INVALID_HANDLE || g_atr_long_trail_handle == INVALID_HANDLE || g_atr_short_stop_handle == INVALID_HANDLE || g_atr_short_target_handle == INVALID_HANDLE || g_ha_handle == INVALID_HANDLE)
      return INIT_FAILED;
   return INIT_SUCCEEDED;
}}

void OnDeinit(const int reason)
{{
   if(g_sma_fast_handle != INVALID_HANDLE) IndicatorRelease(g_sma_fast_handle);
   if(g_lwma_fast_handle != INVALID_HANDLE) IndicatorRelease(g_lwma_fast_handle);
   if(g_lwma_short_handle != INVALID_HANDLE) IndicatorRelease(g_lwma_short_handle);
   if(g_smma_mid_handle != INVALID_HANDLE) IndicatorRelease(g_smma_mid_handle);
   if(g_wt_handle != INVALID_HANDLE) IndicatorRelease(g_wt_handle);
   if(g_atr_long_stop_handle != INVALID_HANDLE) IndicatorRelease(g_atr_long_stop_handle);
   if(g_atr_long_target_handle != INVALID_HANDLE) IndicatorRelease(g_atr_long_target_handle);
   if(g_atr_long_trail_handle != INVALID_HANDLE) IndicatorRelease(g_atr_long_trail_handle);
   if(g_atr_short_stop_handle != INVALID_HANDLE) IndicatorRelease(g_atr_short_stop_handle);
   if(g_atr_short_target_handle != INVALID_HANDLE) IndicatorRelease(g_atr_short_target_handle);
   if(g_ha_handle != INVALID_HANDLE) IndicatorRelease(g_ha_handle);
}}

void OnTick()
{{
   if(!IsNewBar())
      return;
   datetime bar_time = iTime(_Symbol, InpTimeframe, 0);
   if(!InTimeWindow(bar_time))
   {{
      CloseManagedPositions();
      CancelManagedOrders();
      return;
   }}
   CancelExpiredOrders(bar_time);
   if(HasOpenPosition() || HasPendingOrders())
      return;

   double atr_long_stop = 0.0, atr_long_target = 0.0, atr_long_trail = 0.0, atr_short_stop = 0.0, atr_short_target = 0.0;
   if(!CopyValue(g_atr_long_stop_handle, 0, 1, atr_long_stop) || !CopyValue(g_atr_long_target_handle, 0, 1, atr_long_target) || !CopyValue(g_atr_long_trail_handle, 0, 1, atr_long_trail) || !CopyValue(g_atr_short_stop_handle, 0, 1, atr_short_stop) || !CopyValue(g_atr_short_target_handle, 0, 1, atr_short_target))
      return;

   double highest_1 = HighestHigh(HighestPeriod, 2);
   double lowest_1 = LowestLow(LowestPeriod, 2);
   datetime long_expiry = bar_time + PeriodSeconds(InpTimeframe) * LongExpiryBars;
   datetime short_expiry = bar_time + PeriodSeconds(InpTimeframe) * ShortExpiryBars;

   if(LongSignal())
   {{
      double sl = highest_1 - LongStopATR * atr_long_stop;
      double tp = highest_1 + LongTargetATR * atr_long_target;
      if(trade.BuyStop(InpLots, highest_1, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, long_expiry, "{strategy_id}_long"))
         g_long_expiry = long_expiry;
      return;
   }}
   if(ShortSignal())
   {{
      double sl = lowest_1 + ShortStopATR * atr_short_stop;
      double tp = lowest_1 - ShortTargetATR * atr_short_target;
      if(trade.SellStop(InpLots, lowest_1, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, short_expiry, "{strategy_id}_short"))
         g_short_expiry = short_expiry;
   }}
}}
"""


def export_native_strategy_to_mt5(strategy_id: str, *, output_name: str | None = None) -> dict[str, str]:
    record = get_native_strategy_record(strategy_id)
    if not record:
        raise ValueError(f"Unknown native strategy: {strategy_id}")

    template_name = str(record.get("template_name", ""))
    payload = dict(record.get("template_payload", {}))
    if template_name not in {"xau_discovery_grammar", "xau_breakout_session", "sqx_xau_highest_breakout"}:
        raise ValueError(f"MT5 exporter currently supports xau_discovery_grammar/xau_breakout_session/sqx_xau_highest_breakout only, got {template_name}")

    strategy_slug = output_name or strategy_id
    if template_name == "sqx_xau_highest_breakout":
        source = _render_sqx_xau_highest_breakout_ea(strategy_slug, payload)
    else:
        source = _render_xau_discovery_ea(strategy_slug, payload)
    out_dir = MT5_GENERATED_DIR / strategy_slug
    report_dir = out_dir / "ReportMT5"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ea_path = out_dir / f"{strategy_slug}.mq5"
    ea_path.write_text(source, encoding="utf-8")
    spec_dump = out_dir / f"{strategy_slug}.native_payload.json"
    spec_dump.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    gitkeep = report_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")
    return {
        "strategy_id": strategy_id,
        "ea_path": str(ea_path),
        "report_dir": str(report_dir),
        "payload_path": str(spec_dump),
    }
