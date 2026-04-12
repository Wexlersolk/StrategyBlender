#property strict

#include <Trade/Trade.mqh>

CTrade trade;

input group "Strategy"
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;
input double InpLots = 1.0;
input int InpMagicNumber = 1100001;

input group "Discovery Blocks"
input string InpEntryArchetype = "ema_reclaim";
input string InpVolatilityFilter = "atr_expansion";
input string InpSessionFilter = "london_ny";
input string InpStopModel = "atr";
input string InpTargetModel = "fixed_rr";
input string InpExitModel = "atr_time_stop";

input group "Parameters"
input int HighestPeriod = 24;
input int LowestPeriod = 24;
input int ATRPeriod = 21;
input int FastEMA = 13;
input int SlowEMA = 55;
input int BBWRPeriod = 24;
input double BBWRMin = 0.025;
input double EntryBufferATR = 0.15;
input double StopLossATR = 1.1;
input double ProfitTargetATR = 3.6;
input double TrailATR = 1.1;
input int ExitAfterBars = 12;
input int PullbackLookback = 5;
input double PullbackATR = 0.4;
input double BreakEvenATR = 1.1;
input double TimeStopATR = 0.3;
input double MaxDistancePct = 3.5;

input group "Session"
input int SignalFromHour = 6;
input int SignalFromMinute = 0;
input int SignalToHour = 18;
input int SignalToMinute = 0;

datetime g_last_bar_time = 0;
int g_ema_fast_handle = INVALID_HANDLE;
int g_ema_slow_handle = INVALID_HANDLE;
int g_atr_handle = INVALID_HANDLE;


bool CopyValue(const int handle, const int buffer, const int shift, double &value)
{
   double tmp[];
   ArraySetAsSeries(tmp, true);
   if(CopyBuffer(handle, buffer, shift, 1, tmp) < 1)
      return false;
   value = tmp[0];
   return true;
}


bool IsNewBar()
{
   datetime current_bar = iTime(_Symbol, InpTimeframe, 0);
   if(current_bar == 0)
      return false;
   if(current_bar != g_last_bar_time)
   {
      g_last_bar_time = current_bar;
      return true;
   }
   return false;
}


bool InTimeWindow(const datetime when)
{
   MqlDateTime dt;
   TimeToStruct(when, dt);
   int hhmm = dt.hour * 100 + dt.min;
   int from_hhmm = SignalFromHour * 100 + SignalFromMinute;
   int to_hhmm = SignalToHour * 100 + SignalToMinute;
   if(from_hhmm <= to_hhmm)
      return (hhmm >= from_hhmm && hhmm <= to_hhmm);
   return (hhmm >= from_hhmm || hhmm <= to_hhmm);
}


bool HasOpenPosition()
{
   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {
      if(!PositionSelectByTicket(PositionGetTicket(idx)))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      return true;
   }
   return false;
}


void CloseManagedPositions()
{
   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {
      ulong ticket = PositionGetTicket(idx);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      trade.PositionClose(ticket);
   }
}


double LowestLow(const int lookback, const int start_shift)
{
   double value = DBL_MAX;
   for(int shift = start_shift; shift < start_shift + lookback; ++shift)
   {
      double low = iLow(_Symbol, InpTimeframe, shift);
      if(low == 0.0)
         continue;
      value = MathMin(value, low);
   }
   return (value == DBL_MAX) ? 0.0 : value;
}


double HighestHigh(const int lookback, const int start_shift)
{
   double value = -DBL_MAX;
   for(int shift = start_shift; shift < start_shift + lookback; ++shift)
   {
      double high = iHigh(_Symbol, InpTimeframe, shift);
      if(high == 0.0)
         continue;
      value = MathMax(value, high);
   }
   return (value == -DBL_MAX) ? 0.0 : value;
}


bool AtrExpansionOk()
{
   double atr_now = 0.0;
   if(!CopyValue(g_atr_handle, 0, 1, atr_now))
      return false;

   double atr_values[];
   ArraySetAsSeries(atr_values, true);
   int copied = CopyBuffer(g_atr_handle, 0, 1, 48, atr_values);
   if(copied < 12)
      return false;

   double sum = 0.0;
   for(int i = 0; i < copied; ++i)
      sum += atr_values[i];

   double atr_mean = sum / copied;
   return atr_now >= (atr_mean * 1.05);
}


void ApplyAtrTimeStop()
{
   double atr_value = 0.0;
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return;

   double current_close = iClose(_Symbol, InpTimeframe, 0);
   if(current_close <= 0.0)
      return;

   for(int idx = PositionsTotal() - 1; idx >= 0; --idx)
   {
      ulong ticket = PositionGetTicket(idx);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);

      if(type == POSITION_TYPE_BUY && (entry - current_close) > atr_value * TimeStopATR)
         trade.PositionClose(ticket);
      if(type == POSITION_TYPE_SELL && (current_close - entry) > atr_value * TimeStopATR)
         trade.PositionClose(ticket);
   }
}


bool LongSignal(double &atr_value)
{
   double ema_fast_1 = 0.0, ema_fast_2 = 0.0, ema_slow_1 = 0.0;
   if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1))
      return false;
   if(!CopyValue(g_ema_fast_handle, 0, 2, ema_fast_2))
      return false;
   if(!CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
      return false;
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;

   double close_1 = iClose(_Symbol, InpTimeframe, 1);
   double close_2 = iClose(_Symbol, InpTimeframe, 2);
   double open_1 = iOpen(_Symbol, InpTimeframe, 1);
   if(close_1 <= 0.0 || open_1 <= 0.0)
      return false;

   double rolling_low = LowestLow(PullbackLookback, 1);
   double distance_pct = MathAbs(open_1 - close_1) / open_1 * 100.0;
   if(distance_pct > MaxDistancePct)
      return false;
   if(!AtrExpansionOk())
      return false;

   return (
      close_1 > ema_slow_1 &&
      rolling_low < ema_fast_1 &&
      close_1 > ema_fast_1 &&
      close_2 <= ema_fast_2
   );
}


bool ShortSignal(double &atr_value)
{
   double ema_fast_1 = 0.0, ema_fast_2 = 0.0, ema_slow_1 = 0.0;
   if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1))
      return false;
   if(!CopyValue(g_ema_fast_handle, 0, 2, ema_fast_2))
      return false;
   if(!CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
      return false;
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;

   double close_1 = iClose(_Symbol, InpTimeframe, 1);
   double close_2 = iClose(_Symbol, InpTimeframe, 2);
   double open_1 = iOpen(_Symbol, InpTimeframe, 1);
   if(close_1 <= 0.0 || open_1 <= 0.0)
      return false;

   double rolling_high = HighestHigh(PullbackLookback, 1);
   double distance_pct = MathAbs(open_1 - close_1) / open_1 * 100.0;
   if(distance_pct > MaxDistancePct)
      return false;
   if(!AtrExpansionOk())
      return false;

   return (
      close_1 < ema_slow_1 &&
      rolling_high > ema_fast_1 &&
      close_1 < ema_fast_1 &&
      close_2 >= ema_fast_2
   );
}


int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   g_ema_fast_handle = iMA(_Symbol, InpTimeframe, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_ema_slow_handle = iMA(_Symbol, InpTimeframe, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_atr_handle = iATR(_Symbol, InpTimeframe, ATRPeriod);
   if(g_ema_fast_handle == INVALID_HANDLE || g_ema_slow_handle == INVALID_HANDLE || g_atr_handle == INVALID_HANDLE)
      return INIT_FAILED;
   return INIT_SUCCEEDED;
}


void OnDeinit(const int reason)
{
   if(g_ema_fast_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_fast_handle);
   if(g_ema_slow_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_slow_handle);
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
}


void OnTick()
{
   if(!IsNewBar())
      return;

   datetime bar_time = iTime(_Symbol, InpTimeframe, 0);
   bool session_ok = InTimeWindow(bar_time);

   if(!session_ok && HasOpenPosition())
   {
      CloseManagedPositions();
      return;
   }

   if(InpExitModel == "atr_time_stop")
      ApplyAtrTimeStop();

   if(!session_ok || HasOpenPosition())
      return;

   double atr_value = 0.0;
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return;

   if(LongSignal(atr_value))
   {
      double sl = ask - (atr_value * StopLossATR);
      double tp = ask + (atr_value * ProfitTargetATR);
      trade.Buy(InpLots, _Symbol, 0.0, sl, tp, "targeted_00_ema_reclaim_long");
      return;
   }

   if(ShortSignal(atr_value))
   {
      double sl = bid + (atr_value * StopLossATR);
      double tp = bid - (atr_value * ProfitTargetATR);
      trade.Sell(InpLots, _Symbol, 0.0, sl, tp, "targeted_00_ema_reclaim_short");
   }
}
