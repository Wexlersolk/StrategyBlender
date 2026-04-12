#property strict

#include <Trade/Trade.mqh>

CTrade trade;

input group "Strategy"
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;
input double InpLots = 1.0;
input int InpMagicNumber = 8715622;

input group "Discovery Blocks"
input string InpEntryArchetype = "pullback_trend";
input string InpVolatilityFilter = "bb_width";
input string InpSessionFilter = "london_ny";
input string InpStopModel = "atr";
input string InpTargetModel = "fixed_rr";
input string InpExitModel = "session_close";

input group "Parameters"
input int HighestPeriod = 24;
input int LowestPeriod = 24;
input int ATRPeriod = 14;
input int FastEMA = 13;
input int SlowEMA = 55;
input int BBWRPeriod = 24;
input double BBWRMin = 0.01;
input double EntryBufferATR = 0.15;
input double StopLossATR = 1.4;
input double ProfitTargetATR = 2.8;
input double TrailATR = 1.1;
input int ExitAfterBars = 18;
input int PullbackLookback = 5;
input double PullbackATR = 0.6;
input double BreakEvenATR = 0.8;
input double TimeStopATR = 0.4;
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
int g_bands_handle = INVALID_HANDLE;


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
      return (hhmm >= from_hhmm && hhmm < to_hhmm);
   return (hhmm >= from_hhmm || hhmm < to_hhmm);
}


bool HasOpenPosition()
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


double BollingerWidthRatio(const int shift)
{
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
}


bool LongSignal(double &atr_value)
{
   double ema_fast_1 = 0.0, ema_slow_1 = 0.0;
   if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1))
      return false;
   if(!CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
      return false;
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;

   double close_1 = iClose(_Symbol, InpTimeframe, 1);
   double open_1 = iOpen(_Symbol, InpTimeframe, 1);
   if(close_1 <= 0.0 || open_1 <= 0.0)
      return false;

   double rolling_low = LowestLow(PullbackLookback, 1);
   double distance_pct = MathAbs(open_1 - close_1) / open_1 * 100.0;
   bool volatility_ok = true;
   if(InpVolatilityFilter == "bb_width")
      volatility_ok = BollingerWidthRatio(1) >= BBWRMin;

   if(!volatility_ok || distance_pct > MaxDistancePct)
      return false;

   return (
      close_1 > ema_slow_1 &&
      rolling_low <= ema_fast_1 &&
      close_1 > ema_fast_1
   );
}


bool ShortSignal(double &atr_value)
{
   double ema_fast_1 = 0.0, ema_slow_1 = 0.0;
   if(!CopyValue(g_ema_fast_handle, 0, 1, ema_fast_1))
      return false;
   if(!CopyValue(g_ema_slow_handle, 0, 1, ema_slow_1))
      return false;
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;

   double close_1 = iClose(_Symbol, InpTimeframe, 1);
   double open_1 = iOpen(_Symbol, InpTimeframe, 1);
   if(close_1 <= 0.0 || open_1 <= 0.0)
      return false;

   double rolling_high = HighestHigh(PullbackLookback, 1);
   double distance_pct = MathAbs(open_1 - close_1) / open_1 * 100.0;
   bool volatility_ok = true;
   if(InpVolatilityFilter == "bb_width")
      volatility_ok = BollingerWidthRatio(1) >= BBWRMin;

   if(!volatility_ok || distance_pct > MaxDistancePct)
      return false;

   return (
      close_1 < ema_slow_1 &&
      rolling_high >= ema_fast_1 &&
      close_1 < ema_fast_1
   );
}


int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   g_ema_fast_handle = iMA(_Symbol, InpTimeframe, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_ema_slow_handle = iMA(_Symbol, InpTimeframe, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_atr_handle = iATR(_Symbol, InpTimeframe, ATRPeriod);
   g_bands_handle = iBands(_Symbol, InpTimeframe, BBWRPeriod, 0, 2.0, PRICE_CLOSE);
   if(g_ema_fast_handle == INVALID_HANDLE || g_ema_slow_handle == INVALID_HANDLE || g_atr_handle == INVALID_HANDLE || g_bands_handle == INVALID_HANDLE)
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
   if(g_bands_handle != INVALID_HANDLE)
      IndicatorRelease(g_bands_handle);
}


void OnTick()
{
   if(!IsNewBar())
      return;

   datetime bar_time = iTime(_Symbol, InpTimeframe, 0);
   bool session_ok = InTimeWindow(bar_time);

   if(InpExitModel == "session_close" && !session_ok && HasOpenPosition())
   {
      CloseManagedPositions();
      return;
   }

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
      trade.Buy(InpLots, _Symbol, 0.0, sl, tp, "cand_8c7f15d62239_0000_long");
      return;
   }

   if(ShortSignal(atr_value))
   {
      double sl = bid + (atr_value * StopLossATR);
      double tp = bid - (atr_value * ProfitTargetATR);
      trade.Sell(InpLots, _Symbol, 0.0, sl, tp, "cand_8c7f15d62239_0000_short");
   }
}
