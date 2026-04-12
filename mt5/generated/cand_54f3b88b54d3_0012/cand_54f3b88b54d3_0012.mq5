#property strict

#include <Trade/Trade.mqh>

CTrade trade;

input group "Strategy"
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;
input double InpLots = 1.0;
input int InpMagicNumber = 5438812;

input group "Discovery Blocks"
input string InpEntryArchetype = "breakout_stop";
input string InpVolatilityFilter = "bb_width";
input string InpSessionFilter = "london_ny";
input string InpStopModel = "channel";
input string InpTargetModel = "fixed_rr";
input string InpExitModel = "session_close";

input group "Parameters"
input int HighestPeriod = 24;
input int LowestPeriod = 24;
input int ATRPeriod = 14;
input int FastEMA = 21;
input int SlowEMA = 55;
input int BBWRPeriod = 24;
input double BBWRMin = 0.015;
input double EntryBufferATR = 0.15;
input double StopLossATR = 1.4;
input double ProfitTargetATR = 2.8;
input double TrailATR = 1.1;
input int ExitAfterBars = 18;
input int PullbackLookback = 8;
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
int g_atr_handle = INVALID_HANDLE;
int g_bands_handle = INVALID_HANDLE;
datetime g_long_expiry = 0;
datetime g_short_expiry = 0;


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


bool HasPendingOrders()
{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(type == ORDER_TYPE_BUY_STOP || type == ORDER_TYPE_SELL_STOP)
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


void CancelManagedOrders()
{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(type == ORDER_TYPE_BUY_STOP || type == ORDER_TYPE_SELL_STOP)
         trade.OrderDelete(ticket);
   }
}


void CancelExpiredOrders(const datetime bar_time)
{
   for(int idx = OrdersTotal() - 1; idx >= 0; --idx)
   {
      ulong ticket = OrderGetTicket(idx);
      if(ticket == 0 || !OrderSelect(ticket))
         continue;
      if(OrderGetString(ORDER_SYMBOL) != _Symbol)
         continue;
      if((int)OrderGetInteger(ORDER_MAGIC) != InpMagicNumber)
         continue;

      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      string comment = OrderGetString(ORDER_COMMENT);
      if(type == ORDER_TYPE_BUY_STOP && comment == "cand_54f3b88b54d3_0012_long" && g_long_expiry != 0 && bar_time >= g_long_expiry)
         trade.OrderDelete(ticket);
      if(type == ORDER_TYPE_SELL_STOP && comment == "cand_54f3b88b54d3_0012_short" && g_short_expiry != 0 && bar_time >= g_short_expiry)
         trade.OrderDelete(ticket);
   }
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


bool LongSignal(double &atr_value, double &entry_price, double &sl, double &tp)
{
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;

   double close_0 = iClose(_Symbol, InpTimeframe, 0);
   double open_0 = iOpen(_Symbol, InpTimeframe, 0);
   if(close_0 <= 0.0 || open_0 <= 0.0)
      return false;

   double highest_level = HighestHigh(HighestPeriod, 1);
   double lowest_level = LowestLow(LowestPeriod, 1);
   bool volatility_ok = true;
   if(InpVolatilityFilter == "bb_width")
      volatility_ok = BollingerWidthRatio(0) >= BBWRMin;

   double distance_pct = MathAbs(open_0 - close_0) / open_0 * 100.0;
   bool long_signal = close_0 > highest_level;
   if(!volatility_ok || !long_signal || distance_pct > MaxDistancePct)
      return false;

   entry_price = highest_level + (atr_value * EntryBufferATR);
   sl = MathMin(lowest_level, open_0 - (atr_value * 0.5));
   tp = entry_price + (atr_value * ProfitTargetATR);
   return true;
}


bool ShortSignal(double &atr_value, double &entry_price, double &sl, double &tp)
{
   if(!CopyValue(g_atr_handle, 0, 1, atr_value))
      return false;

   double close_0 = iClose(_Symbol, InpTimeframe, 0);
   double open_0 = iOpen(_Symbol, InpTimeframe, 0);
   if(close_0 <= 0.0 || open_0 <= 0.0)
      return false;

   double highest_level = HighestHigh(HighestPeriod, 1);
   double lowest_level = LowestLow(LowestPeriod, 1);
   bool volatility_ok = true;
   if(InpVolatilityFilter == "bb_width")
      volatility_ok = BollingerWidthRatio(0) >= BBWRMin;

   double distance_pct = MathAbs(open_0 - close_0) / open_0 * 100.0;
   bool short_signal = close_0 < lowest_level;
   if(!volatility_ok || !short_signal || distance_pct > MaxDistancePct)
      return false;

   entry_price = lowest_level - (atr_value * EntryBufferATR);
   sl = MathMax(highest_level, open_0 + (atr_value * 0.5));
   tp = entry_price - (atr_value * ProfitTargetATR);
   return true;
}


int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   g_atr_handle = iATR(_Symbol, InpTimeframe, ATRPeriod);
   g_bands_handle = iBands(_Symbol, InpTimeframe, BBWRPeriod, 0, 2.0, PRICE_CLOSE);
   if(g_atr_handle == INVALID_HANDLE || g_bands_handle == INVALID_HANDLE)
      return INIT_FAILED;
   return INIT_SUCCEEDED;
}


void OnDeinit(const int reason)
{
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

   CancelExpiredOrders(bar_time);

   if(InpExitModel == "session_close" && !session_ok)
   {
      if(HasOpenPosition())
         CloseManagedPositions();
      if(HasPendingOrders())
         CancelManagedOrders();
      return;
   }

   if(!session_ok || HasOpenPosition() || HasPendingOrders())
      return;

   double atr_value = 0.0;
   double entry_price = 0.0;
   double sl = 0.0;
   double tp = 0.0;

   if(LongSignal(atr_value, entry_price, sl, tp))
   {
      datetime expiry = bar_time + (PeriodSeconds(InpTimeframe) * 3);
      if(trade.BuyStop(InpLots, entry_price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "cand_54f3b88b54d3_0012_long"))
         g_long_expiry = expiry;
      return;
   }

   if(ShortSignal(atr_value, entry_price, sl, tp))
   {
      datetime expiry = bar_time + (PeriodSeconds(InpTimeframe) * 3);
      if(trade.SellStop(InpLots, entry_price, _Symbol, sl, tp, ORDER_TIME_SPECIFIED, expiry, "cand_54f3b88b54d3_0012_short"))
         g_short_expiry = expiry;
   }
}
