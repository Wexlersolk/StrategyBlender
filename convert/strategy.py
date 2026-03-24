import logging
import os
import json
import sys

from strategytester5.tester import StrategyTester, MetaTrader5 as mt5
from strategytester5.trade_classes.Trade import CTrade

# ============================================================
# 1. Initialize MT5
# ============================================================
if not mt5.initialize():
    raise RuntimeError("Failed to initialize MT5. Make sure MT5 terminal is running.")

# ============================================================
# 2. Load configuration from JSON
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "tester.json"), 'r', encoding='utf-8') as file:
        tester_configs = json.load(file)
except Exception as e:
    mt5.shutdown()
    raise RuntimeError(f"Failed to load config: {e}")

# ============================================================
# 3. Initialize the StrategyTester
# ============================================================
tester = StrategyTester(
    tester_config=tester_configs["tester"],
    mt5_instance=mt5,
    logging_level=logging.INFO,
    broker_data_dir="ICMarketsSC-Demo"  # Replace with your broker's data folder name
)

# ============================================================
# 4. Strategy Parameters
# ============================================================
SYMBOL = "EURUSD"
TIMEFRAME = "PERIOD_H1"
MAGIC_NUMBER = 10012026
SLIPPAGE = 100
STOP_LOSS_PIPS = 500      # 50 pips stop loss
TAKE_PROFIT_PIPS = 1000   # 100 pips take profit
LOT_SIZE = 0.1

# Get symbol info (pip value, point size, etc.)
symbol_info = tester.symbol_info(symbol=SYMBOL)

# Initialize CTrade for easy order management
m_trade = CTrade(
    simulator=tester,
    magic_number=MAGIC_NUMBER,
    filling_type_symbol=SYMBOL,
    deviation_points=SLIPPAGE
)

# ============================================================
# 5. Helper Functions
# ============================================================
def pos_exists(magic: int, position_type: int) -> bool:
    """Check if a position with given magic number and type already exists"""
    for position in tester.positions_get():
        if position.type == position_type and position.magic == magic:
            return True
    return False

def get_signal() -> int:
    """
    Your strategy logic goes here.
    Returns: 1 for BUY, -1 for SELL, 0 for NO SIGNAL
    """
    # Example: Simple moving average crossover
    # Get rates from the tester
    rates = tester.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 100)
    
    if rates is None or len(rates) < 50:
        return 0
    
    # Calculate simple moving averages (pseudo-code)
    # For a real strategy, you'd compute actual indicators
    close_prices = [rate.close for rate in rates]
    
    # Just for demonstration - replace with your actual logic
    sma_fast = sum(close_prices[:10]) / 10
    sma_slow = sum(close_prices[:30]) / 30
    
    if sma_fast > sma_slow:
        return 1  # Buy signal
    elif sma_fast < sma_slow:
        return -1  # Sell signal
    else:
        return 0

# ============================================================
# 6. Main Strategy Logic (OnTick)
# ============================================================
def on_tick():
    """Called on every tick - similar to MQL5's OnTick()"""
    
    # Get current price
    tick_info = tester.symbol_info_tick(symbol=SYMBOL)
    ask = tick_info.ask
    bid = tick_info.bid
    point = symbol_info.point
    
    # Get trading signal
    signal = get_signal()
    
    # Check for existing positions
    has_buy = pos_exists(MAGIC_NUMBER, mt5.POSITION_TYPE_BUY)
    has_sell = pos_exists(MAGIC_NUMBER, mt5.POSITION_TYPE_SELL)
    
    # --- BUY Logic ---
    if signal == 1 and not has_buy:
        # Close any existing sell position
        if has_sell:
            for position in tester.positions_get():
                if position.type == mt5.POSITION_TYPE_SELL and position.magic == MAGIC_NUMBER:
                    m_trade.position_close(position=position)
        
        # Open buy position
        sl_price = ask - STOP_LOSS_PIPS * point
        tp_price = ask + TAKE_PROFIT_PIPS * point
        
        result = m_trade.buy(
            volume=LOT_SIZE,
            symbol=SYMBOL,
            price=ask,
            sl=sl_price,
            tp=tp_price,
            comment="AI Strategy Buy"
        )
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"BUY opened at {ask}, SL: {sl_price}, TP: {tp_price}")
        else:
            print(f"BUY failed: {result.comment}")
    
    # --- SELL Logic ---
    elif signal == -1 and not has_sell:
        # Close any existing buy position
        if has_buy:
            for position in tester.positions_get():
                if position.type == mt5.POSITION_TYPE_BUY and position.magic == MAGIC_NUMBER:
                    m_trade.position_close(position=position)
        
        # Open sell position
        sl_price = bid + STOP_LOSS_PIPS * point
        tp_price = bid - TAKE_PROFIT_PIPS * point
        
        result = m_trade.sell(
            volume=LOT_SIZE,
            symbol=SYMBOL,
            price=bid,
            sl=sl_price,
            tp=tp_price,
            comment="AI Strategy Sell"
        )
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"SELL opened at {bid}, SL: {sl_price}, TP: {tp_price}")
        else:
            print(f"SELL failed: {result.comment}")

# ============================================================
# 7. Run the Backtest
# ============================================================
if __name__ == "__main__":
    print("Starting backtest...")
    tester.OnTick(ontick_func=on_tick)
    print("Backtest completed!")
    
    # Optional: Get results summary
    print(f"\n--- Results ---")
    print(f"Total trades: {len(tester.deals_container)}")
    
    # Calculate total profit from deals
    total_profit = sum(deal.profit for deal in tester.deals_container if hasattr(deal, 'profit'))
    print(f"Total profit: {total_profit:.2f}")
    
    # Shutdown MT5
    mt5.shutdown()
