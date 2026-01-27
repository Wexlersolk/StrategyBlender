"""
MetaTrader 5 Adapter for StrategyBlender
Connects to MT5 running on Wine (Arch Linux)
"""

import os
import json
import asyncio
import aiofiles
from datetime import datetime
from typing import Dict, List, Any, Optional
import MetaTrader5 as mt5
import pytz

from .base import BaseAdapter

class MetaTrader5Adapter(BaseAdapter):
    """Adapter for MetaTrader 5 running on Wine"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mt5_path = config.get("mt5_path", "")
        self.login = config.get("login", 0)
        self.password = config.get("password", "")
        self.server = config.get("server", "")
        self.symbols = config.get("symbols", ["EURUSD", "GBPUSD", "XAUUSD"])
        
        # File-based communication (fallback)
        self.signal_file = config.get("signal_file", "")
        self.order_file = config.get("order_file", "")
        
        self.initialized = False
        self.connection_type = "direct"  # or "file"
        
    async def connect(self) -> bool:
        """Initialize connection to MT5"""
        try:
            # Try direct connection first
            if self._initialize_mt5_direct():
                self.connection_type = "direct"
                print("✅ Connected to MT5 via direct API")
                return True
            
            # Fallback to file-based connection
            elif self._initialize_mt5_file():
                self.connection_type = "file"
                print("⚠️ Using file-based connection to MT5")
                return True
            
            else:
                print("❌ Could not connect to MT5")
                return False
                
        except Exception as e:
            print(f"❌ Error connecting to MT5: {e}")
            return False
    
    def _initialize_mt5_direct(self) -> bool:
        """Initialize MT5 with direct API connection"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                print(f"MT5 initialize failed, error: {mt5.last_error()}")
                return False
            
            # Login to account
            authorized = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server
            )
            
            if authorized:
                account_info = mt5.account_info()
                print(f"✅ Connected to account: {account_info.login}")
                print(f"   Balance: ${account_info.balance}")
                print(f"   Server: {account_info.server}")
                return True
            else:
                print(f"Login failed, error: {mt5.last_error()}")
                return False
                
        except Exception as e:
            print(f"Direct MT5 connection failed: {e}")
            return False
    
    def _initialize_mt5_file(self) -> bool:
        """Initialize file-based connection"""
        if not self.signal_file:
            print("No signal file specified for file-based connection")
            return False
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.signal_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.order_file), exist_ok=True)
        
        # Create empty files if they don't exist
        if not os.path.exists(self.signal_file):
            with open(self.signal_file, 'w') as f:
                json.dump({"signals": []}, f)
        
        if not os.path.exists(self.order_file):
            with open(self.order_file, 'w') as f:
                json.dump({"orders": []}, f)
        
        return True
    
    async def fetch_signals(self) -> List[Dict[str, Any]]:
        """Fetch signals from MT5"""
        if self.connection_type == "direct":
            return await self._fetch_signals_direct()
        else:
            return await self._fetch_signals_file()
    
    async def _fetch_signals_direct(self) -> List[Dict[str, Any]]:
        """Fetch signals via MT5 API"""
        signals = []
        
        try:
            # Get positions (open trades)
            positions = mt5.positions_get()
            if positions:
                for position in positions:
                    signal = {
                        "signal_id": f"mt5_pos_{position.ticket}",
                        "timestamp": datetime.fromtimestamp(position.time, tz=pytz.UTC).isoformat(),
                        "source": "metatrader5",
                        "strategy_id": position.comment or "unknown",
                        "symbol": position.symbol,
                        "action": "buy" if position.type == 0 else "sell",
                        "price": position.price_open,
                        "volume": position.volume,
                        "profit": position.profit,
                        "metadata": {
                            "ticket": position.ticket,
                            "type": position.type,
                            "magic": position.magic,
                            "comment": position.comment
                        }
                    }
                    signals.append(signal)
            
            # Get pending orders
            orders = mt5.orders_get()
            if orders:
                for order in orders:
                    signal = {
                        "signal_id": f"mt5_ord_{order.ticket}",
                        "timestamp": datetime.fromtimestamp(order.time_setup, tz=pytz.UTC).isoformat(),
                        "source": "metatrader5",
                        "strategy_id": order.comment or "unknown",
                        "symbol": order.symbol,
                        "action": self._map_mt5_order_type(order.type),
                        "price": order.price_open,
                        "volume": order.volume_initial,
                        "metadata": {
                            "ticket": order.ticket,
                            "type": order.type,
                            "magic": order.magic,
                            "comment": order.comment
                        }
                    }
                    signals.append(signal)
                    
        except Exception as e:
            print(f"Error fetching MT5 signals: {e}")
        
        return signals
    
    async def _fetch_signals_file(self) -> List[Dict[str, Any]]:
        """Read signals from JSON file"""
        try:
            if not os.path.exists(self.signal_file):
                return []
            
            async with aiofiles.open(self.signal_file, 'r') as f:
                content = await f.read()
                if not content.strip():
                    return []
                
                data = json.loads(content)
                signals = data.get("signals", [])
                
                # Clear the file after reading
                await self._clear_signal_file()
                
                return signals
                
        except Exception as e:
            print(f"Error reading signal file: {e}")
            return []
    
    async def _clear_signal_file(self):
        """Clear the signal file after reading"""
        try:
            async with aiofiles.open(self.signal_file, 'w') as f:
                await f.write(json.dumps({"signals": []}))
        except Exception as e:
            print(f"Error clearing signal file: {e}")
    
    async def send_order(self, order_data: Dict[str, Any]) -> bool:
        """Send order to MT5"""
        if self.connection_type == "direct":
            return await self._send_order_direct(order_data)
        else:
            return await self._send_order_file(order_data)
    
    async def _send_order_direct(self, order_data: Dict[str, Any]) -> bool:
        """Send order via MT5 API"""
        try:
            symbol = order_data.get("symbol")
            order_type = order_data.get("action")
            volume = order_data.get("volume", 0.01)
            price = order_data.get("price", 0)
            
            # Map action to MT5 order type
            if order_type == "buy":
                order_type_mt5 = mt5.ORDER_TYPE_BUY
            elif order_type == "sell":
                order_type_mt5 = mt5.ORDER_TYPE_SELL
            else:
                print(f"Unsupported order type: {order_type}")
                return False
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type_mt5,
                "price": price,
                "deviation": 10,
                "magic": 234000,
                "comment": f"strategyblender_{order_data.get('strategy_id', '')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Order failed, retcode: {result.retcode}")
                return False
            
            print(f"✅ Order executed: {symbol} {order_type} {volume} @ {price}")
            return True
            
        except Exception as e:
            print(f"Error sending order to MT5: {e}")
            return False
    
    async def _send_order_file(self, order_data: Dict[str, Any]) -> bool:
        """Write order to JSON file for MT5 to read"""
        try:
            # Read existing orders
            if os.path.exists(self.order_file):
                async with aiofiles.open(self.order_file, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        data = json.loads(content)
                    else:
                        data = {"orders": []}
            else:
                data = {"orders": []}
            
            # Add new order
            data["orders"].append({
                **order_data,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "pending"
            })
            
            # Write back to file
            async with aiofiles.open(self.order_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            
            print(f"✅ Order written to file: {order_data.get('symbol')} {order_data.get('action')}")
            return True
            
        except Exception as e:
            print(f"Error writing order to file: {e}")
            return False
    
    def _map_mt5_order_type(self, order_type: int) -> str:
        """Map MT5 order type to our action"""
        mapping = {
            0: "buy",      # ORDER_TYPE_BUY
            1: "sell",     # ORDER_TYPE_SELL
            2: "buy_limit",
            3: "sell_limit",
            4: "buy_stop",
            5: "sell_stop"
        }
        return mapping.get(order_type, "unknown")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from MT5"""
        if self.connection_type != "direct":
            return {}
        
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.margin_free,
                    "leverage": account_info.leverage,
                    "currency": account_info.currency,
                    "server": account_info.server
                }
        except Exception as e:
            print(f"Error getting account info: {e}")
        
        return {}
    
    async def disconnect(self) -> None:
        """Disconnect from MT5"""
        if self.initialized and self.connection_type == "direct":
            mt5.shutdown()
            self.initialized = False
            print("Disconnected from MT5")

# Factory function to create MT5 adapter
def create_mt5_adapter(config: Dict[str, Any]) -> MetaTrader5Adapter:
    """Factory function to create MT5 adapter"""
    return MetaTrader5Adapter(config)

# Quick test
if __name__ == "__main__":
    print("Testing MT5 Adapter...")
    
    config = {
        "login": 123456,
        "password": "your_password",
        "server": "ICMarkets-Demo",
        "signal_file": "/tmp/mt5_signals.json",
        "order_file": "/tmp/mt5_orders.json"
    }
    
    adapter = MetaTrader5Adapter(config)
    
    # Test file-based connection
    import asyncio
    loop = asyncio.get_event_loop()
    
    async def test():
        if await adapter.connect():
            print("Connected successfully")
            
            # Test writing a signal
            test_signal = {
                "symbol": "EURUSD",
                "action": "buy",
                "price": 1.0850,
                "volume": 0.01,
                "strategy_id": "test_strategy"
            }
            
            await adapter.send_order(test_signal)
            
            # Test reading signals
            signals = await adapter.fetch_signals()
            print(f"Found {len(signals)} signals")
    
    loop.run_until_complete(test())
