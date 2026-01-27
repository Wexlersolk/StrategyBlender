"""
Base adapter interface for all trading platforms
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class BaseAdapter(ABC):
    """Base class for all trading platform adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.last_update = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the trading platform"""
        pass
    
    @abstractmethod
    async def fetch_signals(self) -> List[Dict[str, Any]]:
        """Fetch latest signals/trades from the platform"""
        pass
    
    @abstractmethod
    async def send_order(self, order_data: Dict[str, Any]) -> bool:
        """Send order to the platform"""
        pass
    
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal format and content"""
        required_fields = ["symbol", "action", "timestamp"]
        
        # Check required fields
        if not all(field in signal for field in required_fields):
            return False
        
        # Validate action
        valid_actions = ["buy", "sell", "hold", "close", "cancel"]
        if signal["action"].lower() not in valid_actions:
            return False
        
        # Validate timestamp
        try:
            datetime.fromisoformat(signal["timestamp"].replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return False
        
        return True
    
    def convert_to_standard(self, raw_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert platform-specific signal to standard format"""
        standard_signal = {
            "signal_id": raw_signal.get("id", ""),
            "timestamp": raw_signal.get("timestamp", datetime.utcnow().isoformat()),
            "source": self.__class__.__name__.replace("Adapter", "").lower(),
            "strategy_id": raw_signal.get("strategy_id", "unknown"),
            "symbol": raw_signal.get("symbol", ""),
            "action": raw_signal.get("action", "hold").lower(),
            "price": float(raw_signal.get("price", 0)),
            "volume": float(raw_signal.get("volume", 0)),
            "confidence": float(raw_signal.get("confidence", 0.5)),
            "metadata": raw_signal.get("metadata", {})
        }
        return standard_signal
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information (optional)"""
        return {}
    
    async def disconnect(self) -> None:
        """Disconnect from platform (optional)"""
        self.connected = False
