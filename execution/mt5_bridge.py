import MetaTrader5 as mt5
import config.settings as settings

class MT5Bridge:
    """Communicates with MT5 terminal to place/modify orders and update EA parameters."""
    
    def __init__(self):
        if not mt5.initialize(path=settings.MT5_PATH):
            raise Exception("MT5 init failed")
        # optional login
        # mt5.login(...)
    
    def set_global_variable(self, name, value):
        """Set a MT5 global variable (double)."""
        mt5.global_variable_set(name, float(value))
    
    def get_global_variable(self, name):
        return mt5.global_variable_get(name)
    
    def send_parameters_to_ea(self, symbol, params):
        """
        Write parameters to global variables that the EA reads.
        Example: prefix variable names with symbol and parameter.
        """
        for key, val in params.items():
            var_name = f"{symbol}_{key}"
            self.set_global_variable(var_name, val)
    
    def modify_order_sl_tp(self, ticket, sl, tp):
        """Modify stop loss and take profit of an open order."""
        order = mt5.orders_get(ticket=ticket)
        if order:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "order": ticket,
                "sl": sl,
                "tp": tp,
            }
            result = mt5.order_send(request)
            return result
        return None
    
    def shutdown(self):
        mt5.shutdown()
