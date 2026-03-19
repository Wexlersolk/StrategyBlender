import MetaTrader5 as mt5
import config.settings as settings

class MT5Bridge:
    """Communicates with MT5 terminal to place/modify orders and update EA parameters."""
    
    def __init__(self):
        if not mt5.initialize(path=settings.MT5_PATH):
            raise Exception("MT5 init failed")
        
        # Login if credentials are provided
        if settings.MT5_LOGIN != 0 and settings.MT5_PASSWORD and settings.MT5_SERVER:
            authorized = mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER
            )
            if not authorized:
                error = mt5.last_error()
                raise Exception(f"MT5 login failed: {error}")
    
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
