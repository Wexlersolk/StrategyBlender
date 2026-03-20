import json
import os
from mt5linux import MetaTrader5
import config.settings as settings

# Shared mt5linux instance — bridge server must be running on port 18812
_mt5 = MetaTrader5(host='localhost', port=18812)


class MT5Bridge:
    """Communicates with MT5 terminal to place/modify orders and update EA parameters."""

    def __init__(self):
        if not _mt5.initialize(path=settings.MT5_PATH):
            raise Exception(f"MT5 init failed: {_mt5.last_error()}")

        if settings.MT5_LOGIN != 0 and settings.MT5_PASSWORD and settings.MT5_SERVER:
            authorized = _mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER
            )
            if not authorized:
                raise Exception(f"MT5 login failed: {_mt5.last_error()}")

    def send_parameters_to_ea(self, symbol, params):
        """
        Write parameters to a JSON file that the EA reads from disk.
        mt5linux does not support global_variable_set, so we use a shared
        params file in the MT5 Files directory instead.

        The EA should be coded to read:
            MQL5/Files/strategy_params.json
        """
        params_file = os.path.expanduser(
            "~/.wine/drive_c/users/wexlersolk/AppData/Roaming/MetaQuotes/"
            "Terminal/Common/Files/strategy_params.json"
        )

        # Load existing params file if present, then update this symbol's entry
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    all_params = json.load(f)
            except Exception:
                all_params = {}
        else:
            all_params = {}

        all_params[symbol] = params

        # Write back
        os.makedirs(os.path.dirname(params_file), exist_ok=True)
        with open(params_file, 'w') as f:
            json.dump(all_params, f, indent=2)

        print(f"Parameters written for {symbol}: {params}")

    def modify_order_sl_tp(self, ticket, sl, tp):
        """Modify stop loss and take profit of an open order."""
        request = {
            "action": _mt5.TRADE_ACTION_SLTP,
            "order": ticket,
            "sl": sl,
            "tp": tp,
        }
        result = _mt5.order_send(request)
        return result

    def get_account_info(self):
        """Return account info object."""
        return _mt5.account_info()

    def shutdown(self):
        _mt5.shutdown()
