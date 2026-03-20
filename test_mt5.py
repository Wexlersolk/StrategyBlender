from mt5linux import MetaTrader5
import os

# Instantiate — connects to the bridge running on port 18812
mt5 = MetaTrader5(host='localhost', port=18812)

mt5_path = os.path.expanduser(
    os.getenv("MT5_PATH", "~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe")
)

if not mt5.initialize(path=mt5_path):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

print("MT5 version:", mt5.version())
mt5.shutdown()
