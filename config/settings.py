import os

# MT5 settings
MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"  # adjust for your installation
MT5_LOGIN = 123456
MT5_PASSWORD = "your_password"
MT5_SERVER = "YourBroker"

# Data settings
DB_PATH = "sqlite:///market_data.db"  # or postgresql://user:pass@localhost/db
INFLUXDB_URL = "http://localhost:8086"  # optional
INFLUXDB_TOKEN = "my-token"
INFLUXDB_ORG = "my-org"
INFLUXDB_BUCKET = "market_data"

# Meta-learning settings
TASK_LENGTH = 30          # days per task
SUPPORT_SIZE = 10         # days for adaptation
QUERY_SIZE = 20           # days for loss computation
META_LEARNING_RATE = 0.001
INNER_LEARNING_RATE = 0.01
META_BATCH_SIZE = 16
NUM_ITERATIONS = 1000
MODEL_SAVE_PATH = "meta_weights.pth"

# Trading symbols and parameters
SYMBOLS_CONFIG = "config/symbols.yaml"

# Execution
PARAMETER_BOUNDS = {
    "lot_size": (0.01, 1.0),
    "sl_atr": (1.0, 5.0),
    "tp_atr": (1.0, 10.0),
    "rsi_period": (7, 21)
}
MAX_DRAWDOWN_PERCENT = 0.15  # 15%
