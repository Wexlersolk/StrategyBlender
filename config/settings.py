import os

# -------------------- Optional MT5 Data Download Settings --------------------
MT5_PATH = os.getenv(
    "MT5_PATH",
    os.path.expanduser("~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"),
)
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

# -------------------- Database Settings --------------------
# Use SQLite by default unless a PostgreSQL URL is constructed
DB_DRIVER = os.getenv("DB_DRIVER", "sqlite")
if DB_DRIVER == "postgresql":
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "market_data")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_PATH = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DB_PATH = os.getenv("DB_PATH", "sqlite:///market_data.db")

# -------------------- InfluxDB (optional) --------------------
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "market_data")

# -------------------- Meta‑learning Settings --------------------
TASK_LENGTH = int(os.getenv("TASK_LENGTH", 30))
SUPPORT_SIZE = int(os.getenv("SUPPORT_SIZE", 10))
QUERY_SIZE = int(os.getenv("QUERY_SIZE", 20))
META_LEARNING_RATE = float(os.getenv("META_LEARNING_RATE", 0.001))
INNER_LEARNING_RATE = float(os.getenv("INNER_LEARNING_RATE", 0.01))
META_BATCH_SIZE = int(os.getenv("META_BATCH_SIZE", 16))
NUM_ITERATIONS = int(os.getenv("NUM_ITERATIONS", 1000))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "meta_weights.pth")

# -------------------- Symbols --------------------
SYMBOLS_CONFIG = os.getenv("SYMBOLS_CONFIG", "config/symbols.yaml")

# -------------------- Risk & Execution --------------------
PARAMETER_BOUNDS = {
    "mmLots":           (1.0,  45.0),
    "StopLossCoef1":    (1.0,   5.0),
    "ProfitTargetCoef1":(1.0,  10.0),
    "StopLossCoef2":    (1.0,   5.0),
    "ProfitTargetCoef2":(1.0,  10.0),
    "TrailingActCef1":  (0.5,   3.0),
}
MAX_DRAWDOWN_PERCENT = float(os.getenv("MAX_DRAWDOWN_PERCENT", 0.15))
