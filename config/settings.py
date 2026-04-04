import os

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

# -------------------- Execution Model --------------------
DEFAULT_COMMISSION_PER_LOT = float(os.getenv("DEFAULT_COMMISSION_PER_LOT", 0.0))
DEFAULT_SPREAD_PIPS = float(os.getenv("DEFAULT_SPREAD_PIPS", 0.0))
DEFAULT_SLIPPAGE_PIPS = float(os.getenv("DEFAULT_SLIPPAGE_PIPS", 0.0))

# -------------------- Research Runtime --------------------
RESEARCH_WORKER_POOL_SIZE = max(1, int(os.getenv("RESEARCH_WORKER_POOL_SIZE", "2")))
RESEARCH_WORKER_POLL_INTERVAL = float(os.getenv("RESEARCH_WORKER_POLL_INTERVAL", "2.0"))
AUTH_SESSION_TTL_HOURS = max(1, int(os.getenv("AUTH_SESSION_TTL_HOURS", "12")))
