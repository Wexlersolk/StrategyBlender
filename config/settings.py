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
DEFAULT_TICK_SIZE = float(os.getenv("DEFAULT_TICK_SIZE", 1.0))
DEFAULT_TICK_VALUE = float(os.getenv("DEFAULT_TICK_VALUE", 0.0))
DEFAULT_CONTRACT_SIZE = float(os.getenv("DEFAULT_CONTRACT_SIZE", 0.0))
DEFAULT_SWAP_PER_LOT_LONG = float(os.getenv("DEFAULT_SWAP_PER_LOT_LONG", 0.0))
DEFAULT_SWAP_PER_LOT_SHORT = float(os.getenv("DEFAULT_SWAP_PER_LOT_SHORT", 0.0))
DEFAULT_SESSION_TIMEZONE_OFFSET_HOURS = float(os.getenv("DEFAULT_SESSION_TIMEZONE_OFFSET_HOURS", 0.0))

XAUUSD_DEFAULT_COMMISSION_PER_LOT = float(os.getenv("XAUUSD_DEFAULT_COMMISSION_PER_LOT", 0.0))
XAUUSD_DEFAULT_SPREAD_PIPS = float(os.getenv("XAUUSD_DEFAULT_SPREAD_PIPS", 0.25))
XAUUSD_DEFAULT_SLIPPAGE_PIPS = float(os.getenv("XAUUSD_DEFAULT_SLIPPAGE_PIPS", 0.0))
XAUUSD_DEFAULT_TICK_SIZE = float(os.getenv("XAUUSD_DEFAULT_TICK_SIZE", 0.01))
XAUUSD_DEFAULT_TICK_VALUE = float(os.getenv("XAUUSD_DEFAULT_TICK_VALUE", 1.0))
XAUUSD_DEFAULT_CONTRACT_SIZE = float(os.getenv("XAUUSD_DEFAULT_CONTRACT_SIZE", 100.0))
XAUUSD_DEFAULT_SWAP_PER_LOT_LONG = float(os.getenv("XAUUSD_DEFAULT_SWAP_PER_LOT_LONG", 0.0))
XAUUSD_DEFAULT_SWAP_PER_LOT_SHORT = float(os.getenv("XAUUSD_DEFAULT_SWAP_PER_LOT_SHORT", 0.0))
XAUUSD_DEFAULT_SESSION_TIMEZONE_OFFSET_HOURS = float(os.getenv("XAUUSD_DEFAULT_SESSION_TIMEZONE_OFFSET_HOURS", 0.0))


def symbol_execution_defaults(symbol: str | None = None) -> dict[str, float | bool]:
    normalized = str(symbol or "").strip().upper()
    if normalized == "XAUUSD":
        return {
            "commission_per_lot": XAUUSD_DEFAULT_COMMISSION_PER_LOT,
            "spread_pips": XAUUSD_DEFAULT_SPREAD_PIPS,
            "slippage_pips": XAUUSD_DEFAULT_SLIPPAGE_PIPS,
            "tick_size": XAUUSD_DEFAULT_TICK_SIZE,
            "tick_value": XAUUSD_DEFAULT_TICK_VALUE,
            "contract_size": XAUUSD_DEFAULT_CONTRACT_SIZE,
            "swap_per_lot_long": XAUUSD_DEFAULT_SWAP_PER_LOT_LONG,
            "swap_per_lot_short": XAUUSD_DEFAULT_SWAP_PER_LOT_SHORT,
            "session_timezone_offset_hours": XAUUSD_DEFAULT_SESSION_TIMEZONE_OFFSET_HOURS,
            "use_bar_spread": True,
        }
    return {
        "commission_per_lot": DEFAULT_COMMISSION_PER_LOT,
        "spread_pips": DEFAULT_SPREAD_PIPS,
        "slippage_pips": DEFAULT_SLIPPAGE_PIPS,
        "tick_size": DEFAULT_TICK_SIZE,
        "tick_value": DEFAULT_TICK_VALUE,
        "contract_size": DEFAULT_CONTRACT_SIZE,
        "swap_per_lot_long": DEFAULT_SWAP_PER_LOT_LONG,
        "swap_per_lot_short": DEFAULT_SWAP_PER_LOT_SHORT,
        "session_timezone_offset_hours": DEFAULT_SESSION_TIMEZONE_OFFSET_HOURS,
        "use_bar_spread": False,
    }

# -------------------- Research Runtime --------------------
RESEARCH_WORKER_POOL_SIZE = max(1, int(os.getenv("RESEARCH_WORKER_POOL_SIZE", "2")))
RESEARCH_WORKER_POLL_INTERVAL = float(os.getenv("RESEARCH_WORKER_POLL_INTERVAL", "2.0"))
AUTH_SESSION_TTL_HOURS = max(1, int(os.getenv("AUTH_SESSION_TTL_HOURS", "12")))
