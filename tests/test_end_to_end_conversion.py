import importlib
import math
import sys
from pathlib import Path

from services.backtest_service import backtest_result_payload, run_backtest
from services.conversion_service import convert_ea_source, persist_converted_ea


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_generated_strategy(converted):
    module_name = converted.strategy_module
    module = (
        importlib.reload(sys.modules[module_name])
        if module_name in sys.modules
        else importlib.import_module(module_name)
    )
    return getattr(module, converted.strategy_class)


def _run_case(file_name: str, symbol: str, timeframe: str, date_from: str, date_to: str):
    source = (ROOT / "mine" / file_name).read_text(encoding="utf-8", errors="replace")
    converted = convert_ea_source(
        source=source,
        strategy_name=Path(file_name).stem,
        symbol=symbol,
        timeframe=timeframe,
        ea_id="pytest",
    )
    persist_converted_ea(converted, Path(file_name).stem)
    strat_cls = _load_generated_strategy(converted)
    result = run_backtest(
        strat_cls,
        symbol=symbol,
        timeframe=timeframe,
        date_from=date_from,
        date_to=date_to,
        overrides=converted.params,
        intrabar_steps=1,
    )
    return converted, backtest_result_payload(result)


def test_end_to_end_us30_conversion_payload():
    converted, payload = _run_case("US30_H1.mq5", "US30.cash", "H1", "2024-01-01", "2024-03-31")

    assert any("WPR/Stochastic" in warning for warning in converted.warnings)
    assert payload["summary"]["num_trades"] == 3
    assert payload["summary"]["num_months"] == 1
    assert math.isclose(payload["summary"]["total_profit"], -228.4183357947004, rel_tol=0.0, abs_tol=1e-9)


def test_end_to_end_usdjpy_conversion_payload():
    converted, payload = _run_case("USDJPY_H4.mq5", "USDJPY", "H4", "2024-01-01", "2024-03-31")

    assert any("EMA/VWAP/WaveTrend" in warning for warning in converted.warnings)
    assert payload["summary"]["num_trades"] == 3
    assert payload["summary"]["num_months"] == 1
    assert math.isclose(payload["summary"]["total_profit"], 403.3339256149115, rel_tol=0.0, abs_tol=1e-9)


def test_end_to_end_xau_conversion_payload():
    converted, payload = _run_case("XAUUSD_H1.mq5", "XAUUSD", "H1", "2024-01-01", "2024-03-31")

    assert any("Stochastic/OsMA/Bollinger" in warning for warning in converted.warnings)
    assert payload["summary"]["num_trades"] == 5
    assert payload["summary"]["num_months"] == 2
    assert math.isclose(payload["summary"]["total_profit"], 8046.5589999999565, rel_tol=0.0, abs_tol=1e-9)
