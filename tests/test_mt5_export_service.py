from pathlib import Path

from services.mt5_export_service import export_native_strategy_to_mt5
from services.native_strategy_lab import CATALOG_PATH, get_native_strategy_record, register_generated_strategy


def test_export_native_strategy_to_mt5_generates_payload_driven_ea():
    strategy_id = "pytest_mt5_export"
    payload = {
        "name": "Payload Export",
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "entry_archetype": "ema_reclaim",
        "volatility_filter": "atr_expansion",
        "session_filter": "london_ny",
        "stop_model": "atr",
        "target_model": "fixed_rr",
        "exit_model": "atr_time_stop",
        "params": {
            "ATRPeriod": 21,
            "FastEMA": 13,
            "SlowEMA": 55,
            "StopLossATR": 1.1,
            "ProfitTargetATR": 3.6,
            "TimeStopATR": 0.3,
            "mmLots": 1.0,
        },
    }
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    out_dir = Path("mt5/generated") / strategy_id
    try:
        register_generated_strategy(
            template_name="xau_discovery_grammar",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
        )
        exported = export_native_strategy_to_mt5(strategy_id)
        ea_path = Path(exported["ea_path"])
        source = ea_path.read_text(encoding="utf-8")
        assert ea_path.exists()
        assert 'InpEntryArchetype = "ema_reclaim"' in source
        assert 'InpVolatilityFilter = "atr_expansion"' in source
        assert 'InpExitModel = "atr_time_stop"' in source
    finally:
        record = get_native_strategy_record(strategy_id)
        if record:
            Path(record["strategy_path"]).unlink(missing_ok=True)
            Path(record["spec_path"]).unlink(missing_ok=True)
        if out_dir.exists():
            for path in sorted(out_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                else:
                    path.rmdir()
            out_dir.rmdir()
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")


def test_export_sqx_xau_highest_breakout_to_mt5():
    strategy_id = "pytest_mt5_export_sqx_xau"
    payload = {
        "name": "SQX Highest Export",
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "long_signal_mode": "sma_bias",
        "short_signal_mode": "lwma_lowest_count",
        "params": {
            "HighestPeriod": 245,
            "LowestPeriod": 245,
            "LongStopATR": 2.0,
            "LongTargetATR": 3.5,
            "ShortStopATR": 1.5,
            "ShortTargetATR": 4.8,
            "LongExpiryBars": 10,
            "ShortExpiryBars": 18,
            "mmLots": 1.0,
        },
    }
    original_catalog = CATALOG_PATH.read_text(encoding="utf-8") if CATALOG_PATH.exists() else None
    out_dir = Path("mt5/generated") / strategy_id
    try:
        register_generated_strategy(
            template_name="sqx_xau_highest_breakout",
            payload=payload,
            strategy_id=strategy_id,
            origin="pytest",
        )
        exported = export_native_strategy_to_mt5(strategy_id)
        ea_path = Path(exported["ea_path"])
        source = ea_path.read_text(encoding="utf-8")
        assert ea_path.exists()
        assert 'InpLongSignalMode = "sma_bias"' in source
        assert 'InpShortSignalMode = "lwma_lowest_count"' in source
        assert "trade.BuyStop" in source
        assert "trade.SellStop" in source
    finally:
        record = get_native_strategy_record(strategy_id)
        if record:
            Path(record["strategy_path"]).unlink(missing_ok=True)
            Path(record["spec_path"]).unlink(missing_ok=True)
        if out_dir.exists():
            for path in sorted(out_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                else:
                    path.rmdir()
            out_dir.rmdir()
        if original_catalog is None:
            CATALOG_PATH.unlink(missing_ok=True)
        else:
            CATALOG_PATH.write_text(original_catalog, encoding="utf-8")
