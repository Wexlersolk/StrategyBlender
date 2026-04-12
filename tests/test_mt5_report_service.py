from pathlib import Path

from services.mt5_report_service import (
    compare_mt5_to_native,
    compare_trade_sequences,
    mt5_correlation_acceptance,
    mt5_validation_acceptance,
    parse_mt5_order_rows,
    parse_mt5_report,
    parse_mt5_trade_sequence,
)


def test_parse_mt5_report_and_compare():
    report_path = Path("mt5/generated/cand_8c7f15d62239_0000/ReportMT5/ReportTester-1513009549.html")
    if not report_path.exists():
        return

    mt5 = parse_mt5_report(report_path)
    assert mt5["expert"] == "cand_8c7f15d62239_0000"
    assert mt5["symbol"] == "XAUUSD"
    assert mt5["total_trades"] == 3224

    comparison = compare_mt5_to_native(
        mt5,
        {
            "total_profit": 41133.3165,
            "profit_factor": 1.0410,
            "equity_dd_pct": 27.5491,
            "num_trades": 2777,
        },
    )
    assert comparison["trade_count_ratio_mt5_to_native"] > 1.0
    assert comparison["drawdown_ratio_mt5_to_native"] > 1.0


def test_mt5_validation_acceptance():
    assert mt5_validation_acceptance(
        {"total_trades": 120, "profit_factor": 1.25, "equity_drawdown_relative_pct": 18.0},
        min_trades=10,
        min_profit_factor=1.12,
        max_drawdown_pct=25.0,
    )
    assert not mt5_validation_acceptance(
        {"total_trades": 120, "profit_factor": 1.02, "equity_drawdown_relative_pct": 18.0},
        min_trades=10,
        min_profit_factor=1.12,
        max_drawdown_pct=25.0,
    )


def test_mt5_correlation_acceptance():
    assert mt5_correlation_acceptance(
        {
            "profit_ratio_mt5_to_native": 0.9,
            "profit_factor_ratio_mt5_to_native": 0.85,
            "drawdown_ratio_mt5_to_native": 1.5,
            "trade_count_ratio_mt5_to_native": 1.1,
        }
    )
    assert not mt5_correlation_acceptance(
        {
            "profit_ratio_mt5_to_native": 0.46,
            "profit_factor_ratio_mt5_to_native": 0.43,
            "drawdown_ratio_mt5_to_native": 8.49,
            "trade_count_ratio_mt5_to_native": 1.68,
        }
    )


def test_parse_mt5_trade_sequence_and_compare():
    report_path = Path("mt5/generated/targeted_00_ema_reclaim_compiled/ReportMT5/ReportTester-1513009549.html")
    if not report_path.exists():
        return
    rows = parse_mt5_order_rows(report_path)
    assert rows
    trades = parse_mt5_trade_sequence(report_path, strategy_tag="targeted_00_ema_reclaim_compile")
    assert trades
    comparison = compare_trade_sequences(
        [{"entry_time": trades[0]["entry_time"], "exit_time": trades[0]["exit_time"], "entry_side": trades[0]["entry_side"]}],
        [trades[0]],
        limit=1,
    )
    assert comparison["first_divergence"] is None
