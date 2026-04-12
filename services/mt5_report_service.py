from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def _html_lines(report_path: str | Path) -> list[str]:
    path = Path(report_path)
    html = path.read_text(encoding="utf-16", errors="ignore")
    text = re.sub(r"<[^>]+>", "\n", html)
    return [line.strip() for line in text.splitlines() if line.strip()]


def mt5_validation_acceptance(
    mt5_metrics: dict[str, float | int | str],
    *,
    min_trades: int,
    min_profit_factor: float,
    max_drawdown_pct: float,
) -> bool:
    trades = int(mt5_metrics.get("total_trades", 0) or 0)
    pf = float(mt5_metrics.get("profit_factor", 0.0) or 0.0)
    dd = float(mt5_metrics.get("equity_drawdown_relative_pct", 0.0) or 0.0)
    return (
        trades >= int(min_trades)
        and pf >= float(min_profit_factor)
        and dd <= float(max_drawdown_pct)
    )


def parse_mt5_report(report_path: str | Path) -> dict[str, float | int | str]:
    lines = _html_lines(report_path)

    pairs: dict[str, str] = {}
    for idx, line in enumerate(lines[:-1]):
        if line.endswith(":"):
            pairs[line[:-1]] = lines[idx + 1]

    def _num(key: str) -> float:
        raw = pairs.get(key, "").replace(" ", "").replace(",", "").replace("%", "")
        raw = raw.split("(")[0]
        try:
            return float(raw)
        except ValueError:
            return 0.0

    def _int(key: str) -> int:
        return int(round(_num(key)))

    return {
        "expert": pairs.get("Expert", ""),
        "symbol": pairs.get("Symbol", ""),
        "period": pairs.get("Period", ""),
        "company": pairs.get("Company", ""),
        "currency": pairs.get("Currency", ""),
        "initial_deposit": _num("Initial Deposit"),
        "history_quality": _num("History Quality"),
        "total_net_profit": _num("Total Net Profit"),
        "profit_factor": _num("Profit Factor"),
        "balance_drawdown_relative_pct": _num("Balance Drawdown Relative"),
        "equity_drawdown_relative_pct": _num("Equity Drawdown Relative"),
        "sharpe_ratio": _num("Sharpe Ratio"),
        "total_trades": _int("Total Trades"),
        "profit_trades": _int("Profit Trades (% of total)"),
        "loss_trades": _int("Loss Trades (% of total)"),
    }


def parse_mt5_order_rows(report_path: str | Path) -> list[dict[str, Any]]:
    html = Path(report_path).read_text(encoding="utf-16", errors="ignore")
    start = html.find("<b>Orders</b>")
    if start < 0:
        return []
    end = html.find("<b>Deals</b>", start)
    if end < 0:
        end = len(html)
    section = html[start:end]
    row_re = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
    cell_re = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
    rows: list[dict[str, Any]] = []
    for match in row_re.finditer(section):
        row_html = match.group(1)
        cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cell_re.findall(row_html)]
        if len(cells) != 11:
            continue
        if not re.match(r"\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}", cells[0]):
            continue
        rows.append(
            {
                "open_time": cells[0],
                "order": cells[1],
                "symbol": cells[2],
                "type": cells[3].lower(),
                "volume": cells[4],
                "price": cells[5],
                "stop_loss": cells[6],
                "take_profit": cells[7],
                "time": cells[8],
                "state": cells[9].lower(),
                "comment": cells[10],
            }
        )
    return rows


def parse_mt5_trade_sequence(report_path: str | Path, *, strategy_tag: str) -> list[dict[str, Any]]:
    orders = parse_mt5_order_rows(report_path)
    sequence: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in orders:
        if row.get("state") != "filled":
            continue
        comment = str(row.get("comment", ""))
        row_type = str(row.get("type", "")).lower()
        if strategy_tag in comment:
            is_long_entry = row_type.startswith("buy")
            current = {
                "entry_time": row.get("time") or row.get("open_time"),
                "entry_side": "long" if is_long_entry else "short",
                "entry_type": row_type,
                "mt5_entry_comment": comment,
                "mt5_stop_loss": row.get("stop_loss"),
                "mt5_take_profit": row.get("take_profit"),
            }
            continue
        if current is None:
            continue
        current["exit_time"] = row.get("time") or row.get("open_time")
        current["exit_type"] = row_type
        current["mt5_exit_comment"] = comment
        sequence.append(current)
        current = None
    return sequence


def compare_mt5_to_native(mt5_metrics: dict[str, float | int | str], native_backtest: dict[str, float | int | str]) -> dict[str, float]:
    native_profit = float(native_backtest.get("total_profit", 0.0) or 0.0)
    native_pf = float(native_backtest.get("profit_factor", 0.0) or 0.0)
    native_dd = float(native_backtest.get("equity_dd_pct", native_backtest.get("max_drawdown_pct", 0.0)) or 0.0)
    native_trades = float(native_backtest.get("num_trades", 0.0) or 0.0)

    mt5_profit = float(mt5_metrics.get("total_net_profit", 0.0) or 0.0)
    mt5_pf = float(mt5_metrics.get("profit_factor", 0.0) or 0.0)
    mt5_dd = float(mt5_metrics.get("equity_drawdown_relative_pct", 0.0) or 0.0)
    mt5_trades = float(mt5_metrics.get("total_trades", 0.0) or 0.0)

    return {
        "profit_ratio_mt5_to_native": (mt5_profit / native_profit) if abs(native_profit) > 1e-9 else 0.0,
        "profit_factor_ratio_mt5_to_native": (mt5_pf / native_pf) if abs(native_pf) > 1e-9 else 0.0,
        "drawdown_ratio_mt5_to_native": (mt5_dd / native_dd) if abs(native_dd) > 1e-9 else 0.0,
        "trade_count_ratio_mt5_to_native": (mt5_trades / native_trades) if abs(native_trades) > 1e-9 else 0.0,
    }


def mt5_correlation_acceptance(
    comparison: dict[str, float],
    *,
    min_profit_ratio: float = 0.7,
    min_profit_factor_ratio: float = 0.75,
    min_trade_count_ratio: float = 0.8,
    max_trade_count_ratio: float = 1.25,
    max_drawdown_ratio: float = 2.5,
) -> bool:
    profit_ratio = float(comparison.get("profit_ratio_mt5_to_native", 0.0) or 0.0)
    pf_ratio = float(comparison.get("profit_factor_ratio_mt5_to_native", 0.0) or 0.0)
    dd_ratio = float(comparison.get("drawdown_ratio_mt5_to_native", 0.0) or 0.0)
    trades_ratio = float(comparison.get("trade_count_ratio_mt5_to_native", 0.0) or 0.0)
    return (
        profit_ratio >= min_profit_ratio
        and pf_ratio >= min_profit_factor_ratio
        and min_trade_count_ratio <= trades_ratio <= max_trade_count_ratio
        and dd_ratio <= max_drawdown_ratio
    )


def compare_trade_sequences(native_trades: list[dict[str, Any]], mt5_trades: list[dict[str, Any]], *, limit: int = 30) -> dict[str, Any]:
    compared: list[dict[str, Any]] = []
    divergence: dict[str, Any] | None = None
    n = min(limit, len(native_trades), len(mt5_trades))
    for idx in range(n):
        native = native_trades[idx]
        mt5 = mt5_trades[idx]
        issue = "aligned"
        if native.get("entry_side") != mt5.get("entry_side"):
            issue = "signal_generation"
        elif native.get("entry_time") != mt5.get("entry_time"):
            issue = "signal_generation"
        elif native.get("exit_time") != mt5.get("exit_time"):
            mt5_comment = str(mt5.get("mt5_exit_comment", "")).lower()
            if "sl" in mt5_comment or "tp" in mt5_comment:
                issue = "stop_tp_handling"
            elif mt5.get("exit_time", "").endswith("18:00:00") or native.get("exit_time", "").endswith("18:00:00"):
                issue = "session_exit_timing"
            else:
                issue = "order_lifecycle"
        compared.append(
            {
                "index": idx,
                "issue": issue,
                "native": native,
                "mt5": mt5,
            }
        )
        if divergence is None and issue != "aligned":
            divergence = compared[-1]
            break
    return {
        "compared": compared,
        "first_divergence": divergence,
        "native_trades": len(native_trades),
        "mt5_trades": len(mt5_trades),
    }
