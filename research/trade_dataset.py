from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from engine.results import BacktestResults


def decision_records_frame(results: BacktestResults) -> pd.DataFrame:
    rows: list[dict] = []
    for record in results.decision_records:
        row = {
            "decision_id": record.decision_id,
            "time": record.time,
            "bar_idx": record.bar_idx,
            "symbol": record.symbol,
            "timeframe": record.timeframe,
            "order_type": record.order_type,
            "direction": record.direction,
            "requested_lots": record.requested_lots,
            "final_lots": record.final_lots,
            "allow_trade": record.allow_trade,
            "policy_name": record.policy_name,
            "policy_tag": record.policy_tag,
            "comment": record.comment,
        }
        for key, value in record.market_features.items():
            row[f"feature_{key}"] = value
        for key, value in record.policy_notes.items():
            row[f"policy_{key}"] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    return frame.sort_values(["time", "decision_id"]).reset_index(drop=True)


def closed_trades_frame(results: BacktestResults) -> pd.DataFrame:
    rows: list[dict] = []
    for trade in results.trades:
        row = asdict(trade)
        row["direction"] = trade.direction.value
        row["close_reason"] = trade.close_reason.value
        row["net_profit"] = trade.net_profit
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["opened_time"] = pd.to_datetime(frame["opened_time"], errors="coerce")
    frame["closed_time"] = pd.to_datetime(frame["closed_time"], errors="coerce")
    return frame.sort_values(["closed_time", "id"]).reset_index(drop=True)


def build_trade_dataset(results: BacktestResults) -> pd.DataFrame:
    decisions = decision_records_frame(results)
    trades = closed_trades_frame(results)
    if decisions.empty:
        return pd.DataFrame()

    if trades.empty:
        out = decisions.copy()
        out["realized_trade_count"] = 0
        out["realized_net_profit"] = 0.0
        out["realized_win"] = 0
        out["trade_taken"] = out["allow_trade"].astype(int)
        return out

    grouped = (
        trades.groupby("decision_id", dropna=False)
        .agg(
            realized_trade_count=("id", "count"),
            realized_net_profit=("net_profit", "sum"),
            realized_avg_profit=("net_profit", "mean"),
            realized_win_rate=("net_profit", lambda x: float((x > 0).mean())),
            first_open_time=("opened_time", "min"),
            last_close_time=("closed_time", "max"),
        )
        .reset_index()
    )
    grouped["realized_win"] = (grouped["realized_net_profit"] > 0).astype(int)

    out = decisions.merge(grouped, how="left", on="decision_id")
    fill_zero = [
        "realized_trade_count",
        "realized_net_profit",
        "realized_avg_profit",
        "realized_win_rate",
        "realized_win",
    ]
    for col in fill_zero:
        out[col] = out[col].fillna(0.0)
    out["trade_taken"] = (out["realized_trade_count"] > 0).astype(int)
    return out.sort_values(["time", "decision_id"]).reset_index(drop=True)
