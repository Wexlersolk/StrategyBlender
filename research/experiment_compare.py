from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _load_frame(record: dict, key: str, preview_key: str) -> pd.DataFrame:
    files = record.get("files", {}) or {}
    path = files.get(key)
    if path and Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(record.get(preview_key, []))


def _flatten(prefix: str, payload, out: dict[str, str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(payload, list):
        out[prefix] = json.dumps(payload, default=str, sort_keys=True)
    else:
        out[prefix] = str(payload)


def compare_experiments(base: dict, candidate: dict) -> dict:
    base_agg = _load_frame(base, "aggregate_metrics", "aggregate_preview")
    cand_agg = _load_frame(candidate, "aggregate_metrics", "aggregate_preview")
    metrics = []
    if not base_agg.empty and not cand_agg.empty:
        merged = base_agg.merge(cand_agg, on="overlay", suffixes=("_base", "_candidate"))
        for _, row in merged.iterrows():
            metrics.append(
                {
                    "overlay": row["overlay"],
                    "base_total_oos_net_profit": float(row.get("total_oos_net_profit_base", 0.0)),
                    "candidate_total_oos_net_profit": float(row.get("total_oos_net_profit_candidate", 0.0)),
                    "delta_total_oos_net_profit": float(row.get("total_oos_net_profit_candidate", 0.0) - row.get("total_oos_net_profit_base", 0.0)),
                    "base_win_window_rate": float(row.get("win_window_rate_base", 0.0)),
                    "candidate_win_window_rate": float(row.get("win_window_rate_candidate", 0.0)),
                    "delta_win_window_rate": float(row.get("win_window_rate_candidate", 0.0) - row.get("win_window_rate_base", 0.0)),
                }
            )

    base_params: dict[str, str] = {}
    cand_params: dict[str, str] = {}
    _flatten("", base.get("metadata", {}), base_params)
    _flatten("", base.get("lineage", {}), base_params)
    _flatten("", candidate.get("metadata", {}), cand_params)
    _flatten("", candidate.get("lineage", {}), cand_params)
    keys = sorted(set(base_params) | set(cand_params))
    parameter_changes = [
        {
            "field": key,
            "base": base_params.get(key, ""),
            "candidate": cand_params.get(key, ""),
        }
        for key in keys
        if base_params.get(key, "") != cand_params.get(key, "")
    ]

    explanation: list[str] = []
    best_overlay_delta = None
    if metrics:
        ml_row = next((row for row in metrics if row["overlay"] == "ML Overlay"), None)
        if ml_row:
            best_overlay_delta = ml_row["delta_total_oos_net_profit"]
            if abs(best_overlay_delta) > 1e-9:
                explanation.append(
                    "ML Overlay OOS net profit changed by "
                    f"{best_overlay_delta:+.2f}."
                )
    important_changes = [row["field"] for row in parameter_changes if any(token in row["field"] for token in ["artifact", "dataset", "execution_config", "train_bars", "test_bars", "benchmark", "strategy_versions"])]
    if important_changes:
        explanation.append("Input differences touched " + ", ".join(important_changes[:6]) + ".")
    if not explanation:
        explanation.append("No major metadata or aggregate-metric differences were detected.")

    return {
        "metric_diff": pd.DataFrame(metrics),
        "parameter_changes": pd.DataFrame(parameter_changes),
        "explanation": " ".join(explanation),
    }
