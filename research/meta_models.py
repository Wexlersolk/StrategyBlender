from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.policy import POLICY_FEATURE_NAMES


def _safe_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def build_policy_feature_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset is None or dataset.empty:
        return pd.DataFrame(columns=POLICY_FEATURE_NAMES)

    frame = pd.DataFrame(index=dataset.index)
    time_col = pd.to_datetime(dataset.get("time"), errors="coerce")

    frame["requested_lots"] = _safe_series(dataset, "requested_lots", 0.0)
    frame["realized_vol"] = _safe_series(dataset, "feature_realized_vol", 0.0)
    frame["balance_drawdown_pct"] = _safe_series(dataset, "feature_balance_drawdown_pct", 0.0)
    frame["equity"] = _safe_series(dataset, "feature_equity", 0.0)
    frame["balance"] = _safe_series(dataset, "feature_balance", 0.0)
    direction = dataset["direction"] if "direction" in dataset.columns else pd.Series("", index=dataset.index)
    order_type = dataset["order_type"] if "order_type" in dataset.columns else pd.Series("", index=dataset.index)
    frame["is_long"] = (direction == "long").astype(float)
    frame["is_short"] = (direction == "short").astype(float)
    frame["is_market_order"] = (order_type == "market").astype(float)
    frame["is_pending_order"] = 1.0 - frame["is_market_order"]
    frame["hour_of_day"] = pd.Series(time_col.dt.hour, index=dataset.index).fillna(0).astype(float)
    frame["weekday"] = pd.Series(time_col.dt.weekday, index=dataset.index).fillna(0).astype(float)
    return frame.astype(float)


def _standardize_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame.to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds < 1e-9] = 1.0
    return (values - means) / stds, means, stds


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(int)
    tp = float(np.sum((preds == 1) & (y_true == 1)))
    tn = float(np.sum((preds == 0) & (y_true == 0)))
    fp = float(np.sum((preds == 1) & (y_true == 0)))
    fn = float(np.sum((preds == 0) & (y_true == 1)))
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "positive_rate": float(np.mean(preds)) if len(preds) else 0.0,
    }


def fit_filter_model(
    dataset: pd.DataFrame,
    *,
    threshold: float = 0.55,
    positive_return_cutoff: float = 0.0,
    iterations: int = 500,
    learning_rate: float = 0.05,
    l2: float = 0.01,
) -> dict:
    features = build_policy_feature_frame(dataset)
    if features.empty:
        raise ValueError("Dataset is empty.")

    target = (_safe_series(dataset, "realized_net_profit", 0.0) > float(positive_return_cutoff)).astype(float).to_numpy()
    if len(np.unique(target)) < 2:
        raise ValueError("Filter model needs both positive and negative outcomes.")

    x, means, stds = _standardize_features(features)
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0

    for _ in range(int(iterations)):
        logits = x @ weights + bias
        probs = _sigmoid(logits)
        error = probs - target
        grad_w = (x.T @ error) / len(x) + float(l2) * weights
        grad_b = float(error.mean())
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b

    probabilities = _sigmoid(x @ weights + bias)
    metrics = _classification_metrics(target, probabilities, float(threshold))
    metrics["avg_probability"] = float(probabilities.mean())

    return {
        "kind": "filter",
        "feature_names": list(features.columns),
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "coefficients": weights.tolist(),
        "intercept": float(bias),
        "probability_threshold": float(threshold),
        "positive_return_cutoff": float(positive_return_cutoff),
        "metrics": metrics,
        "sample_size": int(len(features)),
    }


def fit_sizing_model(
    dataset: pd.DataFrame,
    *,
    min_multiplier: float = 0.50,
    max_multiplier: float = 1.50,
    l2: float = 1.0,
) -> dict:
    features = build_policy_feature_frame(dataset)
    if features.empty:
        raise ValueError("Dataset is empty.")

    target = _safe_series(dataset, "realized_net_profit", 0.0).to_numpy(dtype=float)
    if len(target) < 3:
        raise ValueError("Sizing model needs at least 3 samples.")

    x, means, stds = _standardize_features(features)
    x_design = np.column_stack([x, np.ones(len(x), dtype=float)])
    ridge = np.eye(x_design.shape[1], dtype=float) * float(l2)
    ridge[-1, -1] = 0.0
    coefs = np.linalg.solve(x_design.T @ x_design + ridge, x_design.T @ target)

    predictions = x_design @ coefs
    mse = float(np.mean((predictions - target) ** 2))
    denom = float(np.sum((target - target.mean()) ** 2))
    r2 = 0.0 if denom < 1e-9 else float(1.0 - np.sum((predictions - target) ** 2) / denom)
    scale = float(np.std(target, ddof=0))
    if scale < 1e-6:
        scale = max(abs(float(target.mean())), 1.0)

    return {
        "kind": "sizing",
        "feature_names": list(features.columns),
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "coefficients": coefs[:-1].tolist(),
        "intercept": float(coefs[-1]),
        "target_scale": float(scale),
        "min_multiplier": float(min_multiplier),
        "max_multiplier": float(max_multiplier),
        "metrics": {
            "mse": mse,
            "r2": r2,
            "avg_predicted_profit": float(predictions.mean()),
        },
        "sample_size": int(len(features)),
    }


def predict_filter_probabilities(dataset: pd.DataFrame, model: dict) -> pd.Series:
    features = build_policy_feature_frame(dataset).reindex(columns=model["feature_names"], fill_value=0.0)
    values = features.to_numpy(dtype=float)
    means = np.asarray(model["feature_means"], dtype=float)
    stds = np.asarray(model["feature_stds"], dtype=float)
    stds[stds < 1e-9] = 1.0
    x = (values - means) / stds
    probs = _sigmoid(x @ np.asarray(model["coefficients"], dtype=float) + float(model["intercept"]))
    return pd.Series(probs, index=dataset.index, dtype=float)


def predict_sizing_signal(dataset: pd.DataFrame, model: dict) -> pd.Series:
    features = build_policy_feature_frame(dataset).reindex(columns=model["feature_names"], fill_value=0.0)
    values = features.to_numpy(dtype=float)
    means = np.asarray(model["feature_means"], dtype=float)
    stds = np.asarray(model["feature_stds"], dtype=float)
    stds[stds < 1e-9] = 1.0
    x = (values - means) / stds
    preds = x @ np.asarray(model["coefficients"], dtype=float) + float(model["intercept"])
    return pd.Series(preds, index=dataset.index, dtype=float)


@dataclass
class OverlayResearchArtifact:
    filter_model: dict
    sizing_model: dict
    dataset_summary: dict
    benchmark_name: str
    benchmark_settings: dict

    def to_dict(self) -> dict:
        return {
            "filter_model": self.filter_model,
            "sizing_model": self.sizing_model,
            "dataset_summary": self.dataset_summary,
            "benchmark_name": self.benchmark_name,
            "benchmark_settings": self.benchmark_settings,
        }
