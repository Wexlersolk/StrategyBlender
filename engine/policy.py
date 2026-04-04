from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from engine.position import Direction, OrderType


@dataclass
class TradeIntent:
    order_type: OrderType
    direction: Direction
    price: float
    stop_loss: float
    take_profit: float
    lots: float
    bar_idx: int
    time: Any
    expiry_bars: int = 1
    comment: str = ""
    trailing_stop: float = 0.0
    trail_activation: float = 0.0
    exit_after_bars: int = 0


@dataclass
class OverlayDecision:
    allow_trade: bool = True
    size_multiplier: float = 1.0
    adjusted_lots: float | None = None
    tag: str = "pass"
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def final_lots(self) -> float | None:
        return self.adjusted_lots


@dataclass
class DecisionRecord:
    decision_id: int
    time: Any
    bar_idx: int
    symbol: str
    timeframe: str
    order_type: str
    direction: str
    requested_lots: float
    final_lots: float
    allow_trade: bool
    policy_name: str
    policy_tag: str
    comment: str
    market_features: dict[str, float] = field(default_factory=dict)
    policy_notes: dict[str, Any] = field(default_factory=dict)


POLICY_FEATURE_NAMES = [
    "requested_lots",
    "realized_vol",
    "balance_drawdown_pct",
    "equity",
    "balance",
    "is_long",
    "is_short",
    "is_market_order",
    "is_pending_order",
    "hour_of_day",
    "weekday",
]


def extract_policy_feature_map(intent: TradeIntent, state: dict[str, Any]) -> dict[str, float]:
    timestamp = intent.time
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if not isinstance(timestamp, datetime):
        hour = 0.0
        weekday = 0.0
    else:
        hour = float(timestamp.hour)
        weekday = float(timestamp.weekday())

    is_market_order = float(intent.order_type == OrderType.MARKET)
    is_long = float(intent.direction == Direction.LONG)
    is_short = float(intent.direction == Direction.SHORT)

    return {
        "requested_lots": float(intent.lots),
        "realized_vol": float(state.get("realized_vol", 0.0) or 0.0),
        "balance_drawdown_pct": float(state.get("balance_drawdown_pct", 0.0) or 0.0),
        "equity": float(state.get("equity", 0.0) or 0.0),
        "balance": float(state.get("balance", 0.0) or 0.0),
        "is_long": is_long,
        "is_short": is_short,
        "is_market_order": is_market_order,
        "is_pending_order": 1.0 - is_market_order,
        "hour_of_day": hour,
        "weekday": weekday,
    }


class OverlayPolicy:
    name = "OverlayPolicy"

    def evaluate(self, intent: TradeIntent, state: dict[str, Any]) -> OverlayDecision:
        return OverlayDecision()


class NullOverlayPolicy(OverlayPolicy):
    name = "NullOverlayPolicy"


class VolatilityTargetPolicy(OverlayPolicy):
    name = "VolatilityTargetPolicy"

    def __init__(
        self,
        *,
        target_vol: float = 0.20,
        lookback: int = 20,
        min_multiplier: float = 0.25,
        max_multiplier: float = 1.50,
        floor_vol: float = 1e-6,
    ):
        self.target_vol = float(target_vol)
        self.lookback = int(lookback)
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)
        self.floor_vol = float(floor_vol)

    def evaluate(self, intent: TradeIntent, state: dict[str, Any]) -> OverlayDecision:
        realized_vol = float(state.get("realized_vol", 0.0) or 0.0)
        if realized_vol <= self.floor_vol:
            mult = self.max_multiplier
        else:
            mult = self.target_vol / realized_vol
        mult = max(self.min_multiplier, min(self.max_multiplier, mult))
        return OverlayDecision(
            allow_trade=True,
            size_multiplier=mult,
            tag="vol_target",
            notes={"realized_vol": realized_vol, "target_vol": self.target_vol},
        )


class DrawdownThrottlePolicy(OverlayPolicy):
    name = "DrawdownThrottlePolicy"

    def __init__(
        self,
        *,
        soft_drawdown_pct: float = 5.0,
        hard_drawdown_pct: float = 10.0,
        soft_multiplier: float = 0.50,
    ):
        self.soft_drawdown_pct = float(soft_drawdown_pct)
        self.hard_drawdown_pct = float(hard_drawdown_pct)
        self.soft_multiplier = float(soft_multiplier)

    def evaluate(self, intent: TradeIntent, state: dict[str, Any]) -> OverlayDecision:
        drawdown_pct = float(state.get("balance_drawdown_pct", 0.0) or 0.0)
        if drawdown_pct >= self.hard_drawdown_pct:
            return OverlayDecision(
                allow_trade=False,
                size_multiplier=0.0,
                tag="dd_block",
                notes={"balance_drawdown_pct": drawdown_pct},
            )
        if drawdown_pct >= self.soft_drawdown_pct:
            return OverlayDecision(
                allow_trade=True,
                size_multiplier=self.soft_multiplier,
                tag="dd_throttle",
                notes={"balance_drawdown_pct": drawdown_pct},
            )
        return OverlayDecision(
            allow_trade=True,
            size_multiplier=1.0,
            tag="dd_pass",
            notes={"balance_drawdown_pct": drawdown_pct},
        )


class CompositeOverlayPolicy(OverlayPolicy):
    name = "CompositeOverlayPolicy"

    def __init__(self, policies: list[OverlayPolicy]):
        self.policies = list(policies)

    def evaluate(self, intent: TradeIntent, state: dict[str, Any]) -> OverlayDecision:
        final = OverlayDecision()
        notes: dict[str, Any] = {}
        tags: list[str] = []
        for policy in self.policies:
            decision = policy.evaluate(intent, state)
            tags.append(decision.tag)
            notes[policy.name] = decision.notes
            final.allow_trade = final.allow_trade and decision.allow_trade
            final.size_multiplier *= float(decision.size_multiplier)
        final.tag = "+".join(tags) if tags else "pass"
        final.notes = notes
        return final


class ModelOverlayPolicy(OverlayPolicy):
    name = "ModelOverlayPolicy"

    def __init__(
        self,
        *,
        filter_model: dict | None = None,
        sizing_model: dict | None = None,
        base_policy: OverlayPolicy | None = None,
    ):
        self.filter_model = filter_model or {}
        self.sizing_model = sizing_model or {}
        self.base_policy = base_policy or NullOverlayPolicy()

    @staticmethod
    def _linear_score(model: dict, features: dict[str, float]) -> float:
        names = model.get("feature_names", [])
        means = np.asarray(model.get("feature_means", [0.0] * len(names)), dtype=float)
        stds = np.asarray(model.get("feature_stds", [1.0] * len(names)), dtype=float)
        coefs = np.asarray(model.get("coefficients", [0.0] * len(names)), dtype=float)
        intercept = float(model.get("intercept", 0.0))
        values = np.asarray([float(features.get(name, 0.0)) for name in names], dtype=float)
        stds[stds < 1e-9] = 1.0
        x = (values - means) / stds
        return float(x @ coefs + intercept)

    def evaluate(self, intent: TradeIntent, state: dict[str, Any]) -> OverlayDecision:
        base = self.base_policy.evaluate(intent, state) if self.base_policy else OverlayDecision()
        if not base.allow_trade:
            return base

        features = extract_policy_feature_map(intent, state)
        notes: dict[str, Any] = {"base_policy": base.tag}
        allow_trade = bool(base.allow_trade)
        size_multiplier = float(base.size_multiplier)
        tags: list[str] = [base.tag] if base.tag else []

        if self.filter_model:
            score = self._linear_score(self.filter_model, features)
            probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
            threshold = float(self.filter_model.get("probability_threshold", 0.55))
            allow_trade = allow_trade and probability >= threshold
            tags.append("ml_pass" if probability >= threshold else "ml_block")
            notes["filter_probability"] = float(probability)
            notes["filter_threshold"] = threshold

        if self.sizing_model:
            predicted_profit = self._linear_score(self.sizing_model, features)
            scale = max(float(self.sizing_model.get("target_scale", 1.0)), 1e-6)
            min_multiplier = float(self.sizing_model.get("min_multiplier", 0.5))
            max_multiplier = float(self.sizing_model.get("max_multiplier", 1.5))
            normalized = np.clip(predicted_profit / scale, -3.0, 3.0)
            sizing_multiplier = min_multiplier + (max_multiplier - min_multiplier) * (
                1.0 / (1.0 + np.exp(-normalized))
            )
            size_multiplier *= float(sizing_multiplier)
            tags.append("ml_size")
            notes["predicted_profit"] = float(predicted_profit)
            notes["sizing_multiplier"] = float(sizing_multiplier)

        return OverlayDecision(
            allow_trade=allow_trade,
            size_multiplier=size_multiplier,
            tag="+".join([tag for tag in tags if tag]) or "pass",
            notes=notes,
        )
