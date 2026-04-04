from __future__ import annotations

import argparse
import sys

from research.experiment_registry import get_experiment_record, save_overlay_snapshot
from research.overlay_evaluation import evaluate_portfolio_overlay_walk_forward
from research.state_store import load_dataset_snapshot
from ui.views import ai_training


def _eas_from_dataset_snapshot(dataset_snapshot: dict | None) -> dict[str, dict]:
    metadata = dict((dataset_snapshot or {}).get("metadata", {}) or {})
    snapshots = list(metadata.get("strategy_snapshots", []))
    eas: dict[str, dict] = {}
    for idx, snapshot in enumerate(snapshots):
        ea_id = str(snapshot.get("ea_id") or f"ea_{idx}")
        eas[ea_id] = {
            "id": ea_id,
            "name": snapshot.get("name", ea_id),
            "symbol": snapshot.get("symbol", ""),
            "timeframe": snapshot.get("timeframe", ""),
            "source": snapshot.get("source", ""),
            "engine_source": snapshot.get("engine_source", ""),
            "review_source": snapshot.get("review_source", ""),
            "strategy_module": snapshot.get("strategy_module", ""),
            "strategy_class": snapshot.get("strategy_class", ""),
            "strategy_path": snapshot.get("strategy_path", ""),
        }
    return eas


def rerun_saved_experiment(run_id: str) -> dict:
    record = get_experiment_record(run_id)
    if not record:
        raise ValueError(f"Experiment not found: {run_id}")

    metadata = dict(record.get("metadata", {}) or {})
    lineage = dict(record.get("lineage", {}) or {})
    dataset_id = str((lineage.get("dataset_snapshot", {}) or {}).get("dataset_id", ""))
    dataset_snapshot = load_dataset_snapshot(dataset_id) if dataset_id else None
    eas = _eas_from_dataset_snapshot(dataset_snapshot)

    artifact = metadata.get("artifact")
    portfolio_ids = list(metadata.get("portfolio_ids", []))
    scope = metadata.get("scope", {})
    execution_config = metadata.get("execution_config", {})
    if not artifact:
        raise ValueError("Saved experiment does not include an artifact snapshot.")
    if not portfolio_ids:
        raise ValueError("Saved experiment does not include portfolio_ids.")

    missing = [ea_id for ea_id in portfolio_ids if ea_id not in eas]
    if missing:
        raise ValueError(f"Saved experiment is missing strategy snapshots for: {', '.join(missing)}")

    reports_by_asset = {}
    for ea_id in portfolio_ids:
        reports_by_asset[ea_id] = ai_training._run_walk_forward_comparison(  # noqa: SLF001
            eas[ea_id],
            artifact,
            date_from=str(scope["date_from"]),
            date_to=str(scope["date_to"]),
            train_bars=int(metadata["train_bars"]),
            test_bars=int(metadata["test_bars"]),
            embargo_bars=int(metadata["embargo_bars"]),
            execution_config=execution_config,
        )
    report = next(iter(reports_by_asset.values())) if len(reports_by_asset) == 1 else evaluate_portfolio_overlay_walk_forward(reports_by_asset=reports_by_asset)
    new_lineage = dict(lineage)
    new_lineage["rerun_of_run_id"] = run_id
    manifest, snapshot = save_overlay_snapshot(
        title=f"CLI Re-run {record.get('title', 'Experiment')}",
        metadata=metadata,
        lineage=new_lineage,
        aggregate_metrics=report.aggregate_metrics,
        split_metrics=report.split_metrics,
        windows=report.windows,
        artifacts=ai_training._report_artifacts(report, metadata.get("artifact")),  # noqa: SLF001
    )
    snapshot["run_id"] = manifest["run_id"]
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-run a saved overlay experiment from its persisted manifest.")
    parser.add_argument("--run-id", required=True, help="Saved experiment run_id to reproduce.")
    args = parser.parse_args(argv)
    snapshot = rerun_saved_experiment(args.run_id)
    print(snapshot["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
