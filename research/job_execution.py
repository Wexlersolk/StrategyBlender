from __future__ import annotations

from concurrent.futures import CancelledError
from datetime import datetime, timezone

from research.experiment_registry import get_experiment_record, save_overlay_snapshot
from research.overlay_evaluation import evaluate_portfolio_overlay_walk_forward
from research.state_store import load_dataset_snapshot, load_job, save_audit_event, save_job
from ui.views import ai_training


class JobReporter:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def _job(self) -> dict:
        job = load_job(self.job_id)
        if not job:
            raise ValueError(f"Job not found: {self.job_id}")
        return job

    def event(self, message: str, *, stage: str, level: str = "info") -> None:
        job = self._job()
        events = list(job.get("events", []))
        events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "stage": stage,
            "message": message,
        })
        job["events"] = events[-100:]
        job["stage"] = stage
        save_job(job)
        save_audit_event(entity_type="job", entity_id=self.job_id, event_type="job_event", level=level, payload={"stage": stage, "message": message}, owner_id=job.get("owner_id"))

    def progress(self, progress_pct: float, *, stage: str, message: str | None = None) -> None:
        job = self._job()
        job["progress_pct"] = max(0.0, min(100.0, float(progress_pct)))
        job["stage"] = stage
        save_job(job)
        if message:
            self.event(message, stage=stage, level="info")

    def check_cancelled(self) -> None:
        if self._job().get("cancel_requested"):
            raise CancelledError("Job was cancelled.")


def _eas_from_experiment_record(record: dict) -> dict[str, dict]:
    lineage = dict(record.get("lineage", {}) or {})
    dataset_id = str((lineage.get("dataset_snapshot", {}) or {}).get("dataset_id", ""))
    dataset_snapshot = load_dataset_snapshot(dataset_id) if dataset_id else None
    metadata = dict((dataset_snapshot or {}).get("metadata", {}) or {})
    snapshots = list(metadata.get("strategy_snapshots", []))
    eas = {}
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


def _build_rerun_lineage(record: dict, eas: dict, metadata: dict) -> dict:
    portfolio_ids = list(metadata.get("portfolio_ids", []))
    source_lineage = dict(record.get("lineage", {}) or {})
    return {
        **source_lineage,
        "rerun_of_run_id": record.get("run_id", ""),
        "strategy_versions": [ai_training._strategy_version_record(eas[ea_id]) for ea_id in portfolio_ids if ea_id in eas],  # noqa: SLF001
        "execution_assumptions": dict(metadata.get("execution_config", {}) or {}),
        "execution_assumptions_hash": ai_training._stable_hash(metadata.get("execution_config", {})),  # noqa: SLF001
        "lineage_version": 1,
    }


def run_overlay_rerun_job(job: dict, reporter: JobReporter) -> dict:
    source_run_id = str(job.get("payload", {}).get("source_run_id", ""))
    record = get_experiment_record(source_run_id)
    if not record:
        raise ValueError(f"Source experiment not found: {source_run_id}")

    eas = _eas_from_experiment_record(record)
    metadata = dict(record.get("metadata", {}) or {})
    artifact = metadata.get("artifact")
    portfolio_ids = list(metadata.get("portfolio_ids", []))
    scope = metadata.get("scope", {})
    execution_config = metadata.get("execution_config", {})
    if not artifact:
        raise ValueError("This experiment does not include an artifact snapshot to reproduce.")
    if not portfolio_ids:
        raise ValueError("This experiment does not include a portfolio scope to reproduce.")

    missing_ids = [ea_id for ea_id in portfolio_ids if ea_id not in eas]
    if missing_ids:
        raise ValueError(f"Missing strategies required for reproduction: {', '.join(missing_ids)}")

    reporter.progress(5.0, stage="validating", message="Validated rerun inputs.")
    reports_by_asset = {}
    total = max(len(portfolio_ids), 1)
    for idx, ea_id in enumerate(portfolio_ids, start=1):
        reporter.check_cancelled()
        reporter.progress(
            10.0 + ((idx - 1) / total) * 70.0,
            stage="evaluating",
            message=f"Running walk-forward evaluation for {ea_id} ({idx}/{total}).",
        )
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

    reporter.check_cancelled()
    reporter.progress(85.0, stage="assembling", message="Aggregating overlay results.")
    report = (
        next(iter(reports_by_asset.values()))
        if len(reports_by_asset) == 1
        else evaluate_portfolio_overlay_walk_forward(reports_by_asset=reports_by_asset)
    )
    title = f"Re-run {record.get('title', 'Experiment')}"
    lineage = _build_rerun_lineage(record, eas, metadata)
    reporter.progress(95.0, stage="saving", message="Saving canonical experiment record.")
    manifest, snapshot = save_overlay_snapshot(
        title=title,
        metadata=metadata,
        lineage=lineage,
        aggregate_metrics=report.aggregate_metrics,
        split_metrics=report.split_metrics,
        windows=report.windows,
        artifacts=ai_training._report_artifacts(report, metadata.get("artifact")),  # noqa: SLF001
    )
    snapshot["run_id"] = manifest["run_id"]
    snapshot["title"] = manifest["title"]
    reporter.progress(100.0, stage="saved", message=f"Saved experiment {manifest['run_id']}.")
    return snapshot


JOB_HANDLERS = {
    "overlay_rerun": run_overlay_rerun_job,
}
