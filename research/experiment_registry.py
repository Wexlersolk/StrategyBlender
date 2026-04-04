from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.state_store import (
    canonical_hash,
    list_experiment_manifests,
    load_experiment_manifest,
    save_experiment_manifest,
)


ROOT = Path(__file__).resolve().parent.parent
EXPORT_ROOT = ROOT / "data" / "exports" / "overlay_reports"


@dataclass
class ExperimentRecord:
    run_id: str
    created_at: str
    kind: str
    title: str
    metadata: dict
    lineage: dict = field(default_factory=dict)
    aggregate_preview: list[dict] = field(default_factory=list)
    split_preview: list[dict] = field(default_factory=list)
    export_dir: str = ""
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"


def dataframe_preview(df: pd.DataFrame, limit: int = 10) -> list[dict]:
    if df is None or df.empty:
        return []
    preview = df.head(int(limit)).copy()
    return json.loads(preview.to_json(orient="records", date_format="iso"))


def create_experiment_record(
    *,
    kind: str,
    title: str,
    metadata: dict,
    lineage: dict | None = None,
    aggregate_metrics: pd.DataFrame | None = None,
    split_metrics: pd.DataFrame | None = None,
    export_dir: str = "",
    files: dict | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        run_id=new_run_id(kind),
        created_at=datetime.now(timezone.utc).isoformat(),
        kind=kind,
        title=title,
        metadata=dict(metadata or {}),
        lineage=dict(lineage or {}),
        aggregate_preview=dataframe_preview(aggregate_metrics),
        split_preview=dataframe_preview(split_metrics),
        export_dir=str(export_dir or ""),
        files={k: str(v) for k, v in (files or {}).items()},
    )


def build_overlay_snapshot(
    *,
    title: str,
    metadata: dict,
    lineage: dict | None,
    aggregate_metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    windows: list | None = None,
    artifacts: dict | None = None,
) -> tuple[ExperimentRecord, dict]:
    record = create_experiment_record(
        kind="overlay_wfo",
        title=title,
        metadata=metadata,
        lineage=lineage,
        aggregate_metrics=aggregate_metrics,
        split_metrics=split_metrics,
    )
    files = export_overlay_report_bundle(
        record=record,
        aggregate_metrics=aggregate_metrics,
        split_metrics=split_metrics,
        metadata=metadata,
        artifacts=artifacts,
    )
    record.export_dir = files["export_dir"]
    record.files = {k: v for k, v in files.items() if k != "export_dir"}
    snapshot = {
        "ea_id": metadata.get("portfolio_ids", [""])[0] if len(metadata.get("portfolio_ids", [])) == 1 else "",
        "portfolio_ids": list(metadata.get("portfolio_ids", [])),
        "run_id": record.run_id,
        "title": record.title,
        "export_dir": record.export_dir,
        "files": record.files,
        "metadata": metadata,
        "lineage": record.lineage,
        "split_metrics": split_metrics,
        "aggregate_metrics": aggregate_metrics,
        "windows": list(windows or []),
        "artifacts": dict(artifacts or {}),
    }
    return record, snapshot


def save_overlay_snapshot(
    *,
    title: str,
    metadata: dict,
    lineage: dict | None,
    aggregate_metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    windows: list | None = None,
    artifacts: dict | None = None,
) -> tuple[dict, dict]:
    record, snapshot = build_overlay_snapshot(
        title=title,
        metadata=metadata,
        lineage=lineage,
        aggregate_metrics=aggregate_metrics,
        split_metrics=split_metrics,
        windows=windows,
        artifacts=artifacts,
    )
    manifest = record.to_dict()
    manifest["content_hash"] = canonical_hash(
        {
            "metadata": manifest["metadata"],
            "lineage": manifest.get("lineage", {}),
            "aggregate_preview": manifest["aggregate_preview"],
            "split_preview": manifest["split_preview"],
            "files": manifest["files"],
        }
    )
    save_experiment_manifest(manifest)
    return manifest, snapshot


def get_experiment_record(run_id: str) -> dict | None:
    return load_experiment_manifest(run_id)


def list_experiment_records() -> list[dict]:
    return list_experiment_manifests()


def export_experiment_report(run_id: str, *, output_path: Path | None = None) -> Path:
    record = get_experiment_record(run_id)
    if not record:
        raise ValueError(f"Experiment not found: {run_id}")
    export_dir = Path(record.get("export_dir", ""))
    if not export_dir.exists():
        raise ValueError(f"Export directory not found for experiment: {run_id}")
    destination = Path(output_path) if output_path else export_dir.parent / f"{run_id}_report"
    archive = shutil.make_archive(str(destination), "zip", root_dir=str(export_dir))
    return Path(archive)


def export_overlay_report_bundle(
    *,
    record: ExperimentRecord,
    aggregate_metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    metadata: dict,
    artifacts: dict | None = None,
    output_root: Path | None = None,
) -> dict[str, str]:
    root = Path(output_root or EXPORT_ROOT)
    export_dir = root / record.run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = export_dir / "aggregate_metrics.csv"
    split_path = export_dir / "split_metrics.csv"
    metadata_path = export_dir / "metadata.json"
    artifacts_path = export_dir / "artifacts.json"

    aggregate_metrics.to_csv(aggregate_path, index=False)
    split_metrics.to_csv(split_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    normalized_artifacts = dict(artifacts or {})
    for name, payload in list(normalized_artifacts.items()):
        if isinstance(payload, pd.DataFrame):
            artifact_path = export_dir / f"{name}.csv"
            payload.to_csv(artifact_path, index=False)
            normalized_artifacts[name] = str(artifact_path)
    artifacts_path.write_text(json.dumps(normalized_artifacts, indent=2, default=str), encoding="utf-8")

    return {
        "aggregate_metrics": str(aggregate_path),
        "split_metrics": str(split_path),
        "metadata": str(metadata_path),
        "artifacts": str(artifacts_path),
        "export_dir": str(export_dir),
    }
