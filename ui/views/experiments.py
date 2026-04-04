from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from research.experiment_compare import compare_experiments
from research.experiment_registry import export_experiment_report, list_experiment_records
from research.state_store import list_jobs
from ui.job_runner import cancel_background_job, enqueue_background_job, job_audit_log, new_job_record, poll_background_jobs
from ui.components.common import empty_state, error_box, section_header
from ui.state import autosave


def _display_label(record: dict) -> str:
    title = str(record.get("title", "Experiment"))
    run_id = str(record.get("run_id", ""))[:20]
    created = str(record.get("created_at", ""))
    return f"{title} | {created} | {run_id}"


def _load_frame_from_record(record: dict, key: str, preview_key: str) -> pd.DataFrame:
    files = record.get("files", {}) or {}
    path = files.get(key)
    if path and Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    preview = record.get(preview_key, [])
    return pd.DataFrame(preview)


def _load_report_snapshot(record: dict) -> dict:
    metadata = dict(record.get("metadata", {}) or {})
    return {
        "ea_id": "",
        "portfolio_ids": metadata.get("portfolio_ids", []),
        "run_id": record.get("run_id", ""),
        "title": record.get("title", ""),
        "export_dir": record.get("export_dir", ""),
        "files": record.get("files", {}),
        "metadata": metadata,
        "lineage": record.get("lineage", {}),
        "split_metrics": _load_frame_from_record(record, "split_metrics", "split_preview"),
        "aggregate_metrics": _load_frame_from_record(record, "aggregate_metrics", "aggregate_preview"),
    }


def _load_artifacts(record: dict) -> dict:
    files = record.get("files", {}) or {}
    artifacts_path = files.get("artifacts")
    if not artifacts_path or not Path(artifacts_path).exists():
        return {}
    try:
        payload = json.loads(Path(artifacts_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    loaded = {}
    for key, value in payload.items():
        if isinstance(value, str) and value.endswith(".csv") and Path(value).exists():
            try:
                loaded[key] = pd.read_csv(value)
                continue
            except Exception:
                pass
        loaded[key] = value
    return loaded


def _integrate_completed_job(_job: dict, result: dict):
    if not isinstance(result, dict):
        return
    if result.get("run_id"):
        st.session_state["_selected_experiment_run_id"] = result.get("run_id", "")


def _queue_rerun_job(record: dict, eas: dict) -> dict:
    del eas
    job = new_job_record(
        kind="overlay_rerun",
        title=f"Re-run {record.get('title', 'Experiment')}",
        payload={"source_run_id": record.get("run_id", "")},
        max_retries=1,
    )
    enqueue_background_job(job)
    st.session_state["job_registry"] = list_jobs()[:100]
    autosave()
    return job


def _render_job_registry():
    jobs = list_jobs()
    if not jobs:
        return
    st.markdown("#### Background Jobs")
    listing = pd.DataFrame(jobs)[["job_id", "title", "status", "stage", "progress_pct", "attempt", "max_retries", "submitted_at", "result_run_id"]]
    st.dataframe(listing, use_container_width=True, hide_index=True)

    selected_job_id = st.session_state.get("_selected_experiment_job_id", jobs[0].get("job_id", ""))
    job_ids = [job.get("job_id", "") for job in jobs]
    if selected_job_id not in job_ids:
        selected_job_id = job_ids[0]
    selected_job_id = st.selectbox(
        "Job Details",
        job_ids,
        index=job_ids.index(selected_job_id),
        format_func=lambda job_id: next(
            (
                f"{job.get('title', 'Job')} | {job.get('status', '')} | {job_id}"
                for job in jobs
                if job.get("job_id") == job_id
            ),
            job_id,
        ),
        key="experiment_job_select",
    )
    st.session_state["_selected_experiment_job_id"] = selected_job_id
    selected_job = next(job for job in jobs if job.get("job_id") == selected_job_id)

    job_summary = pd.DataFrame(
        [
            {"field": "job_id", "value": selected_job.get("job_id", "")},
            {"field": "kind", "value": selected_job.get("kind", "")},
            {"field": "status", "value": selected_job.get("status", "")},
            {"field": "stage", "value": selected_job.get("stage", "")},
            {"field": "progress_pct", "value": selected_job.get("progress_pct", 0.0)},
            {"field": "attempt", "value": selected_job.get("attempt", 0)},
            {"field": "max_retries", "value": selected_job.get("max_retries", 0)},
            {"field": "cancel_requested", "value": selected_job.get("cancel_requested", False)},
            {"field": "submitted_at", "value": selected_job.get("submitted_at", "")},
            {"field": "started_at", "value": selected_job.get("started_at", "")},
            {"field": "finished_at", "value": selected_job.get("finished_at", "")},
            {"field": "source_run_id", "value": selected_job.get("payload", {}).get("source_run_id", "")},
            {"field": "result_run_id", "value": selected_job.get("result_run_id", "")},
        ]
    )
    st.dataframe(job_summary, use_container_width=True, hide_index=True)
    progress = float(selected_job.get("progress_pct", 0.0))
    st.progress(min(max(progress / 100.0, 0.0), 1.0), text=f"{selected_job.get('stage', '')} ({progress:.0f}%)")
    if selected_job.get("status") in {"queued", "running"}:
        if st.button("Cancel Job", type="secondary", use_container_width=True, key="experiment_cancel_job"):
            cancel_background_job(selected_job["job_id"])
            st.rerun()
    if selected_job.get("error"):
        error_box(f"Job failed: {selected_job['error']}")
    events = pd.DataFrame(list(selected_job.get("events", [])))
    if not events.empty:
        st.markdown("#### Job Events")
        st.dataframe(events.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    audit = pd.DataFrame(job_audit_log(selected_job["job_id"], limit=50))
    if not audit.empty:
        st.markdown("#### Job Audit Trail")
        st.dataframe(audit, use_container_width=True, hide_index=True)


def render():
    section_header(
        "Experiments",
        "Inspect saved research runs, review exported OOS metrics, and reproduce prior overlay evaluations.",
    )
    poll_background_jobs(st.session_state, context={"eas": st.session_state.get("eas", {})}, on_success=_integrate_completed_job)

    registry = list_experiment_records()
    st.session_state["experiment_registry"] = registry[:100]
    _render_job_registry()
    if not registry:
        empty_state("No experiments saved yet. Run a walk-forward overlay evaluation to populate the registry.", "🧪")
        return

    selected_run_id = st.session_state.get("_selected_experiment_run_id", registry[0].get("run_id"))
    run_ids = [record.get("run_id", "") for record in registry]
    default_index = run_ids.index(selected_run_id) if selected_run_id in run_ids else 0
    selected_run_id = st.selectbox(
        "Experiment",
        run_ids,
        index=default_index,
        format_func=lambda run_id: _display_label(next(record for record in registry if record.get("run_id") == run_id)),
        key="experiment_select",
    )
    st.session_state["_selected_experiment_run_id"] = selected_run_id
    record = next(record for record in registry if record.get("run_id") == selected_run_id)

    summary = pd.DataFrame(
        [
            {"field": "run_id", "value": record.get("run_id", "")},
            {"field": "title", "value": record.get("title", "")},
            {"field": "kind", "value": record.get("kind", "")},
            {"field": "created_at", "value": record.get("created_at", "")},
            {"field": "export_dir", "value": record.get("export_dir", "")},
        ]
    )
    st.markdown("#### Run Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    aggregate = _load_frame_from_record(record, "aggregate_metrics", "aggregate_preview")
    split = _load_frame_from_record(record, "split_metrics", "split_preview")
    if not aggregate.empty:
        st.markdown("#### Aggregate Metrics")
        st.dataframe(aggregate, use_container_width=True, hide_index=True)
    if not split.empty:
        st.markdown("#### Split Metrics")
        st.dataframe(split, use_container_width=True, hide_index=True)

    st.markdown("#### Metadata Snapshot")
    st.code(json.dumps(record.get("metadata", {}), indent=2, default=str), language="json")
    if record.get("lineage"):
        st.markdown("#### Lineage Snapshot")
        st.code(json.dumps(record.get("lineage", {}), indent=2, default=str), language="json")
    try:
        archive_path = export_experiment_report(record.get("run_id", ""))
        st.caption(f"Export archive: {archive_path}")
    except Exception:
        pass

    artifacts = _load_artifacts(record)
    validation = artifacts.get("validation", {})
    if validation:
        st.markdown("#### Validation Rails")
        st.code(json.dumps(validation, indent=2, default=str), language="json")
    for key, title in [
        ("window_details", "Per-Window Diagnostics"),
        ("calibration", "Calibration"),
        ("trade_logs", "Trade Logs"),
        ("equity_curves", "Equity Curves"),
    ]:
        frame = artifacts.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            st.markdown(f"#### {title}")
            st.dataframe(frame, use_container_width=True, hide_index=True)

    if len(registry) > 1:
        compare_options = [run_id for run_id in run_ids if run_id != selected_run_id]
        if compare_options:
            compare_run_id = st.selectbox("Compare Against", compare_options, key="experiment_compare_select")
            candidate = next(item for item in registry if item.get("run_id") == compare_run_id)
            comparison = compare_experiments(record, candidate)
            st.markdown("#### Comparison")
            if comparison["explanation"]:
                st.caption(comparison["explanation"])
            if not comparison["metric_diff"].empty:
                st.dataframe(comparison["metric_diff"], use_container_width=True, hide_index=True)
            if not comparison["parameter_changes"].empty:
                st.markdown("#### Parameter Changes")
                st.dataframe(comparison["parameter_changes"], use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Into AI Validation", type="secondary", use_container_width=True, key="experiment_load"):
            st.session_state["_ai_model_wfo"] = _load_report_snapshot(record)
            st.session_state["_ai_model_wfo_run_id"] = record.get("run_id", "")
            st.session_state["page"] = "AI Training"
            autosave()
            st.rerun()
    with col2:
        if st.button("Queue Re-run Job", type="primary", use_container_width=True, key="experiment_rerun"):
            try:
                job = _queue_rerun_job(record, st.session_state.get("eas", {}))
                st.session_state["_selected_experiment_job_id"] = job["job_id"]
                st.rerun()
            except Exception as exc:
                error_box(f"Experiment re-run failed: {exc}")
