"""Publish stage artifacts with provenance and explicit completion state."""

import gzip
import hashlib
import os
import json
import platform
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .config import (DetectionConfig, EvaluationConfig, FeatureConfig, IncidentConfig, IngestConfig, InputError,
                     MLConfig, RunConfig)
from .features import build_windows
from .ingest import Enrichment, IngestionResult, NormalizedLog, read_logs
from .parse import parse_messages


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json_text(value) + "\n", encoding="utf-8")


def _write_lines(path: Path, values) -> str:
    """Stream JSON lines, optionally gzip-compressed, returning the content hash."""
    digest = hashlib.sha256()
    with path.open("wb") as raw:
        # Omit filename and timestamp so compressed artifacts are reproducible.
        handle = (gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6)
                  if path.suffix == ".gz" else raw)
        try:
            for value in values:
                line = (json_text(value) + "\n").encode("utf-8")
                digest.update(line)
                handle.write(line)
        finally:
            if handle is not raw:
                handle.close()
    return digest.hexdigest()


def _open_data(path: Path, mode="rt"):
    """Read both legacy plain streams and compact gzip streams without extraction."""
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    options = {"encoding": "utf-8"} if "b" not in mode else {}
    return gzip.open(path, mode, **options) if path.suffix == ".gz" else path.open(mode, **options)


def _content_hashes(manifest: dict) -> dict:
    """Stage dependencies identify logical content, independently of storage encoding."""
    return {**manifest["artifacts"], **manifest.get("content_artifacts", {})}


def _artifact_name(manifest: dict, name: str) -> str:
    if name in manifest["artifacts"]:
        return name
    if name + ".gz" in manifest["artifacts"]:
        return name + ".gz"
    raise InputError(f"required artifact missing from manifest: {name}")


def _validate_artifacts(staging: Path, result: IngestionResult) -> None:
    """Validate serialized counts, ordering, references, and disjoint row coverage."""
    seen = set()
    previous = None
    for name, count in (("normalized.jsonl", len(result.records)), ("rejects.jsonl", len(result.rejects))):
        observed = 0
        with _open_data(staging / name) as handle:
            for line in handle:
                row = json.loads(line)
                ordinal = row["source_record_number"]
                if (type(ordinal) is not int or ordinal in seen or
                        not 1 <= ordinal <= result.quality["parsed_count"] or
                        row["record_id"] != f"{result.input_sha256}:{ordinal}" or
                        row["schema_version"] != SCHEMA_VERSION):
                    raise ValueError("invalid source reference or artifact schema")
                if name == "normalized.jsonl":
                    key = row["timestamp"], ordinal
                    if previous is not None and key < previous:
                        raise ValueError("normalized records out of order")
                    previous = key
                seen.add(ordinal)
                observed += 1
        if observed != count:
            raise ValueError(f"artifact count mismatch: {name}")
    if len(seen) != result.quality["parsed_count"]:
        raise ValueError("source accounting mismatch")
    if json.loads((staging / "quality.json").read_text(encoding="utf-8")) != result.quality:
        raise ValueError("quality artifact mismatch")


def _publish(input_path: Path, config: IngestConfig, output: Path, *, compact: bool = False) -> dict:
    start = time.perf_counter()
    phase = 2 if isinstance(config, FeatureConfig) else 1
    stage = "eda_and_features" if phase == 2 else "ingestion_and_parsing"
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    # Reserve staging before work. Failures leave a diagnostic, visibly incomplete run.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    resolved = config.to_dict()
    config_hash = hashlib.sha256(json_text(resolved).encode("utf-8")).hexdigest()
    _write_json(staging / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "phase": phase, "status": "incomplete",
        "pipeline_complete": False,
    })
    result = read_logs(input_path)
    run_id = hashlib.sha256(f"{result.input_sha256}:{config_hash}:{__version__}".encode()).hexdigest()
    result.quality["timezone"] = config.source_timezone
    result.quality["parse_coverage"] = parse_messages(result.records)
    _write_json(staging / "config.json", resolved)
    _write_json(staging / "quality.json", result.quality)
    content_hashes = {"normalized.jsonl": _write_lines(
        staging / ("normalized.jsonl.gz" if compact else "normalized.jsonl"),
        (r.to_dict() for r in result.records))}
    _write_lines(staging / "rejects.jsonl", (asdict(r) for r in result.rejects))
    _validate_artifacts(staging, result)
    if not result.records:
        raise InputError(f"no usable records; diagnostics: {staging}")
    if result.quality["rejected_fraction"] > config.max_rejected_fraction:
        raise InputError(f"rejection fraction exceeds {config.max_rejected_fraction}; diagnostics: {staging}")
    split_ranges = None
    dependencies = {}
    if isinstance(config, FeatureConfig):
        from .report import write_eda_charts

        features = build_windows(result.records, config)
        split_ranges = features.split_ranges
        content_hashes["windows.jsonl"] = _write_lines(
            staging / ("windows.jsonl.gz" if compact else "windows.jsonl"), features.windows)
        _write_json(staging / "template_reference.json", {
            "schema_version": SCHEMA_VERSION, "fit_split": "train",
            "train_range": split_ranges["train"], "services": features.template_reference,
        })
        split_counts = {name: sum(bounds["start"] <= r.timestamp.isoformat() < bounds["end"]
                                  for r in result.records)
                        for name, bounds in split_ranges.items()}
        _write_json(staging / "eda.json", {
            "schema_version": SCHEMA_VERSION, "timezone": config.source_timezone,
            "source_time_bounds": result.quality["source_time_bounds"],
            "parse_coverage": result.quality["parse_coverage"],
            "split_ranges": split_ranges, "split_record_counts": split_counts,
            "services": features.service_statistics,
            "window_counts": {str(minutes): dict(sorted(Counter(w["scope"] for w in features.windows
                              if w["window_minutes"] == minutes).items()))
                              for minutes in (config.window_minutes, config.alternate_window_minutes)},
            "limitations": [
                "Counts are log records, not distinct user transactions across services.",
                "Empty bins indicate no accepted records, not proven outages or collection coverage.",
                "Template references use training only; training novelty is in-sample membership.",
                "Generic templates are syntactic heuristics; unmatched messages remain included.",
                "No detection, model fitting, incident labels, or accuracy evaluation is performed.",
            ],
        })
        # Read back the feature stream before marking the run complete.
        totals = Counter()
        count = 0
        with _open_data(staging / "windows.jsonl") as handle:
            for line in handle:
                window = json.loads(line)
                bounds = split_ranges[window["split"]]
                if not (bounds["start"] <= window["window_start"] < window["window_end"] <= bounds["end"]):
                    raise ValueError("serialized window crosses split bounds")
                if window["record_count"] != window["business"]["record_count"] + window["health"]["record_count"]:
                    raise ValueError("serialized health/business accounting mismatch")
                if window["scope"] == "service":
                    totals[window["window_minutes"]] += window["record_count"]
                count += 1
        if count != len(features.windows) or any(totals[m] != len(result.records)
                                                for m in (5, 15)):
            raise ValueError("serialized window accounting mismatch")
        write_eda_charts(features.windows, staging / "charts", config.source_timezone)
        dependencies = {name: version(name) for name in (
            "matplotlib", "numpy", "contourpy", "cycler", "fonttools", "kiwisolver",
            "packaging", "pillow", "pyparsing", "python-dateutil", "six", "tzdata",
        )}
    artifacts = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.relative_to(staging).as_posix()] = _file_hash(path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "stage": stage,
        "status": "complete",
        "pipeline_complete": False,
        "run_id": run_id,
        "input_path": str(input_path),
        "input_sha256": result.input_sha256,
        "input_size_bytes": result.input_size_bytes,
        "source_columns": result.columns,
        "config_sha256": config_hash,
        "software_version": __version__,
        "python_version": platform.python_version(),
        "dependencies": dependencies,
        "timezone": config.source_timezone,
        "seed": None,
        "split_ranges": split_ranges,
        "runtime_seconds": round(time.perf_counter() - start, 6),
        "counts": {key: result.quality[key] for key in ("parsed_count", "accepted_count", "rejected_count")},
        "artifacts": artifacts,
        **({"content_artifacts": content_hashes} if compact else {}),
    }
    _write_json(staging / "manifest.json", manifest)
    # The deterministic staging name prevents concurrent publishers of this run.
    if output.exists():
        raise FileExistsError(f"output appeared during processing: {output}")
    staging.rename(output)
    print(json_text({"stage": stage, "run_id": run_id,
                     "counts": manifest["counts"], "elapsed_seconds": manifest["runtime_seconds"],
                     "parse_coverage": result.quality["parse_coverage"]}), file=sys.stderr)
    return manifest


def ingest(input_path: Path, config: IngestConfig, output: Path) -> dict:
    """Publish the existing Phase 1 ingestion/parsing contract."""
    return _publish(input_path, config, output)


def features(input_path: Path, config: FeatureConfig, output: Path, *, compact: bool = False) -> dict:
    """Run ingestion/parsing followed by Phase 2 features and EDA only."""
    return _publish(input_path, config, output, compact=compact)


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_delivery(staging: Path, config: RunConfig) -> None:
    """Check final exports against signals and source evidence without reloading all features."""
    from .incidents import incidents_csv, validate_incident_report, validate_signals

    report = json.loads((staging / "incidents.json").read_text(encoding="utf-8"))
    validate_incident_report(report, config.incidents)
    if (staging / "incidents.csv").read_bytes() != incidents_csv(report["incidents"]).encode("utf-8"):
        raise ValueError("incident JSON and CSV content disagree")
    signals = {}
    for stage in ("rules", "ml"):
        with (staging / stage / "signals.jsonl").open(encoding="utf-8") as handle:
            signals[stage] = [json.loads(line) for line in handle]
    source = json.loads((staging / "features/manifest.json").read_text(encoding="utf-8"))
    # Retain only fields needed to resolve evidence, not another copy of messages/features.
    references = {}
    with _open_data(staging / "features/normalized.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            references[row["record_id"]] = tuple(row[k] for k in ("timestamp", "service", "endpoint", "source_ip"))
    validate_signals(signals["rules"], signals["ml"], source["split_ranges"], set(references))
    all_signals = signals["rules"] + signals["ml"]
    if sorted(s["signal_id"] for s in all_signals) != sorted(
            sid for incident in report["incidents"] for sid in incident["signal_ids"]):
        raise ValueError("signals are missing or duplicated in final incidents")
    signal_map = {s["signal_id"]: s for s in all_signals}
    for incident in report["incidents"]:
        expected = {rid for sid in incident["signal_ids"] for rid in signal_map[sid]["evidence_record_ids"]}
        if set(incident["evidence"]["record_ids"]) != expected:
            raise ValueError("incident evidence differs from contributing signals")
    for signal in all_signals:
        for rid in signal["evidence_record_ids"]:
            timestamp, service, endpoint, ip = references[rid]
            entity = endpoint if signal["scope"] == "endpoint" else ip if signal["scope"] == "auth_ip" else service
            if (not signal["window_start"] <= timestamp < signal["window_end"]
                    or service != signal["service"] or entity != signal["entity_key"]):
                raise ValueError("signal evidence lies outside its window or entity")
    quality = json.loads((staging / "quality.json").read_text(encoding="utf-8"))
    if (not quality["accounting_complete"] or quality["accepted_count"] != len(references)
            or quality["parsed_count"] != quality["accepted_count"] + quality["rejected_count"]):
        raise ValueError("incomplete final source accounting")


def run(input_path: Path, config: RunConfig, output: Path, *, debug_artifacts=False) -> dict:
    """Train by default; retain the legacy evaluation workflow only on explicit request."""
    if not debug_artifacts:
        return train(input_path, config, output)
    import shutil

    start = time.perf_counter()
    staging = output.with_name(output.name + ".incomplete")
    if any(p.name.endswith(".incomplete") for p in (output, *output.parents)):
        raise InputError("output must not be inside an .incomplete directory")
    if output.exists() or staging.exists() or output.is_symlink() or staging.is_symlink():
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    if not input_path.is_file():
        raise InputError(f"input file does not exist: {input_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    state = {"schema_version": SCHEMA_VERSION, "phase": 7, "stage": "end_to_end",
             "status": "incomplete", "pipeline_complete": False, "active_stage": "features"}
    _write_json(staging / "manifest.json", state)
    _write_json(staging / "config.json", config.to_dict())
    stages = {}

    def execute(name, operation, *args, **kwargs):
        state["active_stage"] = name
        _write_json(staging / "manifest.json", state)
        stages[name] = operation(*args, **kwargs)

    try:
        execute("features", features, input_path, config.features, staging / "features", compact=True)
        execute("rules", detect, staging / "features", config.rules, staging / "rules")
        execute("ml", ml, staging / "features", config.ml, staging / "ml", staging / "rules")
        if not stages["ml"]["counts"]["available_models"]:
            state["status"] = "partial"
            raise ValueError("ML unavailable: no supported model; inspect ml/ml_summary.json in staging")
        execute("reports", report_incidents, staging / "features", config.incidents,
                staging / "reports", staging / "rules", staging / "ml")
        state["active_stage"] = "delivery_validation"
        _write_json(staging / "manifest.json", state)
        for name in ("incidents.json", "incidents.csv", "summary.md"):
            (staging / name).hardlink_to(staging / "reports" / name)
        for name in ("quality.json", "rejects.jsonl"):
            (staging / name).hardlink_to(staging / "features" / name)
        shutil.copytree(staging / "features/charts", staging / "charts", copy_function=os.link)
        (staging / "charts/incidents.png").hardlink_to(staging / "reports/charts/incidents.png")
        # No labels enter this command. Publish split/coverage and independent detector counts only.
        (staging / "evaluation").mkdir()
        ml_summary = json.loads((staging / "ml/ml_summary.json").read_text(encoding="utf-8"))
        _write_json(staging / "evaluation/comparison.json", {
            "schema_version": SCHEMA_VERSION, "split_ranges": stages["features"]["split_ranges"],
            "counts": stages["reports"]["counts"], "ml_available_models": ml_summary["available_models"],
            "ml_unavailable_models": ml_summary["unavailable_models"],
            "accuracy_metrics": None, "reason": "No authoritative incident labels supplied; counts are candidates.",
        })
        _validate_delivery(staging, config)
        if _file_hash(input_path) != stages["features"]["input_sha256"]:
            raise InputError("source file changed during processing; rerun from stable input")
        # Validate every stage's serialized artifacts before relocating its provenance links.
        for name, manifest in stages.items():
            for artifact, digest in manifest["artifacts"].items():
                if _file_hash(staging / name / artifact) != digest:
                    raise ValueError(f"{name} artifact hash mismatch: {artifact}")
            for key in ("source_run", "rules_run", "ml_run"):
                if manifest.get(key):
                    relative = Path(manifest[key]["path"]).relative_to(staging.resolve())
                    manifest[key]["path"] = str(output.resolve() / relative)
            _write_json(staging / name / "manifest.json", manifest)
        config_hash = hashlib.sha256(json_text(config.to_dict()).encode()).hexdigest()
        run_id = hashlib.sha256(json_text([stages["features"]["input_sha256"], config_hash, __version__]).encode()).hexdigest()
        counts = {**stages["features"]["counts"], **stages["reports"]["counts"],
                  "windows": stages["rules"]["counts"]["windows"],
                  "available_models": stages["ml"]["counts"]["available_models"]}
        paths = sorted(p for p in staging.rglob("*") if p.is_file() and p != staging / "manifest.json")
        manifest = {
            "schema_version": SCHEMA_VERSION, "phase": 7, "stage": "end_to_end",
            "status": "complete", "pipeline_complete": True, "run_id": run_id,
            "input_path": str(input_path.resolve()), "input_sha256": stages["features"]["input_sha256"],
            "input_size_bytes": stages["features"]["input_size_bytes"], "config_sha256": config_hash,
            "software_version": __version__, "python_version": platform.python_version(),
            "dependencies": {k: v for stage in stages.values() for k, v in stage["dependencies"].items()},
            "timezone": config.features.source_timezone, "seed": config.ml.random_state,
            "split_ranges": stages["features"]["split_ranges"], "counts": counts,
            "stage_run_ids": {k: v["run_id"] for k, v in stages.items()},
            "artifacts": {p.relative_to(staging).as_posix(): _file_hash(p) for p in paths},
            "artifact_sizes_bytes": {p.relative_to(staging).as_posix(): p.stat().st_size for p in paths},
            "runtime_seconds": round(time.perf_counter() - start, 6),
        }
        _write_json(staging / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(f"output appeared during processing: {output}")
        staging.rename(output)
    except Exception as exc:
        state["error"] = str(exc)
        try:
            _write_json(staging / "manifest.json", state)
        except OSError:
            pass  # Disk failure must not replace the original exception.
        if isinstance(exc, (InputError, OSError, ImportError)):
            raise
        raise ValueError(f"{state['active_stage']}: {exc}; diagnostics: {staging}") from exc
    print(json_text({"stage": "end_to_end", "run_id": run_id, "counts": counts,
                     "elapsed_seconds": manifest["runtime_seconds"],
                     "summary": str(output / "summary.md")}), file=sys.stderr)
    return manifest


def _read_feature_run(source: Path) -> tuple:
    """Read/hash the exact consumed bytes; refuse partial or incompatible stages."""
    try:
        if source.name.endswith(".incomplete"):
            raise InputError("cannot detect from an incomplete directory")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["schema_version"] != SCHEMA_VERSION or manifest["phase"] != 2
                or manifest["status"] != "complete" or manifest["stage"] != "eda_and_features"):
            raise InputError("detection requires a completed Phase 2 feature run")

        def rows(name):
            digest = hashlib.sha256()
            stored = _artifact_name(manifest, name)
            if stored.endswith(".gz") and _file_hash(source / stored) != manifest["artifacts"][stored]:
                raise InputError(f"feature artifact hash mismatch: {stored}")
            with _open_data(source / stored, "rb") as handle:
                for line in handle:
                    digest.update(line)
                    yield json.loads(line)
            if digest.hexdigest() != _content_hashes(manifest)[name]:
                raise InputError(f"feature artifact hash mismatch: {name}")

        configs = list(rows("config.json"))
        feature_config = FeatureConfig(**configs[0])
        if len(configs) != 1 or hashlib.sha256(json_text(feature_config.to_dict()).encode()).hexdigest() != manifest["config_sha256"]:
            raise InputError("feature configuration mismatch")
        ranges = manifest["split_ranges"]
        previous = None
        for split in ("train", "validation", "test"):
            bounds = ranges[split]
            start, end = (datetime.fromisoformat(bounds[k]) for k in ("start", "end"))
            if (start.tzinfo or end.tzinfo or start >= end or start.time().isoformat() != "00:00:00"
                    or end.time().isoformat() != "00:00:00" or (previous and previous != start)):
                raise InputError("invalid chronological split ranges")
            previous = end
        windows = list(rows("windows.jsonl"))
        seen = set()
        totals = Counter()
        for window in windows:
            bounds = ranges[window["split"]]
            start, end = (datetime.fromisoformat(window[k]) for k in ("window_start", "window_end"))
            minutes = window["window_minutes"]
            if (window["feature_schema_version"] != SCHEMA_VERSION or minutes not in (5, 15)
                    or window["scope"] not in {"service", "endpoint", "auth_ip"}
                    or not bounds["start"] <= window["window_start"] < window["window_end"] <= bounds["end"]
                    or start.tzinfo or end.tzinfo or start.minute % minutes or start.second or start.microsecond
                    or end - start != timedelta(minutes=minutes)):
                raise InputError("invalid feature window or split boundary")
            key = (minutes, window["window_start"], window["service"], window["scope"], window["entity_key"])
            if key in seen:
                raise InputError("duplicate feature window")
            seen.add(key)
            if window["record_count"] != window["business"]["record_count"] + window["health"]["record_count"]:
                raise InputError("feature health/business counts mismatch")
            if window["scope"] == "service":
                totals[minutes] += window["record_count"]
        records = []
        ids = set()
        for row in rows("normalized.jsonl"):
            if (row["schema_version"] != SCHEMA_VERSION or row["record_id"] in ids
                    or row["record_id"] != f'{manifest["input_sha256"]}:{row["source_record_number"]}'):
                raise InputError("invalid normalized source reference")
            ids.add(row["record_id"])
            row["timestamp"] = datetime.fromisoformat(row["timestamp"])
            if not ranges["train"]["start"] <= row["timestamp"].isoformat() < ranges["test"]["end"]:
                raise InputError("normalized event outside feature splits")
            row["enrichment"] = Enrichment(**row["enrichment"])
            # Raw messages remain in the source run; only the evidence resolver needs them here.
            row["original_fields"] = {}
            records.append(NormalizedLog(**row))
        if not records or len(records) != manifest["counts"]["accepted_count"] or any(
                totals[m] != len(records) for m in (5, 15)):
            raise InputError("feature/normalized record accounting mismatch")
        return manifest, feature_config, windows, records
    except InputError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise InputError(f"cannot read Phase 2 artifacts: {exc}") from exc


def detect(input_path: Path, config: DetectionConfig, output: Path) -> dict:
    """Publish only Phase 3 artifacts, retaining hashes/links to the feature run."""
    from .rules import detect_rules

    start = time.perf_counter()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    source, feature_config, windows, records = _read_feature_run(input_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _write_json(staging / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "phase": 3, "status": "incomplete", "pipeline_complete": False,
    })
    result = detect_rules(windows, records, source["split_ranges"], feature_config.health_endpoints, config)
    _write_json(staging / "config.json", config.to_dict())
    _write_json(staging / "baselines.json", result.baselines)
    _write_json(staging / "detection_summary.json", result.diagnostics)
    _write_lines(staging / "signals.jsonl", result.signals)
    # Read-back validation must precede publication, including the no-signal case.
    with (staging / "signals.jsonl").open(encoding="utf-8") as handle:
        serialized = [json.loads(line) for line in handle]
    if serialized != result.signals or len({s["signal_id"] for s in serialized}) != len(serialized):
        raise ValueError("serialized signals mismatch or duplicate identity")
    for name, expected in (("baselines.json", result.baselines),
                           ("detection_summary.json", result.diagnostics), ("config.json", config.to_dict())):
        if json.loads((staging / name).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"serialized detection artifact mismatch: {name}")
    config_hash = hashlib.sha256(json_text(config.to_dict()).encode()).hexdigest()
    source_hashes = {name: _content_hashes(source)[name] for name in ("config.json", "windows.jsonl", "normalized.jsonl")}
    run_id = hashlib.sha256(json_text([source_hashes, source["split_ranges"], config_hash, __version__]).encode()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION, "phase": 3, "stage": "rules_and_baselines",
        "status": "complete", "pipeline_complete": False, "run_id": run_id,
        "source_run": {"path": str(input_path.resolve()), "run_id": source["run_id"], "artifacts": source_hashes},
        "input_sha256": source["input_sha256"], "config_sha256": config_hash,
        "software_version": __version__, "python_version": platform.python_version(), "dependencies": {},
        "timezone": source["timezone"], "seed": None, "split_ranges": source["split_ranges"],
        "baseline_mode": "frozen_training", "runtime_seconds": round(time.perf_counter() - start, 6),
        "counts": {"windows": len(windows), "signals": len(result.signals), "accepted_count": len(records)},
        "artifacts": {path.name: _file_hash(path)
                      for path in sorted(staging.iterdir()) if path.name != "manifest.json"},
    }
    _write_json(staging / "manifest.json", manifest)
    if output.exists():
        raise FileExistsError(f"output appeared during processing: {output}")
    staging.rename(output)
    print(json_text({"stage": manifest["stage"], "run_id": run_id, "counts": manifest["counts"],
                     "elapsed_seconds": manifest["runtime_seconds"]}), file=sys.stderr)
    return manifest


def _read_signal_run(run: Path, source: dict, phase: int, stage: str, label: str) -> tuple[dict, list[dict]]:
    """Read a completed signal stage produced from the same feature stream, verifying its hash."""
    try:
        if run.name.endswith(".incomplete"):
            raise InputError(f"cannot read an incomplete {label} directory")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["schema_version"] != SCHEMA_VERSION or manifest["phase"] != phase
                or manifest["status"] != "complete" or manifest["stage"] != stage):
            raise InputError(f"{label} input requires a completed Phase {phase} run")
        if manifest["source_run"]["artifacts"]["windows.jsonl"] != _content_hashes(source)["windows.jsonl"]:
            raise InputError(f"{label} run was produced from a different feature stream")
        content = (run / "signals.jsonl").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["artifacts"]["signals.jsonl"]:
            raise InputError(f"{label} artifact hash mismatch: signals.jsonl")
        return manifest, [json.loads(line) for line in content.decode("utf-8").splitlines()]
    except InputError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"cannot read Phase {phase} artifacts: {exc}") from exc


def _read_rule_run(rules: Path, source: dict) -> list[dict]:
    """Read Phase 3 signals produced from the same feature stream, for comparison only."""
    return _read_signal_run(rules, source, 3, "rules_and_baselines", "rules")[1]


def ml(input_path: Path, config: MLConfig, output: Path, rules: Path | None = None) -> dict:
    """Publish only Phase 4 ML artifacts, retaining hashes/links to the feature and rule runs."""
    import pickle

    from .ml import fit_and_score

    start = time.perf_counter()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    source, feature_config, windows, records = _read_feature_run(input_path)
    rule_signals = _read_rule_run(rules, source) if rules is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _write_json(staging / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "phase": 4, "status": "incomplete", "pipeline_complete": False,
    })
    result = fit_and_score(windows, records, source["split_ranges"], feature_config.health_endpoints,
                           config, rule_signals)
    (staging / "model").mkdir()
    _write_json(staging / "config.json", config.to_dict())
    _write_json(staging / "ml_summary.json", result.summary)
    _write_json(staging / "model/metadata.json", result.metadata)
    (staging / "model/estimators.pkl").write_bytes(pickle.dumps(result.estimators, protocol=5))
    _write_lines(staging / "signals.jsonl", result.signals)
    with (staging / "signals.jsonl").open(encoding="utf-8") as handle:
        serialized = [json.loads(line) for line in handle]
    if serialized != result.signals or len({s["signal_id"] for s in serialized}) != len(serialized):
        raise ValueError("serialized signals mismatch or duplicate identity")
    for name, expected in (("ml_summary.json", result.summary), ("model/metadata.json", result.metadata),
                           ("config.json", config.to_dict())):
        if json.loads((staging / name).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"serialized ML artifact mismatch: {name}")
    if set(pickle.loads((staging / "model/estimators.pkl").read_bytes())) != set(result.metadata["models"]):
        raise ValueError("serialized estimators do not match model metadata")
    config_hash = hashlib.sha256(json_text(config.to_dict()).encode()).hexdigest()
    source_hashes = {name: _content_hashes(source)[name] for name in ("config.json", "windows.jsonl", "normalized.jsonl")}
    rules_run = None
    if rules is not None:
        rule_manifest = json.loads((rules / "manifest.json").read_text(encoding="utf-8"))
        rules_run = {"path": str(rules.resolve()), "run_id": rule_manifest["run_id"],
                     "artifacts": {"signals.jsonl": rule_manifest["artifacts"]["signals.jsonl"]}}
    run_id = hashlib.sha256(json_text([source_hashes, rules_run and rules_run["artifacts"],
                                       source["split_ranges"], config_hash, __version__]).encode()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION, "phase": 4, "stage": "ml_baseline",
        "status": "complete", "pipeline_complete": False, "run_id": run_id,
        "source_run": {"path": str(input_path.resolve()), "run_id": source["run_id"], "artifacts": source_hashes},
        "rules_run": rules_run, "input_sha256": source["input_sha256"], "config_sha256": config_hash,
        "software_version": __version__, "python_version": platform.python_version(),
        "dependencies": {name: version(name) for name in (
            "scikit-learn", "numpy", "scipy", "joblib", "cloudpickle", "threadpoolctl", "narwhals")},
        "timezone": source["timezone"], "seed": config.random_state, "split_ranges": source["split_ranges"],
        "runtime_seconds": round(time.perf_counter() - start, 6),
        "counts": {"windows": len(windows), "signals": len(result.signals), "accepted_count": len(records),
                   "available_models": result.summary["available_models"]},
        "artifacts": {path.relative_to(staging).as_posix(): _file_hash(path)
                      for path in sorted(staging.rglob("*")) if path.is_file() and path.name != "manifest.json"},
    }
    _write_json(staging / "manifest.json", manifest)
    if output.exists():
        raise FileExistsError(f"output appeared during processing: {output}")
    staging.rename(output)
    print(json_text({"stage": manifest["stage"], "run_id": run_id, "counts": manifest["counts"],
                     "elapsed_seconds": manifest["runtime_seconds"]}), file=sys.stderr)
    return manifest


def report_incidents(input_path: Path, config: IncidentConfig, output: Path, rules: Path, ml_run: Path) -> dict:
    """Publish Phase 5 grouped incidents from existing Phase 3 and Phase 4 signals of one feature run."""
    import csv

    from .incidents import (group_incidents, incident_counts, incidents_csv, report_limitations,
                            summary_markdown, validate_incident_report, validate_signals)
    from .report import write_incident_chart

    start = time.perf_counter()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    source, _, windows, records = _read_feature_run(input_path)
    rule_manifest, rule_signals = _read_signal_run(rules, source, 3, "rules_and_baselines", "rules")
    ml_manifest, ml_signals = _read_signal_run(ml_run, source, 4, "ml_baseline", "ML")
    source_hashes = {name: _content_hashes(source)[name] for name in ("config.json", "windows.jsonl", "normalized.jsonl")}
    for label, manifest in (("rules", rule_manifest), ("ML", ml_manifest)):
        if (manifest["source_run"]["artifacts"] != source_hashes or manifest["split_ranges"] != source["split_ranges"]
                or manifest["input_sha256"] != source["input_sha256"] or manifest["timezone"] != source["timezone"]):
            raise InputError(f"{label} run is incompatible with the feature run (artifacts, splits, input or timezone)")
    rule_hash = rule_manifest["artifacts"]["signals.jsonl"]
    if ml_manifest["rules_run"] is not None and ml_manifest["rules_run"]["artifacts"]["signals.jsonl"] != rule_hash:
        raise InputError("ML run was compared against a different rules run")
    validate_signals(rule_signals, ml_signals, source["split_ranges"], {r.record_id for r in records})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _write_json(staging / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "phase": 5, "status": "incomplete", "pipeline_complete": False,
    })
    config_hash = hashlib.sha256(json_text(config.to_dict()).encode()).hexdigest()
    ml_hash = ml_manifest["artifacts"]["signals.jsonl"]
    run_id = hashlib.sha256(json_text([source_hashes, rule_hash, ml_hash, source["split_ranges"],
                                       config_hash, __version__]).encode()).hexdigest()
    incidents = group_incidents(rule_signals, ml_signals, records, config)
    report = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "timezone": source["timezone"],
              "incidents": incidents, "limitations": report_limitations(source["timezone"])}
    validate_incident_report(report, config)
    counts = incident_counts(rule_signals, ml_signals, incidents)
    _write_json(staging / "config.json", config.to_dict())
    _write_json(staging / "incidents.json", report)
    (staging / "incidents.csv").write_text(incidents_csv(incidents), encoding="utf-8", newline="")
    (staging / "summary.md").write_text(summary_markdown(report, counts), encoding="utf-8")
    write_incident_chart(windows, incidents, staging / "charts/incidents.png", source["timezone"])
    # Read back JSON and CSV before publication, including the no-incident case.
    serialized = json.loads((staging / "incidents.json").read_text(encoding="utf-8"))
    validate_incident_report(serialized, config)
    with (staging / "incidents.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if serialized != report or [r["incident_id"] for r in rows] != [i["incident_id"] for i in incidents]:
        raise ValueError("serialized incident reports disagree")
    if json.loads((staging / "config.json").read_text(encoding="utf-8")) != config.to_dict():
        raise ValueError("serialized incident artifact mismatch: config.json")
    manifest = {
        "schema_version": SCHEMA_VERSION, "phase": 5, "stage": "incident_reports",
        "status": "complete", "pipeline_complete": False, "run_id": run_id,
        "source_run": {"path": str(input_path.resolve()), "run_id": source["run_id"], "artifacts": source_hashes},
        "rules_run": {"path": str(rules.resolve()), "run_id": rule_manifest["run_id"],
                      "artifacts": {"signals.jsonl": rule_hash}},
        "ml_run": {"path": str(ml_run.resolve()), "run_id": ml_manifest["run_id"],
                   "artifacts": {"signals.jsonl": ml_hash}},
        "input_sha256": source["input_sha256"], "config_sha256": config_hash,
        "software_version": __version__, "python_version": platform.python_version(),
        "dependencies": {"matplotlib": version("matplotlib")},
        "timezone": source["timezone"], "seed": None, "split_ranges": source["split_ranges"],
        "runtime_seconds": round(time.perf_counter() - start, 6), "counts": counts,
        "artifacts": {path.relative_to(staging).as_posix(): _file_hash(path)
                      for path in sorted(staging.rglob("*")) if path.is_file() and path.name != "manifest.json"},
    }
    _write_json(staging / "manifest.json", manifest)
    if output.exists():
        raise FileExistsError(f"output appeared during processing: {output}")
    staging.rename(output)
    print(json_text({"stage": manifest["stage"], "run_id": run_id, "counts": {
        k: counts[k] for k in ("rule_signals", "ml_signals", "incidents", "rule_only", "ml_only", "corroborated")},
        "elapsed_seconds": manifest["runtime_seconds"]}), file=sys.stderr)
    return manifest


def evaluate(input_path: Path, config: EvaluationConfig, output: Path, labels_path: Path) -> dict:
    """Publish Phase 6 rules/ML/combined metrics for a completed Phase 5 run against fixture labels."""
    from .evaluate import evaluate_incidents, summary_markdown, validate_labels
    from .incidents import group_incidents

    start = time.perf_counter()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    try:
        if input_path.name.endswith(".incomplete"):
            raise InputError("cannot evaluate an incomplete directory")
        report_manifest = json.loads((input_path / "manifest.json").read_text(encoding="utf-8"))
        if (report_manifest["schema_version"] != SCHEMA_VERSION or report_manifest["phase"] != 5
                or report_manifest["status"] != "complete" or report_manifest["stage"] != "incident_reports"):
            raise InputError("evaluation requires a completed Phase 5 incident run")
        content = {name: (input_path / name).read_bytes() for name in ("incidents.json", "config.json")}
        if any(hashlib.sha256(v).hexdigest() != report_manifest["artifacts"][k] for k, v in content.items()):
            raise InputError("incident artifact hash mismatch")
        published = json.loads(content["incidents.json"])
        incident_config = IncidentConfig(**json.loads(content["config.json"]))
        label_bytes = labels_path.read_bytes()
        labels = json.loads(label_bytes)
        events = validate_labels(labels)
        source_path = Path(report_manifest["source_run"]["path"])
        rules_path, ml_path = (Path(report_manifest[k]["path"]) for k in ("rules_run", "ml_run"))
    except InputError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"cannot read Phase 5 run or labels: {exc}") from exc
    source, _, windows, records = _read_feature_run(source_path)
    rule_manifest, rule_signals = _read_signal_run(rules_path, source, 3, "rules_and_baselines", "rules")
    ml_manifest, ml_signals = _read_signal_run(ml_path, source, 4, "ml_baseline", "ML")
    if (source["run_id"] != report_manifest["source_run"]["run_id"]
            or rule_manifest["artifacts"]["signals.jsonl"] != report_manifest["rules_run"]["artifacts"]["signals.jsonl"]
            or ml_manifest["artifacts"]["signals.jsonl"] != report_manifest["ml_run"]["artifacts"]["signals.jsonl"]):
        raise InputError("Phase 5 run no longer matches its source feature, rules, or ML runs")
    try:
        summary_bytes = (ml_path / "ml_summary.json").read_bytes()
        rule_config = DetectionConfig(**json.loads((rules_path / "config.json").read_text(encoding="utf-8")))
        ml_config = MLConfig(**json.loads((ml_path / "config.json").read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise InputError(f"cannot read rules/ML configuration: {exc}") from exc
    if hashlib.sha256(summary_bytes).hexdigest() != ml_manifest["artifacts"]["ml_summary.json"]:
        raise InputError("ML artifact hash mismatch: ml_summary.json")
    ml_models = {m["model_key"]: m["status"] if m["status"] == "available" else f"unavailable: {m['reason']}"
                 for m in json.loads(summary_bytes)["models"]}
    # Each mode regroups with the same Phase 5 code and configuration, so matching is identical.
    modes = {"rules": group_incidents(rule_signals, [], records, incident_config),
             "ml": group_incidents([], ml_signals, records, incident_config),
             "combined": group_incidents(rule_signals, ml_signals, records, incident_config)}
    if modes["combined"] != published["incidents"]:
        raise InputError("regrouped combined incidents differ from the published Phase 5 report")
    guards = {"minimum_records": rule_config.minimum_records, "minimum_statuses": rule_config.minimum_statuses,
              "minimum_latencies": rule_config.minimum_latencies,
              "minimum_auth_attempts": rule_config.minimum_auth_attempts,
              "ml_minimum_records": ml_config.minimum_records}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _write_json(staging / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "phase": 6, "status": "incomplete", "pipeline_complete": False,
    })
    label_hash = hashlib.sha256(label_bytes).hexdigest()
    result = evaluate_incidents(modes, events, source["split_ranges"], config, ml_signals, rule_signals,
                                windows, ml_models, guards, labels.get("normal_patterns", []))
    evaluation = {
        "schema_version": SCHEMA_VERSION, "config": config.to_dict(),
        "labels": {"path": str(labels_path), "sha256": label_hash, "seed": labels.get("seed"),
                   "generator": labels.get("generator"), "generator_version": labels.get("generator_version")},
        "split_ranges": source["split_ranges"],
        "incident_counts": {mode: dict(sorted(Counter(i["split"] for i in incidents).items()))
                            for mode, incidents in modes.items()},
        "ml_models": ml_models, **result,
        "limitations": [
            "Metrics describe synthetic labelled fixtures only, not the unlabelled supplied logs.",
            "Every unmatched prediction counts as a false positive, including extra scopes or categories of a "
            "labelled event (fp_kind duplicate_of_labelled_event).",
            "Validation-split ML alerts lie in the threshold-calibration day; the primary split is fully held out.",
            "recurring_normal_pattern uses fixture metadata to explain false positives; it never affects matching.",
        ],
    }
    _write_json(staging / "config.json", config.to_dict())
    _write_json(staging / "evaluation.json", evaluation)
    (staging / "summary.md").write_text(summary_markdown(evaluation), encoding="utf-8")
    if json.loads((staging / "evaluation.json").read_text(encoding="utf-8")) != evaluation:
        raise ValueError("serialized evaluation mismatch")
    config_hash = hashlib.sha256(json_text(config.to_dict()).encode()).hexdigest()
    run_id = hashlib.sha256(json_text([report_manifest["run_id"], label_hash, config_hash, __version__]).encode()).hexdigest()
    primary = result["results"][config.primary_split]
    manifest = {
        "schema_version": SCHEMA_VERSION, "phase": 6, "stage": "evaluation", "status": "complete",
        "pipeline_complete": False, "run_id": run_id,
        "incident_run": {"path": str(input_path.resolve()), "run_id": report_manifest["run_id"],
                         "artifacts": {"incidents.json": report_manifest["artifacts"]["incidents.json"]}},
        "labels": {"path": str(labels_path.resolve()), "sha256": label_hash},
        "input_sha256": source["input_sha256"], "config_sha256": config_hash,
        "software_version": __version__, "python_version": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy")},
        "timezone": source["timezone"], "seed": labels.get("seed"), "split_ranges": source["split_ranges"],
        "runtime_seconds": round(time.perf_counter() - start, 6),
        "counts": {mode: {k: primary[mode][k] for k in ("expected", "predicted", "tp", "fp", "fn")} for mode in modes},
        "artifacts": {path.name: _file_hash(path)
                      for path in sorted(staging.iterdir()) if path.name != "manifest.json"},
    }
    _write_json(staging / "manifest.json", manifest)
    if output.exists():
        raise FileExistsError(f"output appeared during processing: {output}")
    staging.rename(output)
    print(json_text({"stage": "evaluation", "run_id": run_id, "counts": manifest["counts"],
                     "elapsed_seconds": manifest["runtime_seconds"]}), file=sys.stderr)
    return manifest


@contextmanager
def _publication(output: Path):
    """Reserve a sibling directory; publish all files together or clean up on failure."""
    import shutil

    staging = output.with_name(output.name + ".incomplete")
    if any(p.name.endswith(".incomplete") for p in (output, *output.parents)):
        raise InputError("output must not be inside an .incomplete directory")
    if any(p.exists() or p.is_symlink() for p in (output, staging)):
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        yield staging
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during processing: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def train(input_path: Path, config: RunConfig, output: Path) -> dict:
    """Reuse a matching verified bundle, or fit with bounded raw-record storage."""
    import pickle
    import sqlite3
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    from . import PARSER_VERSION
    from .baselines import fit_baselines
    from .features import build_training_windows
    from .ml import fit_and_score

    started = time.perf_counter()
    staging = output.with_name(output.name + ".incomplete")
    if output.is_dir() and not output.is_symlink() and not (staging.exists() or staging.is_symlink()):
        metadata, estimators = load_model_bundle(output)
        if (metadata.get("reuse", {}).get("config") != config.to_dict()
                or metadata.get("input_sha256") != _file_hash(input_path)):
            raise FileExistsError(f"existing model bundle does not match input/config: {output}; "
                                  "choose a different --output directory to train a new bundle")
        return {"phase": 7, "status": "complete", "stage": "training", "reused": True,
                "run_id": metadata["run_id"], "counts": {"available_models": len(estimators)},
                "runtime_seconds": round(time.perf_counter() - started, 6)}
    with _publication(output) as staging:
        # SQLite supplies event-time order even for shuffled CSVs, using a bounded page cache.
        with TemporaryDirectory(prefix="log-anomaly-") as temporary:
            connection = sqlite3.connect(Path(temporary) / "records.sqlite")
            try:
                connection.execute("PRAGMA cache_size = -4096")
                connection.execute("PRAGMA temp_store = FILE")
                connection.execute("CREATE TABLE records (time TEXT, ordinal INTEGER, service TEXT, data BLOB)")
                parse_counts = Counter()

                def store(record):
                    parse_messages([record])
                    parse_counts[(record.service, record.parse_status)] += 1
                    record.original_fields = {}
                    connection.execute("INSERT INTO records VALUES (?, ?, ?, ?)", (
                        record.timestamp.isoformat(), record.source_record_number, record.service,
                        pickle.dumps(record, protocol=5)))

                source = read_logs(input_path, record_sink=store)
                if (not source.quality["accepted_count"] or
                        source.quality["rejected_fraction"] > config.features.max_rejected_fraction):
                    raise InputError("training requires usable records within the configured rejection limit")
                connection.execute("CREATE INDEX event_order ON records (time, ordinal)")
                connection.commit()
                services = [r[0] for r in connection.execute("SELECT DISTINCT service FROM records ORDER BY service")]
                source.quality["timezone"] = config.features.source_timezone
                source.quality["parse_coverage"] = {}
                for service in services:
                    counts = {status: parse_counts[(service, status)]
                              for status in ("matched", "partial", "unmatched", "empty")}
                    total = sum(counts.values())
                    source.quality["parse_coverage"][service] = {
                        "total": total, **counts, "matched_fraction": counts["matched"] / total}

                def records(end):
                    for (data,) in connection.execute(
                            "SELECT data FROM records WHERE time < ? ORDER BY time, ordinal", (end,)):
                        yield pickle.loads(data)

                bounds = [SimpleNamespace(timestamp=datetime.fromisoformat(t))
                          for t in source.quality["source_time_bounds"].values()]
                prepared = build_training_windows(records, bounds, services, config.features)
            finally:
                connection.close()
        result = fit_and_score(prepared.windows, [], prepared.split_ranges,
                               config.features.health_endpoints, config.ml, training_only=True)
        if not result.estimators:
            raise ValueError("ML unavailable: no supported model; supply sufficient training/validation windows")
        cutoff = prepared.split_ranges["train"]["end"]
        lower = (datetime.fromisoformat(cutoff) - timedelta(days=config.rules.traffic_recurrence_days)).isoformat()
        training = [w for w in prepared.windows if w["split"] == "train"]
        reuse = {
            "config": config.to_dict(), "template_reference": prepared.template_reference,
            "baselines": fit_baselines(training, prepared.split_ranges["train"], config.rules),
            "entities": sorted({(w["window_minutes"], w["scope"], w["service"], w["entity_key"])
                                for w in training}),
            "recurrence": [{k: w[k] for k in ("window_minutes", "service", "scope", "entity_key",
                                              "window_start", "window_end", "split")}
                           | {"business": {"record_count": w["business"]["record_count"]}}
                           for w in training if w["scope"] == "service" and w["window_start"] >= lower],
            "model_status": [{k: m[k] for k in ("model_key", "status", "reason") if k in m}
                             for m in result.summary["models"]],
        }
        with (staging / "estimators.pkl").open("wb") as handle:
            pickle.dump(result.estimators, handle, protocol=5)
        config_hash = hashlib.sha256(json_text(config.to_dict()).encode()).hexdigest()
        metadata = {**result.metadata, "reuse_version": "1.0", "reuse": reuse,
                    "software_version": __version__, "parser_version": PARSER_VERSION,
                    "python_version": platform.python_version(),
                    "dependencies": {name: version(name) for name in ("scikit-learn", "numpy", "scipy")},
                    "input_sha256": source.input_sha256, "input_size_bytes": source.input_size_bytes,
                    "source_columns": source.columns, "quality": source.quality,
                    "config_sha256": config_hash,
                    "run_id": hashlib.sha256(json_text([source.input_sha256, config_hash, __version__]).encode()).hexdigest(),
                    "estimators_sha256": _file_hash(staging / "estimators.pkl")}
        metadata["metadata_sha256"] = hashlib.sha256(json_text(metadata).encode()).hexdigest()
        _write_json(staging / "metadata.json", metadata)
        # Verify the exact serialized bundle before making the directory visible.
        _load_reusable_bundle(staging)
        if _file_hash(input_path) != source.input_sha256:
            raise InputError("source file changed during training")
    return {"phase": 7, "status": "complete", "stage": "training", "run_id": metadata["run_id"],
            "counts": {"available_models": len(result.estimators)},
            "runtime_seconds": round(time.perf_counter() - started, 6)}


def _load_reusable_bundle(path: Path) -> tuple[dict, dict]:
    """Check integrity and runtime compatibility before unpickling a trusted local bundle."""
    import pickle
    from . import PARSER_VERSION
    from .ml import validate_model_schema

    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        checksum = metadata.pop("metadata_sha256")
        if hashlib.sha256(json_text(metadata).encode()).hexdigest() != checksum:
            raise InputError("model metadata hash mismatch")
        metadata["metadata_sha256"] = checksum
        if (metadata["reuse_version"] != "1.0" or metadata["software_version"] != __version__
                or metadata["parser_version"] != PARSER_VERSION
                or metadata["python_version"].split(".")[:2] != platform.python_version().split(".")[:2]):
            raise InputError("incompatible model software/parser/Python version")
        for name in ("scikit-learn", "numpy", "scipy"):
            if metadata["dependencies"][name] != version(name):
                raise InputError(f"model was built with a different {name} version")
        reuse = metadata["reuse"]
        resolved = RunConfig(**reuse["config"])
        if (resolved.ml.to_dict() != metadata["config"]
                or hashlib.sha256(json_text(resolved.to_dict()).encode()).hexdigest() != metadata["config_sha256"]
                or reuse["baselines"]["train_range"] != metadata["split_ranges"]["train"]):
            raise InputError("model metadata disagrees with training configuration/ranges")
        with (path / "estimators.pkl").open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != metadata["estimators_sha256"]:
                raise InputError("model artifact hash mismatch: estimators.pkl")
            handle.seek(0)
            estimators = pickle.load(handle)
        if set(estimators) != set(metadata["models"]):
            raise InputError("model metadata and estimators disagree")
        validate_model_schema(metadata, estimators)
        return metadata, estimators
    except InputError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError, EOFError,
            pickle.UnpicklingError) as exc:
        raise InputError(f"cannot load reusable model bundle: {exc}") from exc


def load_model_bundle(run: Path) -> tuple[dict, dict]:
    """Load a trusted reusable or legacy Phase 4 bundle after integrity/version checks."""
    import pickle

    try:
        if any(p.name.endswith(".incomplete") for p in (run, *run.parents)):
            raise InputError("cannot load a model from an incomplete directory")
        if (run / "metadata.json").is_file():
            return _load_reusable_bundle(run)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["schema_version"] != SCHEMA_VERSION or manifest["phase"] != 4
                or manifest["status"] != "complete" or manifest["stage"] != "ml_baseline"):
            raise InputError("model loading requires a completed Phase 4 run")
        if manifest["dependencies"]["scikit-learn"] != version("scikit-learn"):
            raise InputError("model was built with a different scikit-learn version")
        content = {}
        for name in ("model/metadata.json", "model/estimators.pkl"):
            content[name] = (run / name).read_bytes()
            if hashlib.sha256(content[name]).hexdigest() != manifest["artifacts"][name]:
                raise InputError(f"model artifact hash mismatch: {name}")
        metadata = json.loads(content["model/metadata.json"])
        # Hash-verified bytes from this project's own publication only.
        estimators = pickle.loads(content["model/estimators.pkl"])
        if set(estimators) != set(metadata["models"]):
            raise InputError("model metadata and estimators disagree")
        from .ml import validate_model_schema
        validate_model_schema(metadata, estimators)
        return metadata, estimators
    except InputError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError, EOFError,
            pickle.UnpicklingError) as exc:
        raise InputError(f"cannot load Phase 4 model bundle: {exc}") from exc


def infer(input_path: Path, config: RunConfig, output: Path, model_bundle: Path) -> dict:
    """Score a new batch with frozen references; atomically publish three incident reports."""
    from .baselines import Recurrence
    from .features import _floor
    from .incidents import (group_incidents, incident_counts, incidents_csv, report_limitations,
                            summary_markdown, validate_incident_report, validate_signals)
    from .ml import Evidence, _signal, score_windows
    from .rules import detect_rules

    start = time.perf_counter()
    staging = output.with_name(output.name + ".incomplete")
    if any(p.name.endswith(".incomplete") for p in (output, *output.parents, model_bundle, *model_bundle.parents)):
        raise InputError("inference requires complete source and output directories")
    if output.exists() or staging.exists() or output.is_symlink() or staging.is_symlink():
        raise FileExistsError(f"output or staging path already exists: {output}, {staging}")
    if (model_bundle / "metadata.json").is_file():
        metadata, estimators = load_model_bundle(model_bundle)
        reuse = metadata["reuse"]
        for section in ("features", "rules", "ml"):
            if reuse["config"][section] != config.to_dict()[section]:
                raise InputError(f"incompatible inference configuration: {section}")
        baselines, templates = reuse["baselines"], reuse["template_reference"]
        entities = {tuple(e) for e in reuse["entities"]}
        recurrence = Recurrence(config.rules)
        for window in reuse["recurrence"]:
            recurrence.add(window)
        source = {"run_id": metadata["run_id"], "artifacts": {
            "metadata.json": _file_hash(model_bundle / "metadata.json"),
            "estimators.pkl": metadata["estimators_sha256"]}}
    else:
        try:
            source = json.loads((model_bundle / "manifest.json").read_text())
            if (source["stage"] != "end_to_end" or source["status"] != "complete"
                    or not source["pipeline_complete"] or source["software_version"] != __version__):
                raise InputError("inference requires a compatible completed training run directory")

            def verified(name):
                name = _artifact_name(source, name)
                path = model_bundle / name
                if _file_hash(path) != source["artifacts"][name]:
                    raise InputError(f"training artifact hash mismatch: {name}")
                return path

            saved_config = json.loads(verified("config.json").read_text())
            for section in ("features", "rules", "ml"):
                if saved_config[section] != config.to_dict()[section]:
                    raise InputError(f"incompatible inference configuration: {section}")
            verified("ml/manifest.json")
            metadata, estimators = load_model_bundle(model_bundle / "ml")
            if metadata["config"] != config.ml.to_dict() or metadata["split_ranges"] != source["split_ranges"]:
                raise InputError("model metadata disagrees with training run")
            baselines = json.loads(verified("rules/baselines.json").read_text())
            templates = json.loads(verified("features/template_reference.json").read_text())["services"]
            if baselines["train_range"] != source["split_ranges"]["train"]:
                raise InputError("rule baseline training range mismatch")
            # Stream saved windows: retain only compact training entity/recurrence context.
            recurrence = Recurrence(config.rules)
            entities = set()
            cutoff = source["split_ranges"]["train"]["end"]
            lower = (datetime.fromisoformat(cutoff) - timedelta(days=config.rules.traffic_recurrence_days)).isoformat()
            with _open_data(verified("features/windows.jsonl")) as handle:
                for line in handle:
                    window = json.loads(line)
                    if window["split"] == "train":
                        entities.add((window["window_minutes"], window["scope"], window["service"], window["entity_key"]))
                        if window["window_start"] >= lower:
                            recurrence.add(window)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise InputError(f"cannot use training bundle: {exc}") from exc

    result = read_logs(input_path)
    if not result.records or result.quality["rejected_fraction"] > config.features.max_rejected_fraction:
        raise InputError("inference requires usable records within the configured rejection limit")
    first, last = result.records[0].timestamp, result.records[-1].timestamp
    calibration_end = metadata["split_ranges"]["validation"]["end"]
    if first.isoformat() < calibration_end:
        raise InputError("inference records must be at or after the calibration period end")
    result.quality["timezone"] = config.features.source_timezone
    result.quality["parse_coverage"] = parse_messages(result.records)
    ranges = {**metadata["split_ranges"], "test": {
        "start": _floor(first, 15).isoformat(),
        "end": (_floor(last, 15) + timedelta(minutes=15)).isoformat()}}
    features = build_windows(result.records, config.features, inference_ranges=ranges,
                             template_reference=templates)
    for window in features.windows:
        window["incomplete_window"] = (window["window_start"] < first.isoformat()
                                        or window["window_end"] > last.isoformat())
    complete = [w for w in features.windows if not w["incomplete_window"]]
    rules = detect_rules(complete, result.records, ranges, config.features.health_endpoints,
                         config.rules, frozen_baselines=baselines, recurrence=recurrence)
    del complete, recurrence, baselines
    scores = score_windows(metadata, estimators, features.windows)
    evidence = Evidence(result.records, config.features.health_endpoints)
    ml_signals = []
    for window, score in zip(features.windows, scores):
        if score["alert"]:
            entry = metadata["models"][score["model_key"]]
            seen = (window["window_minutes"], window["scope"], window["service"], window["entity_key"]) in entities
            ml_signals.append(_signal(window, score["score"], entry["threshold"], entry["preprocessing"],
                                      score["model_key"], entry["training_window_count"], seen, evidence, config.ml))
    validate_signals(rules.signals, ml_signals, ranges, {r.record_id for r in result.records})
    incidents = group_incidents(rules.signals, ml_signals, result.records, config.incidents)
    run_id = hashlib.sha256(json_text([result.input_sha256, source["run_id"],
        source["artifacts"], config.to_dict(), "inference", __version__]).encode()).hexdigest()
    limitations = [
        "Inference uses frozen training references and validation thresholds; no fitting or recalibration occurs.",
        "The test split denotes new inference observations, not a labelled evaluation experiment.",
        "Boundary windows outside the first/last accepted event times are unscored by both detectors; retain overlapping context when batching.",
        "Observed event span does not prove collection completeness. No cross-batch state or incident merging is maintained.",
        "Historical baselines can become stale; periodically review the saved model against new operating conditions.",
    ]
    report = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "timezone": config.features.source_timezone,
              "incidents": incidents,
              "limitations": report_limitations(config.features.source_timezone, inference=True) + limitations}
    validate_incident_report(report, config.incidents)
    counts = incident_counts(rules.signals, ml_signals, incidents)
    # Keep source accounting and unsupported-model coverage in the final JSON, not sidecars.
    for incident in incidents:
        incident["evidence"].pop("chart", None)
    report.update(
        provenance={"input_path": str(input_path.resolve()), "input_sha256": result.input_sha256,
                    "training_run_id": source["run_id"], "model_artifacts": source["artifacts"],
                    "config": config.to_dict(), "split_ranges": ranges, "quality": result.quality},
        counts=counts,
        coverage={"window_status_counts": dict(sorted(Counter(s["status"] for s in scores).items())),
                  "unavailable_models": sorted({s["model_key"] for s in scores if s["status"] == "ml_unavailable"})})
    validate_incident_report(report, config.incidents)
    with _publication(output) as staging:
        _write_json(staging / "incidents.json", report)
        (staging / "incidents.csv").write_text(incidents_csv(incidents), encoding="utf-8", newline="")
        summary = summary_markdown(report, counts, inference=True)
        (staging / "summary.md").write_text(summary, encoding="utf-8")
        serialized = json.loads((staging / "incidents.json").read_text(encoding="utf-8"))
        validate_incident_report(serialized, config.incidents)
        if (serialized != report
                or (staging / "incidents.csv").read_bytes() != incidents_csv(incidents).encode()
                or (staging / "summary.md").read_text(encoding="utf-8") != summary):
            raise ValueError("serialized incident reports disagree")
        if _file_hash(input_path) != result.input_sha256:
            raise InputError("source file changed during inference")
    return {"schema_version": SCHEMA_VERSION, "phase": 7, "stage": "inference", "status": "complete",
            "pipeline_complete": True, "run_id": run_id, "input_sha256": result.input_sha256,
            "counts": counts, "runtime_seconds": round(time.perf_counter() - start, 6)}
