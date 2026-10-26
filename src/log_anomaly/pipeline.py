"""Publish stage artifacts with provenance and explicit completion state."""

import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .config import DetectionConfig, FeatureConfig, IngestConfig, InputError
from .features import build_windows
from .ingest import Enrichment, IngestionResult, NormalizedLog, read_logs
from .parse import parse_messages


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json_text(value) + "\n", encoding="utf-8")


def _write_lines(path: Path, values) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json_text(value) + "\n")


def _validate_artifacts(staging: Path, result: IngestionResult) -> None:
    """Validate serialized counts, ordering, references, and disjoint row coverage."""
    seen = set()
    previous = None
    for name, count in (("normalized.jsonl", len(result.records)), ("rejects.jsonl", len(result.rejects))):
        observed = 0
        with (staging / name).open(encoding="utf-8") as handle:
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


def _publish(input_path: Path, config: IngestConfig, output: Path) -> dict:
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
    _write_lines(staging / "normalized.jsonl", (r.to_dict() for r in result.records))
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
        _write_lines(staging / "windows.jsonl", features.windows)
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
        with (staging / "windows.jsonl").open(encoding="utf-8") as handle:
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
            artifacts[path.relative_to(staging).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
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


def features(input_path: Path, config: FeatureConfig, output: Path) -> dict:
    """Run ingestion/parsing followed by Phase 2 features and EDA only."""
    return _publish(input_path, config, output)


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
            with (source / name).open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    yield json.loads(line)
            if digest.hexdigest() != manifest["artifacts"][name]:
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
    source_hashes = {name: source["artifacts"][name] for name in ("config.json", "windows.jsonl", "normalized.jsonl")}
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
        "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(staging.iterdir()) if path.name != "manifest.json"},
    }
    _write_json(staging / "manifest.json", manifest)
    if output.exists():
        raise FileExistsError(f"output appeared during processing: {output}")
    staging.rename(output)
    print(json_text({"stage": manifest["stage"], "run_id": run_id, "counts": manifest["counts"],
                     "elapsed_seconds": manifest["runtime_seconds"]}), file=sys.stderr)
    return manifest
