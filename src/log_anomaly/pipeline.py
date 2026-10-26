"""Publish stage artifacts with provenance and explicit completion state."""

import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .config import FeatureConfig, IngestConfig, InputError
from .features import build_windows
from .ingest import IngestionResult, read_logs
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
