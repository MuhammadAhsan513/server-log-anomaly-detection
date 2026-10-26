"""Event-time aggregates and training-only template references; no detection."""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import FeatureConfig, InputError
from .ingest import NormalizedLog

FEATURE_SCHEMA_VERSION = "1.0"
HEALTH_STATUS = re.compile(r"^(?:GET|HEAD) (\S+) -> ([1-5][0-9]{2})(?:\s|$)")


@dataclass
class FeatureResult:
    windows: list[dict]
    split_ranges: dict[str, dict[str, str]]
    template_reference: dict[str, list[str]]
    service_statistics: dict[str, dict]


def chronological_splits(records: list[NormalizedLog], config: FeatureConfig) -> dict:
    if not records:
        raise InputError("features require accepted records")
    first = min(r.timestamp for r in records)
    last = max(r.timestamp for r in records)
    start = (datetime.fromisoformat(config.split_start_day) if config.split_start_day
             else first.replace(hour=0, minute=0, second=0, microsecond=0))
    try:
        train_end = start + timedelta(days=config.train_days)
        validation_end = train_end + timedelta(days=config.validation_days)
        end = last.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    except OverflowError as exc:
        raise InputError("split dates exceed supported datetime range") from exc
    if first < start:
        raise InputError("accepted records precede split_start_day")
    if end <= validation_end:
        raise InputError("input must span training, validation, and at least one test day")
    return {
        "train": {"start": start.isoformat(), "end": train_end.isoformat()},
        "validation": {"start": train_end.isoformat(), "end": validation_end.isoformat()},
        "test": {"start": validation_end.isoformat(), "end": end.isoformat()},
    }


def _split(timestamp: datetime, ranges: dict) -> str:
    value = timestamp.isoformat()
    for name, bounds in ranges.items():
        if bounds["start"] <= value < bounds["end"]:
            return name
    raise ValueError("window outside chronological splits")


def _floor(timestamp: datetime, minutes: int) -> datetime:
    return timestamp.replace(minute=timestamp.minute // minutes * minutes, second=0, microsecond=0)


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _latencies(values: list[int]) -> dict:
    """Linear interpolation at (n - 1) * q, including singleton windows."""
    values = sorted(values)

    def quantile(q):
        if not values:
            return None
        index = (len(values) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return {"valid_count": len(values), "median_ms": quantile(0.5),
            "p95_ms": quantile(0.95), "max_ms": max(values) if values else None}


def _statistics(records: list[NormalizedLog], reference: set[str] | None,
                minimum_records: int) -> dict:
    statuses = [r.http_status for r in records if r.http_status is not None]
    levels = Counter(r.log_level for r in records)
    attempts = [r for r in records if r.service == "auth-service"
                and r.enrichment.authentication_outcome in {"success", "failure"}]
    failures = sum(r.enrichment.authentication_outcome == "failure" for r in attempts)
    templates = [r.enrichment.message_template for r in records
                 if r.enrichment.message_template is not None]
    novel = sum(t not in reference for t in templates) if reference else None
    latency = _latencies([r.response_time_ms for r in records if r.response_time_ms is not None])
    result = {
        "record_count": len(records), "low_support": len(records) < minimum_records,
        "valid_status_count": len(statuses),
        "level_counts": {level: levels[level] for level in ("INFO", "WARN", "ERROR", "UNKNOWN")},
        "level_rates": {level: _rate(levels[level], len(records))
                        for level in ("INFO", "WARN", "ERROR", "UNKNOWN")},
        "valid_latency_count": latency["valid_count"],
        "median_latency_ms": latency["median_ms"], "p95_latency_ms": latency["p95_ms"],
        "max_latency_ms": latency["max_ms"],
        "unique_ip_count": len({r.source_ip for r in records if r.source_ip is not None}),
        "auth_attempt_count": len(attempts), "auth_failure_count": failures,
        "auth_failure_rate": _rate(failures, len(attempts)),
        "valid_template_count": len(templates), "novel_template_count": novel,
        "template_novelty_rate": _rate(novel, len(templates)) if novel is not None else None,
        "template_reference_available": bool(reference),
        "upstream_latency": _latencies([r.enrichment.upstream_latency_ms for r in records
                                        if r.enrichment.upstream_latency_ms is not None]),
        "search_latency": _latencies([r.enrichment.took_ms for r in records
                                      if r.enrichment.took_ms is not None]),
    }
    for category in range(1, 6):
        count = sum(s // 100 == category for s in statuses)
        result[f"http_{category}xx_count"] = count
        result[f"http_{category}xx_rate"] = _rate(count, len(statuses))
    return result


def _health_status(record: NormalizedLog) -> int | None:
    match = HEALTH_STATUS.match(record.message)
    return int(match[2]) if match and match[1] == record.endpoint else None


def _populations(records: list[NormalizedLog], config: FeatureConfig,
                 reference: set[str] | None) -> dict:
    business = [r for r in records if r.endpoint not in config.health_endpoints]
    health = [r for r in records if r.endpoint in config.health_endpoints]
    health_stats = _statistics(health, reference, config.minimum_records)
    # Only explicit HTTP status evidence counts; absence is not a successful check.
    known = failures = 0
    for record in health:
        statuses = [s for s in (record.http_status, _health_status(record)) if s is not None]
        known += bool(statuses)
        failures += any(s >= 500 for s in statuses)
    health_stats.update(failure_count=failures, failure_valid_count=known,
                        failure_rate=_rate(failures, known))
    return {"record_count": len(records),
            "business": _statistics(business, reference, config.minimum_records),
            "health": health_stats}


def build_windows(records: list[NormalizedLog], config: FeatureConfig) -> FeatureResult:
    """Build both resolutions. Service bins are dense; endpoint/auth-IP bins sparse.

    The analysis extent comprises whole source-local days, not asserted collection
    coverage. Templates are fit once per service on training records only. Training
    novelty is in-sample membership, not a causal novelty detector.
    """
    ranges = chronological_splits(records, config)
    vocabulary: dict[str, set[str]] = defaultdict(set)
    by_service = defaultdict(list)
    for record in records:
        by_service[record.service].append(record)
        if _split(record.timestamp, ranges) == "train" and record.enrichment.message_template is not None:
            vocabulary[record.service].add(record.enrichment.message_template)
    windows = []
    for minutes in (config.window_minutes, config.alternate_window_minutes):
        groups = defaultdict(list)
        for record in records:
            start = _floor(record.timestamp, minutes)
            groups[(start, record.service, "service", record.service)].append(record)
            if record.endpoint is not None:
                groups[(start, record.service, "endpoint", record.endpoint)].append(record)
            if (record.service == "auth-service" and record.source_ip is not None
                    and record.enrichment.authentication_outcome in {"success", "failure"}
                    and record.endpoint not in config.health_endpoints):
                groups[(start, record.service, "auth_ip", record.source_ip)].append(record)
        start = datetime.fromisoformat(ranges["train"]["start"])
        end = datetime.fromisoformat(ranges["test"]["end"])
        while start < end:
            for service in by_service:
                groups.setdefault((start, service, "service", service), [])
            start += timedelta(minutes=minutes)
        for (start, service, scope, entity), members in sorted(groups.items()):
            end = start + timedelta(minutes=minutes)
            split = _split(start, ranges)
            if end.isoformat() > ranges[split]["end"]:
                raise ValueError("window straddles chronological split")
            angle = 2 * math.pi * (start.hour + start.minute / 60) / 24
            windows.append({
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "window_start": start.isoformat(), "window_end": end.isoformat(),
                "window_minutes": minutes, "service": service, "scope": scope,
                "entity_key": entity, "split": split,
                "hour_sin": math.sin(angle), "hour_cos": math.cos(angle),
                "is_weekend": start.weekday() >= 5,
                **_populations(members, config, vocabulary.get(service)),
            })
    return FeatureResult(
        windows, ranges, {service: sorted(values) for service, values in sorted(vocabulary.items())},
        {service: _populations(members, config, vocabulary.get(service))
         for service, members in sorted(by_service.items())},
    )
