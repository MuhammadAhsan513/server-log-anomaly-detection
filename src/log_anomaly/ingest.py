"""CSV ingestion with explicit accounting and stable source references."""

import csv
import hashlib
import io
import ipaddress
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import PARSER_VERSION, SCHEMA_VERSION
from .config import InputError

REQUIRED_COLUMNS = (
    "timestamp", "service", "log_level", "http_status", "response_time_ms",
    "endpoint", "source_ip", "message",
)
KNOWN_SERVICES = frozenset((
    "api-gateway", "auth-service", "orders-service", "payment-service", "search-service",
))
TIMESTAMP_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")
INTEGER_PATTERN = re.compile(r"[0-9]+")


@dataclass
class Enrichment:
    method: str | None = None
    upstream_service: str | None = None
    upstream_latency_ms: int | None = None
    request_id: str | None = None
    message_path: str | None = None
    message_query: str | None = None
    query_parameters: dict[str, list[str]] | None = None
    took_ms: int | None = None
    query: str | None = None
    query_metadata: dict | None = None
    user_id: str | None = None
    authentication_outcome: str | None = None
    message_template: str | None = None


@dataclass
class NormalizedLog:
    record_id: str
    source_record_number: int
    timestamp: datetime
    service: str
    log_level: str
    http_status: int | None
    response_time_ms: int | None
    endpoint: str | None
    source_ip: str | None
    message: str
    original_fields: dict[str, str]
    quality_flags: list[str]
    schema_version: str = SCHEMA_VERSION
    parser_version: str = PARSER_VERSION
    parse_status: str = "unmatched"
    enrichment: Enrichment = field(default_factory=Enrichment)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class RejectedLog:
    record_id: str
    source_record_number: int
    reasons: list[str]
    original_fields: list[str]
    schema_version: str = SCHEMA_VERSION


@dataclass
class IngestionResult:
    input_sha256: str
    input_size_bytes: int
    columns: list[str]
    records: list[NormalizedLog]
    rejects: list[RejectedLog]
    quality: dict


def _integer(value: str, minimum: int, maximum: int | None = None) -> int | None:
    value = value.strip()
    if not INTEGER_PATTERN.fullmatch(value):
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    return number if number >= minimum and (maximum is None or number <= maximum) else None


def _normalize(raw: dict[str, str], ordinal: int, digest: str) -> tuple[NormalizedLog | None, list[str]]:
    reasons = []
    timestamp = None
    try:
        if not TIMESTAMP_PATTERN.fullmatch(raw["timestamp"]):
            raise ValueError
        timestamp = datetime.strptime(raw["timestamp"], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        reasons.append("invalid_timestamp")
    service = raw["service"].strip()
    if not service:
        reasons.append("missing_service")
    if reasons:
        return None, reasons
    flags = []
    if service not in KNOWN_SERVICES:
        flags.append("unknown_service")
    level = raw["log_level"].strip().upper()
    if level not in {"INFO", "WARN", "ERROR"}:
        level = "UNKNOWN"
        flags.append("unknown_log_level")
    status = _integer(raw["http_status"], 100, 599)
    if status is None:
        flags.append("invalid_http_status")
    latency = _integer(raw["response_time_ms"], 0)
    if latency is None:
        flags.append("invalid_response_time_ms")
    endpoint = raw["endpoint"] if raw["endpoint"].strip() else None
    if endpoint is None:
        flags.append("missing_endpoint")
    elif not endpoint.startswith("/"):
        flags.append("invalid_endpoint")
    try:
        ip = str(ipaddress.ip_address(raw["source_ip"].strip()))
    except ValueError:
        ip = None
        flags.append("invalid_source_ip")
    if not raw["message"].strip():
        flags.append("empty_message")
    assert timestamp is not None
    return NormalizedLog(
        f"{digest}:{ordinal}", ordinal, timestamp, service, level, status, latency,
        endpoint, ip, raw["message"], raw, flags,
    ), []


def read_logs(path: Path) -> IngestionResult:
    """Read one immutable byte snapshot; retain duplicates and quarantine row errors.

    Accept a single UTF-8 BOM at file start. Fatal framing/encoding errors never
    return a result that could be mistaken for complete record accounting.
    """
    try:
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise InputError(f"cannot read UTF-8 input {path}: {exc}") from exc
    digest = hashlib.sha256(content).hexdigest()
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    records: list[NormalizedLog] = []
    rejects: list[RejectedLog] = []
    seen = set()
    duplicate_candidates = 0
    parsed = 0
    try:
        columns = next(reader, None)
        if not columns:
            raise InputError("empty input: required CSV header is absent")
        if len(set(columns)) != len(columns) or any(not c for c in columns):
            raise InputError("CSV header has duplicate or empty column names")
        missing = set(REQUIRED_COLUMNS) - set(columns)
        if missing:
            raise InputError(f"missing required columns: {sorted(missing)}")
        for ordinal, values in enumerate(reader, start=1):
            parsed += 1
            if len(values) != len(columns):
                rejects.append(RejectedLog(f"{digest}:{ordinal}", ordinal,
                                           ["column_count"], values))
                continue
            raw = dict(zip(columns, values))
            record, reasons = _normalize(raw, ordinal, digest)
            if record is None:
                rejects.append(RejectedLog(f"{digest}:{ordinal}", ordinal, reasons, values))
                continue
            key = tuple(values)
            if key in seen:
                duplicate_candidates += 1
            seen.add(key)
            records.append(record)
    except csv.Error as exc:
        raise InputError(f"CSV framing error near physical line {reader.line_num}, "
                         f"data record {parsed + 1}; accounting incomplete: {exc}") from exc
    original_order = [r.source_record_number for r in records]
    records.sort(key=lambda r: (r.timestamp, r.source_record_number))
    quality = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": digest,
        "parsed_count": parsed,
        "accepted_count": len(records),
        "rejected_count": len(rejects),
        "rejected_fraction": len(rejects) / parsed if parsed else 0.0,
        "accounting_complete": True,
        "rejection_reasons": dict(sorted(Counter(reason for r in rejects for reason in r.reasons).items())),
        "field_quality_counts": dict(sorted(Counter(flag for r in records for flag in r.quality_flags).items())),
        "duplicate_candidates": duplicate_candidates,
        "ordering_changes": sum(a != b.source_record_number for a, b in zip(original_order, records)),
        "source_time_bounds": {
            "first": records[0].timestamp.isoformat() if records else None,
            "last": records[-1].timestamp.isoformat() if records else None,
        },
        "parse_coverage": {},
    }
    return IngestionResult(digest, len(content), columns, records, rejects, quality)
