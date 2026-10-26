from dataclasses import replace

import pytest

from log_anomaly.config import FeatureConfig, InputError
from log_anomaly.features import build_windows
from log_anomaly.ingest import read_logs
from log_anomaly.parse import parse_messages

CONFIG = FeatureConfig(train_days=1, validation_days=1, minimum_records=4)


def parsed(csv_file, rows):
    records = read_logs(csv_file(rows)).records
    parse_messages(records)
    return records


def select(result, start, service="api-gateway", scope="service", minutes=5):
    return next(w for w in result.windows if w["window_start"] == start
                and w["service"] == service and w["scope"] == scope and w["window_minutes"] == minutes)


def test_hand_computed_statistics_health_empty_and_boundaries(csv_file, row):
    rows = [
        {**row, "timestamp": "2025-02-03 00:00:00", "response_time_ms": "10"},
        {**row, "timestamp": "2025-02-03 00:04:59", "http_status": "404", "response_time_ms": "20", "log_level": "WARN"},
        {**row, "timestamp": "2025-02-03 00:03:00", "http_status": "bad", "response_time_ms": "bad", "log_level": "other", "source_ip": "bad", "message": ""},
        {**row, "timestamp": "2025-02-03 00:04:00", "http_status": "500", "response_time_ms": "100", "log_level": "ERROR"},
        {**row, "timestamp": "2025-02-03 00:02:00", "endpoint": "/health", "http_status": "503", "response_time_ms": "1000", "message": "GET /health -> 503"},
        {**row, "timestamp": "2025-02-03 00:02:01", "endpoint": "/health", "http_status": "bad", "message": "HEAD /health -> 503"},
        {**row, "timestamp": "2025-02-03 00:02:02", "endpoint": "/health", "http_status": "bad", "message": "unknown health response"},
        {**row, "timestamp": "2025-02-03 00:05:00", "response_time_ms": "9"},
        {**row, "timestamp": "2025-02-03 00:15:00"},
        {**row, "timestamp": "2025-02-03 23:59:59"},
        {**row, "timestamp": "2025-02-04 00:00:00"},
        {**row, "timestamp": "2025-02-04 23:59:59"},
        {**row, "timestamp": "2025-02-05 00:00:00"},
        {**row, "timestamp": "2025-02-03 00:00:00", "service": "orders-service",
         "http_status": "404", "response_time_ms": "500"},
    ]
    result = build_windows(parsed(csv_file, rows), CONFIG)
    first = select(result, "2025-02-03T00:00:00")
    business, health = first["business"], first["health"]
    assert first["record_count"] == 7
    assert business["record_count"] == 4 and business["low_support"] is False
    assert business["valid_status_count"] == 3
    assert business["http_4xx_count"] == business["http_5xx_count"] == 1
    assert business["http_4xx_rate"] == business["http_5xx_rate"] == pytest.approx(1 / 3)
    assert business["level_rates"] == dict.fromkeys(("INFO", "WARN", "ERROR", "UNKNOWN"), 0.25)
    assert business["valid_latency_count"] == 3
    assert business["median_latency_ms"] == 20
    assert business["p95_latency_ms"] == pytest.approx(92)
    assert business["max_latency_ms"] == 100
    endpoints = {w["entity_key"]: w for w in result.windows
                 if w["window_start"] == "2025-02-03T00:00:00"
                 and w["service"] == "api-gateway" and w["scope"] == "endpoint"
                 and w["window_minutes"] == 5}
    assert endpoints[row["endpoint"]]["business"] == business
    assert endpoints[row["endpoint"]]["health"]["record_count"] == 0
    assert endpoints["/health"]["health"] == health
    assert endpoints["/health"]["business"]["record_count"] == 0
    other = select(result, "2025-02-03T00:00:00", "orders-service", "endpoint")
    assert other["entity_key"] == row["endpoint"]
    assert other["business"]["record_count"] == 1
    assert other["business"]["http_4xx_rate"] == 1
    assert other["business"]["p95_latency_ms"] == 500
    assert business["unique_ip_count"] == 1
    assert business["upstream_latency"] == {"valid_count": 3, "median_ms": 8, "p95_ms": 8, "max_ms": 8}
    assert health["record_count"] == 3
    assert health["failure_count"] == health["failure_valid_count"] == 2
    assert health["failure_rate"] == 1
    assert first["hour_sin"] == 0 and first["hour_cos"] == 1 and not first["is_weekend"]
    singleton = select(result, "2025-02-03T00:05:00")["business"]
    assert singleton["record_count"] == 1 and singleton["low_support"] is True
    assert singleton["median_latency_ms"] == singleton["p95_latency_ms"] == 9
    empty = select(result, "2025-02-03T00:10:00")
    assert empty["record_count"] == 0
    for field in ("http_5xx_rate", "median_latency_ms", "auth_failure_rate", "template_novelty_rate"):
        assert empty["business"][field] is None
    assert select(result, "2025-02-03T00:00:00", minutes=15)["record_count"] == 8
    assert select(result, "2025-02-03T00:15:00", minutes=15)["record_count"] == 1
    for minutes in (5, 15):
        assert sum(w["record_count"] for w in result.windows
                   if w["scope"] == "service" and w["window_minutes"] == minutes) == len(rows)
    assert all(w["record_count"] > 0 for w in result.windows if w["scope"] != "service")
    for window in result.windows:
        bounds = result.split_ranges[window["split"]]
        assert bounds["start"] <= window["window_start"] < window["window_end"] <= bounds["end"]
    assert select(result, "2025-02-04T00:00:00")["split"] == "validation"
    assert select(result, "2025-02-05T00:00:00")["split"] == "test"
    assert select(result, "2025-02-03T23:55:00")["split"] == "train"
    assert select(result, "2025-02-04T23:45:00", minutes=15)["split"] == "validation"


def test_auth_ip_scope_requires_context_and_valid_ip(csv_file, row):
    base = {**row, "service": "auth-service", "endpoint": "/login", "timestamp": "2025-02-03 12:00:00"}
    rows = [
        {**base, "http_status": "401", "message": "Login failed for user u1: invalid password"},
        {**base, "message": "Login succeeded for user u2"},
        {**base, "http_status": "401", "message": "Token rejected for user u3"},
        {**base, "http_status": "401", "message": "Login failed for user u4", "source_ip": "bad"},
        {**base, "http_status": "403", "message": "arbitrary forbidden"},
        {**row, "timestamp": "2025-02-04 00:00:00", "message": "Login failed for user u5"},
        {**row, "timestamp": "2025-02-05 00:00:00"},
    ]
    result = build_windows(parsed(csv_file, rows), CONFIG)
    auth = select(result, "2025-02-03T12:00:00", "auth-service")["business"]
    assert auth["auth_attempt_count"] == 3 and auth["auth_failure_count"] == 2
    assert auth["auth_failure_rate"] == pytest.approx(2 / 3)
    ip = select(result, "2025-02-03T12:00:00", "auth-service", "auth_ip")
    assert ip["entity_key"] == row["source_ip"]
    assert ip["business"]["auth_attempt_count"] == 2
    assert ip["business"]["auth_failure_rate"] == 0.5
    assert len([w for w in result.windows if w["scope"] == "auth_ip"]) == 2


def test_template_reference_and_training_features_do_not_learn_from_holdout(csv_file, row):
    rows = [row, {**row, "timestamp": "2025-02-04 00:00:00", "message": "new validation format"},
            {**row, "timestamp": "2025-02-05 00:00:00", "message": "new test format"}]
    original = build_windows(parsed(csv_file, rows), CONFIG)
    rows[-1].update(message="completely changed future", response_time_ms="99999")
    rows[1]["message"] = "different validation format"
    changed = build_windows(parsed(csv_file, rows), CONFIG)
    assert original.template_reference == changed.template_reference
    assert [w for w in original.windows if w["split"] == "train"] == [
        w for w in changed.windows if w["split"] == "train"]
    assert select(original, "2025-02-04T00:00:00")["business"]["template_novelty_rate"] == 1
    assert select(original, "2025-02-05T00:00:00")["business"]["template_novelty_rate"] == 1


def test_unknown_reference_search_and_missing_denominators(csv_file, row):
    rows = [row, {**row, "timestamp": "2025-02-04 00:00:00"},
            {**row, "service": "search-service", "timestamp": "2025-02-08 00:00:00",
             "http_status": "", "response_time_ms": "", "endpoint": "", "source_ip": "",
             "message": 'query="tea" hits=2 took=23ms'}]
    result = build_windows(parsed(csv_file, rows), CONFIG)
    window = select(result, "2025-02-08T00:00:00", "search-service")
    business = window["business"]
    assert window["is_weekend"] is True
    assert business["record_count"] == 1
    assert business["http_5xx_rate"] is business["p95_latency_ms"] is None
    assert business["template_novelty_rate"] is None
    assert business["template_reference_available"] is False
    assert business["search_latency"] == {"valid_count": 1, "median_ms": 23, "p95_ms": 23, "max_ms": 23}
    assert not any(w["scope"] == "endpoint" and w["service"] == "search-service" for w in result.windows)


def test_split_configuration_rejects_discarded_records_or_short_input(csv_file, row):
    records = parsed(csv_file, [row])
    with pytest.raises(InputError, match="test day"):
        build_windows(records, CONFIG)
    with pytest.raises(InputError, match="precede"):
        build_windows(records, replace(CONFIG, split_start_day="2025-02-04"))
