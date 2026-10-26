from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from log_anomaly.baselines import History, fit_baselines, robust_score, series_key
from log_anomaly.config import DetectionConfig, FeatureConfig
from log_anomaly.features import build_windows
from log_anomaly.ingest import read_logs
from log_anomaly.parse import parse_messages
from log_anomaly.rules import detect_rules

FEATURES = FeatureConfig(train_days=1, validation_days=1)
CONFIG = DetectionConfig()


def batch(row, when, count=20, failures=0, latency=100, **extra):
    return [{**row, "timestamp": when, "response_time_ms": str(latency),
             "http_status": "500" if i < failures else "200",
             "log_level": "ERROR" if i < failures else "INFO", **extra} for i in range(count)]


def prepared(csv_file, row, rows, feature_config=FEATURES):
    # Establish all splits without adding support to a tested entity.
    markers = [{**row, "service": "marker-service", "timestamp": f"2025-02-0{day} 00:00:00"}
               for day in (3, 4, 5)]
    records = read_logs(csv_file(rows + markers)).records
    parse_messages(records)
    result = build_windows(records, feature_config)
    return result, records


def run(features, records, config=CONFIG, feature_config=FEATURES):
    return detect_rules(features.windows, records, features.split_ranges, feature_config.health_endpoints, config)


def history_rows(row, start="2025-02-03 20:00:00", failures=0):
    first = datetime.fromisoformat(start)
    return [record for i in range(36)
            for record in batch(row, (first + timedelta(minutes=5*i)).isoformat(sep=" "), failures=failures)]


def for_detector(result, detector, **filters):
    return [s for s in result.signals if s["detector_id"] == detector
            and all(s[key] == value for key, value in filters.items())]


def test_strong_patterns_at_both_resolutions_with_traceable_evidence(csv_file, row):
    rows = history_rows(row)
    rows += batch(row, "2025-02-04 20:00:00", count=40, failures=20, latency=1600)
    rows += batch(row, "2025-02-05 20:00:00", count=90, failures=45, latency=1600)
    for ip in ("192.0.2.10", "192.0.2.11"):
        rows += batch(row, "2025-02-05 20:00:00", count=10, service="auth-service", endpoint="/login",
                      source_ip=ip, message="Login failed for user u1: invalid password", http_status="401")
    rows += batch(row, "2025-02-05 20:00:00", count=2, endpoint="/health", http_status="",
                  message="GET /health -> 503", latency=999999)
    features, records = prepared(csv_file, row, rows)
    result = run(features, records)
    for detector in ("http_error_burst", "http_error_rate_rise", "latency_degradation",
                     "service_error_increase", "traffic_surge", "repeated_auth_failures"):
        assert {s["window_minutes"] for s in for_detector(result, detector, split="test")} == {5, 15}
    assert {s["entity_key"] for s in for_detector(result, "repeated_auth_failures")} == {"192.0.2.10", "192.0.2.11"}
    health = for_detector(result, "health_failure")
    assert len(health) == 1 and health[0]["window_minutes"] == 15
    assert health[0]["support_count"] == 2
    source = {r.record_id: r for r in records}
    for signal in result.signals:
        assert len(signal["evidence_record_ids"]) == signal["support_count"]
        assert signal["threshold"] and signal["baseline_metrics"] and signal["limitations"]
        assert signal["service"] in signal["explanation"]
        assert signal["window_start"] in signal["explanation"]
        assert str(signal["support_count"]) in signal["explanation"]
        for record_id in signal["evidence_record_ids"]:
            record = source[record_id]
            assert record.service == signal["service"]
            assert signal["window_start"] <= record.timestamp.isoformat() < signal["window_end"]
            assert (record.endpoint == "/health") == (signal["category"] == "health")
            if signal["scope"] == "endpoint":
                assert record.endpoint == signal["entity_key"]
            if signal["scope"] == "auth_ip":
                assert record.source_ip == signal["entity_key"]
    assert result.signals == run(features, records).signals
    assert len({s["signal_id"] for s in result.signals}) == len(result.signals)


def test_normal_background_and_low_volume_do_not_trigger(csv_file, row):
    rows = history_rows(row, failures=1)
    rows += batch(row, "2025-02-04 20:00:00", failures=2, latency=120)
    rows += batch(row, "2025-02-05 20:00:00", failures=1, latency=130)
    rows += batch(row, "2025-02-05 02:00:00", count=4, failures=4, latency=9000)
    rows += batch(row, "2025-02-05 02:20:00", count=25, http_status="", response_time_ms="")
    rows += batch(row, "2025-02-05 02:40:00", count=20, service="auth-service", endpoint="/tokens",
                  http_status="401", message="Token rejected for user u1")
    features, records = prepared(csv_file, row, rows)
    result = run(features, records)
    assert result.signals == []
    reasons = {r["reason"] for r in result.diagnostics["decision_counts"]}
    assert {"missing_metric", "low_support", "baseline_unavailable", "below_threshold"} <= reasons
    assert not for_detector(result, "health_failure")  # Empty health bins are never failures.


def test_fifteen_minute_support_absolute_rules_without_history_and_custom_health(csv_file, row):
    rows = [r for minute in (0, 5, 10) for r in batch(row, f"2025-02-05 01:{minute:02}:00", count=8, failures=2)]
    rows += batch(row, "2025-02-05 01:00:00", count=30, failures=30, latency=99999,
                  endpoint="/ready", message="GET /ready -> 503")
    config = replace(FEATURES, health_endpoints=("/ready",))
    features, records = prepared(csv_file, row, rows, config)
    result = run(features, records, feature_config=config)
    errors = for_detector(result, "http_error_burst")
    assert errors and {s["window_minutes"] for s in errors} == {15}
    assert all(s["support_count"] == 24 and s["observed_metrics"]["http_5xx_count"] == 6 for s in errors)
    assert all(s["baseline_metrics"]["http_5xx_rate"]["status"] == "baseline_unavailable" for s in errors)
    assert not for_detector(result, "latency_degradation")
    assert not for_detector(result, "service_error_increase")
    assert len(for_detector(result, "health_failure")) == 1


def test_threshold_edges_missing_denominators_and_auth_context(csv_file, row):
    rows = history_rows(row)
    rows += batch(row, "2025-02-05 20:00:00", count=25, failures=5, latency=1000)
    rows += batch(row, "2025-02-05 20:15:00", count=20, failures=4, latency=1001)
    rows += batch(row, "2025-02-05 20:30:00", count=19, failures=19, latency=9000)
    rows += batch(row, "2025-02-05 21:00:00", count=25, http_status="", response_time_ms="")
    rows += batch(row, "2025-02-05 20:00:00", count=10, service="auth-service", endpoint="/login",
                  message="Login failed for user u1", http_status="401")
    rows += batch(row, "2025-02-05 20:00:00", count=2, service="auth-service", endpoint="/login",
                  message="Login succeeded for user u2")
    rows += batch(row, "2025-02-05 20:15:00", count=12, service="auth-service", endpoint="/token",
                  message="Token rejected for user u3", http_status="401")
    rows += batch(row, "2025-02-05 20:30:00", count=12, service="auth-service", endpoint="/login",
                  message="Login failed for user u3", source_ip="invalid")
    features, records = prepared(csv_file, row, rows)
    result = run(features, records)
    assert {s["window_start"] for s in for_detector(result, "http_error_burst")} == {"2025-02-05T20:00:00"}
    assert {s["window_start"] for s in for_detector(result, "latency_degradation")} == {"2025-02-05T20:15:00"}
    auth = for_detector(result, "repeated_auth_failures")
    assert len(auth) == 2 and all(s["support_count"] == 12 for s in auth)
    assert all(s["observed_metrics"]["auth_failure_count"] == 10 for s in auth)
    tightened = run(features, records, replace(CONFIG, http_minimum_errors=6, auth_failure_rate=0.9))
    assert not for_detector(tightened, "http_error_burst")
    assert not for_detector(tightened, "repeated_auth_failures")


def test_latency_requires_both_strict_limits(csv_file, row):
    rows = history_rows(row)
    # Median window p95 = 500, so 1500 is exactly 3x and must not alert.
    rows = [{**r, "response_time_ms": "500"} for r in rows]
    rows += batch(row, "2025-02-05 20:00:00", latency=1500)
    rows += batch(row, "2025-02-05 20:15:00", latency=1501)
    features, records = prepared(csv_file, row, rows)
    result = run(features, records)
    assert {s["window_start"] for s in for_detector(result, "latency_degradation")} == {"2025-02-05T20:15:00"}


def test_robust_score_comparison_is_strict_and_needs_valid_support(csv_file, row):
    rows = history_rows(row)
    rows += batch(row, "2025-02-05 20:00:00", count=125, failures=5)  # score exactly 4
    rows += batch(row, "2025-02-05 20:15:00", count=100, failures=5)  # score 5
    rows += batch(row, "2025-02-05 20:30:00", count=19, failures=5, latency=2000)
    rows += batch(row, "2025-02-05 20:30:00", count=20, http_status="", response_time_ms="")
    features, records = prepared(csv_file, row, rows)
    result = run(features, records)
    assert {s["window_start"] for s in for_detector(result, "http_error_rate_rise")} == {"2025-02-05T20:15:00"}
    assert not for_detector(result, "http_error_burst")
    assert not for_detector(result, "latency_degradation")
    assert not for_detector(result, "service_error_increase", window_start="2025-02-05T20:00:00")
    assert all(s["score"] == pytest.approx(5) for s in for_detector(result, "http_error_rate_rise"))


def test_patterns_generalize_when_dates_and_identifiers_change(csv_file, row):
    rows = history_rows(row)
    rows += batch(row, "2025-02-05 20:00:00", count=40, failures=20, latency=2000)
    rows += batch(row, "2025-02-05 20:00:00", count=10, service="auth-service", endpoint="/login",
                  message="Login failed for user u1", http_status="401")
    features, records = prepared(csv_file, row, rows)
    original = run(features, records)
    shifted = deepcopy(records)
    for record in shifted:
        record.timestamp += timedelta(days=17)
        record.source_ip = "198.51.100.77"
        record.record_id = "shifted:" + str(record.source_record_number)
        if record.service == "api-gateway":
            record.service = "unfamiliar-service"
            record.endpoint = "/other-path"
    new_features = build_windows(shifted, FEATURES)
    other = run(new_features, shifted)
    assert sorted((s["detector_id"], s["window_minutes"], s["scope"], s["score"]) for s in original.signals) == sorted(
        (s["detector_id"], s["window_minutes"], s["scope"], s["score"]) for s in other.signals)


def test_baseline_mad_daypart_expiry_and_no_cross_entity_borrowing(csv_file, row):
    features, _ = prepared(csv_file, row, history_rows(row))
    template = next(w for w in features.windows if w["business"]["record_count"] == 20 and w["scope"] == "service")
    config = replace(CONFIG, minimum_history_windows=3)
    history = History(config)
    for offset, value in enumerate((100, 200, 300)):
        window = deepcopy(template)
        window["window_start"] = f"2025-02-03T20:{offset*5:02}:00"
        window["window_end"] = f"2025-02-03T20:{offset*5+5:02}:00"
        window["business"]["p95_latency_ms"] = value
        history.add(window)
    key = series_key(template)
    ref = history.reference(key, "p95_latency_ms", "2025-02-03T20:15:00", 3)
    assert ref["median"] == 200 and ref["mad"] == 100
    assert ref["scale"] == pytest.approx(148.26) and ref["daypart_matched"]
    assert robust_score(400, ref) == pytest.approx(200 / 148.26)
    broad = history.reference(key, "p95_latency_ms", "2025-02-04T02:00:00", 0)
    assert broad["median"] == 200 and not broad["daypart_matched"]
    for other in ((15, *key[1:]), (5, "new-service", *key[2:]), (5, key[1], "endpoint", "/new")):
        assert history.reference(other, "p95_latency_ms", "2025-02-04T02:00:00", 0)["status"] == "baseline_unavailable"
    expired = history.reference(key, "p95_latency_ms", "2025-02-04T20:05:00", 3)
    assert expired["history_window_count"] == 2 and expired["median"] is None


def test_zero_mad_floor_minimum_history_and_overlap_exclusion(csv_file, row):
    features, _ = prepared(csv_file, row, history_rows(row))
    windows = [w for w in features.windows if w["scope"] == "service" and w["window_minutes"] == 5
               and w["business"]["record_count"] == 20]
    history = History(CONFIG)
    for window in windows[:12]:
        history.add(window)
    key = series_key(windows[0])
    # Current/overlapping window is excluded even if supplied to history.
    ref = history.reference(key, "p95_latency_ms", windows[11]["window_start"], 3)
    assert ref["history_window_count"] == 11 and ref["status"] == "baseline_unavailable"
    for metric, epsilon in (("p95_latency_ms", 1), ("record_count", 1), ("http_5xx_rate", 0.01)):
        ref = history.reference(key, metric, windows[11]["window_end"], 3)
        assert ref["history_window_count"] == 12 and ref["mad"] == 0 and ref["scale"] == epsilon
        assert robust_score(ref["median"] + 4 * epsilon, ref) == pytest.approx(4)
    with pytest.raises(ValueError, match="training"):
        history.add({**windows[0], "split": "test"})


def test_future_training_and_heldout_values_cannot_change_earlier_references(csv_file, row):
    rows = history_rows(row)
    rows += batch(row, "2025-02-03 23:00:00", failures=10, latency=2000)
    rows += batch(row, "2025-02-04 20:00:00", failures=10, latency=2000)
    rows += batch(row, "2025-02-05 20:00:00", failures=10, latency=2000)
    features, records = prepared(csv_file, row, rows)
    original = run(features, records)
    changed = deepcopy(features)
    for window in changed.windows:
        if window["split"] != "train":
            window["business"]["p95_latency_ms"] = 999999 if window["business"]["valid_latency_count"] else None
    other = run(changed, records)
    assert original.baselines == other.baselines
    assert [s for s in original.signals if s["split"] == "train"] == [s for s in other.signals if s["split"] == "train"]
    # Validation mutation cannot alter any test result or fitted baseline.
    changed = deepcopy(features)
    for window in changed.windows:
        if window["split"] == "validation" and window["business"]["valid_latency_count"]:
            window["business"]["p95_latency_ms"] = 1
    other = run(changed, records)
    assert original.baselines == other.baselines
    assert [s for s in original.signals if s["split"] == "test"] == [s for s in other.signals if s["split"] == "test"]
    # A later training spike cannot change the earlier training spike's reference.
    later = deepcopy(next(w for w in features.windows if w["window_start"] == "2025-02-03T23:00:00"
                          and w["scope"] == "service" and w["window_minutes"] == 5))
    history = History(CONFIG)
    for window in sorted(features.windows, key=lambda w: w["window_start"]):
        if window["split"] == "train" and window["window_start"] < later["window_start"]:
            history.add(window)
    before = history.reference(series_key(later), "p95_latency_ms", later["window_start"], 3)
    history.add({**later, "window_start": "2025-02-03T23:05:00", "window_end": "2025-02-03T23:10:00"})
    assert before == history.reference(series_key(later), "p95_latency_ms", later["window_start"], 3)
    assert fit_baselines(features.windows, features.split_ranges["train"], CONFIG) == original.baselines


def test_recurring_daily_volume_is_suppressed_but_new_surges_are_not(csv_file, row):
    """Phase 6 regression: a nightly batch at the same clock time alerted every day."""
    features_config = FeatureConfig(train_days=3, validation_days=1)
    rows = []
    for day in range(5):
        first = datetime(2025, 2, 3 + day)
        rows += [r for i in range(72) for r in batch(row, (first + timedelta(minutes=5 * i)).isoformat(sep=" "))]
        rows += batch(row, (first + timedelta(hours=3)).isoformat(sep=" "), count=60, endpoint="/api/export")
    rows += batch(row, "2025-02-07 05:00:00", count=60, endpoint="/api/products")  # one-off, new time of day
    features, records = prepared(csv_file, row, rows, features_config)

    def surges(config):
        result = run(features, records, config, features_config)
        return result, {(s["window_minutes"], s["window_start"]) for s in for_detector(
            result, "traffic_surge", scope="service", service=row["service"]) if s["split"] != "train"}

    _, before = surges(replace(CONFIG, traffic_recurrence_days=0))
    result, after = surges(CONFIG)
    assert {(5, "2025-02-06T03:00:00"), (5, "2025-02-07T03:00:00"), (5, "2025-02-07T05:00:00")} <= before
    assert after == {(5, "2025-02-07T05:00:00"), (15, "2025-02-07T05:00:00")}
    signal = next(s for s in for_detector(result, "traffic_surge", window_minutes=5)
                  if s["window_start"] == "2025-02-07T05:00:00")
    context = signal["baseline_metrics"]["time_of_day_recurrence"]
    assert (context["status"], context["prior_days"], context["cutoff"]) == ("available", 3, "2025-02-06T00:00:00")
    assert context["same_time_max"] == 20 and "same-time-of-day maximum 20" in signal["explanation"]
    suppressed = [d for d in result.diagnostics["decision_counts"] if d["reason"] == "recurring_time_of_day"]
    assert {d["split"] for d in suppressed} == {"train", "validation", "test"}
    # Training days without enough prior days keep alerting, with the limitation stated.
    first_day = [s for s in for_detector(result, "traffic_surge") if s["window_start"] == "2025-02-03T03:00:00"]
    assert first_day and any("recurrence context unavailable" in t for t in first_day[0]["limitations"])
