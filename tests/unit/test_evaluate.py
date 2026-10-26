import pytest

from log_anomaly.config import EvaluationConfig, InputError
from log_anomaly.evaluate import clip_truths, iou, match, score_split, validate_labels

DAY = {"start": "2025-02-05T00:00:00", "end": "2025-02-06T00:00:00"}


def event(event_id="e1", category="http_errors", service="orders-service", start="10:00", end="10:30",
          source_ip=None, day="2025-02-05"):
    return {"event_id": event_id, "category": category, "service": service, "endpoint": None,
            "source_ip": source_ip, "start": f"{day}T{start}:00", "end": f"{day}T{end}:00"}


def incident(incident_id, start, end, category="http_errors", service="orders-service", scope="service",
             source_ip=None, signal_ids=("rule",), day="2025-02-05"):
    return {"incident_id": incident_id, "start_time": f"{day}T{start}:00", "end_time": f"{day}T{end}:00",
            "category": category, "services": [service], "split": "test", "severity": "warning",
            "entity": {"scope": scope, "endpoint": None, "source_ip": source_ip}, "signal_ids": list(signal_ids)}


def truths(*events):
    return clip_truths(validate_labels({"schema_version": "1.0", "events": list(events)}), DAY)


def scored(incidents, expected, ml=None, patterns=()):
    return score_split(incidents, expected, DAY, EvaluationConfig(), ml or {}, list(patterns))


def test_iou_threshold_boundary_is_inclusive_and_adjacent_intervals_do_not_overlap():
    expected = truths(event(end="10:30"))
    # [10:00, 10:09) against [10:00, 10:30) has IoU exactly 0.30; one minute less falls below.
    assert iou(*[(t["start_dt"], t["end_dt"]) for t in expected * 2]) == 1.0
    assert len(match([incident("a", "10:00", "10:09")], expected, 0.3, {})) == 1
    assert match([incident("a", "10:00", "10:08")], expected, 0.3, {}) == []
    assert match([incident("a", "10:30", "11:00")], expected, 0.1, {}) == []


def test_one_to_one_matching_counts_duplicates_as_unmatched_and_maximizes_total_iou():
    expected = truths(event("e1", start="10:00", end="10:30"), event("e2", start="10:20", end="11:00"))
    predictions = [incident("wide", "10:00", "11:00"), incident("early", "10:00", "10:30"),
                   incident("dup", "10:05", "10:30", scope="endpoint")]
    result = scored(predictions, expected)
    assert {(m["incident_id"], m["event_id"]) for m in result["matches"]} == {("early", "e1"), ("wide", "e2")}
    assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 0)
    assert result["unmatched_predictions"][0]["fp_kind"] == "duplicate_of_labelled_event"
    # Only one prediction may claim a truth, however many fragments overlap it.
    single = scored(predictions, truths(event("e1")))
    assert (single["tp"], single["fp"], single["recall"]["value"]) == (1, 2, 1.0)


def test_category_service_ip_and_ml_feature_compatibility():
    expected = truths(event("auth", "auth_failures", "auth-service", source_ip="192.0.2.9"),
                      event("lat", "latency", "search-service", start="12:00", end="12:30"))
    predictions = [
        incident("other-ip", "10:00", "10:30", "auth_failures", "auth-service", "auth_ip", "192.0.2.8"),
        incident("wrong-service", "12:00", "12:30", "latency", "orders-service"),
        incident("http", "12:00", "12:30", "http_errors", "search-service"),
        incident("ml", "12:00", "12:30", "multivariate", "search-service", signal_ids=("s1",)),
    ]
    ml = {"s1": {"feature_evidence": [{"feature": "business.p95_latency_ms", "robust_deviation": 5.0}]}}
    result = scored(predictions, expected, ml)
    assert [(m["incident_id"], m["event_id"]) for m in result["matches"]] == [("ml", "lat")]
    # Without a latency feature increase the ML candidate cannot claim the latency event.
    ml["s1"]["feature_evidence"][0]["robust_deviation"] = -5.0
    assert scored(predictions, expected, ml)["tp"] == 0


def test_metrics_null_denominators_and_recurring_false_positive_classification():
    result = scored([incident("a", "10:00", "10:30"), incident("b", "03:30", "03:35", "traffic")],
                    truths(event()), patterns=[{"service": "orders-service", "daily_clock_time": "3:30:00"}])
    assert (result["precision"]["value"], result["recall"]["value"]) == (0.5, 1.0)
    assert result["f1"]["value"] == pytest.approx(2 / 3)
    assert result["fp_kinds"] == {"recurring_normal_pattern": 1} and result["unmatched_per_day"] == 1.0
    assert result["matches"][0]["start_error_minutes"] == 0
    empty = scored([], [])
    assert empty["precision"] == {"value": None, "reason": "no predicted incidents in this split"}
    assert empty["recall"]["value"] is None and empty["f1"]["value"] is None
    missed = scored([], truths(event()))
    assert (missed["precision"]["value"], missed["recall"]["value"], missed["fn"]) == (None, 0.0, 1)


def test_labels_are_clipped_to_the_split_and_invalid_labels_are_rejected():
    crossing = event(start="23:50", end="23:59", day="2025-02-04") | {"end": "2025-02-05T00:20:00"}
    [clipped] = truths(crossing, event("outside", day="2025-02-07"))
    assert (clipped["start_dt"].isoformat(), clipped["clipped"]) == ("2025-02-05T00:00:00", True)
    for bad in ({**event(), "category": "unknown"}, {**event(), "end": event()["start"]},
                {k: v for k, v in event().items() if k != "source_ip"}):
        with pytest.raises(InputError):
            validate_labels({"schema_version": "1.0", "events": [bad]})
    with pytest.raises(InputError):
        validate_labels({"schema_version": "1.0", "events": [event(), event()]})
