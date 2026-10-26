import copy
import csv
import io
import random
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from log_anomaly.config import IncidentConfig, InputError
from log_anomaly.incidents import (CSV_FIELDS, group_incidents, incident_counts, incidents_csv,
                                   report_limitations, summary_markdown, validate_incident_report,
                                   validate_signals)
from log_anomaly.ingest import Enrichment, NormalizedLog

SPLITS = {"train": {"start": "2025-02-03T00:00:00", "end": "2025-02-04T00:00:00"},
          "validation": {"start": "2025-02-04T00:00:00", "end": "2025-02-05T00:00:00"},
          "test": {"start": "2025-02-05T00:00:00", "end": "2025-02-06T00:00:00"}}
CONFIG = IncidentConfig()


def record(n, when, service="orders-service", status=200, upstream=None, message="ok", **fields):
    return NormalizedLog(
        record_id=f"h:{n}", source_record_number=n, timestamp=datetime.fromisoformat(f"2025-02-05T{when}"),
        service=service, log_level="ERROR" if status >= 500 else "INFO", http_status=status,
        response_time_ms=fields.get("latency", 100), endpoint=fields.get("endpoint", "/api/orders"),
        source_ip=fields.get("ip", "192.0.2.1"), message=message, original_fields={},
        quality_flags=fields.get("flags", []), enrichment=Enrichment(upstream_service=upstream))


def signal(sid, start, ids, minutes=5, detector="http_error_burst", category="http_errors", scope="service",
           service="orders-service", entity=None, split="test", observed=None, features=None, score=1.0):
    begin = datetime.fromisoformat(f"{SPLITS[split]['start'][:10]}T{start}")
    result = {
        "signal_id": sid, "detector_id": detector, "detector_version": "1.0", "category": category,
        "scope": scope, "entity_key": entity or service, "service": service,
        "window_start": begin.isoformat(), "window_end": (begin + timedelta(minutes=minutes)).isoformat(),
        "window_minutes": minutes, "split": split, "score": score, "threshold": {"count_gte": 5},
        "observed_metrics": observed or {"http_5xx_count": 6, "http_5xx_rate": 0.3, "valid_status_count": 20},
        "baseline_metrics": {"http_5xx_rate": {"median": 0.01, "status": "available"}},
        "support_count": len(ids), "evidence_record_ids": [f"h:{i}" for i in ids],
        "limitations": [f"{detector} caveat"], "explanation": f"{detector} saw rate 30% vs limit 20% ({sid}).",
    }
    if detector == "isolation_forest":
        result["feature_evidence"] = features or []
        result["limitations"] = ["ML-only candidate: review severity and low confidence unless corroborated."]
    return result


def ml(sid, start, ids, feature="business.http_5xx_rate", deviation=5.0, **kwargs):
    features = [{"feature": feature, "observed": 0.4, "training_median": 0.0, "robust_deviation": deviation}]
    return signal(sid, start, ids, detector="isolation_forest", category="multivariate",
                  features=features, score=0.7, **kwargs)


def records(n=12):
    return [record(i, f"12:{5 * (i - 1):02d}:30", status=500 if i % 2 else 200) for i in range(1, n + 1)]


def by_category(incidents):
    return {(i["category"], i["entity"]["scope"], i["services"][0]): i for i in incidents}


def test_burst_merges_resolutions_and_deduplicates_evidence():
    rows = records(3) + [record(4, "12:12:00", status=500)]
    signals = [signal("a", "12:00:00", [1]), signal("b", "12:05:00", [2]), signal("c", "12:10:00", [3, 4]),
               signal("d", "12:00:00", [1, 2, 3, 4], minutes=15, detector="http_error_rate_rise")]
    [incident] = group_incidents(signals, [], rows, CONFIG)
    assert (incident["start_time"], incident["end_time"]) == ("2025-02-05T12:00:00", "2025-02-05T12:15:00")
    assert incident["evidence"]["record_ids"] == ["h:1", "h:2", "h:3", "h:4"]
    assert incident["signal_ids"] == ["a", "b", "c", "d"]
    assert incident["detector_ids"] == ["http_error_burst", "http_error_rate_rise"]
    assert (incident["first_evidence_time"], incident["last_evidence_time"]) == \
        ("2025-02-05T12:00:30", "2025-02-05T12:12:00")
    assert incident["evidence"]["gap_minutes"] == 0
    assert [s["http_status"] for s in incident["evidence"]["samples"]] == [500, 500, 500, 200]
    assert any("each source record is counted once" in text for text in incident["limitations"])


def test_gap_scope_entity_and_split_boundaries():
    rows = records(12) + [record(20, "12:00:10", service="auth-service", ip="192.0.2.8"),
                          record(21, "12:00:20", service="auth-service", ip="192.0.2.9")]
    signals = [
        signal("gap-1", "12:00:00", [1]), signal("gap-2", "12:10:00", [3]),  # one five-minute gap merges
        signal("far", "12:30:00", [7]),  # ten-minute gap stays separate
        signal("endpoint", "12:00:00", [1], scope="endpoint", entity="/api/orders"),
        signal("ip-1", "12:00:00", [20], detector="repeated_auth_failures", category="auth_failures",
               scope="auth_ip", service="auth-service", entity="192.0.2.8"),
        signal("ip-2", "12:00:00", [21], detector="repeated_auth_failures", category="auth_failures",
               scope="auth_ip", service="auth-service", entity="192.0.2.9"),
    ]
    incidents = group_incidents(signals, [], rows, CONFIG)
    service = [i for i in incidents if i["entity"]["scope"] == "service"]
    assert [i["signal_ids"] for i in service] == [["gap-1", "gap-2"], ["far"]]
    merged = service[0]
    assert merged["evidence"]["gap_minutes"] == 5 and len(merged["evidence"]["active_windows"]) == 2
    assert any("5 minute(s) without contributing signals" in text for text in merged["limitations"])
    assert service[1]["incident_id"] not in merged["related_incident_ids"]
    endpoint = next(i for i in incidents if i["entity"]["scope"] == "endpoint")
    assert endpoint["entity"]["endpoint"] == "/api/orders" and endpoint["incident_id"] in merged["related_incident_ids"]
    ips = sorted(i["entity"]["source_ip"] for i in incidents if i["category"] == "auth_failures")
    assert ips == ["192.0.2.8", "192.0.2.9"]

    boundary = [signal("late", "23:55:00", [1], split="validation"), signal("early", "00:00:00", [2], split="test")]
    assert len(group_incidents(boundary, [], records(2), CONFIG)) == 2


def test_ml_corroboration_and_ml_only_candidates():
    rows = records(12) + [record(30, "12:01:00", service="search-service"),
                          record(31, "12:06:00", service="search-service")]
    rules = [signal("rule", "12:00:00", [1, 2])]
    ml_signals = [
        ml("corroborates", "12:00:00", [1, 2, 3], minutes=15),
        ml("unrelated-feature", "12:00:00", [1], scope="endpoint", entity="/api/orders",
           feature="business.http_4xx_rate"),
        ml("decrease", "12:00:00", [1, 2], deviation=-3.0, minutes=5, service="payment-service"),
        ml("search-1", "12:00:00", [30], service="search-service", feature="business.p95_latency_ms"),
        ml("search-2", "12:05:00", [31], service="search-service", feature="business.p95_latency_ms"),
    ]
    rows += [record(i, "12:00:40", service="payment-service") for i in (40, 41)]
    ml_signals[2]["evidence_record_ids"] = ["h:40", "h:41"]
    incidents = group_incidents(rules, ml_signals, rows, CONFIG)
    corroborated = by_category(incidents)[("http_errors", "service", "orders-service")]
    assert corroborated["detector_ids"] == ["http_error_burst", "isolation_forest"]
    assert (corroborated["severity"], corroborated["confidence"]) == ("warning", "high")
    assert corroborated["end_time"] == "2025-02-05T12:15:00"
    assert corroborated["observed_metrics"]["ml_anomaly_score"] == 0.7
    assert corroborated["baseline_metrics"] == {"http_5xx_rate": 0.01, "business.http_5xx_rate": 0.0}
    assert "not probabilities" in corroborated["explanation"]
    assert "rate 30% vs limit 20%" in corroborated["explanation"]
    assert not any(text.startswith("ML-only") for text in corroborated["limitations"])

    candidates = [i for i in incidents if i["category"] == "multivariate"]
    assert all((i["severity"], i["confidence"]) == ("review", "low") for i in candidates)
    assert sorted(i["signal_ids"] for i in candidates) == [["decrease"], ["search-1", "search-2"],
                                                          ["unrelated-feature"]]
    endpoint = next(i for i in candidates if i["entity"]["scope"] == "endpoint")
    assert corroborated["incident_id"] in endpoint["related_incident_ids"]
    search = next(i for i in candidates if i["services"] == ["search-service"])
    assert "resemble latency" in search["explanation"]
    counts = incident_counts(rules, ml_signals, incidents)
    assert (counts["corroborated"], counts["ml_only"], counts["rule_only"]) == (1, 3, 0)
    assert (counts["ml_signals_attached"], counts["ml_signals_unattached"]) == (1, 4)


@pytest.mark.parametrize(("windows", "expected"), [
    ([(15, 20, 33)], "high"),  # one sustained fifteen-minute window
    ([(5, 8, 12), (5, 7, 12), (5, 9, 12)], "high"),  # consecutive five-minute windows
    ([(5, 8, 12), (5, 4, 12), (5, 9, 12)], "warning"),  # middle window below 50%
    ([(15, 20, 25)], "warning"),  # too few valid statuses
])
def test_high_severity_requires_sustained_supported_failure(windows, expected):
    signals, start = [], datetime(2025, 2, 5, 12)
    for n, (minutes, errors, valid) in enumerate(windows):
        signals.append(signal(f"s{n}", start.time().isoformat(), [n + 1], minutes=minutes,
                              observed={"http_5xx_count": errors, "http_5xx_rate": errors / valid,
                                        "valid_status_count": valid}))
        start += timedelta(minutes=minutes)
    [incident] = group_incidents(signals, [], records(), CONFIG)
    assert incident["severity"] == expected
    assert ("Sustained condition" in incident["explanation"]) == (expected == "high")


def test_traffic_is_review_by_policy_and_configurable():
    traffic = [signal("t", "12:00:00", [1], detector="traffic_surge", category="traffic",
                      observed={"record_count": 40, "robust_score": 6.0})]
    [incident] = group_incidents(traffic, [], records(), CONFIG)
    assert (incident["severity"], incident["confidence"]) == ("review", "medium")
    assert incident["explanation"].startswith("Suspicious activity requiring investigation")
    [incident] = group_incidents(traffic, [], records(), replace(CONFIG, review_categories=()))
    assert incident["severity"] == "warning"


def test_gateway_link_requires_routing_evidence_and_overlap():
    rows = [record(1, "12:01:00", service="api-gateway", status=502, upstream="orders-service"),
            record(2, "12:02:00", service="api-gateway", status=502, upstream="search-service"),
            record(3, "12:01:30", status=500), record(4, "12:31:00", service="search-service", status=500)]
    signals = [signal("gw", "12:00:00", [1, 2], service="api-gateway"), signal("orders", "12:00:00", [3]),
               signal("search", "12:30:00", [4], service="search-service")]
    incidents = by_category(group_incidents(signals, [], rows, CONFIG))
    gateway = incidents[("http_errors", "service", "api-gateway")]
    orders = incidents[("http_errors", "service", "orders-service")]
    search = incidents[("http_errors", "service", "search-service")]
    assert gateway["related_incident_ids"] == [orders["incident_id"]]
    assert orders["related_incident_ids"] == [gateway["incident_id"]] and search["related_incident_ids"] == []
    assert gateway["root_cause_hypothesis"].startswith("Hypothesis: consistent with orders-service")
    assert "do not establish the cause" in gateway["root_cause_hypothesis"]
    assert orders["root_cause_hypothesis"] is None
    assert gateway["evidence"]["upstream_services"] == {"orders-service": 1, "search-service": 1}


@pytest.mark.parametrize("mutation", ["duplicate", "unknown_record", "ml_category", "rule_detector",
                                      "outside_split", "support", "ml_train", "missing_field"])
def test_incompatible_signals_are_rejected(mutation):
    rules, ml_signals = [signal("r", "12:00:00", [1])], [ml("m", "12:00:00", [1])]
    if mutation == "duplicate":
        ml_signals[0]["signal_id"] = "r"
    elif mutation == "unknown_record":
        rules[0]["evidence_record_ids"] = ["other-input:1"]
    elif mutation == "ml_category":
        ml_signals[0]["category"] = "http_errors"
    elif mutation == "rule_detector":
        rules[0]["detector_id"] = "isolation_forest"
    elif mutation == "outside_split":
        rules[0]["split"] = "train"
    elif mutation == "support":
        rules[0]["support_count"] = 2
    elif mutation == "ml_train":
        ml_signals[0] = ml("m", "12:00:00", [1], split="train")
    else:
        del rules[0]["observed_metrics"]
    with pytest.raises(InputError):
        validate_signals(rules, ml_signals, SPLITS, {"h:1"})


def test_reports_are_deterministic_escaped_and_schema_valid():
    rows = records(12) + [record(13, "12:00:40", status=500, message="=HYPERLINK(\"x\")|<b>", flags=["invalid_ip"])]
    rules = [signal(f"r{n}", f"12:{5 * n:02d}:00", [n + 1] + [13] * (n == 0)) for n in range(4)]
    rules.append(signal("flagged", "12:00:00", [13], scope="endpoint", entity="/api/orders"))
    ml_signals = [ml("m", "12:00:00", [1, 2])]
    shuffled = copy.deepcopy(rules)
    random.Random(3).shuffle(shuffled)
    incidents = group_incidents(rules, ml_signals, rows, CONFIG)
    assert incidents == group_incidents(shuffled, ml_signals, rows, CONFIG)
    report = {"schema_version": "1.0", "run_id": "r", "timezone": "unspecified", "incidents": incidents,
              "limitations": report_limitations("unspecified")}
    validate_incident_report(report, CONFIG)
    flagged = next(i for i in incidents if i["entity"]["scope"] == "endpoint")
    assert flagged["evidence"]["quality_flag_counts"] == {"invalid_ip": 1}
    assert any("data-quality flags: invalid_ip" in text for text in flagged["limitations"])

    incidents[0]["explanation"] = "=cmd()"
    parsed = list(csv.DictReader(io.StringIO(incidents_csv(incidents))))
    assert [r["incident_id"] for r in parsed] == [i["incident_id"] for i in incidents]
    assert parsed[0]["explanation"] == "'=cmd()" and parsed[0]["root_cause_hypothesis"] == ""
    assert "record_ids" not in parsed[0]["evidence"] and '"record_count"' in parsed[0]["evidence"]
    summary = summary_markdown(report, incident_counts(rules, ml_signals, incidents))
    assert "\\|&lt;b&gt;" in summary and "<b>" not in summary

    unsorted = {**report, "incidents": incidents[::-1]}
    with pytest.raises(ValueError, match="sorted"):
        validate_incident_report(unsorted, CONFIG)
    empty = {**report, "incidents": []}
    validate_incident_report(empty, CONFIG)
    assert incidents_csv([]) == ",".join(CSV_FIELDS) + "\n"
    assert "No incidents" in summary_markdown(empty, incident_counts([], [], []))


@pytest.mark.parametrize(("key", "value"), [("merge_gap_minutes", 3), ("merge_gap_minutes", -5),
                                            ("sample_evidence_limit", 0), ("high_http_5xx_rate", 1.5),
                                            ("review_categories", ["multivariate"])])
def test_incident_config_rejects_invalid_values(key, value):
    with pytest.raises(InputError):
        IncidentConfig(**{key: value})
