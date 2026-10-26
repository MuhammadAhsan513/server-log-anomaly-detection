import pytest

from log_anomaly.ingest import read_logs
from log_anomaly.parse import message_template, parse_messages


def parse_one(csv_file, row, service, message):
    record = read_logs(csv_file([{**row, "service": service, "message": message}])).records[0]
    parse_messages([record])
    assert record.message == message
    return record


def test_gateway_path_and_query_do_not_overwrite_endpoint(csv_file, row):
    record = parse_one(csv_file, row, "api-gateway",
                       "method=GET path=/different?q=tea+cup&q=雪&empty= upstream=search-service "
                       "upstream_latency_ms=42 req_id=variable-id")
    fields = record.enrichment
    assert (fields.method, fields.upstream_service, fields.upstream_latency_ms, fields.request_id) == (
        "GET", "search-service", 42, "variable-id")
    assert fields.message_path == "/different"
    assert fields.message_query == "q=tea+cup&q=雪&empty="
    assert fields.query_parameters == {"q": ["tea cup", "雪"], "empty": [""]}
    assert record.endpoint == "/api/search"
    assert record.parse_status == "matched"


@pytest.mark.parametrize("message,status", [
    ("method=GET path=/health status=200 upstreams_healthy=4/4", "matched"),
    ("method=GET path=/api/search cache=HIT", "matched"),
    ("upstream=search-service upstream_latency_ms=bad", "partial"),
    ("method=GET path=/x upstream_latency_ms=-3", "partial"),
    ("unexpected message 雪", "unmatched"), ("  ", "empty"),
])
def test_gateway_partial_unknown_and_empty(csv_file, row, message, status):
    record = parse_one(csv_file, row, "api-gateway", message)
    assert record.parse_status == status
    assert record.enrichment.upstream_latency_ms is None


@pytest.mark.parametrize("message,status,took,query,metadata", [
    ('query="café 雪" hits=12 took=43ms shards=6', "matched", 43, "café 雪", {"hits": 12, "shards": 6}),
    ('suggest prefix="wat" results=5 took=17ms', "matched", 17, None, {"prefix": "wat", "results": 5}),
    ('product lookup sku=SKU-123 found took=145ms', "matched", 145, None, {"sku": "SKU-123"}),
    ('query="tea" not modified (etag match)', "matched", None, "tea", None),
    ('query="tea" took=bad hits=NaN', "partial", None, "tea", None),
    ('query="tea" timed out after 8000ms', "partial", None, "tea", None),
    ('arbitrary took=43ms', "unmatched", None, None, None),
])
def test_search(csv_file, row, message, status, took, query, metadata):
    record = parse_one(csv_file, row, "search-service", message)
    assert record.parse_status == status
    assert record.enrichment.took_ms == took
    assert record.enrichment.query == query
    assert record.enrichment.query_metadata == metadata


@pytest.mark.parametrize("message,user,outcome,status", [
    ("Login failed for user guest: invalid credentials", "guest", "failure", "matched"),
    ("Login succeeded for user u123 (auth=password, mfa=false)", "u123", "success", "matched"),
    ("Access token refreshed for user u456 expires_in=900s", "u456", None, "matched"),
    ("Logout rejected for user u456: session not found", "u456", None, "matched"),
    ("Rejected login request: malformed JSON body", None, None, "partial"),
    ("Login failed", None, None, "partial"),
    ("Forbidden on /admin status=403", None, None, "unmatched"),
])
def test_auth_requires_recognized_context(csv_file, row, message, user, outcome, status):
    record = parse_one(csv_file, {**row, "http_status": "401"}, "auth-service", message)
    assert record.enrichment.user_id == user
    assert record.enrichment.authentication_outcome == outcome
    assert record.parse_status == status


def test_other_service_does_not_become_auth_attempt(csv_file, row):
    record = parse_one(csv_file, {**row, "http_status": "403"}, "orders-service",
                       "Login failed for user guest: invalid credentials")
    assert record.enrichment.authentication_outcome is None
    assert record.parse_status == "unmatched"


def test_generic_templates_and_coverage(csv_file, row):
    assert message_template("user u123 from 192.0.2.4 req_id=abc123 latency=42ms") == (
        "user <ID> from <IP> req_id=<ID> latency=<NUM>ms")
    assert message_template("client 2001:db8::1 took 12ms") == "client <IP> took <NUM>ms"
    assert message_template(" ") is None
    records = read_logs(csv_file([row, {**row, "message": "new unknown"}, {**row, "message": ""}])).records
    coverage = parse_messages(records)
    assert coverage["api-gateway"] == {
        "total": 3, "matched": 1, "partial": 0, "unmatched": 1, "empty": 1, "matched_fraction": 1 / 3,
    }
    before = [r.to_dict() for r in records]
    assert parse_messages(records) == coverage
    assert [r.to_dict() for r in records] == before
