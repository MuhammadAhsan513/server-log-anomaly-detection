import hashlib

import pytest

from log_anomaly.config import InputError
from log_anomaly.ingest import REQUIRED_COLUMNS, read_logs


def test_quoted_unicode_multiline_bom_and_record_ordinals(csv_file, row):
    message = 'Unicode café 雪, "quoted"\nsecond line'
    path = csv_file([{**row, "message": message}, row])
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    result = read_logs(path)
    assert result.records[0].message == message
    assert result.records[1].source_record_number == 2
    assert result.records[0].record_id == hashlib.sha256(path.read_bytes()).hexdigest() + ":1"
    assert result.records[0].timestamp.tzinfo is None
    assert not result.records[0].to_dict()["timestamp"].endswith("Z")


@pytest.mark.parametrize("timestamp", ["", "2025-2-03 12:00:00", "2025-02-30 12:00:00",
                                        "2025-02-03T12:00:00", "2025-02-03 24:00:00"])
def test_bad_dates_quarantined_valid_rows_survive(csv_file, row, timestamp):
    result = read_logs(csv_file([{**row, "timestamp": timestamp}, row]))
    assert len(result.records) == len(result.rejects) == 1
    assert result.rejects[0].reasons == ["invalid_timestamp"]
    assert result.quality["parsed_count"] == 2


def test_missing_service_and_multiple_reasons(csv_file, row):
    result = read_logs(csv_file([{**row, "service": " ", "timestamp": "bad"}]))
    assert result.rejects[0].reasons == ["invalid_timestamp", "missing_service"]


@pytest.mark.parametrize("status", ["", "99", "600", "200.0", "NaN", "-1", "True"])
def test_invalid_status_nullable(csv_file, row, status):
    record = read_logs(csv_file([{**row, "http_status": status}])).records[0]
    assert record.http_status is None
    assert "invalid_http_status" in record.quality_flags


@pytest.mark.parametrize("latency", ["", "-1", "1.5", "NaN", "9" * 5000])
def test_invalid_latency_nullable(csv_file, row, latency):
    record = read_logs(csv_file([{**row, "response_time_ms": latency}])).records[0]
    assert record.response_time_ms is None
    assert "invalid_response_time_ms" in record.quality_flags


def test_optional_fields_unknowns_and_originals(csv_file, row):
    raw = {**row, "service": "new-service", "log_level": "debug", "source_ip": "bad",
           "endpoint": "", "message": "", "extra": "kept"}
    result = read_logs(csv_file([raw], (*REQUIRED_COLUMNS, "extra")))
    record = result.records[0]
    assert record.original_fields == raw
    assert record.source_ip is record.endpoint is None
    assert record.log_level == "UNKNOWN"
    assert set(record.quality_flags) == {"unknown_service", "unknown_log_level", "invalid_source_ip",
                                         "missing_endpoint", "empty_message"}
    assert result.quality["field_quality_counts"]["empty_message"] == 1


@pytest.mark.parametrize("status", ["100", "599"])
def test_valid_boundaries_normalization_ipv6(csv_file, row, status):
    record = read_logs(csv_file([{**row, "log_level": " warn ", "http_status": status,
                                  "response_time_ms": "0", "source_ip": "2001:db8::1"}])).records[0]
    assert record.log_level == "WARN"
    assert record.http_status == int(status)
    assert record.response_time_ms == 0
    assert record.source_ip == "2001:db8::1"
    assert record.quality_flags == []


def test_invalid_endpoint_retained_flagged(csv_file, row):
    record = read_logs(csv_file([{**row, "endpoint": "not-a-path"}])).records[0]
    assert record.endpoint == "not-a-path"
    assert "invalid_endpoint" in record.quality_flags


def test_stable_order_duplicates_and_column_counts(csv_file, row):
    earlier = {**row, "timestamp": "2025-02-03 11:00:00"}
    result = read_logs(csv_file([row, earlier, earlier, ["short"], [], list(row.values()) + ["extra"]]))
    assert [r.source_record_number for r in result.records] == [2, 3, 1]
    assert result.quality["duplicate_candidates"] == 1
    assert result.quality["ordering_changes"] == 3
    assert result.quality["parsed_count"] == 6
    assert result.quality["accepted_count"] + result.quality["rejected_count"] == 6
    assert all(r.reasons == ["column_count"] for r in result.rejects)


@pytest.mark.parametrize("content", [b"", b"timestamp,service\n", b"\xff", b'timestamp,timestamp\n'])
def test_fatal_input(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_bytes(content)
    with pytest.raises(InputError):
        read_logs(path)


def test_fatal_framing_reports_location(csv_file, row):
    path = csv_file([row])
    with path.open("a") as handle:
        handle.write('"unterminated\nquoted field')
    with pytest.raises(InputError, match=r"physical line .*data record 2; accounting incomplete"):
        read_logs(path)


def test_header_only_and_missing_file(csv_file, tmp_path):
    result = read_logs(csv_file([]))
    assert result.quality["parsed_count"] == 0
    assert result.quality["source_time_bounds"] == {"first": None, "last": None}
    with pytest.raises(InputError, match="cannot read"):
        read_logs(tmp_path / "absent.csv")
