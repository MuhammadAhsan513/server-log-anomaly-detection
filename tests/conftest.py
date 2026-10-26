import csv
from pathlib import Path

import pytest

from log_anomaly.ingest import REQUIRED_COLUMNS


@pytest.fixture
def row():
    return {
        "timestamp": "2025-02-03 12:00:00", "service": "api-gateway", "log_level": "INFO",
        "http_status": "200", "response_time_ms": "12", "endpoint": "/api/search",
        "source_ip": "192.0.2.7",
        "message": "method=GET path=/api/search?q=tea upstream=search-service upstream_latency_ms=8 req_id=abc123",
    }


@pytest.fixture
def csv_file(tmp_path):
    def write(rows, columns=REQUIRED_COLUMNS, name="input.csv") -> Path:
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row.get(c, "") for c in columns] if isinstance(row, dict) else row)
        return path
    return write
