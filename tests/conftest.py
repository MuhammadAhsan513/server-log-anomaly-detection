import csv
import random
from datetime import datetime, timedelta
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


@pytest.fixture
def synthetic_rows(row):
    """Seeded orders traffic with background errors/slow requests plus one injected fault.

    The fault time is a generator argument only; detectors never receive it.
    """
    def generate(seed=7, fault="2025-02-05 12:00:00"):
        rng = random.Random(seed)
        rows = []
        start = datetime(2025, 2, 3)
        for step in range(3 * 288):
            when = start + timedelta(minutes=5 * step)
            injected = when.isoformat(sep=" ") == fault
            for i in range(6):
                status = 500 if injected else rng.choices([200, 404, 500], [94, 5, 1])[0]
                latency = 2000 + i if injected else rng.randint(80, 140) if rng.random() < 0.98 else rng.randint(400, 900)
                rows.append({**row, "service": "orders-service", "endpoint": "/api/orders",
                             "timestamp": (when + timedelta(seconds=40 * i)).isoformat(sep=" "),
                             "http_status": str(status), "log_level": "ERROR" if status == 500 else "INFO",
                             "response_time_ms": str(latency),
                             "message": f"order {rng.randint(1, 999)} processed"})
        return rows
    return generate
