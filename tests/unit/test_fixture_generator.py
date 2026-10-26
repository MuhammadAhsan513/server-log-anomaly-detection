"""Evaluation labels are separate from the detector's CSV input."""

import csv
import importlib.util
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fixture_generator", ROOT / "scripts/generate_fixture_logs.py")
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_fixture_is_seeded_labelled_and_keeps_labels_out_of_logs(tmp_path):
    first_logs, first_labels = generator.write(11, tmp_path / "first", days=3)
    second_logs, second_labels = generator.write(11, tmp_path / "second", days=3)
    assert first_logs.read_bytes() == second_logs.read_bytes()
    assert first_labels.read_bytes() == second_labels.read_bytes()

    labels = json.loads(first_labels.read_text())
    with first_logs.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == generator.COLUMNS
        rows = list(reader)
    assert len(rows) > 10000
    assert all("event_id" not in row and "injection" not in row for row in rows)
    assert {e["category"] for e in labels["events"]} >= {
        "http_errors", "auth_failures", "latency", "service_errors", "traffic"}
    assert len({e["event_id"] for e in labels["events"]}) == len(labels["events"])

    # Every strong injected event is visible in the CSV under the labelled service
    # and time bounds; the evaluator alone knows which rows belong to the event.
    for event in labels["events"]:
        if event["difficulty"] == "weak":
            continue
        selected = [row for row in rows if row["service"] == event["service"]
                    and event["start"] <= row["timestamp"].replace(" ", "T") < event["end"]]
        assert selected, event["event_id"]
        if event["category"] == "auth_failures":
            assert sum(row["source_ip"] == event["source_ip"] and row["http_status"] == "401"
                       for row in selected) >= 10
        elif event["category"] == "http_errors":
            assert sum(int(row["http_status"]) >= 500 for row in selected) >= 5
        elif event["category"] == "service_errors":
            assert Counter(row["log_level"] for row in selected)["ERROR"] >= 5
        elif event["category"] == "latency":
            assert max(int(row["response_time_ms"]) for row in selected) > 1000
        elif event["category"] == "traffic":
            assert len(selected) >= 100

    assert all(datetime.fromisoformat(e["start"]) < datetime.fromisoformat(e["end"])
               for e in labels["events"])
