import csv
import hashlib
import json
from pathlib import Path

import pytest

from log_anomaly.cli import main
from log_anomaly.config import FeatureConfig
from log_anomaly.pipeline import features

ROOT = Path(__file__).resolve().parents[2]
ML_CONFIG = {**json.loads((ROOT / "configs/ml.json").read_text()), "resolutions": [5], "scopes": ["service"],
             "minimum_records": 5, "minimum_training_windows": 50, "minimum_validation_windows": 10,
             "n_estimators": 50}
# The fixture has six records per five minutes, so fifteen-minute rule windows need lower support guards.
RULE_CONFIG = {**json.loads((ROOT / "configs/detection.json").read_text()), "minimum_statuses": 10,
               "minimum_latencies": 10, "minimum_records": 10}


@pytest.fixture
def runs(csv_file, synthetic_rows, tmp_path):
    features(csv_file(synthetic_rows()), FeatureConfig(train_days=1, validation_days=1), tmp_path / "features")
    (tmp_path / "ml.json").write_text(json.dumps(ML_CONFIG))
    (tmp_path / "detection.json").write_text(json.dumps(RULE_CONFIG))
    assert main(["detect", "--input", str(tmp_path / "features"), "--config", str(tmp_path / "detection.json"),
                 "--output", str(tmp_path / "rules")]) == 0
    assert main(["ml", "--input", str(tmp_path / "features"), "--config", str(tmp_path / "ml.json"),
                 "--rules", str(tmp_path / "rules"), "--output", str(tmp_path / "ml")]) == 0
    return tmp_path


def cli(tmp_path, output, rules="rules", ml="ml"):
    return main(["incidents", "--input", str(tmp_path / "features"), "--rules", str(tmp_path / rules),
                 "--ml", str(tmp_path / ml), "--config", str(ROOT / "configs/incidents.json"),
                 "--output", str(tmp_path / output)])


def hashes(tmp_path):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("features", "rules", "ml") for p in (tmp_path / d).rglob("*") if p.is_file()}


def test_evaluation_cli_reproducible_and_rejects_invalid_labels(runs):
    assert cli(runs, "incidents-for-evaluation") == 0
    label_path = runs / "labels.json"
    label_path.write_text(json.dumps({"schema_version": "1.0", "seed": 7, "events": []}))
    command = ["evaluate", "--input", str(runs / "incidents-for-evaluation"),
               "--labels", str(label_path), "--config", str(ROOT / "configs/evaluation.json")]
    before = hashes(runs)
    for name in ("evaluation-a", "evaluation-b"):
        assert main([*command, "--output", str(runs / name)]) == 0
    first = json.loads((runs / "evaluation-a/evaluation.json").read_text())
    second = json.loads((runs / "evaluation-b/evaluation.json").read_text())
    assert first == second
    assert first["results"]["test"]["rules"]["tp"] == 0
    assert first["results"]["test"]["rules"]["recall"]["value"] is None
    assert hashes(runs) == before
    assert json.loads((runs / "evaluation-a/manifest.json").read_text())["pipeline_complete"] is False

    label_path.write_text(json.dumps({"schema_version": "1.0", "events": [{"event_id": "bad"}]}))
    assert main([*command, "--output", str(runs / "invalid-evaluation")]) == 2
    assert not (runs / "invalid-evaluation").exists()


def test_phase5_cli_publishes_reproducible_traceable_reports(runs):
    sources = hashes(runs)
    assert cli(runs, "first") == cli(runs, "second") == 0
    first = json.loads((runs / "first/manifest.json").read_text())
    second = json.loads((runs / "second/manifest.json").read_text())
    assert (first["phase"], first["stage"], first["status"]) == (5, "incident_reports", "complete")
    assert not first["pipeline_complete"]
    assert first["run_id"] == second["run_id"] and first["artifacts"] == second["artifacts"]
    assert set(first["artifacts"]) == {"config.json", "incidents.json", "incidents.csv", "summary.md",
                                       "charts/incidents.png"}
    for name, run in (("rules", "rules_run"), ("ml", "ml_run")):
        manifest = json.loads((runs / name / "manifest.json").read_text())
        assert first[run]["artifacts"]["signals.jsonl"] == manifest["artifacts"]["signals.jsonl"]
    assert sources == hashes(runs)

    report = json.loads((runs / "first/incidents.json").read_text())
    rule_signals = [json.loads(line) for line in (runs / "rules/signals.jsonl").read_text().splitlines()]
    ml_signals = [json.loads(line) for line in (runs / "ml/signals.jsonl").read_text().splitlines()]
    incidents = report["incidents"]
    assert first["counts"]["rule_signals"] == len(rule_signals) and first["counts"]["incidents"] == len(incidents)
    # Every signal lands in exactly one incident and every evidence ID resolves to a source record.
    assert sorted(s for i in incidents for s in i["signal_ids"]) == sorted(
        s["signal_id"] for s in rule_signals + ml_signals)
    normalized = {json.loads(line)["record_id"]
                  for line in (runs / "features/normalized.jsonl").read_text().splitlines()}
    assert all(set(i["evidence"]["record_ids"]) <= normalized for i in incidents)
    fault = [i for i in incidents if i["category"] == "http_errors" and i["split"] == "test"
             and i["start_time"] <= "2025-02-05T12:00:00" < i["end_time"]]
    assert fault and all(i["severity"] in {"warning", "high"} for i in fault)
    with (runs / "first/incidents.csv").open(newline="") as handle:
        assert [r["incident_id"] for r in csv.DictReader(handle)] == [i["incident_id"] for i in incidents]
    assert "## Overview" in (runs / "first/summary.md").read_text()
    assert cli(runs, "first") == 4


def test_incompatible_or_incomplete_runs_are_rejected(runs):
    # Same feature stream, different rule configuration: the ML run was compared with other signals.
    assert main(["detect", "--input", str(runs / "features"), "--config", str(ROOT / "configs/detection.json"),
                 "--output", str(runs / "rules-other")]) == 0
    assert cli(runs, "other-rules", rules="rules-other") == 2
    assert cli(runs, "swapped", rules="ml", ml="rules") == 2
    (runs / "ml").rename(runs / "ml.incomplete")
    assert cli(runs, "incomplete", ml="ml.incomplete") == 2
    (runs / "ml.incomplete").rename(runs / "ml")
    manifest = json.loads((runs / "ml/manifest.json").read_text())
    manifest["split_ranges"]["test"]["end"] = "2025-02-07T00:00:00"
    (runs / "ml/manifest.json").write_text(json.dumps(manifest))
    assert cli(runs, "splits") == 2
    assert not any((runs / name).exists() or (runs / f"{name}.incomplete").exists()
                   for name in ("other-rules", "swapped", "incomplete", "splits"))
