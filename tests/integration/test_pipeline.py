import csv
import hashlib
import json
from pathlib import Path

import pytest

from log_anomaly.cli import main
from log_anomaly.config import IngestConfig
from log_anomaly.pipeline import ingest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ingest.json"


def cli(path, output, config=CONFIG):
    return main(["ingest", "--input", str(path), "--config", str(config), "--output", str(output)])


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ingest" in capsys.readouterr().out


def test_deterministic_traceable_publication(csv_file, row, tmp_path):
    source = csv_file([row, row, {**row, "timestamp": "bad"}])
    config = IngestConfig(max_rejected_fraction=0.5)
    first = ingest(source, config, tmp_path / "first")
    second = ingest(source, config, tmp_path / "second")
    assert first["run_id"] == second["run_id"]
    assert first["artifacts"] == second["artifacts"]
    assert first["status"] == "complete" and first["pipeline_complete"] is False
    for name, digest in first["artifacts"].items():
        content = (tmp_path / "first" / name).read_bytes()
        assert content == (tmp_path / "second" / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest
    assert not (tmp_path / "first.incomplete").exists()
    assert cli(source, tmp_path / "first") == 4


@pytest.mark.parametrize("rows", [[], [{"timestamp": "bad"}]])
def test_no_usable_rows_retains_incomplete_diagnostics(csv_file, tmp_path, rows):
    assert cli(csv_file(rows), tmp_path / "run") == 2
    assert not (tmp_path / "run").exists()
    staging = tmp_path / "run.incomplete"
    assert json.loads((staging / "manifest.json").read_text())["status"] == "incomplete"
    assert (staging / "quality.json").exists()


def test_rejection_limit_boundary(csv_file, row, tmp_path):
    source = csv_file([row] * 19 + [{**row, "timestamp": "bad"}])
    assert cli(source, tmp_path / "at_limit") == 0
    source = csv_file([row] * 18 + [{**row, "timestamp": "bad"}])
    assert cli(source, tmp_path / "over_limit") == 2
    assert not (tmp_path / "over_limit").exists()


def test_missing_input_and_bad_config_before_staging(tmp_path, csv_file, row):
    assert cli(tmp_path / "missing.csv", tmp_path / "run") == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"unknown":1}')
    assert cli(csv_file([row]), tmp_path / "run", bad) == 2
    assert not (tmp_path / "run.incomplete").exists()


def test_fatal_framing_cannot_publish(csv_file, row, tmp_path):
    source = csv_file([row])
    with source.open("a") as handle:
        handle.write('"unclosed')
    assert cli(source, tmp_path / "run") == 2
    staging = tmp_path / "run.incomplete"
    assert json.loads((staging / "manifest.json").read_text())["status"] == "incomplete"
    assert not (staging / "quality.json").exists()


def test_interrupted_output_requires_fresh_directory(csv_file, row, tmp_path, monkeypatch):
    import log_anomaly.pipeline as pipeline
    original = pipeline._write_lines

    def fail(*args, **kwargs):
        raise OSError("simulated disk full")

    source = csv_file([row])
    monkeypatch.setattr(pipeline, "_write_lines", fail)
    assert cli(source, tmp_path / "run") == 4
    assert not (tmp_path / "run").exists()
    assert json.loads((tmp_path / "run.incomplete/manifest.json").read_text())["status"] == "incomplete"
    monkeypatch.setattr(pipeline, "_write_lines", original)
    assert cli(source, tmp_path / "run") == 4
    assert cli(source, tmp_path / "fresh") == 0


def test_invalid_serialized_artifacts_block_publication(csv_file, row, tmp_path, monkeypatch):
    import log_anomaly.pipeline as pipeline
    original = pipeline._write_lines

    def corrupt(path, values):
        original(path, values)
        if path.name == "normalized.jsonl":
            path.write_text("")

    monkeypatch.setattr(pipeline, "_write_lines", corrupt)
    assert cli(csv_file([row]), tmp_path / "run") == 3
    assert not (tmp_path / "run").exists()


def test_checked_in_malformed_fixture(tmp_path):
    source = ROOT / "tests/fixtures/ingestion.csv"
    manifest = ingest(source, IngestConfig(max_rejected_fraction=0.5), tmp_path / "fixture")
    assert manifest["counts"] == {"parsed_count": 8, "accepted_count": 6, "rejected_count": 2}


def test_full_supplied_file_all_outputs_traceable(tmp_path):
    source = ROOT / "logs/server_application_logs_100k.csv"
    if not source.exists():
        pytest.skip("supplied data missing; full-file Phase 1 exit gate is blocked")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = ingest(source, IngestConfig(), tmp_path / "full")
    assert manifest["counts"] == {"parsed_count": 100000, "accepted_count": 100000, "rejected_count": 0}
    assert before == hashlib.sha256(source.read_bytes()).hexdigest() == manifest["input_sha256"]
    with source.open(encoding="utf-8", newline="") as handle:
        originals = list(csv.DictReader(handle))
    seen = set()
    previous = None
    with (tmp_path / "full/normalized.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            ordinal = record["source_record_number"]
            assert record["original_fields"] == originals[ordinal - 1]
            assert record["record_id"] == f"{before}:{ordinal}"
            key = record["timestamp"], ordinal
            assert previous is None or previous <= key
            previous = key
            seen.add(ordinal)
    assert seen == set(range(1, 100001))


def test_phase2_publication_reproducibility_and_chart_failure(csv_file, row, tmp_path, monkeypatch):
    from log_anomaly.config import FeatureConfig
    import log_anomaly.report as report

    source = csv_file([row, {**row, "timestamp": "2025-02-04 00:00:00"},
                       {**row, "timestamp": "2025-02-05 00:00:00"}])
    config = tmp_path / "features.json"
    config.write_text(json.dumps(FeatureConfig(train_days=1, validation_days=1).to_dict()))

    def run(name):
        return main(["features", "--input", str(source), "--config", str(config),
                     "--output", str(tmp_path / name)])

    assert run("first") == run("second") == 0
    first = json.loads((tmp_path / "first/manifest.json").read_text())
    second = json.loads((tmp_path / "second/manifest.json").read_text())
    assert first["phase"] == 2 and first["stage"] == "eda_and_features"
    assert first["pipeline_complete"] is False and first["status"] == "complete"
    assert first["artifacts"] == second["artifacts"]
    for name, digest in first["artifacts"].items():
        assert hashlib.sha256((tmp_path / "first" / name).read_bytes()).hexdigest() == digest
    assert set(name for name in first["artifacts"] if name.startswith("charts/")) == {
        "charts/volume.png", "charts/status.png", "charts/latency.png", "charts/levels.png"}
    for name in ("volume", "status", "latency", "levels"):
        assert (tmp_path / f"first/charts/{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    eda = json.loads((tmp_path / "first/eda.json").read_text())
    assert eda["split_record_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert eda["parse_coverage"]["api-gateway"]["matched_fraction"] == 1
    assert run("first") == 4

    def fail(*args):
        raise OSError("simulated chart write failure")

    monkeypatch.setattr(report, "write_eda_charts", fail)
    assert run("failed") == 4
    assert not (tmp_path / "failed").exists()
    assert json.loads((tmp_path / "failed.incomplete/manifest.json").read_text())["status"] == "incomplete"
