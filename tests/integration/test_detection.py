import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from log_anomaly.cli import main
from log_anomaly.config import DetectionConfig, FeatureConfig
from log_anomaly.pipeline import detect, features, json_text

CONFIG = Path(__file__).resolve().parents[2] / "configs/detection.json"


@pytest.fixture
def feature_run(csv_file, row, tmp_path):
    rows = [{**row, "timestamp": f"2025-02-0{day} 23:00:00", "http_status": "500"}
            for day in (3, 4, 5) for _ in range(20)]
    features(csv_file(rows), FeatureConfig(train_days=1, validation_days=1), tmp_path / "features")
    return tmp_path / "features"


def cli(source, output):
    return main(["detect", "--input", str(source), "--output", str(output), "--config", str(CONFIG)])


def rewrite(source, name, content):
    path = source / name
    path.write_text(content, encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["artifacts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "manifest.json").write_text(json.dumps(manifest))


def test_phase3_cli_publication_reproducibility_and_empty_results(feature_run, tmp_path):
    source_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in feature_run.iterdir() if p.is_file()}
    assert cli(feature_run, tmp_path / "first") == cli(feature_run, tmp_path / "second") == 0
    first = json.loads((tmp_path / "first/manifest.json").read_text())
    second = json.loads((tmp_path / "second/manifest.json").read_text())
    assert first["phase"] == 3 and first["status"] == "complete" and not first["pipeline_complete"]
    assert first["stage"] == "rules_and_baselines" and first["baseline_mode"] == "frozen_training"
    assert first["run_id"] == second["run_id"] and first["artifacts"] == second["artifacts"]
    assert set(first["artifacts"]) == {"config.json", "signals.jsonl", "baselines.json", "detection_summary.json"}
    assert first["counts"]["signals"] == 12
    assert first["source_run"]["path"] == str(feature_run.resolve())
    assert first["timezone"] == "unspecified"
    assert first["split_ranges"] == json.loads((feature_run / "manifest.json").read_text())["split_ranges"]
    for name, digest in first["artifacts"].items():
        assert hashlib.sha256((tmp_path / "first" / name).read_bytes()).hexdigest() == digest
    assert source_before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in feature_run.iterdir() if p.is_file()}
    assert cli(feature_run, tmp_path / "first") == 4
    empty = detect(feature_run, replace(DetectionConfig(), http_minimum_errors=21), tmp_path / "empty")
    assert empty["counts"]["signals"] == 0
    assert (tmp_path / "empty/signals.jsonl").read_text() == ""
    summary = json.loads((tmp_path / "empty/detection_summary.json").read_text())
    assert summary["signal_count"] == 0 and summary["decision_counts"]


@pytest.mark.parametrize("corruption", ["hash", "split", "duplicate", "incomplete", "schema"])
def test_incompatible_or_corrupted_source_is_rejected(feature_run, tmp_path, corruption):
    windows = [json.loads(line) for line in (feature_run / "windows.jsonl").read_text().splitlines()]
    if corruption == "hash":
        with (feature_run / "windows.jsonl").open("a") as handle:
            handle.write("{}\n")
    elif corruption == "split":
        windows[0]["split"] = "test"
        rewrite(feature_run, "windows.jsonl", "".join(json_text(w) + "\n" for w in windows))
    elif corruption == "duplicate":
        rewrite(feature_run, "windows.jsonl", "".join(json_text(w) + "\n" for w in windows + windows[:1]))
    elif corruption == "schema":
        windows[0]["feature_schema_version"] = "2"
        rewrite(feature_run, "windows.jsonl", "".join(json_text(w) + "\n" for w in windows))
    else:
        target = feature_run.with_suffix(".incomplete")
        feature_run.rename(target)
        feature_run = target
    assert cli(feature_run, tmp_path / "result") == 2
    assert not (tmp_path / "result").exists()


def test_interrupted_and_corrupted_publication_cannot_complete(feature_run, tmp_path, monkeypatch):
    import log_anomaly.pipeline as pipeline

    original = pipeline._write_lines

    def fail(*args):
        raise OSError("simulated disk full")

    monkeypatch.setattr(pipeline, "_write_lines", fail)
    assert cli(feature_run, tmp_path / "failed") == 4
    assert json.loads((tmp_path / "failed.incomplete/manifest.json").read_text())["status"] == "incomplete"
    monkeypatch.setattr(pipeline, "_write_lines", original)
    assert cli(feature_run, tmp_path / "failed") == 4

    def corrupt(path, values):
        original(path, values)
        path.write_text("")

    monkeypatch.setattr(pipeline, "_write_lines", corrupt)
    assert cli(feature_run, tmp_path / "corrupt") == 3
    assert not (tmp_path / "corrupt").exists()
    monkeypatch.setattr(pipeline, "_write_lines", original)
    assert cli(feature_run, tmp_path / "fresh") == 0


def test_evidence_mismatch_blocks_completion(feature_run, tmp_path):
    windows = [json.loads(line) for line in (feature_run / "windows.jsonl").read_text().splitlines()]
    for window in windows:
        if window["business"]["valid_status_count"]:
            window["business"]["valid_status_count"] += 1
    rewrite(feature_run, "windows.jsonl", "".join(json_text(w) + "\n" for w in windows))
    assert cli(feature_run, tmp_path / "invalid") == 3
    assert not (tmp_path / "invalid").exists()
