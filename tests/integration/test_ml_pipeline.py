import hashlib
import json
from pathlib import Path

import pytest

from log_anomaly.cli import main
from log_anomaly.config import FeatureConfig, InputError
from log_anomaly.ml import score_windows
from log_anomaly.pipeline import features, load_model_bundle

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {**json.loads((ROOT / "configs/ml.json").read_text()), "resolutions": [5], "scopes": ["service"],
          "minimum_records": 5, "minimum_training_windows": 50, "minimum_validation_windows": 10,
          "n_estimators": 50}


@pytest.fixture
def runs(csv_file, synthetic_rows, tmp_path):
    features(csv_file(synthetic_rows()), FeatureConfig(train_days=1, validation_days=1), tmp_path / "features")
    config = tmp_path / "ml.json"
    config.write_text(json.dumps(CONFIG))
    detect = ["detect", "--input", str(tmp_path / "features"), "--config", str(ROOT / "configs/detection.json"),
              "--output", str(tmp_path / "rules")]
    assert main(detect) == 0
    return tmp_path, config


def cli(tmp_path, config, output, rules=True):
    args = ["ml", "--input", str(tmp_path / "features"), "--config", str(config), "--output", str(tmp_path / output)]
    return main(args + (["--rules", str(tmp_path / "rules")] if rules else []))


def test_phase4_cli_publishes_reproducible_bundle_without_touching_sources(runs):
    tmp_path, config = runs
    sources = {p: hashlib.sha256(p.read_bytes()).hexdigest()
               for d in ("features", "rules") for p in (tmp_path / d).rglob("*") if p.is_file()}
    assert cli(tmp_path, config, "first") == cli(tmp_path, config, "second") == 0
    first = json.loads((tmp_path / "first/manifest.json").read_text())
    second = json.loads((tmp_path / "second/manifest.json").read_text())
    assert first["phase"] == 4 and first["stage"] == "ml_baseline" and first["status"] == "complete"
    assert not first["pipeline_complete"] and first["seed"] == 42
    assert first["run_id"] == second["run_id"] and first["artifacts"] == second["artifacts"]
    assert set(first["artifacts"]) == {"config.json", "signals.jsonl", "ml_summary.json",
                                       "model/metadata.json", "model/estimators.pkl"}
    assert first["rules_run"]["artifacts"]["signals.jsonl"] == \
        json.loads((tmp_path / "rules/manifest.json").read_text())["artifacts"]["signals.jsonl"]
    assert "scikit-learn" in first["dependencies"]
    summary = json.loads((tmp_path / "first/ml_summary.json").read_text())
    assert summary["available_models"] == first["counts"]["available_models"] == 1
    assert "rule_comparison" in summary["models"][0]["alerts"]["test"]
    signals = [json.loads(line) for line in (tmp_path / "first/signals.jsonl").read_text().splitlines()]
    assert signals and all(s["split"] in {"validation", "test"} for s in signals)
    assert sources == {p: hashlib.sha256(p.read_bytes()).hexdigest()
                       for d in ("features", "rules") for p in (tmp_path / d).rglob("*") if p.is_file()}
    metadata, estimators = load_model_bundle(tmp_path / "first")
    windows = [json.loads(line) for line in (tmp_path / "features/windows.jsonl").read_text().splitlines()]
    window = next(w for w in windows if w["window_start"] == signals[-1]["window_start"]
                  and w["window_minutes"] == 5 and w["scope"] == "service")
    assert score_windows(metadata, estimators, [window])[0]["score"] == signals[-1]["score"]
    assert cli(tmp_path, config, "first") == 4
    assert cli(tmp_path, config, "no-rules", rules=False) == 0
    assert json.loads((tmp_path / "no-rules/ml_summary.json").read_text())["rule_comparison"] == "not_requested"


def test_tampered_bundle_and_mismatched_rules_are_rejected(runs):
    tmp_path, config = runs
    assert cli(tmp_path, config, "model") == 0
    with (tmp_path / "model/model/estimators.pkl").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(InputError, match="hash mismatch"):
        load_model_bundle(tmp_path / "model")
    rules = json.loads((tmp_path / "rules/manifest.json").read_text())
    rules["source_run"]["artifacts"]["windows.jsonl"] = "0" * 64
    (tmp_path / "rules/manifest.json").write_text(json.dumps(rules))
    assert cli(tmp_path, config, "mismatch") == 2
    assert not (tmp_path / "mismatch").exists() and not (tmp_path / "mismatch.incomplete").exists()
    assert main(["ml", "--input", str(tmp_path / "missing"), "--config", str(config),
                 "--output", str(tmp_path / "x")]) == 2
