import json

import pytest

from log_anomaly.config import DetectionConfig, FeatureConfig, IngestConfig, InputError, load_config


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.05", None, float("nan"), float("inf")])
def test_invalid_rejection_limit(value):
    with pytest.raises(InputError):
        IngestConfig(max_rejected_fraction=value)


@pytest.mark.parametrize("change", [{"schema_version": "2"}, {"source_timezone": "not-a-zone"},
                                   {"source_timezone": None}, {"unknown": 5}])
def test_strict_config(tmp_path, change):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**IngestConfig().to_dict(), **change}))
    with pytest.raises(InputError):
        load_config(path)


@pytest.mark.parametrize("text", ['{}', '[]', '{', '{"schema_version":"1.0","schema_version":"1.0"}'])
def test_missing_duplicate_and_malformed_keys(tmp_path, text):
    path = tmp_path / "config.json"
    path.write_text(text)
    with pytest.raises(InputError):
        load_config(path)


def test_default_config_and_explicit_timezone():
    from pathlib import Path
    assert load_config(Path("configs/ingest.json")) == IngestConfig()
    assert IngestConfig(source_timezone="Asia/Karachi").source_timezone == "Asia/Karachi"


@pytest.mark.parametrize("change", [
    {"window_minutes": 0}, {"window_minutes": 10}, {"alternate_window_minutes": 5},
    {"train_days": True}, {"validation_days": -1}, {"minimum_records": 0},
    {"split_start_day": "2025-02-30"}, {"health_endpoints": "/health"},
    {"health_endpoints": ["/health", "/health"]}, {"health_endpoints": [None]},
])
def test_feature_config_validation(change):
    with pytest.raises(InputError):
        FeatureConfig(**change)


def test_stage_configs_remain_separate():
    from pathlib import Path
    assert load_config(Path("configs/features.json"), FeatureConfig) == FeatureConfig()
    with pytest.raises(InputError):
        load_config(Path("configs/features.json"))
    with pytest.raises(InputError):
        load_config(Path("configs/ingest.json"), FeatureConfig)
    assert load_config(Path("configs/detection.json"), DetectionConfig) == DetectionConfig()
    with pytest.raises(InputError):
        load_config(Path("configs/features.json"), DetectionConfig)


@pytest.mark.parametrize("change", [
    {"history_hours": 0}, {"history_hours": 10000}, {"minimum_history_windows": True},
    {"minimum_statuses": 1.5}, {"daypart_hours": 5}, {"rate_epsilon": 0},
    {"latency_epsilon_ms": float("nan")}, {"count_epsilon": float("inf")},
    {"http_error_rate": 1.1}, {"auth_failure_rate": -1}, {"latency_multiplier": "3"},
    {"error_robust_score": False}, {"schema_version": "2"},
])
def test_detection_configuration_rejects_invalid_values(change):
    with pytest.raises(InputError):
        DetectionConfig(**change)
