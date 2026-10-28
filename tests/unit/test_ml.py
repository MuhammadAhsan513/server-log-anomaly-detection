import json
import pickle
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from log_anomaly.config import FeatureConfig, InputError, MLConfig, load_config
from log_anomaly.features import build_windows
from log_anomaly.ingest import read_logs
from log_anomaly.ml import calibrate, fit_and_score, fit_preprocessing, score_windows, transform
from log_anomaly.parse import parse_messages

FEATURES = FeatureConfig(train_days=1, validation_days=1)
CONFIG = MLConfig(resolutions=(5,), scopes=("service",), minimum_records=5,
                  minimum_training_windows=50, minimum_validation_windows=10, n_estimators=50)
@pytest.fixture
def prepared(csv_file, synthetic_rows):
    records = read_logs(csv_file(synthetic_rows())).records
    parse_messages(records)
    return build_windows(records, FEATURES), records


def run(prepared, config=CONFIG, windows=None, rule_signals=None):
    features, records = prepared
    return fit_and_score(features.windows if windows is None else windows, records,
                         features.split_ranges, FEATURES.health_endpoints, config, rule_signals)


def test_injected_fault_is_scored_above_validation_threshold_with_evidence(prepared):
    result = run(prepared)
    model = next(m for m in result.summary["models"] if m["model_key"] == "5:service:orders-service")
    assert model["status"] == "available"
    assert model["supported_window_counts"] == {"train": 288, "validation": 288, "test": 288}
    injected = [s for s in result.signals if s["window_start"] == "2025-02-05T12:00:00"]
    assert len(injected) == 1 and injected[0]["split"] == "test"
    signal = injected[0]
    assert signal["score"] > signal["threshold"]["score_gt"] == model["calibration"]["selected"]["threshold"]
    assert signal["category"] == "multivariate" and signal["detector_id"] == "isolation_forest"
    assert {f["feature"] for f in signal["feature_evidence"]} & {
        "business.http_5xx_rate", "business.level_rates.ERROR", "business.median_latency_ms"}
    assert len(signal["evidence_record_ids"]) == signal["support_count"] == 6
    assert signal["observed_metrics"]["business.http_5xx_rate"] == 1.0
    assert "2025-02-05T12:00:00" in signal["explanation"]
    assert any("does not establish an incident" in text for text in signal["limitations"])
    # Validation p99 is an alert-volume control: about 1% of validation windows exceed it.
    assert model["alerts"]["validation"]["count"] == model["calibration"]["selected"]["validation_alerts"] <= 3


def test_shifted_fault_time_is_detected_without_identity_assumptions(csv_file, synthetic_rows):
    records = read_logs(csv_file(synthetic_rows(seed=11, fault="2025-02-05 03:35:00"))).records
    parse_messages(records)
    result = run((build_windows(records, FEATURES), records))
    assert "2025-02-05T03:35:00" in {s["window_start"] for s in result.signals if s["split"] == "test"}


def test_held_out_values_never_change_fitted_preprocessing_or_model(prepared):
    features, _ = prepared
    baseline = run(prepared)
    for split in ("validation", "test"):
        perturbed = deepcopy(features.windows)
        for window in perturbed:
            if window["split"] == split:
                window["business"]["p95_latency_ms"] = 1e6
                window["business"]["http_4xx_rate"] = 0.9
        result = run(prepared, windows=perturbed)
        for key, model in result.metadata["models"].items():
            reference = baseline.metadata["models"][key]
            assert model["preprocessing"] == reference["preprocessing"]
            assert model["training_window_count"] == reference["training_window_count"]
        assert pickle.dumps(result.estimators) == pickle.dumps(baseline.estimators)
        train = {m["model_key"]: m["score_distribution"]["train"] for m in result.summary["models"]}
        assert train == {m["model_key"]: m["score_distribution"]["train"] for m in baseline.summary["models"]}
        if split == "test":
            # Test data is never used for calibration or the sensitivity experiments.
            assert result.metadata == baseline.metadata
            for new, old in zip(result.summary["models"], baseline.summary["models"]):
                assert new["experiments"] == old["experiments"]


def test_training_is_deterministic_for_fixed_seed_and_seed_changes_are_reported(prepared):
    first, second = run(prepared), run(prepared)
    assert first.signals == second.signals and first.metadata == second.metadata
    assert pickle.dumps(first.estimators) == pickle.dumps(second.estimators)
    model = first.summary["models"][0]
    assert [s["seed"] for s in model["experiments"]["seed_stability"]] == list(CONFIG.stability_seeds)
    assert all(s["score_spearman"] > 0.5 for s in model["experiments"]["seed_stability"])
    other = run(prepared, replace(CONFIG, random_state=1, stability_seeds=(2,)))
    assert other.metadata["seed"] == 1
    assert other.metadata["models"]["5:service:orders-service"]["threshold"] != \
        first.metadata["models"]["5:service:orders-service"]["threshold"]


def test_threshold_is_strict_percentile_and_ties_are_flagged():
    scores = np.arange(100, dtype=float)
    selected = calibrate(scores, 99.0)
    assert selected["threshold"] == pytest.approx(98.01)
    assert selected["validation_alerts"] == 1 and not selected["scores_tied"]
    tied = calibrate(np.full(40, 0.5), 95.0)
    assert tied["threshold"] == 0.5 and tied["validation_alerts"] == 0 and tied["scores_tied"]


def test_preprocessing_drops_null_constant_features_and_imputes_training_medians():
    def window(count, p95, minute=0):
        angle = 0.0
        return {"business": {"record_count": count, "http_4xx_rate": 0.0, "http_5xx_rate": 0.0,
                             "level_rates": {"WARN": 0.0, "ERROR": 0.0}, "median_latency_ms": p95,
                             "p95_latency_ms": p95, "max_latency_ms": p95, "unique_ip_count": 1,
                             "auth_failure_rate": None, "upstream_latency": {"p95_ms": None},
                             "search_latency": {"p95_ms": None}},
                "health": {"record_count": 0, "failure_rate": None},
                "hour_sin": angle, "hour_cos": 1.0, "is_weekend": False}

    train = [window(10 + i, None if i == 0 else 100.0 + i) for i in range(5)]
    train[1]["business"]["p95_latency_ms"] = float("inf")
    prep = fit_preprocessing(train, "service")
    names = [f["name"] for f in prep["features"]]
    assert names == ["business.record_count", "business.median_latency_ms", "business.median_latency_ms__missing",
                     "business.p95_latency_ms", "business.p95_latency_ms__missing", "business.max_latency_ms",
                     "business.max_latency_ms__missing"]
    assert prep["dropped"]["business.auth_failure_rate"] == "all_null_in_training"
    assert prep["dropped"]["business.http_5xx_rate"] == "constant_in_training"
    assert prep["dropped"]["is_weekend"] == "constant_in_training"
    held_out = [window(1000, None), {"business": {}, "health": {}}]
    matrix = transform(held_out, prep)
    assert matrix.shape == (2, len(names)) and np.isfinite(matrix).all()
    assert matrix[0, names.index("business.p95_latency_ms")] == prep["medians"]["business.p95_latency_ms"]
    assert matrix[0, names.index("business.p95_latency_ms__missing")] == 1.0
    assert matrix[0, 0] == pytest.approx(np.log1p(1000))
    # Fitted statistics come from the supplied windows only.
    assert prep == fit_preprocessing(train, "service")


def test_insufficient_and_unknown_services_fall_back_without_borrowing_models(prepared):
    features, records = prepared
    windows = deepcopy(features.windows)
    extra = deepcopy(next(w for w in windows if w["split"] == "test" and w["scope"] == "service"))
    extra.update(service="new-service", entity_key="new-service")
    result = run(prepared, windows=windows + [extra])
    reasons = {m["model_key"]: m.get("reason") for m in result.summary["models"]}
    assert reasons["5:service:new-service"] == "unknown_service"
    assert "5:service:new-service" not in result.estimators
    sparse = run(prepared, replace(CONFIG, minimum_training_windows=1000))
    assert sparse.signals == [] and sparse.estimators == {}
    assert sparse.summary["unavailable_models"] == {"insufficient_training_windows": 1}
    unscored = {(u["split"], u["reason"]): u["count"] for u in sparse.summary["unscored_windows"]}
    assert unscored[("test", "insufficient_training_windows")] == 288


def test_unseen_windows_are_scored_with_the_frozen_bundle(prepared):
    features, _ = prepared
    result = run(prepared)
    metadata = json.loads(json.dumps(result.metadata))
    estimators = pickle.loads(pickle.dumps(result.estimators))
    normal = deepcopy(next(w for w in features.windows if w["window_start"] == "2025-02-05T08:00:00"
                           and w["scope"] == "service" and w["window_minutes"] == 5))
    extreme = deepcopy(normal)
    extreme["business"].update(http_5xx_rate=1.0, median_latency_ms=5000.0, p95_latency_ms=6000.0,
                               max_latency_ms=7000.0, level_rates={**normal["business"]["level_rates"], "ERROR": 1.0})
    sparse = deepcopy(normal)
    sparse["business"]["record_count"] = 1
    unknown = {**deepcopy(normal), "service": "other-service"}
    scored = score_windows(metadata, estimators, [normal, extreme, sparse, unknown])
    assert [s["status"] for s in scored] == ["scored", "scored", "low_support", "ml_unavailable"]
    assert scored[1]["score"] > scored[0]["score"] and scored[1]["alert"] is True
    window = next(s for s in result.signals if s["window_start"] == "2025-02-05T12:00:00")
    original = next(w for w in features.windows if w["window_start"] == window["window_start"]
                    and w["scope"] == "service" and w["window_minutes"] == 5)
    assert score_windows(metadata, estimators, [original])[0]["score"] == window["score"]


def test_rule_comparison_matches_scope_entity_and_window(prepared):
    rule = {"split": "test", "window_minutes": 5, "scope": "service", "service": "orders-service",
            "entity_key": "orders-service", "window_start": "2025-02-05T12:00:00", "detector_id": "http_error_burst"}
    other = {**rule, "window_start": "2025-02-05T13:00:00", "detector_id": "traffic_surge"}
    result = run(prepared, rule_signals=[rule, other])
    comparison = result.summary["models"][0]["alerts"]["test"]["rule_comparison"]
    assert comparison["both"] == 1 and comparison["rule_only"] == 1
    assert comparison["both_detectors"] == {"http_error_burst": 1}
    assert comparison["rule_only_detectors"] == {"traffic_surge": 1}
    assert comparison["ml_only"] == result.summary["models"][0]["alerts"]["test"]["count"] - 1
    assert run(prepared).summary["rule_comparison"] == "not_requested"


@pytest.mark.parametrize("change", [
    {"resolutions": [10]}, {"scopes": ["service", "service"]}, {"max_features": 0},
    {"contamination": 0.9}, {"threshold_percentile": 90.0}, {"stability_seeds": [42]},
    {"daypart_hours": 5}, {"minimum_training_windows": 0}, {"random_state": True},
])
def test_ml_config_rejects_invalid_values(tmp_path, change):
    path = tmp_path / "ml.json"
    path.write_text(json.dumps({**MLConfig().to_dict(), **change}))
    with pytest.raises(InputError):
        load_config(path, MLConfig)


def test_ml_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "ml.json"
    path.write_text(json.dumps({**MLConfig().to_dict(), "test_percentile": 99}))
    with pytest.raises(InputError, match="unknown"):
        load_config(path, MLConfig)


def test_model_schema_rejects_incompatible_saved_features_and_thresholds(prepared):
    from log_anomaly.ml import validate_model_schema

    result = run(prepared)
    validate_model_schema(result.metadata, result.estimators)
    for kind in ('version', 'order', 'path', 'threshold', 'identity', 'median'):
        metadata = deepcopy(result.metadata)
        model = next(iter(metadata['models'].values()))
        prep = model['preprocessing']
        if kind == 'version':
            metadata['bundle_version'] = 'future'
        elif kind == 'order':
            prep['features'].reverse()
        elif kind == 'path':
            prep['features'][0]['path'] = ['unknown']
        elif kind == 'threshold':
            model['threshold']['threshold'] = float('nan')
        elif kind == 'identity':
            model['service'] = 'other-service'
        else:
            prep['medians'].clear()
        with pytest.raises(InputError, match='incompatible model bundle'):
            validate_model_schema(metadata, result.estimators)


def test_new_endpoint_uses_only_its_own_service_scope_model(prepared):
    result = run(prepared, replace(CONFIG, scopes=('endpoint',)))
    window = deepcopy(next(w for w in prepared[0].windows if w['scope'] == 'endpoint'
                           and w['window_minutes'] == 5 and w['split'] == 'test'))
    new_endpoint = {**window, 'entity_key': '/new-endpoint', 'window_start': '2026-02-05T00:00:00',
                    'window_end': '2026-02-05T00:05:00'}
    other_service = {**new_endpoint, 'service': 'unseen-service'}
    scores = score_windows(result.metadata, result.estimators, [window, new_endpoint, other_service])
    assert scores[0] == scores[1]  # Same numerical features; identifiers do not enter the model.
    assert scores[2]['status'] == 'ml_unavailable' and scores[2]['score'] is None
