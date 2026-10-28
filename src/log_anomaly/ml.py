"""Phase 4 Isolation Forest: training-only preprocessing, validation thresholds, frozen scoring."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import median

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

from .config import MLConfig
from .ingest import NormalizedLog
from .rules import Evidence

DETECTOR_ID = "isolation_forest"
DETECTOR_VERSION = "1.0"
BUNDLE_VERSION = "1.0"
HELD_OUT = ("validation", "test")

# (path within a window, apply log1p). Identifiers, support flags, and training
# template novelty (in-sample membership, always zero in training) are excluded.
_BUSINESS = (
    (("business", "record_count"), True), (("business", "http_4xx_rate"), False),
    (("business", "http_5xx_rate"), False), (("business", "level_rates", "WARN"), False),
    (("business", "level_rates", "ERROR"), False), (("business", "median_latency_ms"), True),
    (("business", "p95_latency_ms"), True), (("business", "max_latency_ms"), True),
    (("business", "unique_ip_count"), True), (("business", "auth_failure_rate"), False),
    (("business", "upstream_latency", "p95_ms"), True), (("business", "search_latency", "p95_ms"), True),
)
_TIME = ((("hour_sin",), False), (("hour_cos",), False), (("is_weekend",), False))
FEATURES = {
    "service": _BUSINESS + ((("health", "record_count"), True), (("health", "failure_rate"), False)) + _TIME,
    "endpoint": _BUSINESS + _TIME,
    "auth_ip": ((("business", "auth_attempt_count"), True), (("business", "auth_failure_count"), True),
                (("business", "auth_failure_rate"), False), (("business", "http_4xx_rate"), False),
                (("business", "level_rates", "WARN"), False)) + _TIME,
}


@dataclass
class MLResult:
    signals: list[dict]
    estimators: dict[str, IsolationForest]
    metadata: dict
    summary: dict


def model_key(minutes: int, scope: str, service: str) -> str:
    return f"{minutes}:{scope}:{service}"


def raw_value(window: dict, path: tuple[str, ...]) -> float | None:
    value = window
    for part in path:
        value = value.get(part) if isinstance(value, dict) else None
    if value is None or isinstance(value, (dict, list, str)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _transform(value: float | None, log: bool) -> float | None:
    if value is None or (log and value < 0):
        return None
    return math.log1p(value) if log else value


def fit_preprocessing(windows: list[dict], scope: str) -> dict:
    """Fit imputation and feature selection on the given (training) windows only."""
    features, medians, raw_medians, scales, dropped = [], {}, {}, {}, {}
    for path, log in FEATURES[scope]:
        name = ".".join(path)
        raw = [raw_value(w, path) for w in windows]
        values = [_transform(v, log) for v in raw]
        present = [v for v in values if v is not None]
        if not present:
            dropped[name] = "all_null_in_training"
            continue
        center = median(present)
        imputed = [center if v is None else v for v in values]
        missing = len(present) != len(values)
        if min(imputed) == max(imputed) and not missing:
            dropped[name] = "constant_in_training"
            continue
        mad = median(abs(v - center) for v in present)
        spread = float(np.std(present))
        medians[name] = center
        raw_medians[name] = median(v for v in raw if v is not None)
        scales[name] = 1.4826 * mad if mad > 0 else spread if spread > 0 else 1.0
        if min(imputed) != max(imputed):
            features.append({"name": name, "path": list(path), "log1p": log, "kind": "value"})
        else:
            dropped[name] = "constant_in_training"
        if missing:
            features.append({"name": f"{name}__missing", "path": list(path), "log1p": log, "kind": "missing"})
    return {"features": features, "medians": medians, "raw_medians": raw_medians,
            "scales": scales, "dropped": dropped}


def transform(windows: list[dict], preprocessing: dict) -> np.ndarray:
    """Apply frozen preprocessing in persisted feature order; never refit."""
    rows = []
    for window in windows:
        row = []
        for feature in preprocessing["features"]:
            base = feature["name"].removesuffix("__missing")
            value = _transform(raw_value(window, tuple(feature["path"])), feature["log1p"])
            if feature["kind"] == "missing":
                row.append(float(value is None))
            else:
                row.append(preprocessing["medians"][base] if value is None else value)
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(len(windows), len(preprocessing["features"]))


def fit_forest(matrix: np.ndarray, config: MLConfig, seed: int) -> IsolationForest:
    # Contamination only sets sklearn's internal offset; reports use validation thresholds.
    return IsolationForest(n_estimators=config.n_estimators, max_samples=min(config.max_samples, len(matrix)),
                           max_features=config.max_features, contamination=config.contamination,
                           random_state=seed).fit(matrix)


def anomaly_scores(model: IsolationForest, matrix: np.ndarray) -> np.ndarray:
    """Negative score_samples: larger means more unusual."""
    return -model.score_samples(matrix) if len(matrix) else np.empty(0)


def calibrate(scores: np.ndarray, percentile: float) -> dict:
    tied = bool(scores.max() == scores.min())
    threshold = float(np.percentile(scores, percentile))
    alerts = int((scores > threshold).sum())
    return {"percentile": percentile, "threshold": threshold, "validation_alerts": alerts,
            "validation_alert_rate": alerts / len(scores), "scores_tied": tied}


def _distribution(scores: np.ndarray) -> dict:
    if not len(scores):
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": len(scores), "min": float(scores.min()), "max": float(scores.max()),
            **{f"p{q}": float(np.percentile(scores, q)) for q in (50, 95, 99)}}


def _agreement(primary: np.ndarray, other: np.ndarray, threshold: float, other_threshold: float) -> dict:
    a, b = set(np.flatnonzero(primary > threshold)), set(np.flatnonzero(other > other_threshold))
    rho = spearmanr(primary, other).statistic if len(primary) > 1 else float("nan")
    return {"validation_alerts": len(b), "alert_jaccard": len(a & b) / len(a | b) if a | b else None,
            "score_spearman": float(rho) if math.isfinite(rho) else None}


def _supported(window: dict, config: MLConfig) -> bool:
    return window["business"]["record_count"] >= config.minimum_records


def _feature_evidence(window: dict, preprocessing: dict, limit: int) -> list[dict]:
    """Descriptive deviations from training medians; not causal attributions."""
    rows = []
    for feature in preprocessing["features"]:
        if feature["kind"] != "value":
            continue
        name = feature["name"]
        observed = raw_value(window, tuple(feature["path"]))
        value = _transform(observed, feature["log1p"])
        if value is None:
            continue
        deviation = (value - preprocessing["medians"][name]) / preprocessing["scales"][name]
        rows.append({"feature": name, "observed": observed, "training_median": preprocessing["raw_medians"][name],
                     "robust_deviation": deviation})
    rows.sort(key=lambda r: (-abs(r["robust_deviation"]), r["feature"]))
    return rows[:limit]


def _daypart(window: dict, config: MLConfig) -> int:
    return datetime.fromisoformat(window["window_start"]).hour // config.daypart_hours


def fit_and_score(windows: list[dict], records: list[NormalizedLog], split_ranges: dict,
                  health_endpoints: tuple[str, ...], config: MLConfig,
                  rule_signals: list[dict] | None = None, *, training_only: bool = False) -> MLResult:
    """Fit one model per resolution, scope and known service from supported training windows.

    Preprocessing and trees see training windows only. Thresholds come from
    validation scores alone; test windows are scored with everything frozen.
    Unknown services or insufficient support produce ML-unavailable metadata.
    """
    groups = defaultdict(lambda: {split: [] for split in ("train", *HELD_OUT)})
    for window in sorted(windows, key=lambda w: (w["window_start"], w["service"], w["entity_key"])):
        if training_only and window["split"] == "test":
            continue
        if window["window_minutes"] in config.resolutions and window["scope"] in config.scopes:
            groups[(window["window_minutes"], window["scope"], window["service"])][window["split"]].append(window)
    evidence = Evidence(records, health_endpoints)
    rule_keys = None
    if rule_signals is not None:
        rule_keys = defaultdict(set)
        for s in rule_signals:
            if s["split"] in HELD_OUT:
                rule_keys[(s["window_minutes"], s["scope"], s["service"], s["entity_key"],
                           s["window_start"])].add(s["detector_id"])
    estimators, bundle, models, signals = {}, {}, [], []
    unscored = Counter()

    for (minutes, scope, service), splits in sorted(groups.items()):
        key = model_key(minutes, scope, service)
        supported = {split: [w for w in rows if _supported(w, config)] for split, rows in splits.items()}
        entry = {"model_key": key, "window_minutes": minutes, "scope": scope, "service": service,
                 "window_counts": {split: len(rows) for split, rows in splits.items()},
                 "supported_window_counts": {split: len(rows) for split, rows in supported.items()}}
        models.append(entry)
        for split in HELD_OUT:
            unscored[(key, split, "low_support")] += len(splits[split]) - len(supported[split])
        train = supported["train"]
        preprocessing = fit_preprocessing(train, scope) if len(train) >= config.minimum_training_windows else None
        if preprocessing is None or not preprocessing["features"]:
            entry["status"] = "unavailable"
            entry["reason"] = ("unknown_service" if not splits["train"] else
                               "insufficient_training_windows" if preprocessing is None else "no_usable_features")
            for split in HELD_OUT:
                unscored[(key, split, entry["reason"])] += len(supported[split])
            continue
        matrix = transform(train, preprocessing)
        model = fit_forest(matrix, config, config.random_state)
        scores = {"train": anomaly_scores(model, matrix)}
        matrices = {split: transform(supported[split], preprocessing) for split in HELD_OUT}
        for split in HELD_OUT:
            scores[split] = anomaly_scores(model, matrices[split])
        entry.update(preprocessing={k: preprocessing[k] for k in ("features", "dropped")},
                     max_samples_used=min(config.max_samples, len(train)),
                     score_distribution={split: _distribution(values) for split, values in scores.items()})
        validation = scores["validation"]
        if len(validation) < config.minimum_validation_windows:
            entry.update(status="unavailable", reason="insufficient_validation_windows")
            for split in HELD_OUT:
                unscored[(key, split, entry["reason"])] += len(supported[split])
            continue
        candidates = [calibrate(validation, p) for p in config.candidate_percentiles]
        selected = next(c for c in candidates if c["percentile"] == config.threshold_percentile)
        threshold = selected["threshold"]
        estimators[key] = model
        bundle[key] = {"window_minutes": minutes, "scope": scope, "service": service,
                       "training_window_count": len(train), "max_samples_used": min(config.max_samples, len(train)),
                       "preprocessing": preprocessing, "threshold": selected}
        entry.update(status="available", calibration={"candidates": candidates, "selected": selected})

        if training_only:
            # Sensitivity fits and held-out signal reports do not affect the saved estimator.
            continue

        # Sensitivity experiments use training and validation only, never test.
        stability = []
        for seed in config.stability_seeds:
            other = anomaly_scores(fit_forest(matrix, config, seed), matrices["validation"])
            stability.append({"seed": seed, **_agreement(
                validation, other, threshold, calibrate(other, config.threshold_percentile)["threshold"])})
        cutoff = float(np.percentile(scores["train"], config.training_filter_percentile))
        kept = [w for w, s in zip(train, scores["train"]) if s <= cutoff]
        filtered_prep = fit_preprocessing(kept, scope)
        filtered = anomaly_scores(fit_forest(transform(kept, filtered_prep), config, config.random_state),
                                  transform(supported["validation"], filtered_prep))
        entry["experiments"] = {
            "seed_stability": stability,
            "training_filter": {"training_score_percentile": config.training_filter_percentile,
                                "training_score_cutoff": cutoff, "retained_training_windows": len(kept),
                                "removed_training_windows": len(train) - len(kept),
                                **_agreement(validation, filtered, threshold,
                                             calibrate(filtered, config.threshold_percentile)["threshold"])},
        }
        entry["alerts"] = {}
        train_entities = {w["entity_key"] for w in splits["train"]}
        for split in HELD_OUT:
            flagged = [(w, float(s)) for w, s in zip(supported[split], scores[split]) if s > threshold]
            dayparts = Counter(str(_daypart(w, config)) for w, _ in flagged)
            alerts = {"count": len(flagged), "rate": len(flagged) / len(supported[split]) if supported[split] else None,
                      "weekend": sum(w["is_weekend"] for w, _ in flagged),
                      "by_daypart": dict(sorted(dayparts.items()))}
            if rule_keys is not None:
                mine = {(minutes, scope, service, w["entity_key"], w["window_start"]) for w, _ in flagged}
                theirs = {k for k in rule_keys if k[:3] == (minutes, scope, service)
                          and split_ranges[split]["start"] <= k[4] < split_ranges[split]["end"]}
                alerts["rule_comparison"] = {
                    "rule_signal_windows": len(theirs), "both": len(mine & theirs),
                    "ml_only": len(mine - theirs), "rule_only": len(theirs - mine),
                    "rule_only_detectors": dict(sorted(Counter(d for k in theirs - mine for d in rule_keys[k]).items())),
                    "both_detectors": dict(sorted(Counter(d for k in theirs & mine for d in rule_keys[k]).items())),
                }
            entry["alerts"][split] = alerts
            for window, score in flagged:
                signals.append(_signal(window, score, selected, preprocessing, key, len(train),
                                       window["entity_key"] in train_entities, evidence, config))

    unavailable = Counter(m["reason"] for m in models if m["status"] == "unavailable")
    metadata = {
        "schema_version": "1.0", "bundle_version": BUNDLE_VERSION, "detector_id": DETECTOR_ID,
        "detector_version": DETECTOR_VERSION, "fit_split": "train", "calibration_split": "validation",
        "split_ranges": split_ranges, "seed": config.random_state, "config": config.to_dict(),
        "score_definition": "negative IsolationForest.score_samples; larger is more unusual; alert when score > threshold",
        "models": bundle,
    }
    signals.sort(key=lambda s: (s["window_start"], s["service"], s["category"], s["signal_id"]))
    summary = {
        "schema_version": "1.0", "detector_id": DETECTOR_ID, "threshold_selection": "validation_percentile",
        "selected_percentile": config.threshold_percentile, "signal_count": len(signals),
        "signal_counts": [{"model_key": k, "split": split, "count": c} for (k, split), c in
                          sorted(Counter((s["model_key"], s["split"]) for s in signals).items())],
        "available_models": sum(m["status"] == "available" for m in models),
        "unavailable_models": dict(sorted(unavailable.items())),
        "unscored_windows": [{"model_key": k, "split": split, "reason": reason, "count": c}
                             for (k, split, reason), c in sorted(unscored.items()) if c],
        "rule_comparison": ("held-out windows matched by resolution, scope, service, entity and window start; "
                            "available ML models only" if rule_keys is not None else "not_requested"),
        "models": models,
        "limitations": [
            "No labels exist for the supplied logs; no accuracy, precision, recall, or F1 is computed.",
            "Percentile thresholds control validation alert volume; they are not false-positive guarantees.",
            "Validation alerts are in the calibration period; only test alerts are fully held out.",
            "Training windows are unlabelled and may contain anomalies; training is not declared clean.",
            "Training covers weekdays only; weekend test windows are a distribution shift.",
            "Low-support windows and unavailable models fall back to rules; they receive no ML score.",
        ],
    }
    return MLResult(signals, estimators, metadata, summary)


def _signal(window: dict, score: float, selected: dict, preprocessing: dict, key: str,
            training_count: int, seen: bool, evidence: Evidence, config: MLConfig) -> dict:
    population = "auth" if window["scope"] == "auth_ip" else "business"
    ids = evidence.resolve(window, population)
    support = window["business"]["record_count"]
    if len(ids) != support or len(set(ids)) != len(ids):
        raise ValueError("ML signal evidence does not match window support")
    top = _feature_evidence(window, preprocessing, config.top_features)
    observed = {f["name"]: raw_value(window, tuple(f["path"])) for f in preprocessing["features"]
                if f["kind"] == "value"}
    limitations = [
        "Isolation Forest ranks unusual feature combinations; it does not establish an incident or cause.",
        "Feature deviations are descriptive evidence, not causal attributions.",
        "ML-only candidate: review severity and low confidence unless independently corroborated.",
        f"Threshold is the validation p{selected['percentile']} score, an alert-volume control.",
    ]
    if window["split"] == "validation":
        limitations.append("Validation window: this period calibrated the threshold.")
    if window["is_weekend"]:
        limitations.append("Weekend window; training contained weekdays only.")
    if not seen:
        limitations.append("Entity not observed in training; scored by the service's pooled scope model.")
    if selected["scores_tied"]:
        limitations.append("Validation scores tied; calibration could not separate candidates.")
    deviations = "; ".join(f"{f['feature']} {f['observed']:g} (training median {f['training_median']:g})"
                           for f in top)
    explanation = (f"{window['service']} ({window['scope']}: {window['entity_key']}) during "
                   f"[{window['window_start']}, {window['window_end']}): anomaly score {score:.4f} exceeded "
                   f"validation p{selected['percentile']} threshold {selected['threshold']:.4f} across "
                   f"{support} records. Largest deviations from training medians: {deviations or 'none'}.")
    identity = [DETECTOR_ID, DETECTOR_VERSION, key, window["entity_key"], window["window_start"],
                window["window_end"], config.to_dict(), ids]
    return {
        "signal_id": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "detector_id": DETECTOR_ID, "detector_version": DETECTOR_VERSION, "category": "multivariate",
        "scope": window["scope"], "entity_key": window["entity_key"], "service": window["service"],
        "window_start": window["window_start"], "window_end": window["window_end"],
        "window_minutes": window["window_minutes"], "split": window["split"], "score": score,
        "threshold": {"score_gt": selected["threshold"], "percentile": selected["percentile"],
                      "calibration_split": "validation", "minimum_records": config.minimum_records},
        "observed_metrics": observed,
        "baseline_metrics": {"mode": "frozen_training", "model_key": key, "training_window_count": training_count,
                             "training_medians": preprocessing["raw_medians"]},
        "feature_evidence": top, "model_key": key, "entity_seen_in_training": seen,
        "support_count": support, "evidence_record_ids": ids, "limitations": limitations,
        "explanation": explanation,
    }


def score_windows(metadata: dict, estimators: dict, windows: list[dict]) -> list[dict]:
    """Score unseen windows with a frozen bundle; missing models are reported, never substituted."""
    config = MLConfig(**metadata["config"])
    results = [None] * len(windows)
    groups = defaultdict(list)
    for index, window in enumerate(windows):
        key = model_key(window["window_minutes"], window["scope"], window["service"])
        reason = ("incomplete_window" if window.get("incomplete_window") else
                  "ml_unavailable" if key not in metadata["models"] or key not in estimators else
                  "low_support" if not _supported(window, config) else None)
        if reason:
            results[index] = {"model_key": key, "status": reason, "score": None, "alert": None}
        else:
            groups[key].append(index)
    for key, indices in groups.items():
        model = metadata["models"][key]
        scores = anomaly_scores(estimators[key], transform([windows[i] for i in indices], model["preprocessing"]))
        for index, score in zip(indices, scores):
            results[index] = {"model_key": key, "status": "scored", "score": float(score),
                              "alert": bool(score > model["threshold"]["threshold"])}
    return results


def validate_model_schema(metadata: dict, estimators: dict) -> None:
    """Reject incompatible feature definitions before scoring trusted estimators."""
    from .config import InputError
    from sklearn.utils.validation import check_is_fitted

    try:
        if (metadata["bundle_version"] != BUNDLE_VERSION or metadata["schema_version"] != "1.0"
                or metadata["detector_version"] != DETECTOR_VERSION
                or metadata["detector_id"] != DETECTOR_ID):
            raise ValueError("unsupported model schema/version")
        config = MLConfig(**metadata["config"])
        for key, entry in metadata["models"].items():
            if (key != model_key(entry["window_minutes"], entry["scope"], entry["service"])
                    or entry["scope"] not in config.scopes or entry["window_minutes"] not in config.resolutions):
                raise ValueError("incompatible model identity")
            prep = entry["preprocessing"]
            allowed = []
            for path, log in FEATURES[entry["scope"]]:
                for kind in ("value", "missing"):
                    allowed.append({"name": ".".join(path) + ("__missing" if kind == "missing" else ""),
                                    "path": list(path), "log1p": log, "kind": kind})
            positions = [allowed.index(f) for f in prep["features"]]
            if not positions or positions != sorted(set(positions)):
                raise ValueError("incompatible feature order")
            for feature in prep["features"]:
                base = feature["name"].removesuffix("__missing")
                for field in ("medians", "raw_medians", "scales"):
                    if not math.isfinite(prep[field][base]):
                        raise ValueError("invalid preprocessing statistic")
                if prep["scales"][base] <= 0:
                    raise ValueError("invalid preprocessing scale")
            threshold = entry["threshold"]
            if (not math.isfinite(threshold["threshold"])
                    or threshold["percentile"] != config.threshold_percentile):
                raise ValueError("invalid calibrated threshold")
            estimator = estimators[key]
            if not isinstance(estimator, IsolationForest):
                raise ValueError("unexpected estimator type")
            check_is_fitted(estimator)
            if estimator.n_features_in_ != len(positions):
                raise ValueError("estimator feature count mismatch")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise InputError(f"incompatible model bundle: {exc}") from exc
