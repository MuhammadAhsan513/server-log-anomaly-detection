"""Strict stage-specific configuration."""

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import SCHEMA_VERSION


class InputError(ValueError):
    """An input or configuration cannot be processed reliably."""


@dataclass(frozen=True)
class IngestConfig:
    schema_version: str = SCHEMA_VERSION
    source_timezone: str = "unspecified"
    max_rejected_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise InputError("unsupported schema_version")
        if not isinstance(self.source_timezone, str) or not self.source_timezone:
            raise InputError("source_timezone must be unspecified or an IANA timezone")
        if self.source_timezone != "unspecified":
            try:
                ZoneInfo(self.source_timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise InputError("unknown source_timezone") from exc
        rate = self.max_rejected_fraction
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or not 0 <= rate <= 1):
            raise InputError("max_rejected_fraction must be a finite number in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureConfig(IngestConfig):
    window_minutes: int = 5
    alternate_window_minutes: int = 15
    split_start_day: str | None = None
    train_days: int = 4
    validation_days: int = 1
    health_endpoints: tuple[str, ...] = ("/health",)
    minimum_records: int = 20

    def __post_init__(self) -> None:
        super().__post_init__()
        for key in ("window_minutes", "alternate_window_minutes", "train_days",
                    "validation_days", "minimum_records"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise InputError(f"{key} must be a positive integer")
        if (self.window_minutes, self.alternate_window_minutes) != (5, 15):
            raise InputError("Phase 2 supports five-minute and fifteen-minute windows")
        if self.split_start_day is not None:
            try:
                if date.fromisoformat(self.split_start_day).isoformat() != self.split_start_day:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise InputError("split_start_day must be YYYY-MM-DD or null") from exc
        endpoints = self.health_endpoints
        if (not isinstance(endpoints, (list, tuple)) or not endpoints
                or any(not isinstance(e, str) or not e.startswith("/") for e in endpoints)
                or len(set(endpoints)) != len(endpoints)):
            raise InputError("health_endpoints must be distinct absolute endpoint paths")
        object.__setattr__(self, "health_endpoints", tuple(endpoints))

    def to_dict(self) -> dict:
        return {**asdict(self), "health_endpoints": list(self.health_endpoints)}


@dataclass(frozen=True)
class DetectionConfig:
    schema_version: str = SCHEMA_VERSION
    history_hours: int = 24
    minimum_history_windows: int = 12
    daypart_hours: int = 6
    rate_epsilon: float = 0.01
    latency_epsilon_ms: float = 1.0
    count_epsilon: float = 1.0
    minimum_statuses: int = 20
    http_minimum_errors: int = 5
    http_error_rate: float = 0.20
    http_robust_score: float = 4.0
    minimum_auth_attempts: int = 10
    auth_minimum_failures: int = 10
    auth_failure_rate: float = 0.80
    minimum_latencies: int = 20
    latency_p95_ms: float = 1000.0
    latency_multiplier: float = 3.0
    minimum_records: int = 20
    error_minimum_count: int = 5
    error_robust_score: float = 4.0
    traffic_minimum_count: int = 30
    traffic_robust_score: float = 4.0
    # Same-time-of-day context from prior training days; 0 days disables it.
    traffic_recurrence_days: int = 7
    traffic_recurrence_minimum_days: int = 2
    traffic_recurrence_tolerance_minutes: int = 15
    traffic_recurrence_ratio: float = 1.25
    health_minimum_failures: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise InputError("unsupported schema_version")
        nonnegative_keys = {"traffic_recurrence_days", "traffic_recurrence_tolerance_minutes"}
        integer_keys = {
            "history_hours", "minimum_history_windows", "daypart_hours", "minimum_statuses",
            "http_minimum_errors", "minimum_auth_attempts", "auth_minimum_failures",
            "minimum_latencies", "minimum_records", "error_minimum_count",
            "traffic_minimum_count", "health_minimum_failures", "traffic_recurrence_minimum_days",
        }
        for key, value in asdict(self).items():
            if key == "schema_version":
                continue
            if key in nonnegative_keys:
                if type(value) is not int or value < 0:
                    raise InputError(f"{key} must be a nonnegative integer")
            elif key in integer_keys:
                if type(value) is not int or value <= 0:
                    raise InputError(f"{key} must be a positive integer")
            elif (isinstance(value, bool) or not isinstance(value, (float, int))
                  or not math.isfinite(value) or value <= 0):
                raise InputError(f"{key} must be finite and positive")
        if self.daypart_hours > 24 or 24 % self.daypart_hours:
            raise InputError("daypart_hours must divide 24")
        if self.history_hours > 24 * 365 or self.traffic_recurrence_days > 365:
            raise InputError("history_hours and traffic_recurrence_days must not exceed one year")
        if self.traffic_recurrence_tolerance_minutes >= 12 * 60:
            raise InputError("traffic_recurrence_tolerance_minutes must be under twelve hours")
        if any(getattr(self, key) > 1 for key in ("http_error_rate", "auth_failure_rate", "rate_epsilon")):
            raise InputError("rates must be in (0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MLConfig:
    schema_version: str = SCHEMA_VERSION
    resolutions: tuple[int, ...] = (5, 15)
    scopes: tuple[str, ...] = ("service", "endpoint", "auth_ip")
    minimum_records: int = 20
    minimum_training_windows: int = 100
    minimum_validation_windows: int = 20
    n_estimators: int = 200
    max_samples: int = 256
    max_features: float = 1.0
    contamination: str | float = "auto"
    random_state: int = 42
    candidate_percentiles: tuple[float, ...] = (95.0, 97.5, 99.0)
    threshold_percentile: float = 99.0
    stability_seeds: tuple[int, ...] = (43, 44, 45)
    training_filter_percentile: float = 99.0
    daypart_hours: int = 6
    top_features: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise InputError("unsupported schema_version")
        for key, allowed in (("resolutions", {5, 15}), ("scopes", {"service", "endpoint", "auth_ip"})):
            values = getattr(self, key)
            if (not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values)
                    or any(isinstance(v, bool) or v not in allowed for v in values)):
                raise InputError(f"{key} must be distinct values from {sorted(allowed)}")
            object.__setattr__(self, key, tuple(values))
        for key in ("minimum_records", "minimum_training_windows", "minimum_validation_windows",
                    "n_estimators", "max_samples", "daypart_hours", "top_features"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise InputError(f"{key} must be a positive integer")
        seeds = self.stability_seeds
        if (type(self.random_state) is not int or self.random_state < 0
                or not isinstance(seeds, (list, tuple)) or len(set(seeds)) != len(seeds)
                or any(type(s) is not int or s < 0 or s == self.random_state for s in seeds)):
            raise InputError("random_state and distinct stability_seeds must be other nonnegative integers")
        object.__setattr__(self, "stability_seeds", tuple(seeds))

        def number(value):
            return (not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value))

        if not number(self.max_features) or not 0 < self.max_features <= 1:
            raise InputError("max_features must be a fraction in (0, 1]")
        if self.contamination != "auto" and (not number(self.contamination)
                                             or not 0 < self.contamination <= 0.5):
            raise InputError("contamination must be 'auto' or in (0, 0.5]")
        percentiles = self.candidate_percentiles
        if (not isinstance(percentiles, (list, tuple)) or not percentiles
                or len(set(percentiles)) != len(percentiles)
                or any(not number(p) or not 0 < p < 100 for p in percentiles)):
            raise InputError("candidate_percentiles must be distinct numbers in (0, 100)")
        object.__setattr__(self, "candidate_percentiles", tuple(percentiles))
        if self.threshold_percentile not in self.candidate_percentiles:
            raise InputError("threshold_percentile must be one of candidate_percentiles")
        if not number(self.training_filter_percentile) or not 0 < self.training_filter_percentile < 100:
            raise InputError("training_filter_percentile must be in (0, 100)")
        if self.daypart_hours > 24 or 24 % self.daypart_hours:
            raise InputError("daypart_hours must divide 24")

    def to_dict(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class IncidentConfig:
    schema_version: str = SCHEMA_VERSION
    merge_gap_minutes: int = 5
    sample_evidence_limit: int = 5
    high_minimum_minutes: int = 15
    high_http_5xx_rate: float = 0.5
    high_minimum_statuses: int = 30
    review_categories: tuple[str, ...] = ("traffic",)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise InputError("unsupported schema_version")
        gap = self.merge_gap_minutes
        if type(gap) is not int or gap < 0 or gap % 5:
            raise InputError("merge_gap_minutes must be a nonnegative multiple of five")
        for key in ("sample_evidence_limit", "high_minimum_minutes", "high_minimum_statuses"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise InputError(f"{key} must be a positive integer")
        rate = self.high_http_5xx_rate
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or not 0 < rate <= 1):
            raise InputError("high_http_5xx_rate must be in (0, 1]")
        allowed = {"http_errors", "service_errors", "auth_failures", "latency", "traffic", "health"}
        categories = self.review_categories
        if (not isinstance(categories, (list, tuple)) or len(set(categories)) != len(categories)
                or any(c not in allowed for c in categories)):
            raise InputError(f"review_categories must be distinct values from {sorted(allowed)}")
        object.__setattr__(self, "review_categories", tuple(categories))

    def to_dict(self) -> dict:
        return {**asdict(self), "review_categories": list(self.review_categories)}


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: str = SCHEMA_VERSION
    iou_threshold: float = 0.3
    sensitivity_iou_thresholds: tuple[float, ...] = (0.1, 0.5)
    splits: tuple[str, ...] = ("validation", "test")
    primary_split: str = "test"

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise InputError("unsupported schema_version")

        def fraction(value):
            return (not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value) and 0 < value <= 1)

        thresholds = self.sensitivity_iou_thresholds
        if not fraction(self.iou_threshold) or (
                not isinstance(thresholds, (list, tuple)) or len(set(thresholds)) != len(thresholds)
                or any(not fraction(t) or t == self.iou_threshold for t in thresholds)):
            raise InputError("IoU thresholds must be distinct numbers in (0, 1]")
        splits = self.splits
        if (not isinstance(splits, (list, tuple)) or not splits or len(set(splits)) != len(splits)
                or any(s not in {"train", "validation", "test"} for s in splits)
                or self.primary_split not in splits):
            raise InputError("splits must be distinct train/validation/test names including primary_split")
        object.__setattr__(self, "sensitivity_iou_thresholds", tuple(thresholds))
        object.__setattr__(self, "splits", tuple(splits))

    def to_dict(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class RunConfig:
    """Resolved stage configurations; both detectors are required for delivery."""

    schema_version: str
    features: FeatureConfig
    rules: DetectionConfig
    ml: MLConfig
    incidents: IncidentConfig

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise InputError("unsupported schema_version")
        for name, kind in (("features", FeatureConfig), ("rules", DetectionConfig),
                           ("ml", MLConfig), ("incidents", IncidentConfig)):
            value = getattr(self, name)
            if not isinstance(value, kind):
                object.__setattr__(self, name, _config_from_dict(value, kind))

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version,
                **{name: getattr(self, name).to_dict() for name in ("features", "rules", "ml", "incidents")}}


def _config_from_dict(data: object, config_type: type):
    if not isinstance(data, dict):
        raise InputError("configuration must be an object")
    expected = set(config_type.__dataclass_fields__)
    if set(data) != expected:
        raise InputError(f"configuration keys: missing={sorted(expected - set(data))}, "
                         f"unknown={sorted(set(data) - expected)}")
    try:
        return config_type(**data)
    except (TypeError, ValueError) as exc:
        raise InputError(f"invalid {config_type.__name__}: {exc}") from exc


def load_config(path: Path, config_type: type[IngestConfig] | type[DetectionConfig] | type[MLConfig]
                | type[IncidentConfig] | type[EvaluationConfig] | type[RunConfig] = IngestConfig
                ) -> IngestConfig | DetectionConfig | MLConfig | IncidentConfig | EvaluationConfig | RunConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read configuration: {exc}") from exc
    return _config_from_dict(data, config_type)
