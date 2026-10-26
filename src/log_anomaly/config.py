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


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


def load_config(path: Path, config_type: type[IngestConfig] = IngestConfig) -> IngestConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read configuration: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError("configuration must be an object")
    expected = set(config_type.__dataclass_fields__)
    if set(data) != expected:
        raise InputError(f"configuration keys: missing={sorted(expected - set(data))}, "
                         f"unknown={sorted(set(data) - expected)}")
    return config_type(**data)
