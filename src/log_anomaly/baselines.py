"""Supported, causal median/MAD references; held-out scoring never updates them."""

from collections import defaultdict, deque
from datetime import datetime, timedelta
from statistics import median

from .config import DetectionConfig

METRICS = ("http_5xx_rate", "error_rate", "p95_latency_ms", "record_count")


def series_key(window: dict) -> tuple:
    return (window["window_minutes"], window["service"], window["scope"], window["entity_key"])


def metric_value(window: dict, metric: str, config: DetectionConfig) -> tuple:
    business = window["business"]
    if metric == "http_5xx_rate":
        return business[metric], business["valid_status_count"], config.minimum_statuses
    if metric == "p95_latency_ms":
        return business[metric], business["valid_latency_count"], config.minimum_latencies
    value = business["level_rates"]["ERROR"] if metric == "error_rate" else business[metric]
    return value, business["record_count"], config.minimum_records


def daypart(timestamp: str, config: DetectionConfig) -> int:
    return datetime.fromisoformat(timestamp).hour // config.daypart_hours


class History:
    """One history per entity, resolution and metric, populated after scoring."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.values = defaultdict(deque)

    def add(self, window: dict) -> None:
        if window["split"] != "train":
            raise ValueError("only training windows may update history")
        if window["scope"] not in {"service", "endpoint"}:
            return
        for metric in METRICS:
            value, support, minimum = metric_value(window, metric, self.config)
            if value is not None and support >= minimum:
                self.values[series_key(window) + (metric,)].append((
                    window["window_start"], window["window_end"], value,
                ))

    def reference(self, key: tuple, metric: str, cutoff: str, part: int) -> dict:
        config = self.config
        lower = (datetime.fromisoformat(cutoff) - timedelta(hours=config.history_hours)).isoformat()
        history = self.values.get(key + (metric,), deque())
        while history and history[0][0] < lower:
            history.popleft()
        # Explicit end guard also excludes overlapping/current windows.
        candidates = [r for r in history if r[1] <= cutoff]
        same_part = [r for r in candidates if daypart(r[0], config) == part]
        matched = len(same_part) >= config.minimum_history_windows
        selected = same_part if matched else candidates
        available = len(selected) >= config.minimum_history_windows
        center = median(r[2] for r in selected) if available else None
        mad = median(abs(r[2] - center) for r in selected) if available else None
        epsilon = (config.latency_epsilon_ms if metric == "p95_latency_ms" else
                   config.count_epsilon if metric == "record_count" else config.rate_epsilon)
        return {
            "status": "available" if available else "baseline_unavailable",
            "history_window_count": len(selected), "minimum_history_windows": config.minimum_history_windows,
            "median": center, "mad": mad,
            "scale": max(1.4826 * mad, epsilon) if available else None,
            "epsilon": epsilon, "cutoff": cutoff, "history_start": lower,
            "first_window_start": selected[0][0] if selected else None,
            "last_window_end": selected[-1][1] if selected else None,
            "daypart": part, "daypart_matched": matched,
        }


class Recurrence:
    """Same-clock-time business volume on prior training days (service scope only)."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.counts = defaultdict(dict)

    def add(self, window: dict) -> None:
        if window["split"] != "train":
            raise ValueError("only training windows may update history")
        if window["scope"] == "service":
            self.counts[series_key(window)][window["window_start"]] = (
                window["window_end"], window["business"]["record_count"])

    def reference(self, key: tuple, window_start: str, cutoff: str) -> dict:
        """Maximum count within the tolerance at the same clock time on earlier days ending by cutoff."""
        config = self.config
        current = datetime.fromisoformat(window_start)
        lower = (datetime.fromisoformat(cutoff) - timedelta(days=config.traffic_recurrence_days)).isoformat()
        tolerance = config.traffic_recurrence_tolerance_minutes
        counts = self.counts.get(key, {})
        days, peak = set(), None
        last_day = (current - datetime.fromisoformat(lower)).days + 1
        for back in range(1, last_day + 1):
            for offset in range(-tolerance, tolerance + 1, key[0]):
                start = (current - timedelta(days=back, minutes=offset)).isoformat()
                found = counts.get(start)
                if found is None or start < lower or found[0] > cutoff:
                    continue
                days.add(back)
                peak = found[1] if peak is None else max(peak, found[1])
        available = len(days) >= config.traffic_recurrence_minimum_days
        return {"status": "available" if available else "recurrence_unavailable", "prior_days": len(days),
                "minimum_days": config.traffic_recurrence_minimum_days, "same_time_max": peak if available else None,
                "tolerance_minutes": tolerance, "ratio": config.traffic_recurrence_ratio, "cutoff": cutoff,
                "history_start": lower}


def fit_baselines(windows: list[dict], train_range: dict, config: DetectionConfig) -> dict:
    """Freeze final training history for every daypart; ignore held-out values."""
    history = History(config)
    keys = set()
    for window in sorted(windows, key=lambda w: (w["window_start"], series_key(w))):
        if window["split"] != "train" or window["scope"] not in {"service", "endpoint"}:
            continue
        if not (train_range["start"] <= window["window_start"] < window["window_end"] <= train_range["end"]):
            raise ValueError("training window outside training range")
        keys.add(series_key(window))
        history.add(window)
    return {
        "schema_version": "1.0", "mode": "frozen_training", "fit_split": "train",
        "train_range": train_range, "history_hours": config.history_hours,
        "references": [
            {"window_minutes": key[0], "service": key[1], "scope": key[2], "entity_key": key[3],
             "metric": metric, **history.reference(key, metric, train_range["end"], part)}
            for key in sorted(keys) for metric in METRICS for part in range(24 // config.daypart_hours)
        ],
    }


def robust_score(value: float, reference: dict) -> float | None:
    if reference["status"] != "available":
        return None
    return (value - reference["median"]) / reference["scale"]
