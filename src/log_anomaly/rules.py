"""Phase 3 rule signals and suppression accounting, without incident grouping."""

import hashlib
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from .baselines import History, Recurrence, daypart, fit_baselines, robust_score, series_key
from .config import DetectionConfig
from .features import _health_status
from .ingest import NormalizedLog

DETECTOR_VERSION = "1.0"


def _explanation(detector: str, window: dict, observed: dict, threshold: dict,
                 support: int, reference: dict | None, context: dict | None = None) -> str:
    if detector == "http_error_burst":
        reason = (f"{observed['http_5xx_count']} business HTTP 5xx responses among {support} valid statuses "
                  f"({observed['http_5xx_rate']:.1%}) met the count ≥{threshold['count_gte']} and "
                  f"rate ≥{threshold['rate_gte']:.1%} limits")
    elif detector == "http_error_rate_rise":
        reason = (f"{observed['http_5xx_count']} business HTTP 5xx responses among {support} valid statuses "
                  f"({observed['http_5xx_rate']:.1%}) had robust rate score "
                  f"{observed['robust_score']:.2f} >{threshold['robust_score_gt']}")
    elif detector == "repeated_auth_failures":
        reason = (f"{observed['auth_failure_count']} recognized login failures among {support} attempts "
                  f"({observed['auth_failure_rate']:.1%}) met the count ≥{threshold['failure_count_gte']} "
                  f"and rate ≥{threshold['failure_rate_gte']:.1%} limits")
    elif detector == "latency_degradation":
        reason = (f"business p95 latency {observed['p95_latency_ms']:.1f} ms across {support} valid latencies "
                  f"exceeded {threshold['p95_ms_gt']:.1f} ms and "
                  f"{threshold['baseline_multiplier_gt']}× historical median p95")
    elif detector == "service_error_increase":
        reason = (f"{observed['error_count']} business ERROR records among {support} records "
                  f"({observed['error_rate']:.1%}) had robust rate score "
                  f"{observed['robust_score']:.2f} >{threshold['robust_score_gt']}")
    elif detector == "traffic_surge":
        reason = (f"{support} business records had robust count score "
                  f"{observed['robust_score']:.2f} >{threshold['robust_score_gt']}, "
                  f"with count ≥{threshold['count_gte']}")
        if context and context["status"] == "available":
            reason += (f", and exceeded {context['ratio']}× the same-time-of-day maximum "
                       f"{context['same_time_max']} over {context['prior_days']} prior training days")
    else:
        reason = (f"{observed['failure_count']} explicit failed health checks among {support} known-status "
                  f"checks met the count ≥{threshold['failure_count_gte']} limit")
    baseline = (f" Historical median: {reference['median']:.3f}." if reference and
                reference["status"] == "available" else
                " Historical baseline unavailable." if reference else "")
    return (f"{window['service']} ({window['scope']}: {window['entity_key']}) during "
            f"[{window['window_start']}, {window['window_end']}): {reason}.{baseline}")


@dataclass
class DetectionResult:
    signals: list[dict]
    baselines: dict
    diagnostics: dict


class Evidence:
    """Resolve population support to source IDs using existing normalized records."""

    def __init__(self, records: list[NormalizedLog], health_endpoints: tuple[str, ...]):
        self.health_endpoints = health_endpoints
        self.records = defaultdict(list)
        for record in sorted(records, key=lambda r: (r.timestamp, r.source_record_number)):
            self.records[record.service].append(record)
        self.times = {service: [r.timestamp for r in rows] for service, rows in self.records.items()}

    def resolve(self, window: dict, population: str) -> list[str]:
        rows = self.records[window["service"]]
        times = self.times.get(window["service"], [])
        start = bisect_left(times, datetime.fromisoformat(window["window_start"]))
        end = bisect_left(times, datetime.fromisoformat(window["window_end"]))
        ids = []
        for record in rows[start:end]:
            if window["scope"] == "endpoint" and record.endpoint != window["entity_key"]:
                continue
            if window["scope"] == "auth_ip" and record.source_ip != window["entity_key"]:
                continue
            health = record.endpoint in self.health_endpoints
            if health != (population == "health"):
                continue
            if population == "status" and record.http_status is None:
                continue
            if population == "latency" and record.response_time_ms is None:
                continue
            if population == "auth" and (record.service != "auth-service" or
                    record.enrichment.authentication_outcome not in {"success", "failure"}):
                continue
            if population == "health" and record.http_status is None and _health_status(record) is None:
                continue
            ids.append(record.record_id)
        return ids


def detect_rules(windows: list[dict], records: list[NormalizedLog], split_ranges: dict,
                 health_endpoints: tuple[str, ...], config: DetectionConfig, *,
                 frozen_baselines: dict | None = None,
                 recurrence: Recurrence | None = None) -> DetectionResult:
    """Score training causally, then validation/test against frozen training only.

    Both resolutions run independently. Overlap is retained for Phase 5 grouping.
    Thresholds are explicit configuration, never selected from held-out values.
    """
    if frozen_baselines is not None and any(w["split"] == "train" for w in windows):
        raise ValueError("frozen inference cannot update training history")
    baselines = (frozen_baselines if frozen_baselines is not None
                 else fit_baselines(windows, split_ranges["train"], config))
    frozen = {series_key(r) + (r["metric"], r["daypart"]): r for r in baselines["references"]}
    history = History(config)
    empty_history = History(config)
    volumes = recurrence if recurrence is not None else Recurrence(config)
    evidence = Evidence(records, health_endpoints)
    signals = []
    counts = Counter()

    for window in sorted(windows, key=lambda w: (w["window_start"], series_key(w))):
        business, health = window["business"], window["health"]
        part = daypart(window["window_start"], config)
        references = {}

        def reference(metric):
            if metric not in references:
                if window["split"] == "train":
                    ref = history.reference(series_key(window), metric, window["window_start"], part)
                    mode = "preceding_training"
                else:
                    ref = frozen.get(series_key(window) + (metric, part))
                    if ref is None:
                        ref = empty_history.reference(series_key(window), metric, split_ranges["train"]["end"], part)
                    mode = "frozen_training"
                references[metric] = {**ref, "mode": mode}
            return references[metric]

        def assess(detector, category, population, value, support, minimum, threshold,
                   observed, triggered, metric=None, needs_baseline=False, suppress=None, context=None):
            ref = reference(metric) if metric else None
            if value is None:
                reason = "missing_metric"
            elif support < minimum:
                reason = "low_support"
            elif needs_baseline and ref["status"] != "available":
                reason = "baseline_unavailable"
            elif not triggered:
                reason = "below_threshold"
            else:
                # A suppression returns its reason, e.g. a recurring same-time-of-day pattern.
                reason = (suppress and suppress()) or "emitted"
            counts[(detector, window["service"], window["scope"], window["window_minutes"],
                    window["split"], part, reason)] += 1
            if reason != "emitted":
                return
            ids = evidence.resolve(window, population)
            if len(ids) != support or len(set(ids)) != len(ids):
                raise ValueError("signal evidence does not match window support")
            limitations = ["Candidate thresholds have not been calibrated for production.",
                           "Overlapping scopes/resolutions are separate signals, not independent incidents."]
            if ref:
                if ref["status"] != "available":
                    limitations.append("baseline_unavailable: absolute rule only.")
                elif not ref["daypart_matched"]:
                    limitations.append("Insufficient same-daypart history; broader history used with uncertainty.")
                if ref["mode"] == "frozen_training":
                    limitations.append("Frozen training reference; held-out activity does not update it.")
                limitations.append("Weekly seasonality is not established.")
            if category == "traffic":
                limitations.append("Volume change alone does not establish an outage.")
            if context and context["status"] != "available":
                limitations.append(f"Time-of-day recurrence context unavailable (fewer than "
                                   f"{context['minimum_days']} prior training days); a recurring daily "
                                   f"pattern cannot be excluded.")
            if category == "auth_failures":
                limitations.append("Repeated failures do not establish an attack or identify a user.")
            identity = [detector, DETECTOR_VERSION, *series_key(window), window["window_start"],
                        window["window_end"], config.to_dict(), ids]
            signal_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            signals.append({
                "signal_id": signal_id, "detector_id": detector, "detector_version": DETECTOR_VERSION,
                "category": category, "scope": window["scope"], "entity_key": window["entity_key"],
                "service": window["service"], "window_start": window["window_start"],
                "window_end": window["window_end"], "window_minutes": window["window_minutes"],
                "split": window["split"], "score": observed.get("robust_score", value), "threshold": threshold,
                "observed_metrics": observed,
                "baseline_metrics": ({metric: ref, **({"time_of_day_recurrence": context} if context else {})}
                                     if ref else {"status": "not_required"}),
                "support_count": support, "evidence_record_ids": ids, "limitations": limitations,
                "explanation": _explanation(detector, window, observed, threshold, support, ref, context),
            })

        if window["scope"] in {"service", "endpoint"}:
            rate = business["http_5xx_rate"]
            count = business["http_5xx_count"]
            support = business["valid_status_count"]
            observed = {"http_5xx_count": count, "http_5xx_rate": rate, "valid_status_count": support}
            assess("http_error_burst", "http_errors", "status", rate, support, config.minimum_statuses,
                   {"count_gte": config.http_minimum_errors, "rate_gte": config.http_error_rate,
                    "minimum_statuses": config.minimum_statuses}, observed,
                   rate is not None and count >= config.http_minimum_errors and rate >= config.http_error_rate,
                   "http_5xx_rate")
            score = robust_score(rate, reference("http_5xx_rate")) if rate is not None else None
            assess("http_error_rate_rise", "http_errors", "status", rate, support, config.minimum_statuses,
                   {"count_gte": config.http_minimum_errors, "robust_score_gt": config.http_robust_score,
                    "minimum_statuses": config.minimum_statuses}, {**observed, "robust_score": score},
                   count >= config.http_minimum_errors and score is not None and score > config.http_robust_score,
                   "http_5xx_rate", True)
            p95 = business["p95_latency_ms"]
            ref = reference("p95_latency_ms")
            assess("latency_degradation", "latency", "latency", p95, business["valid_latency_count"],
                   config.minimum_latencies,
                   {"p95_ms_gt": config.latency_p95_ms, "baseline_multiplier_gt": config.latency_multiplier,
                    "minimum_latencies": config.minimum_latencies},
                   {"p95_latency_ms": p95, "valid_latency_count": business["valid_latency_count"]},
                   p95 is not None and ref["median"] is not None and p95 > config.latency_p95_ms
                   and p95 > config.latency_multiplier * ref["median"], "p95_latency_ms", True)

        if window["scope"] == "service":
            count, rate = business["level_counts"]["ERROR"], business["level_rates"]["ERROR"]
            score = robust_score(rate, reference("error_rate")) if rate is not None else None
            assess("service_error_increase", "service_errors", "business", rate, business["record_count"],
                   config.minimum_records,
                   {"count_gte": config.error_minimum_count, "robust_score_gt": config.error_robust_score,
                    "minimum_records": config.minimum_records},
                   {"error_count": count, "error_rate": rate, "robust_score": score},
                   count >= config.error_minimum_count and score is not None and score > config.error_robust_score,
                   "error_rate", True)
            count = business["record_count"]
            score = robust_score(count, reference("record_count"))
            recurrence = {}

            def recurring():
                # Prior training days only: preceding days in training, frozen at training end for held-out.
                cutoff = window["window_start"] if window["split"] == "train" else split_ranges["train"]["end"]
                recurrence.update(volumes.reference(series_key(window), window["window_start"], cutoff))
                if (recurrence["status"] == "available"
                        and count <= config.traffic_recurrence_ratio * recurrence["same_time_max"]):
                    return "recurring_time_of_day"
                return None

            enabled = config.traffic_recurrence_days > 0
            assess("traffic_surge", "traffic", "business", count, count, config.minimum_records,
                   {"count_gte": config.traffic_minimum_count, "robust_score_gt": config.traffic_robust_score,
                    "minimum_records": config.minimum_records,
                    **({"same_time_of_day_ratio_gt": config.traffic_recurrence_ratio} if enabled else {})},
                   {"record_count": count, "robust_score": score},
                   count >= config.traffic_minimum_count and score is not None and score > config.traffic_robust_score,
                   "record_count", True, recurring if enabled else None, recurrence if enabled else None)
            if window["window_minutes"] == 15:
                assess("health_failure", "health", "health", health["failure_rate"],
                       health["failure_valid_count"], config.health_minimum_failures,
                       {"failure_count_gte": config.health_minimum_failures, "window_minutes": 15},
                       {"failure_count": health["failure_count"], "failure_rate": health["failure_rate"],
                        "failure_valid_count": health["failure_valid_count"]},
                       health["failure_count"] >= config.health_minimum_failures)

        if window["scope"] == "auth_ip" and window["service"] == "auth-service":
            rate = business["auth_failure_rate"]
            assess("repeated_auth_failures", "auth_failures", "auth", rate, business["auth_attempt_count"],
                   config.minimum_auth_attempts,
                   {"failure_count_gte": config.auth_minimum_failures, "failure_rate_gte": config.auth_failure_rate,
                    "minimum_attempts": config.minimum_auth_attempts},
                   {"auth_failure_count": business["auth_failure_count"], "auth_failure_rate": rate,
                    "auth_attempt_count": business["auth_attempt_count"]},
                   rate is not None and business["auth_failure_count"] >= config.auth_minimum_failures
                   and rate >= config.auth_failure_rate)
        if window["split"] == "train":
            history.add(window)
            volumes.add(window)

    fields = ("detector_id", "service", "scope", "window_minutes", "split", "daypart", "reason")
    diagnostics = {
        "schema_version": "1.0", "baseline_mode": "frozen_training",
        "training_scoring": "preceding_training", "threshold_selection": "explicit_config",
        "signal_count": len(signals), "decision_counts": [
            {**dict(zip(fields, key)), "count": count} for key, count in sorted(counts.items())
        ],
        "limitations": [
            "Candidate counts on unlabelled logs are not precision, recall, or false-positive rates.",
            "No incident grouping or ML is performed; overlapping evidence remains explicit.",
            "Missing checks and empty windows do not establish service failure or collection coverage.",
        ],
    }
    signals.sort(key=lambda s: (s["window_start"], s["service"], s["category"], s["signal_id"]))
    return DetectionResult(signals, baselines, diagnostics)
