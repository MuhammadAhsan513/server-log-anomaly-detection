"""Phase 5 incident grouping, severity/confidence policy, explanations, and report exports."""

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations

from .config import IncidentConfig, InputError
from .features import _health_status
from .ingest import NormalizedLog

INCIDENT_VERSION = "1.0"
ML_DETECTOR = "isolation_forest"
RULE_CATEGORIES = ("http_errors", "service_errors", "auth_failures", "latency", "traffic", "health")
CATEGORIES = RULE_CATEGORIES + ("multivariate",)
SEVERITIES = ("review", "warning", "high")
CONFIDENCES = ("low", "medium", "high")
FAILURE_CATEGORIES = frozenset({"http_errors", "service_errors", "latency", "health"})
CHART = "charts/incidents.png"
# ML feature deviations above the training median that describe a rule category.
FEATURE_CATEGORIES = {
    "business.http_5xx_rate": "http_errors", "business.level_rates.ERROR": "service_errors",
    "business.median_latency_ms": "latency", "business.p95_latency_ms": "latency",
    "business.max_latency_ms": "latency", "business.upstream_latency.p95_ms": "latency",
    "business.search_latency.p95_ms": "latency", "business.record_count": "traffic",
    "business.auth_failure_rate": "auth_failures", "business.auth_failure_count": "auth_failures",
    "health.failure_rate": "health",
}
SIGNAL_FIELDS = frozenset({
    "signal_id", "detector_id", "detector_version", "category", "scope", "entity_key", "service",
    "window_start", "window_end", "window_minutes", "split", "score", "threshold", "observed_metrics",
    "baseline_metrics", "support_count", "evidence_record_ids", "limitations", "explanation",
})
CSV_FIELDS = (
    "incident_id", "split", "start_time", "end_time", "first_evidence_time", "last_evidence_time",
    "services", "category", "entity", "severity", "confidence", "detector_ids", "signal_ids",
    "observed_metrics", "baseline_metrics", "evidence", "explanation", "root_cause_hypothesis",
    "limitations", "related_incident_ids",
)
MODES = {"train": "training period: rules scored against preceding training history; ML not scored",
         "validation": "validation day: rules frozen; ML threshold calibrated on this day",
         "test": "held-out test: rules and ML frozen"}
INVESTIGATION = {
    "http_errors": "Inspect the 5xx samples and the affected endpoint/upstream for the window.",
    "service_errors": "Read the ERROR messages and compare with concurrent HTTP or latency incidents.",
    "auth_failures": "Review the IP's login attempts; consider misconfiguration, shared proxy, or abuse.",
    "latency": "Compare slow samples with upstream latency and concurrent error incidents.",
    "traffic": "Check whether the volume is expected (batch, crawler, retry storm) before escalating.",
    "health": "Verify health-check failures against service availability and deployment events.",
    "multivariate": "Review the deviating features and samples; dismiss if the combination is benign.",
}


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _ordinal(record_id: str) -> int:
    return int(record_id.rsplit(":", 1)[1])


def _order(signal: dict) -> tuple:
    return signal["window_start"], signal["window_end"], signal["detector_id"], signal["signal_id"]


def _entity(signal: dict) -> tuple:
    return signal["scope"], signal["service"], signal["entity_key"], signal["split"]


def validate_signals(rule_signals: list[dict], ml_signals: list[dict], split_ranges: dict,
                     record_ids: set[str]) -> None:
    """Reject signals that cannot be grouped against the shared feature/evidence contract."""
    seen = set()
    for source, signals in (("rules", rule_signals), ("ml", ml_signals)):
        for s in signals:
            missing = SIGNAL_FIELDS - s.keys()
            if missing:
                raise InputError(f"{source} signal lacks fields: {sorted(missing)}")
            if s["signal_id"] in seen:
                raise InputError(f"duplicate signal identity: {s['signal_id']}")
            seen.add(s["signal_id"])
            is_ml = s["detector_id"] == ML_DETECTOR
            if (is_ml != (source == "ml") or s["category"] not in CATEGORIES
                    or (s["category"] == "multivariate") != is_ml):
                raise InputError(f"{source} signal has an incompatible detector or category")
            if is_ml and (s["split"] == "train" or not isinstance(s.get("feature_evidence"), list)):
                raise InputError("ML signal lacks held-out split or feature evidence")
            bounds = split_ranges.get(s["split"])
            if (s["scope"] not in {"service", "endpoint", "auth_ip"} or s["window_minutes"] not in (5, 15)
                    or bounds is None
                    or not bounds["start"] <= s["window_start"] < s["window_end"] <= bounds["end"]
                    or _time(s["window_end"]) - _time(s["window_start"]) != timedelta(minutes=s["window_minutes"])):
                raise InputError(f"{source} signal window/scope does not match the feature run")
            ids = s["evidence_record_ids"]
            if len(ids) != s["support_count"] or len(set(ids)) != len(ids) or not record_ids.issuperset(ids):
                raise InputError(f"{source} signal evidence does not resolve to source records")


def ml_categories(signal: dict) -> list[str]:
    """Rule categories described by an ML signal's top feature increases, in evidence order."""
    categories = []
    for feature in signal["feature_evidence"]:
        category = FEATURE_CATEGORIES.get(feature["feature"])
        if category and feature["robust_deviation"] > 0 and category not in categories:
            categories.append(category)
    return categories


def _merge(signals: list[dict], gap: timedelta) -> list[dict]:
    """Merge overlapping, adjacent, or gap-bounded windows within one partition."""
    groups = []
    for s in sorted(signals, key=_order):
        start, end = _time(s["window_start"]), _time(s["window_end"])
        if groups and start <= groups[-1]["end"] + gap:
            groups[-1]["signals"].append(s)
            groups[-1]["end"] = max(groups[-1]["end"], end)
        else:
            groups.append({"start": start, "end": end, "signals": [s]})
    return groups


def group_incidents(rule_signals: list[dict], ml_signals: list[dict], records: list[NormalizedLog],
                    config: IncidentConfig) -> list[dict]:
    """Group signals by category, entity and split; attach ML only where its features corroborate."""
    gap = timedelta(minutes=config.merge_gap_minutes)
    partitions = defaultdict(list)
    for s in rule_signals:
        partitions[(s["category"], *_entity(s))].append(s)
    groups = {key: _merge(signals, gap) for key, signals in partitions.items()}
    unattached = defaultdict(list)
    for s in sorted(ml_signals, key=_order):
        start, end = _time(s["window_start"]), _time(s["window_end"])
        target = None
        for category in ml_categories(s):
            target = next((g for g in groups.get((category, *_entity(s)), [])
                           if g["start"] < end and start < g["end"]), None)
            if target:
                break
        if target is None:
            unattached[("multivariate", *_entity(s))].append(s)
            continue
        target["signals"].append(s)
        target["start"], target["end"] = min(target["start"], start), max(target["end"], end)
    groups.update({key: _merge(signals, gap) for key, signals in unattached.items()})
    by_id = {r.record_id: r for r in records}
    incidents = [_incident(key, group, by_id, config) for key, rows in groups.items() for group in rows]
    _link(incidents, gap)
    incidents.sort(key=lambda i: (i["start_time"], i["services"], i["category"], i["incident_id"]))
    return incidents


def _active_windows(signals: list[dict]) -> list[tuple[datetime, datetime]]:
    spans = []
    for start, end in sorted((_time(s["window_start"]), _time(s["window_end"])) for s in signals):
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metrics(rules: list[dict], ml: list[dict]) -> tuple[dict, dict]:
    """Peak observed values across contributing signals and the baseline used at that peak."""
    observed, baseline, peaks = {}, {}, {}
    for s in rules:
        for metric, value in s["observed_metrics"].items():
            if _number(value) and (observed.get(metric) is None or value > observed[metric]):
                observed[metric], peaks[metric] = value, s
            else:
                observed.setdefault(metric, None)
    for metric, s in peaks.items():
        reference = s["baseline_metrics"].get(metric)
        if isinstance(reference, dict):
            baseline[metric] = reference.get("median")
    if ml:
        peak = max(ml, key=lambda s: (s["score"], s["signal_id"]))
        observed["ml_anomaly_score"] = peak["score"]
        for feature in peak["feature_evidence"]:
            observed[feature["feature"]] = feature["observed"]
            baseline[feature["feature"]] = feature["training_median"]
    return observed, baseline


def _sustained_failure(rules: list[dict], config: IncidentConfig) -> dict | None:
    """Longest run of consecutive same-resolution windows meeting the high-severity 5xx policy."""
    windows = {}
    for s in rules:
        metrics = s["observed_metrics"]
        if s["category"] == "http_errors" and metrics.get("valid_status_count"):
            windows[(s["window_minutes"], s["window_start"])] = (
                s["window_end"], metrics["http_5xx_count"], metrics["valid_status_count"])
    best = None
    for minutes in sorted({key[0] for key in windows}):
        run = []
        for (resolution, start), (end, errors, valid) in sorted(windows.items()):
            if resolution != minutes:
                continue
            if errors < config.high_http_5xx_rate * valid:
                run = []
                continue
            if run and run[-1][1] != start:
                run = []
            run.append((start, end, errors, valid))
            duration = (_time(end) - _time(run[0][0])).total_seconds() / 60
            total = sum(r[3] for r in run)
            if (duration >= config.high_minimum_minutes and total >= config.high_minimum_statuses
                    and (best is None or duration > best["minutes"])):
                best = {"start": run[0][0], "end": end, "minutes": int(duration), "window_minutes": minutes,
                        "http_5xx_count": sum(r[2] for r in run), "valid_status_count": total}
    return best


def _samples(rows: list[NormalizedLog], category: str, limit: int) -> list[dict]:
    def relevant(r: NormalizedLog) -> bool:
        if category == "http_errors":
            return (r.http_status or 0) >= 500
        if category == "service_errors":
            return r.log_level == "ERROR"
        if category == "auth_failures":
            return r.enrichment.authentication_outcome == "failure"
        if category == "health":
            return any(s >= 500 for s in (r.http_status, _health_status(r)) if s is not None)
        if category == "latency":
            return r.response_time_ms is not None
        if category == "multivariate":
            return r.log_level != "INFO" or (r.http_status or 0) >= 400
        return True

    slow = category == "latency"
    chosen = sorted(rows, key=lambda r: (not relevant(r), -(r.response_time_ms or 0) if slow else 0,
                                         r.timestamp, r.source_record_number))[:limit]
    return [{"record_id": r.record_id, "timestamp": r.timestamp.isoformat(), "service": r.service,
             "endpoint": r.endpoint, "log_level": r.log_level, "http_status": r.http_status,
             "response_time_ms": r.response_time_ms, "message": r.message} for r in chosen]


def _strongest(signals: list[dict]) -> dict:
    return max(signals, key=lambda s: (s["score"], s["support_count"], s["signal_id"]))


def _incident(key: tuple, group: dict, records: dict, config: IncidentConfig) -> dict:
    category, scope, service, entity_key, split = key
    signals = sorted(group["signals"], key=_order)
    rules = [s for s in signals if s["detector_id"] != ML_DETECTOR]
    ml = [s for s in signals if s["detector_id"] == ML_DETECTOR]
    ids = sorted({i for s in signals for i in s["evidence_record_ids"]}, key=_ordinal)
    rows = [records[i] for i in ids]
    start, end = group["start"], group["end"]
    active = _active_windows(signals)
    covered = sum((e - s for s, e in active), timedelta())
    gap_minutes = int((end - start - covered).total_seconds() // 60)
    observed, baseline = _metrics(rules, ml)
    sustained = _sustained_failure(rules, config) if category == "http_errors" else None
    if not rules or category in config.review_categories:
        severity = "review"
    else:
        severity = "high" if sustained else "warning"
    confidence = "low" if not rules else "high" if ml else "medium"
    detectors = sorted({s["detector_id"] for s in signals})
    resolutions = "/".join(str(m) for m in sorted({s["window_minutes"] for s in signals}))
    where = f"{service} ({scope}: {entity_key})"
    span = f"[{start.isoformat()}, {end.isoformat()})"
    coverage = (f"{len(signals)} signal(s) from {', '.join(detectors)} at {resolutions}-minute resolution "
                f"cover {int(covered.total_seconds() // 60)} of {int((end - start).total_seconds() // 60)} "
                f"minutes; {len(ids)} unique source records.")
    if rules:
        kind = {"review": "Suspicious activity requiring investigation",
                "warning": "Rule-supported incident candidate",
                "high": "Rule-supported sustained failure candidate"}[severity]
        parts = [f"{kind}: {category} on {where} during {span} ({split}). {coverage}"]
        for detector in sorted({s["detector_id"] for s in rules}):
            parts.append(f"Strongest {detector} signal: "
                         f"{_strongest([s for s in rules if s['detector_id'] == detector])['explanation']}")
        if sustained:
            parts.append(
                f"Sustained condition: {sustained['http_5xx_count']} 5xx among {sustained['valid_status_count']} "
                f"valid statuses across consecutive {sustained['window_minutes']}-minute windows "
                f"[{sustained['start']}, {sustained['end']}) met the high-severity policy "
                f"(≥{config.high_minimum_minutes} min, every window ≥{config.high_http_5xx_rate:.0%} 5xx, "
                f"≥{config.high_minimum_statuses} valid statuses).")
        if ml:
            parts.append(f"Corroborated by Isolation Forest in {len(ml)} window(s) whose top feature increases "
                         f"describe {category}: {_strongest(ml)['explanation']}")
    else:
        parts = [f"Suspicious activity requiring investigation: Isolation Forest flagged {where} during {span} "
                 f"({split}) and no overlapping rule signal of a matching category exists. {coverage}",
                 _strongest(ml)["explanation"]]
        described = sorted({c for s in ml for c in ml_categories(s)})
        if described:
            parts.append(f"Feature increases resemble {', '.join(described)}, but no overlapping rule "
                         f"signal of that category was emitted for this entity.")
    if ml:
        parts.append("Anomaly scores rank windows relative to training; they are not probabilities "
                     "and do not identify causes.")
    limitations = []
    if gap_minutes:
        limitations.append(f"Merged span includes {gap_minutes} minute(s) without contributing signals; "
                           "activity is not asserted to be continuous.")
    if len({s["window_minutes"] for s in signals}) > 1:
        limitations.append("Five- and fifteen-minute signals overlap; each source record is counted once.")
    if split == "train":
        limitations.append("Training period: ML was not scored, so independent corroboration is unavailable.")
    if not rules:
        limitations.append("ML-only candidate: no rule detector corroborated it.")
    flags = Counter(f for r in rows for f in r.quality_flags)
    if flags:
        limitations.append(f"{sum(bool(r.quality_flags) for r in rows)} evidence record(s) carry data-quality "
                           f"flags: {', '.join(sorted(flags))}.")
    for s in signals:
        # Rule corroboration supersedes the ML signal's own ML-only caveat.
        limitations.extend(text for text in s["limitations"] if text not in limitations
                           and not (rules and text.startswith("ML-only candidate")))
    evidence = {
        "record_count": len(ids), "record_ids": ids,
        "samples": _samples(rows, category, config.sample_evidence_limit),
        "active_windows": [{"start": s.isoformat(), "end": e.isoformat()} for s, e in active],
        "gap_minutes": gap_minutes,
        "signals": [{"signal_id": s["signal_id"], "detector_id": s["detector_id"],
                     "window_minutes": s["window_minutes"], "window_start": s["window_start"],
                     "window_end": s["window_end"], "score": s["score"], "threshold": s["threshold"],
                     "support_count": s["support_count"]} for s in signals],
        "sustained_failure": sustained,
        "upstream_services": dict(sorted(Counter(r.enrichment.upstream_service for r in rows
                                                 if r.enrichment.upstream_service).items())),
        "quality_flag_counts": dict(sorted(flags.items())),
        "relations": [],
        "chart": CHART,
    }
    identity = [INCIDENT_VERSION, category, scope, service, entity_key, split, start.isoformat(),
                end.isoformat(), sorted({f"{s['detector_id']}:{s['detector_version']}" for s in signals})]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    return {
        "incident_id": f"inc-{digest[:20]}", "split": split,
        "start_time": start.isoformat(), "end_time": end.isoformat(),
        "first_evidence_time": min(r.timestamp for r in rows).isoformat() if rows else None,
        "last_evidence_time": max(r.timestamp for r in rows).isoformat() if rows else None,
        "services": [service], "category": category,
        "entity": {"scope": scope, "endpoint": entity_key if scope == "endpoint" else None,
                   "source_ip": entity_key if scope == "auth_ip" else None},
        "severity": severity, "confidence": confidence, "detector_ids": detectors,
        "signal_ids": sorted(s["signal_id"] for s in signals),
        "observed_metrics": observed, "baseline_metrics": baseline, "evidence": evidence,
        "explanation": " ".join(parts), "root_cause_hypothesis": None,
        "limitations": limitations, "related_incident_ids": [],
    }


def _link(incidents: list[dict], gap: timedelta) -> None:
    """Link without merging: same service within the merge gap, or overlapping gateway routing evidence."""
    relations = defaultdict(dict)
    hypotheses = defaultdict(list)
    for a, b in combinations(incidents, 2):
        if a["split"] != b["split"]:
            continue
        if a["services"] == b["services"]:
            if (_time(a["start_time"]) <= _time(b["end_time"]) + gap
                    and _time(b["start_time"]) <= _time(a["end_time"]) + gap):
                reason = f"same-service {b['category']} / {a['category']} incidents within the merge gap"
                relations[a["incident_id"]][b["incident_id"]] = reason
                relations[b["incident_id"]][a["incident_id"]] = reason
            continue
        if not a["start_time"] < b["end_time"] or not b["start_time"] < a["end_time"]:
            continue
        for gateway, downstream in ((a, b), (b, a)):
            routed = gateway["evidence"]["upstream_services"].get(downstream["services"][0])
            if not routed:
                continue
            reason = (f"{routed} of {gateway['evidence']['record_count']} {gateway['services'][0]} evidence records "
                      f"were routed upstream to {downstream['services'][0]} during overlapping intervals")
            relations[gateway["incident_id"]][downstream["incident_id"]] = reason
            relations[downstream["incident_id"]][gateway["incident_id"]] = reason
            if (gateway["category"] in FAILURE_CATEGORIES and downstream["category"] in FAILURE_CATEGORIES
                    and downstream["confidence"] != "low"):
                hypotheses[gateway["incident_id"]].append((routed, downstream))
    for incident in incidents:
        linked = relations.get(incident["incident_id"], {})
        incident["related_incident_ids"] = sorted(linked)
        incident["evidence"]["relations"] = [{"incident_id": i, "reason": linked[i]} for i in sorted(linked)]
        if linked:
            incident["limitations"].append("Related incidents are linked by time proximity or routing evidence; "
                                           "links do not establish causality.")
        candidates = hypotheses.get(incident["incident_id"])
        if candidates:
            routed, downstream = max(candidates, key=lambda c: (
                c[0], SEVERITIES.index(c[1]["severity"]), c[1]["incident_id"]))
            incident["root_cause_hypothesis"] = (
                f"Hypothesis: consistent with {downstream['services'][0]} degradation (incident "
                f"{downstream['incident_id']}, {downstream['category']}); {routed} of "
                f"{incident['evidence']['record_count']} {incident['services'][0]} evidence records were routed "
                f"there during overlapping intervals. Routing and timing do not establish the cause.")


def report_limitations(timezone: str, *, inference: bool = False) -> list[str]:
    limitations = [
        "No ground-truth labels exist; every incident is an unconfirmed detection requiring human review.",
        "Severity review marks suspicious activity; warning/high mark rule-supported incident candidates.",
        "Confidence is an evidence-quality assessment, not a probability.",
        "Training-split incidents come from rules only; ML was scored on validation and test.",
        "Validation alerts lie in the ML calibration period; only test incidents are fully held out.",
        "Sample messages are reproduced from synthetic logs; redact them before sharing real data.",
    ]
    if inference:
        limitations = limitations[:3] + ["Sample messages are source evidence; redact them before sharing real data."]
    if timezone == "unspecified":
        limitations.append("Source timezone is unspecified; times are source-local, not UTC.")
    return limitations


def incident_counts(rule_signals: list[dict], ml_signals: list[dict], incidents: list[dict]) -> dict:
    def tally(field: str) -> dict:
        return dict(sorted(Counter(i[field] for i in incidents).items()))

    signals_by_split = defaultdict(lambda: {"rules": 0, "ml": 0})
    for source, signals in (("rules", rule_signals), ("ml", ml_signals)):
        for s in signals:
            signals_by_split[s["split"]][source] += 1
    has_ml = [ML_DETECTOR in i["detector_ids"] for i in incidents]
    attached = sum(s["detector_id"] == ML_DETECTOR for i in incidents if i["category"] != "multivariate"
                   for s in i["evidence"]["signals"])
    return {
        "rule_signals": len(rule_signals), "ml_signals": len(ml_signals),
        "signals_by_split": dict(sorted(signals_by_split.items())), "incidents": len(incidents),
        "rule_only": sum(not ml for ml in has_ml), "ml_only": sum(i["category"] == "multivariate" for i in incidents),
        "corroborated": sum(ml and i["category"] != "multivariate" for i, ml in zip(incidents, has_ml)),
        "ml_signals_attached": attached, "ml_signals_unattached": len(ml_signals) - attached,
        "by_split": tally("split"), "by_severity": tally("severity"), "by_confidence": tally("confidence"),
        "by_category": tally("category"),
    }


def validate_incident_report(report: dict, config: IncidentConfig) -> None:
    """Schema, policy and ordering checks performed before publication."""
    required = {"schema_version", "run_id", "timezone", "incidents", "limitations"}
    if not required <= report.keys() or report.keys() - required - {"provenance", "coverage", "counts"}:
        raise ValueError("incident report envelope mismatch")
    ids = [i["incident_id"] for i in report["incidents"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate incident identity")
    keys = [(i["start_time"], i["services"], i["category"], i["incident_id"]) for i in report["incidents"]]
    if keys != sorted(keys):
        raise ValueError("incidents are not sorted by start time, service, category and ID")
    for i in report["incidents"]:
        evidence = i["evidence"]
        start, end = _time(i["start_time"]), _time(i["end_time"])
        checks = (
            set(i) == set(CSV_FIELDS), i["category"] in CATEGORIES, i["severity"] in SEVERITIES,
            i["confidence"] in CONFIDENCES, start < end, start.tzinfo is None,
            bool(i["services"]) and i["services"] == sorted(set(i["services"])),
            set(i["entity"]) == {"scope", "endpoint", "source_ip"},
            bool(i["detector_ids"]) and i["detector_ids"] == sorted(set(i["detector_ids"])),
            bool(i["signal_ids"]) and i["signal_ids"] == sorted(set(i["signal_ids"])),
            len(evidence["signals"]) == len(i["signal_ids"]),
            evidence["record_count"] == len(evidence["record_ids"]) == len(set(evidence["record_ids"])),
            len(evidence["samples"]) <= config.sample_evidence_limit,
            i["first_evidence_time"] is None or (
                i["start_time"] <= i["first_evidence_time"] <= i["last_evidence_time"] < i["end_time"]),
            isinstance(i["explanation"], str) and bool(i["explanation"]),
            i["root_cause_hypothesis"] is None or i["root_cause_hypothesis"].startswith("Hypothesis:"),
            all(isinstance(t, str) for t in i["limitations"]),
            i["incident_id"] not in i["related_incident_ids"] and set(i["related_incident_ids"]) <= set(ids),
            # ML alone never escalates; high confidence requires rule and ML agreement.
            i["category"] != "multivariate" or (i["severity"], i["confidence"]) == ("review", "low"),
            i["confidence"] != "high" or (ML_DETECTOR in i["detector_ids"] and len(i["detector_ids"]) > 1),
        )
        if not all(checks):
            raise ValueError(f"incident failed schema/policy validation: {i['incident_id']}")


def _cell(value) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                           separators=(",", ":"))
    # Prevent spreadsheet formula interpretation of exported text.
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def incidents_csv(incidents: list[dict]) -> str:
    """One row per incident; complete evidence record IDs remain in the authoritative JSON."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    for incident in incidents:
        row = {**incident, "evidence": {k: v for k, v in incident["evidence"].items() if k != "record_ids"}}
        writer.writerow([_cell(row[field]) for field in CSV_FIELDS])
    return buffer.getvalue()


def _md(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    for old, new in (("\r", " "), ("\n", " "), ("|", "\\|"), ("`", "\\`"), ("<", "&lt;"), (">", "&gt;")):
        text = text.replace(old, new)
    return text


def summary_markdown(report: dict, counts: dict, *, inference: bool = False) -> str:
    incidents = report["incidents"]
    signals = counts["signals_by_split"]
    lines = [
        "# Incident summary", "",
        f"Run `{report['run_id']}` · source timezone: {report['timezone']} · no ground-truth labels.", "",
        "## Overview", "",
        f"- Input signals: {counts['rule_signals']} rule, {counts['ml_signals']} ML ("
        + ", ".join(f"{split}: {v['rules']} rule / {v['ml']} ML" for split, v in signals.items()) + ").",
        f"- Incidents: {counts['incidents']} — rule-only {counts['rule_only']}, ML-only {counts['ml_only']}, "
        f"rule + ML corroborated {counts['corroborated']}. ML signals attached to rule incidents: "
        f"{counts['ml_signals_attached']}; kept as multivariate candidates: {counts['ml_signals_unattached']}.",
        "- By split: " + ", ".join(f"{k} {v}" for k, v in counts["by_split"].items()) + ".",
        "- By severity: " + ", ".join(f"{k} {v}" for k, v in counts["by_severity"].items())
        + "; by confidence: " + ", ".join(f"{k} {v}" for k, v in counts["by_confidence"].items()) + ".",
        "",
        "`review` = suspicious activity requiring investigation; `warning`/`high` = rule-supported incident "
        "candidates. None is confirmed: confirmation requires human review or labels.", "",
    ]
    if "coverage" in report:
        lines += ["- Window scoring: " + _md(report["coverage"]["window_status_counts"]) + ".",
                  "- ML unavailable (rules still apply): "
                  + (", ".join(report["coverage"]["unavailable_models"]) or "none") + ".", ""]
    if not incidents:
        lines += ["**No incidents**: no rule or ML signal was emitted for this run.", ""]
    else:
        lines += ["## Incidents", "",
                  "| ID | Split | Window [start, end) | Service | Category | Entity | Severity | Confidence "
                  "| Detectors | Records |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for i in incidents:
            entity = i["entity"]["endpoint"] or i["entity"]["source_ip"] or i["entity"]["scope"]
            lines.append(f"| {i['incident_id']} | {i['split']} | {i['start_time']} – {i['end_time']} | "
                         f"{', '.join(i['services'])} | {i['category']} | {_md(entity)} | {i['severity']} | "
                         f"{i['confidence']} | {', '.join(i['detector_ids'])} | {i['evidence']['record_count']} |")
        lines += ["", ("## Inference incident details" if inference
                          else "## Held-out incident details (validation and test)"), "",
                  "One block per group of related incidents, describing its highest-severity member.", ""]
        by_id = {i["incident_id"]: i for i in incidents}
        shown = set()
        for i in incidents:
            if i["split"] == "train" or i["incident_id"] in shown:
                continue
            group, pending = set(), [i["incident_id"]]
            while pending:
                current = pending.pop()
                if current not in group:
                    group.add(current)
                    pending.extend(by_id[current]["related_incident_ids"])
            shown |= group
            i = max((by_id[g] for g in group), key=lambda g: (
                SEVERITIES.index(g["severity"]), CONFIDENCES.index(g["confidence"]),
                g["evidence"]["record_count"], g["incident_id"]))
            evidence = i["evidence"]
            others = sorted(group - {i["incident_id"]})
            lines += [
                f"### {i['incident_id']}", "",
                f"- Category / severity / confidence: {i['category']} / {i['severity']} / {i['confidence']}",
                f"- Affected: {', '.join(i['services'])}; entity {_md(i['entity'])}",
                f"- Detection mode: {'inference: rules and ML frozen' if inference else MODES[i['split']]}",
                f"- Window [{i['start_time']}, {i['end_time']}); evidence {i['first_evidence_time']} – "
                f"{i['last_evidence_time']}; {evidence['record_count']} source records",
                f"- Peak observed: {_md(i['observed_metrics'])}; baseline: {_md(i['baseline_metrics'])}",
                f"- Detectors: {', '.join(i['detector_ids'])} ({len(i['signal_ids'])} signals)",
                f"- Explanation: {_md(i['explanation'])}",
                f"- Root-cause hypothesis: {_md(i['root_cause_hypothesis'] or 'none established from logs')}",
                f"- Related incidents ({len(others)}): " + (", ".join(
                    f"{o} ({by_id[o]['services'][0]} {by_id[o]['category']}, {by_id[o]['severity']})"
                    for o in others) or "none"),
                *(f"- Sample `{s['record_id'].rsplit(':', 1)[1]}`: {s['timestamp']} {_md(s['endpoint'] or '')} "
                  f"{s['http_status']} {s['response_time_ms']} ms — {_md(s['message'])}" for s in evidence["samples"][:2]),
                f"- Suggested investigation: {INVESTIGATION[i['category']]}",
                f"- Limitations: {_md(' '.join(i['limitations']))}", "",
            ]
    lines += ["## Report limitations", "", *(f"- {text}" for text in report["limitations"]), ""]
    return "\n".join(lines)
