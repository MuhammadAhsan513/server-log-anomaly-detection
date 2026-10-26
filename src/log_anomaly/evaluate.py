"""Phase 6 incident-level evaluation against evaluator-only fixture labels.

Detectors never import this module. Labels enter only here, after rules, ML, and
grouping have completed, and are used to score — never to fit — any detector.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import EvaluationConfig, InputError
from .incidents import ML_DETECTOR, ml_categories

LABEL_CATEGORIES = ("http_errors", "service_errors", "auth_failures", "latency", "traffic", "health")
# A predicted category may match these labelled categories. 5xx bursts are usually
# logged at ERROR level, so HTTP and service-error incidents describe one another.
COMPATIBLE = {
    "http_errors": {"http_errors", "service_errors"}, "service_errors": {"service_errors", "http_errors"},
    "auth_failures": {"auth_failures"}, "latency": {"latency"}, "traffic": {"traffic"}, "health": {"health"},
}
MODES = ("rules", "ml", "combined")
LATENCY_NOTE = ("not_applicable: batch evaluation of completed windows; online detection delay requires the "
                "Phase 8 replay mode and is not batch runtime")


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def validate_labels(labels: dict) -> list[dict]:
    """Strict evaluator-only label schema; returns events with parsed bounds."""
    try:
        events = labels["events"]
        if labels["schema_version"] != "1.0" or not isinstance(events, list):
            raise InputError("unsupported label schema")
        parsed, ids = [], set()
        for e in events:
            if not {"event_id", "category", "service", "endpoint", "source_ip", "start", "end"} <= set(e):
                raise InputError("label event lacks required fields")
            start, end = _time(e["start"]), _time(e["end"])
            if (e["event_id"] in ids or e["category"] not in LABEL_CATEGORIES or not e["service"]
                    or start.tzinfo or end.tzinfo or start >= end):
                raise InputError(f"invalid label event: {e.get('event_id')}")
            ids.add(e["event_id"])
            parsed.append({**e, "start_dt": start, "end_dt": end})
        return parsed
    except InputError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"invalid labels: {exc}") from exc


def clip_truths(events: list[dict], bounds: dict) -> list[dict]:
    """Expected incidents intersecting one split, clipped to it (incidents never cross splits)."""
    low, high = _time(bounds["start"]), _time(bounds["end"])
    return [{**e, "start_dt": max(e["start_dt"], low), "end_dt": min(e["end_dt"], high),
             "clipped": e["start_dt"] < low or e["end_dt"] > high}
            for e in events if e["start_dt"] < high and low < e["end_dt"]]


def iou(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> float:
    overlap = (min(a[1], b[1]) - max(a[0], b[0])).total_seconds()
    if overlap <= 0:
        return 0.0
    return overlap / (max(a[1], b[1]) - min(a[0], b[0])).total_seconds()


def described_categories(incident: dict, ml_signals: dict) -> set[str]:
    """Categories an incident can claim: its own, or for ML-only candidates its feature increases."""
    if incident["category"] != "multivariate":
        return {incident["category"]}
    return {c for sid in incident["signal_ids"] if sid in ml_signals for c in ml_categories(ml_signals[sid])}


def eligible(incident: dict, truth: dict, described: set[str]) -> bool:
    entity = incident["entity"]
    return (truth["service"] in incident["services"] and bool(described & COMPATIBLE[truth["category"]])
            and (truth["source_ip"] is None or entity["source_ip"] == truth["source_ip"])
            and (truth["endpoint"] is None or entity["endpoint"] in (None, truth["endpoint"])))


def match(incidents: list[dict], truths: list[dict], threshold: float, ml_signals: dict) -> list[tuple]:
    """Maximum-total-IoU one-to-one assignment among eligible pairs with IoU ≥ threshold.

    Returns (incident index, truth index, IoU). A second prediction for an already
    matched truth stays unmatched, so duplicates cannot inflate recall.
    """
    if not incidents or not truths:
        return []
    weights = np.zeros((len(incidents), len(truths)))
    for i, incident in enumerate(incidents):
        span = (_time(incident["start_time"]), _time(incident["end_time"]))
        described = described_categories(incident, ml_signals)
        for t, truth in enumerate(truths):
            value = iou(span, (truth["start_dt"], truth["end_dt"]))
            if value >= threshold and eligible(incident, truth, described):
                weights[i, t] = value
    rows, columns = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(t), float(weights[i, t])) for i, t in zip(rows, columns) if weights[i, t] > 0]


def _ratio(numerator: int, denominator: int, reason: str) -> dict:
    return ({"value": numerator / denominator, "reason": None} if denominator
            else {"value": None, "reason": reason})


def _recurring(incident: dict, patterns: list[dict]) -> bool:
    """Overlap with an evaluator-known normal recurring pattern (fixture metadata only)."""
    start, end = _time(incident["start_time"]), _time(incident["end_time"])
    for pattern in patterns:
        if pattern["service"] not in incident["services"]:
            continue
        hours, minutes, seconds = (int(x) for x in pattern["daily_clock_time"].split(":"))
        day = start.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        while day < end:
            run = day + timedelta(hours=hours, minutes=minutes, seconds=seconds)
            if run < end and start < run + timedelta(minutes=5):
                return True
            day += timedelta(days=1)
    return False


def score_split(incidents: list[dict], truths: list[dict], bounds: dict, config: EvaluationConfig,
                ml_signals: dict, patterns: list[dict]) -> dict:
    pairs = match(incidents, truths, config.iou_threshold, ml_signals)
    matched_incidents = {i for i, _, _ in pairs}
    matched_truths = {t for _, t, _ in pairs}
    tp, fp, fn = len(pairs), len(incidents) - len(pairs), len(truths) - len(pairs)
    precision = _ratio(tp, tp + fp, "no predicted incidents in this split")
    recall = _ratio(tp, tp + fn, "no labelled events in this split")
    p, r = precision["value"], recall["value"]
    f1 = ({"value": 2 * p * r / (p + r) if p + r else 0.0, "reason": None} if p is not None and r is not None
          else {"value": None, "reason": "precision or recall undefined"})
    days = (_time(bounds["end"]) - _time(bounds["start"])).total_seconds() / 86400

    unmatched = []
    for index, incident in enumerate(incidents):
        if index in matched_incidents:
            continue
        span = (_time(incident["start_time"]), _time(incident["end_time"]))
        # Fragment: overlaps a labelled event on the same service but lost the one-to-one match.
        if any(t["service"] in incident["services"] and iou(span, (t["start_dt"], t["end_dt"])) > 0
               for t in truths):
            kind = "duplicate_of_labelled_event"
        elif _recurring(incident, patterns):
            kind = "recurring_normal_pattern"
        else:
            kind = "unexplained"
        unmatched.append({"incident_id": incident["incident_id"], "category": incident["category"],
                          "service": incident["services"][0], "scope": incident["entity"]["scope"],
                          "start_time": incident["start_time"], "end_time": incident["end_time"],
                          "severity": incident["severity"], "fp_kind": kind})
    matches = []
    for i, t, value in sorted(pairs, key=lambda p: truths[p[1]]["event_id"]):
        incident, truth = incidents[i], truths[t]
        matches.append({
            "event_id": truth["event_id"], "incident_id": incident["incident_id"], "iou": value,
            "category": truth["category"], "predicted_category": incident["category"],
            "start_error_minutes": (_time(incident["start_time"]) - truth["start_dt"]).total_seconds() / 60,
            "end_error_minutes": (_time(incident["end_time"]) - truth["end_dt"]).total_seconds() / 60,
        })
    by_category = defaultdict(lambda: {"expected": 0, "detected": 0})
    by_difficulty = defaultdict(lambda: {"expected": 0, "detected": 0})
    for t, truth in enumerate(truths):
        for table, key in ((by_category, truth["category"]), (by_difficulty, truth.get("difficulty", "standard"))):
            table[key]["expected"] += 1
            table[key]["detected"] += t in matched_truths
    sensitivity = {}
    for threshold in config.sensitivity_iou_thresholds:
        count = len(match(incidents, truths, threshold, ml_signals))
        sensitivity[str(threshold)] = {"tp": count, "fp": len(incidents) - count, "fn": len(truths) - count}
    errors = [abs(m[k]) for m in matches for k in ("start_error_minutes", "end_error_minutes")]
    return {
        "expected": len(truths), "predicted": len(incidents), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "unmatched_per_day": fp / days, "evaluated_days": days,
        "fp_kinds": dict(sorted(Counter(u["fp_kind"] for u in unmatched).items())),
        "fp_categories": dict(sorted(Counter(u["category"] for u in unmatched).items())),
        "recall_by_category": dict(sorted(by_category.items())),
        "recall_by_difficulty": dict(sorted(by_difficulty.items())),
        "mean_iou": sum(m["iou"] for m in matches) / len(matches) if matches else None,
        "mean_abs_boundary_error_minutes": sum(errors) / len(errors) if errors else None,
        "max_abs_boundary_error_minutes": max(errors) if errors else None,
        "detection_latency": LATENCY_NOTE,
        "iou_sensitivity": sensitivity,
        "matches": matches,
        "missed_event_ids": [truths[t]["event_id"] for t in range(len(truths)) if t not in matched_truths],
        "unmatched_predictions": unmatched,
    }


def diagnose(truth: dict, windows: list[dict], rule_signals: list[dict], ml_signals: list[dict],
             ml_models: dict, guards: dict) -> dict:
    """Why an expected incident was missed: overlapping signals, support, and ML coverage."""
    span = (truth["start_dt"], truth["end_dt"])

    def overlaps(item):
        return (item["service"] == truth["service"]
                and iou((_time(item["window_start"]), _time(item["window_end"])), span) > 0)

    support = {}
    for minutes in (5, 15):
        counts = [w["business"]["record_count"] for w in windows if w["window_minutes"] == minutes
                  and w["scope"] == "service" and overlaps(w)]
        support[str(minutes)] = {"windows": len(counts), "max_business_records": max(counts, default=0),
                                 "windows_meeting_minimum_records": sum(c >= guards["minimum_records"]
                                                                        for c in counts)}
    scope = "auth_ip" if truth["source_ip"] else "service"
    return {
        "event_id": truth["event_id"], "category": truth["category"], "service": truth["service"],
        "difficulty": truth.get("difficulty", "standard"),
        "rule_signal_categories": dict(sorted(Counter(s["category"] for s in rule_signals if overlaps(s)).items())),
        "ml_signals": sum(overlaps(s) for s in ml_signals),
        "service_window_support": support, "support_guards": guards,
        "ml_models": {key: ml_models.get(key, "unavailable: no model key")
                      for key in (f"{m}:{scope}:{truth['service']}" for m in (5, 15))},
    }


def evaluate_incidents(modes: dict[str, list[dict]], events: list[dict], split_ranges: dict,
                       config: EvaluationConfig, ml_signals: list[dict], rule_signals: list[dict],
                       windows: list[dict], ml_models: dict, guards: dict, patterns: list[dict]) -> dict:
    """Score each detector mode per split with identical matching criteria."""
    by_id = {s["signal_id"]: s for s in ml_signals if s["detector_id"] == ML_DETECTOR}
    results = {}
    diagnoses = {}
    for split in config.splits:
        truths = clip_truths(events, split_ranges[split])
        results[split] = {}
        for mode, incidents in modes.items():
            selected = [i for i in incidents if i["split"] == split]
            results[split][mode] = score_split(selected, truths, split_ranges[split], config, by_id, patterns)
        missed = {e for mode in results[split].values() for e in mode["missed_event_ids"]}
        diagnoses[split] = [diagnose(t, windows, rule_signals, ml_signals, ml_models, guards)
                            for t in truths if t["event_id"] in missed]
        results[split]["expected_events"] = [
            {k: t[k] for k in ("event_id", "category", "service", "endpoint", "source_ip")}
            | {"start": t["start_dt"].isoformat(), "end": t["end_dt"].isoformat(), "clipped": t["clipped"],
               "difficulty": t.get("difficulty", "standard")} for t in truths]
    return {"results": results, "missed_event_diagnoses": diagnoses}


def summary_markdown(evaluation: dict) -> str:
    def value(metric):
        return "null" if metric["value"] is None else f"{metric['value']:.3f}"

    lines = ["# Labelled fixture evaluation", "",
             f"Labels: `{evaluation['labels']['path']}` (seed {evaluation['labels']['seed']}). "
             f"Matching: one-to-one maximum total IoU, IoU ≥ {evaluation['config']['iou_threshold']}, "
             "compatible category and same service/entity. Synthetic-fixture metrics are not accuracy "
             "on the supplied unlabelled logs.", ""]
    for split, modes in evaluation["results"].items():
        note = " (primary, fully held out)" if split == evaluation["config"]["primary_split"] else (
            " (rules held out; ML threshold calibrated on this unlabelled day)" if split == "validation" else "")
        lines += [f"## {split}{note}", "",
                  "| Mode | Expected | Predicted | TP | FP | FN | Precision | Recall | F1 | FP kinds |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for mode in MODES:
            m = modes[mode]
            kinds = ", ".join(f"{k} {v}" for k, v in m["fp_kinds"].items()) or "none"
            lines.append(f"| {mode} | {m['expected']} | {m['predicted']} | {m['tp']} | {m['fp']} | {m['fn']} | "
                         f"{value(m['precision'])} | {value(m['recall'])} | {value(m['f1'])} | {kinds} |")
        lines.append("")
        for d in evaluation["missed_event_diagnoses"][split]:
            missed_by = [mode for mode in MODES if d["event_id"] in modes[mode]["missed_event_ids"]]
            lines.append(f"- Missed by {', '.join(missed_by)}: `{d['event_id']}` {d['category']} on {d['service']} "
                         f"({d['difficulty']}); overlapping rule signals {d['rule_signal_categories'] or 'none'}, "
                         f"ML signals {d['ml_signals']}, max business records per 5/15-min window "
                         f"{d['service_window_support']['5']['max_business_records']}/"
                         f"{d['service_window_support']['15']['max_business_records']}; ML models {d['ml_models']}.")
        lines.append("")
    lines += ["Detection latency: " + LATENCY_NOTE + ".", ""]
    return "\n".join(lines)
