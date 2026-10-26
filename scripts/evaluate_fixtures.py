"""Phase 6 fixture experiment: labelled fixtures → existing stages → rules/ML/combined metrics.

Development seeds diagnose failures and choose changes. Final seeds stay untouched
until `--final`, which may be run only after the detection configuration is frozen.
Each detection variant reuses the same features, ML run, grouping, and matching.

    python scripts/evaluate_fixtures.py --output artifacts/phase6-dev
    python scripts/evaluate_fixtures.py --output artifacts/phase6-final --final
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_fixture_logs import write  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = (11, 12, 13)
FINAL_SEEDS = (21, 22, 23)
MODES = ("rules", "ml", "combined")


def stage(*args: str) -> None:
    result = subprocess.run([sys.executable, "-m", "log_anomaly", *args], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"stage failed ({result.returncode}): {' '.join(args)}\n{result.stderr}")


def variants(specs: list[str], output: Path) -> dict[str, Path]:
    """name=path pairs; by default the frozen config and its pre-Phase 6 baseline (recurrence context off)."""
    if specs:
        return {name: Path(path) for name, path in (s.split("=", 1) for s in specs)}
    improved = ROOT / "configs/detection.json"
    baseline = output / "detection-baseline.json"
    baseline.write_text(json.dumps({**json.loads(improved.read_text()), "traffic_recurrence_days": 0}, indent=2))
    return {"baseline": baseline, "improved": improved}


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(evaluations: dict[int, dict]) -> dict:
    """Pool counts over seeds (micro-average); metrics recomputed from pooled counts."""
    pooled = {}
    for split in next(iter(evaluations.values()))["results"]:
        pooled[split] = {}
        for mode in MODES:
            total = Counter()
            kinds, categories, by_category = Counter(), Counter(), {}
            sensitivity = {}
            for evaluation in evaluations.values():
                m = evaluation["results"][split][mode]
                total.update({k: m[k] for k in ("expected", "predicted", "tp", "fp", "fn")})
                total["days"] += m["evaluated_days"]
                kinds.update(m["fp_kinds"])
                categories.update(m["fp_categories"])
                for category, row in m["recall_by_category"].items():
                    entry = by_category.setdefault(category, Counter())
                    entry.update(row)
                for threshold, row in m["iou_sensitivity"].items():
                    sensitivity.setdefault(threshold, Counter()).update(row)
            precision, recall = ratio(total["tp"], total["tp"] + total["fp"]), ratio(total["tp"], total["tp"] + total["fn"])
            pooled[split][mode] = {
                **{k: total[k] for k in ("expected", "predicted", "tp", "fp", "fn")},
                "precision": precision, "recall": recall,
                "f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
                if precision is not None and recall is not None else None,
                "unmatched_per_day": ratio(total["fp"], total["days"]),
                "fp_kinds": dict(sorted(kinds.items())), "fp_categories": dict(sorted(categories.items())),
                "recall_by_category": {c: dict(v) for c, v in sorted(by_category.items())},
                "iou_sensitivity": {t: dict(v) for t, v in sorted(sensitivity.items())},
            }
    return pooled


def markdown(report: dict) -> str:
    def number(value):
        return "null" if value is None else f"{value:.3f}"

    lines = [f"# Phase 6 fixture comparison ({report['seed_set']} seeds {report['seeds']})", "",
             "Pooled over seeds; one-to-one max-IoU matching (IoU ≥ 0.3). Synthetic fixtures only.", ""]
    for split in report["primary_first"]:
        lines += [f"## {split}", "", "| Variant | Mode | Expected | Predicted | TP | FP | FN | Precision | Recall "
                  "| F1 | FP/day | FP kinds |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for variant, pooled in report["pooled"].items():
            for mode in MODES:
                m = pooled[split][mode]
                lines.append(f"| {variant} | {mode} | {m['expected']} | {m['predicted']} | {m['tp']} | {m['fp']} | "
                             f"{m['fn']} | {number(m['precision'])} | {number(m['recall'])} | {number(m['f1'])} | "
                             f"{number(m['unmatched_per_day'])} | "
                             + (", ".join(f"{k} {v}" for k, v in m["fp_kinds"].items()) or "none") + " |")
        lines.append("")
    lines += [f"Stage runtime {report['runtime_seconds']:.0f} s; peak child RSS {report['peak_child_rss_mb']:.0f} MB.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--final", action="store_true", help="use the untouched final seeds (after freezing)")
    parser.add_argument("--seeds", type=int, nargs="+", help="subset of the selected seed set")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=DETECTION_CONFIG")
    args = parser.parse_args()
    allowed = FINAL_SEEDS if args.final else DEV_SEEDS
    seeds = tuple(args.seeds or allowed)
    if not set(seeds) <= set(allowed):
        raise SystemExit(f"seeds must come from {'final' if args.final else 'development'} set {allowed}")
    args.output.mkdir(parents=True)
    configs = variants(args.variant, args.output)
    start = time.perf_counter()
    evaluations = {name: {} for name in configs}
    for seed in seeds:
        base = args.output / f"seed-{seed}"
        logs, labels = write(seed, base / "fixture")
        stage("features", "--input", str(logs), "--config", str(ROOT / "configs/features.json"),
              "--output", str(base / "features"))
        stage("ml", "--input", str(base / "features"), "--config", str(ROOT / "configs/ml.json"),
              "--output", str(base / "ml"))
        for name, config in configs.items():
            stage("detect", "--input", str(base / "features"), "--config", str(config),
                  "--output", str(base / f"rules-{name}"))
            stage("incidents", "--input", str(base / "features"), "--rules", str(base / f"rules-{name}"),
                  "--ml", str(base / "ml"), "--config", str(ROOT / "configs/incidents.json"),
                  "--output", str(base / f"incidents-{name}"))
            stage("evaluate", "--input", str(base / f"incidents-{name}"), "--labels", str(labels),
                  "--config", str(ROOT / "configs/evaluation.json"), "--output", str(base / f"evaluation-{name}"))
            evaluations[name][seed] = json.loads((base / f"evaluation-{name}/evaluation.json").read_text())
        print(f"seed {seed} complete", file=sys.stderr)
    splits = list(next(iter(next(iter(evaluations.values())).values()))["results"])
    report = {
        "seed_set": "final" if args.final else "development", "seeds": list(seeds),
        "variants": {name: str(path) for name, path in configs.items()},
        "primary_first": sorted(splits, key=lambda s: s != "test"),
        "pooled": {name: aggregate(evals) for name, evals in evaluations.items()},
        "per_seed": {name: {str(seed): {split: {mode: {k: e["results"][split][mode][k]
                                                       for k in ("expected", "predicted", "tp", "fp", "fn")}
                                                for mode in MODES} for split in splits}
                            for seed, e in evals.items()} for name, evals in evaluations.items()},
        "runtime_seconds": time.perf_counter() - start,
        "peak_child_rss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "comparison.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
