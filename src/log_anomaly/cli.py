"""End-to-end pipeline and compatible individual stage commands."""

import argparse
import sys
from pathlib import Path

from .config import (DetectionConfig, EvaluationConfig, FeatureConfig, IncidentConfig, IngestConfig, InputError,
                     MLConfig, RunConfig, load_config)
from .pipeline import detect, evaluate, features, ingest, json_text, ml, report_incidents, run, infer, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log anomaly project — ingestion, features, rule/ML detection, "
                                                 "incidents, labelled evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (("train", "fit a reusable model bundle from historical CSV logs"),
                              ("run", "alias for train; --debug-artifacts enables the legacy evaluation workflow"),
                              ("infer", "score new CSV using a trusted local model bundle; never fit"),
                              ("ingest", "validate and parse logs"),
                              ("features", "ingest, aggregate, split, and chart logs; no detection"),
                              ("detect", "score rules from a completed Phase 2 artifact directory"),
                              ("ml", "fit/score Isolation Forest from a completed Phase 2 artifact directory"),
                              ("incidents", "group completed Phase 3 rule and Phase 4 ML signals into reports"),
                              ("evaluate", "score a completed Phase 5 run's rules/ML/combined incidents against "
                                           "evaluator-only fixture labels")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--input", required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--debug-artifacts", action="store_true",
                                 help="retain legacy stage artifacts, charts and sensitivity experiments for evaluation")
        if name == "infer":
            command.add_argument("--model-bundle", type=Path, required=True,
                                 help="trusted local models directory (legacy completed run directories also supported)")
        if name == "ml":
            command.add_argument("--rules", type=Path, help="completed Phase 3 run for rule comparison")
        if name == "incidents":
            command.add_argument("--rules", type=Path, required=True, help="completed Phase 3 rules run")
            command.add_argument("--ml", type=Path, required=True, help="completed Phase 4 ML run")
        if name == "evaluate":
            command.add_argument("--labels", type=Path, required=True, help="evaluator-only fixture labels JSON")
    args = parser.parse_args(argv)
    try:
        if not args.input:
            raise InputError("--input is empty; provide a CSV path or set NEW_LOGS before using --input \"$NEW_LOGS\"")
        args.input = Path(args.input)
        config_type, operation = {"train": (RunConfig, train), "run": (RunConfig, run), "infer": (RunConfig, infer),
                                  "ingest": (IngestConfig, ingest), "features": (FeatureConfig, features),
                                  "detect": (DetectionConfig, detect), "ml": (MLConfig, ml),
                                  "incidents": (IncidentConfig, report_incidents),
                                  "evaluate": (EvaluationConfig, evaluate)}[args.command]
        config = load_config(args.config, config_type)
        staged = {"detect", "ml", "incidents", "evaluate"}
        if args.command in staged and not args.input.is_dir():
            raise InputError(f"input run directory does not exist: {args.input}")
        if args.command not in staged and not args.input.is_file():
            raise InputError(f"input file does not exist or is not a regular file: {args.input}")
        if args.command == "run":
            manifest = operation(args.input, config, args.output, debug_artifacts=args.debug_artifacts)
        elif args.command == "infer":
            manifest = operation(args.input, config, args.output, args.model_bundle)
        elif args.command == "ml":
            if args.rules is not None and not args.rules.is_dir():
                raise InputError(f"rules run directory does not exist: {args.rules}")
            manifest = operation(args.input, config, args.output, args.rules)
        elif args.command == "incidents":
            for label, path in (("rules", args.rules), ("ML", args.ml)):
                if not path.is_dir():
                    raise InputError(f"{label} run directory does not exist: {path}")
            manifest = operation(args.input, config, args.output, args.rules, args.ml)
        elif args.command == "evaluate":
            if not args.labels.is_file():
                raise InputError(f"labels file does not exist: {args.labels}")
            manifest = operation(args.input, config, args.output, args.labels)
        else:
            manifest = operation(args.input, config, args.output)
    except InputError as exc:
        print(f"invalid input/config: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"output failure: {exc}", file=sys.stderr)
        return 4
    except ImportError as exc:
        print(f"required stage dependency unavailable: {exc}; install requirements.lock", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"required stage failed: {exc}", file=sys.stderr)
        return 3
    print(json_text({"phase": manifest["phase"], "status": manifest["status"], "output": str(args.output),
                     **({"reused": True} if manifest.get("reused") else {})}))
    return 0
