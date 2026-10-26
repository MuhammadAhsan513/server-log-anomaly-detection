"""Stage-specific CLI orchestration through the Phase 4 ML baseline."""

import argparse
import sys
from pathlib import Path

from .config import DetectionConfig, FeatureConfig, IngestConfig, InputError, MLConfig, load_config
from .pipeline import detect, features, ingest, json_text, ml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log anomaly project — ingestion, features, rule and ML detection")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (("ingest", "validate and parse logs"),
                              ("features", "ingest, aggregate, split, and chart logs; no detection"),
                              ("detect", "score rules from a completed Phase 2 artifact directory"),
                              ("ml", "fit/score Isolation Forest from a completed Phase 2 artifact directory")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "ml":
            command.add_argument("--rules", type=Path, help="completed Phase 3 run for rule comparison")
    args = parser.parse_args(argv)
    try:
        config_type, operation = {"ingest": (IngestConfig, ingest), "features": (FeatureConfig, features),
                                  "detect": (DetectionConfig, detect), "ml": (MLConfig, ml)}[args.command]
        config = load_config(args.config, config_type)
        if args.command in {"detect", "ml"} and not args.input.is_dir():
            raise InputError(f"feature run directory does not exist: {args.input}")
        if args.command not in {"detect", "ml"} and not args.input.is_file():
            raise InputError(f"input file does not exist or is not a regular file: {args.input}")
        if args.command == "ml":
            if args.rules is not None and not args.rules.is_dir():
                raise InputError(f"rules run directory does not exist: {args.rules}")
            manifest = operation(args.input, config, args.output, args.rules)
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
    print(json_text({"phase": manifest["phase"], "status": manifest["status"], "output": str(args.output)}))
    return 0
