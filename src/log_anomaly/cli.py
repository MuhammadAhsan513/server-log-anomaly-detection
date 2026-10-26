"""Stage-specific CLI orchestration through Phase 3 rules and baselines."""

import argparse
import sys
from pathlib import Path

from .config import DetectionConfig, FeatureConfig, IngestConfig, InputError, load_config
from .pipeline import detect, features, ingest, json_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log anomaly project — ingestion, features, and rule detection")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (("ingest", "validate and parse logs"),
                              ("features", "ingest, aggregate, split, and chart logs; no detection"),
                              ("detect", "score rules from a completed Phase 2 artifact directory")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config_type, operation = {"ingest": (IngestConfig, ingest), "features": (FeatureConfig, features),
                                  "detect": (DetectionConfig, detect)}[args.command]
        config = load_config(args.config, config_type)
        if args.command == "detect" and not args.input.is_dir():
            raise InputError(f"feature run directory does not exist: {args.input}")
        if args.command != "detect" and not args.input.is_file():
            raise InputError(f"input file does not exist or is not a regular file: {args.input}")
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
