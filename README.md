# server-log-anomaly-detection
Python-based Server Log Analysis and Anomaly Detection Pipeline

## Project planning

Read the [engineering documentation](docs/README.md) for requirements, system design, implementation phases, data contracts, detection methods, evaluation, and delivery instructions. Phases 1–3 are implemented; ML and incident reporting remain planned.

## Phase 1 implementation

Ingestion and parsing are now implemented. The full detection pipeline remains
planned. See the [Phase 1 setup and usage guide](docs/phase-1.md) for tested commands,
artifact contracts, scope boundaries, and handoff information.

## Phase 2: features and EDA

From the repository root, using the virtual environment from the Phase 1 setup:

```bash
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m log_anomaly features --input logs/server_application_logs_100k.csv --config configs/features.json --output artifacts/phase2-run-001
.venv/bin/python -m pytest -q
```

Use a fresh output directory. The `features` command reuses ingestion and parsing,
then writes five- and fifteen-minute service, endpoint, and authentication-IP
windows, separate business/health statistics, and train/validation/test splits.
It adds `windows.jsonl`, `template_reference.json`, `eda.json` (service statistics,
parse coverage, split counts), and four PNG charts under `charts/` to the Phase 1
artifacts. See the [Phase 2 contract](docs/04-data-contracts.md#phase-2-implemented-interface)
for field meanings and denominator rules.

Defaults use four training days, one validation day, and the remaining days for
test, starting at the first accepted event's day; a shorter input needs adjusted
positive day counts and at least one test day. Templates are learned only from
training records. Phase 1's `ingest` command remains available. Matplotlib is
needed only for Phase 2; dependencies are pinned for the tested Python 3.14.4
environment. The manifest marks Phase 2 complete and keeps
`pipeline_complete` false.

## Phase 3: rules and historical baselines

```bash
.venv/bin/python -m log_anomaly detect --input artifacts/phase2-run-001 --config configs/detection.json --output artifacts/phase3-run-001
```

The input is a completed Phase 2 directory. Detection verifies and consumes its
`windows.jsonl`, `normalized.jsonl`, and feature configuration without copying
them. Use a fresh output directory. Outputs are `signals.jsonl`, `baselines.json`,
`detection_summary.json`, resolved `config.json`, and `manifest.json` with source
artifact hashes. An empty signal file is a valid no-alert result.

Configurable rules cover business HTTP 5xx bursts/rate rises, recognized login
failures per IP, p95 latency, service ERROR activity, traffic surges, and explicit
health failures. Five- and fifteen-minute signals retain separate identities;
health failures use fifteen-minute windows. Minimum support uses each metric's
valid denominator. Evidence IDs identify the supporting source records.

Training windows use preceding supported history only (24 hours, at least 12
windows by default). Validation and test use frozen training median/MAD
references, with same-daypart history when supported and an explicit broader
history fallback. Missing history never disables absolute rules. The summary
counts suppressions by reason, service, scope, resolution, split, and daypart.
Thresholds remain documented starting hypotheses, not production calibration.
See the [Phase 3 contract](docs/04-data-contracts.md#phase-3-implemented-interface).
This stage has no additional dependencies, ML, incident grouping, or accuracy
evaluation; `pipeline_complete` remains false.
