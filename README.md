# server-log-anomaly-detection
Python-based Server Log Analysis and Anomaly Detection Pipeline

## Project planning

Read the [engineering documentation](docs/README.md) for requirements, system design, implementation phases, data contracts, detection methods, evaluation, and delivery instructions. Phases 1–2 are implemented; the full detection pipeline remains planned.

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
environment. Detection models, statistical baselines, alerts, and incident
grouping are not implemented. The manifest marks Phase 2 complete and keeps
`pipeline_complete` false.
