# Server Log Anomaly Detection

A local Python batch pipeline that turns server logs into traceable incident
candidates. It validates and parses CSV records, builds event-time features,
runs statistical rules and Isolation Forest independently, and groups their
signals into JSON, CSV, and Markdown reports. Phases 1–7 are implemented.
No alert is automatically confirmed, and the pipeline performs no remediation.

```text
Raw CSV → validation / parsing → 5- and 15-minute features
                                     ├─ rules / historical baselines
                                     └─ training-only Isolation Forest
                                               ↓
                                  incident grouping → reports / charts
```

The implementation uses the standard library, NumPy, scikit-learn, and
Matplotlib. `pipeline.py` coordinates the existing stage functions in one
process; there is no external orchestration service.

## Setup

Verified on Ubuntu 26.04.1, x86-64, Python **3.14.4**. Package metadata permits
Python ≥3.11, but the supplied dependency lock and release verification use
3.14.4; other Python/OS combinations have not been verified.
From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m log_anomaly --help
python -m pytest -q
```

Install the lock first: it includes the pinned build tooling, runtime libraries,
and test dependencies. Download access to the Python package index is needed
for a new environment. All subsequent analysis is local.

## Input and a fresh-clone demonstration

**The original 100,000-row CSV and `docs/` are intentionally absent from GitHub.**
They are not required for setup, tests, or the demonstration. The repository
includes a deterministic synthetic generator with realistic daily/weekend
traffic, background errors, health checks, and injected faults:

```bash
python scripts/generate_fixture_logs.py --seed 11 --output artifacts/demo-input
python -m log_anomaly run --input artifacts/demo-input/logs.csv --config configs/default.json --output artifacts/demo-run
```

This generates seven days of logs for five e-commerce services. It also writes
`artifacts/demo-input/labels.json` for the evaluator alone; `run` never reads
labels. Seeds change fault timing, service, IP, and intensity. The generator
refuses to overwrite its destination. Generated data, models, and reports stay
under the Git-ignored `artifacts/` directory; do not add them to Git.

If you have the original dataset, place it locally under `logs/` and run:

```bash
python -m log_anomaly run --input logs/server_application_logs_100k.csv --config configs/default.json --output artifacts/full-run
```

For other data, supply a UTF-8 CSV (optional BOM, quoted commas/newlines) with
these eight required columns:

```csv
timestamp,service,log_level,http_status,response_time_ms,endpoint,source_ip,message
```

Timestamps must be `YYYY-MM-DD HH:MM:SS`. Services must be nonempty; known
services are `api-gateway`, `auth-service`, `orders-service`, `payment-service`,
and `search-service`. Statuses are integers 100–599, latency is a nonnegative
integer in milliseconds, endpoints are paths, and source IPs must parse as IP
addresses. Unknown services/templates are retained with limited ML coverage.
Bad dates/empty services/wrong-width rows are quarantined; invalid numeric/IP
fields become null with quality flags, not zero. Missing columns or broken CSV
framing are fatal. Duplicate records are retained and counted as candidates.

Default splits use the first accepted event's day: four training days, one
validation day, then test. Input must reach at least day six. For shorter data,
change the positive `train_days`/`validation_days` in the `features` section,
keeping at least one test day and enough supported windows for ML. Missing
calendar coverage is not proof of normal operation. Timezone stays
`unspecified` unless you have authoritative source metadata.

## Outputs and interpretation

Open `artifacts/demo-run/summary.md` first. Every successful run has a root
`manifest.json` with `phase: 7`, `status: complete`, and
`pipeline_complete: true`. All paths below are relative to the chosen output:

| Output | Contents |
| --- | --- |
| `incidents.json` | Authoritative nested report: IDs, half-open time bounds, service/entity, category, severity, confidence, metrics, baselines, evidence IDs/samples, explanations, limitations, and related incidents |
| `incidents.csv` | One row per incident; nested fields are JSON in quoted cells; full evidence IDs remain in JSON; formula-like text is escaped |
| `summary.md` | Counts, incident table, and evidence-based details for held-out related groups |
| `signals.jsonl` | Rule signals followed by ML signals, with thresholds, support, source IDs, and split |
| `quality.json`, `rejects.jsonl` | Complete accepted/rejected accounting, parser coverage, quality flags, and quarantined records |
| `charts/` | Volume, status, latency, log-level, and annotated incident PNG charts |
| `features/` | Normalized records, windows, training template reference, EDA statistics, and Phase 2 manifest |
| `rules/` | Rule signals, frozen baselines, suppression counts, config, and Phase 3 manifest |
| `ml/` | Independent ML signals/summary and `model/{metadata.json,estimators.pkl}`, with Phase 4 manifest |
| `reports/` | Original Phase 5 report bundle, also usable by the separate `evaluate` command |
| `evaluation/comparison.json` | Splits, rule/ML/grouped counts, and ML coverage; accuracy is null because `run` receives no labels |
| `config.json`, `manifest.json` | Resolved nested config, source/config/artifact hashes, seed, versions, counts, sizes, stage IDs, and runtime |

The nested stage bundles preserve the individual command interfaces; a few
small report/chart exports are also copied to the root for convenience.
Source IDs are `<input SHA-256>:<one-based CSV record ordinal>`, not physical
line numbers. Resolve them through `features/normalized.jsonl` and the original
CSV. Counts are log records, not distinct user transactions across services.
Window totals overlap across scopes/resolutions and must not be summed as logs.

`review` denotes ML-only or traffic candidates; `warning`/`high` denote
rule-supported candidates. Confidence measures evidence strength, not a
probability. Service, endpoint, and auth-IP scopes remain separate; related
incidents may be linked. A root-cause hypothesis is explicitly marked as such.
No-alert runs have an empty incidents array, a CSV header, and an explicit
no-alert summary.

Example output excerpt from the seed-11 demonstration (the JSON report retains
the remaining metrics, source IDs, samples, and limitations):

```json
{
  "incident_id": "inc-ba3deaddbc1ee39d0cee",
  "start_time": "2026-11-14T00:00:00",
  "end_time": "2026-11-14T00:30:00",
  "services": ["auth-service"],
  "category": "auth_failures",
  "entity": {"scope": "auth_ip", "endpoint": null, "source_ip": "203.0.113.212"},
  "severity": "warning",
  "split": "test",
  "detector_ids": ["repeated_auth_failures"],
  "evidence": {"record_count": 122}
}
```

The strongest contributing window contains 109 recognized login failures among
109 attempts, meeting the ≥10 failures and ≥80% rate limits. This is a
rule-supported candidate; the report does not assert malicious intent.
The complete demo accounts for 94,754 records, 76,239 windows, 120 rule signals,
20 ML signals, and 54 incidents, with nine available models.

## Detection and chronological separation

`configs/default.json` embeds the strict, fully resolved feature, rule, ML, and
incident configs. Its rule section equals the improved Phase 6
`configs/detection.json`. Unknown/missing keys and invalid values fail early.

Rules detect supported HTTP 5xx bursts, recognized login failures, latency
increases, service ERROR activity, traffic surges, and explicit health failures.
For example, the absolute HTTP rule requires ≥5 errors, ≥20 valid business
statuses, and a 5xx rate ≥0.20. Latency needs ≥20 valid observations, p95 >1000 ms,
and >3× historical window p95. Health traffic has separate statistics. Training
references use preceding windows; validation/test references stay frozen at
training end. Median/MAD history prefers supported dayparts.

The recurrence-aware traffic rule compares a candidate against similar clock
times on prior training days: seven-day lookback, at least two days, ±15 minutes,
and a 1.25 ratio. It suppresses ordinary recurring batches. It uses timestamps
as time context, never special anomaly dates, IDs, or labels.

Isolation Forest uses supported numeric window features: counts, status/level
rates, latency, IP counts, authentication outcomes, upstream/search latency,
health, and time context. Raw identifiers are not numerical model features.
Preprocessing, missingness handling, feature selection, and 200 trees are fitted
on training only (seed 42). Each resolution/scope/service needs ≥100 training
windows with ≥20 records and ≥20 supported validation windows. Strict score
`> validation p99` controls alert volume; it is not a false-positive guarantee.
Only validation and test windows emit ML signals. Validation is the calibration
period; test remains held out. Existing sensitivity experiments are retained;
Phase 7 changes no detector thresholds or model choices.

## Evaluation, observed failure, and limitations

Phase 6 uses deterministic development seeds 11–13 and frozen final seeds 21–23.
Matching is maximum-total-IoU, one-to-one, with IoU ≥0.30 and compatible
category/service/entity. Extra predictions cannot inflate recall. Final held-out
results pooled over 27 labelled events:

| Mode | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Improved rules | 21 | 13 | 6 | 0.618 | 0.778 | 0.689 |
| ML only | 12 | 39 | 15 | 0.235 | 0.444 | 0.308 |
| Combined | 21 | 51 | 6 | 0.292 | 0.778 | 0.424 |

The observed failure was ordinary nightly batches triggering traffic alerts.
Development validation reduced rule FP from 23 to 17 while retaining 13 TP.
After freezing the recurrence-aware configuration, final held-out rule FP fell
from **17 to 13**, with TP unchanged at 21 and recall unchanged at 0.778;
rule F1 rose from 0.646 to 0.689. The persistent recurrence regression is in
`tests/unit/test_rules.py`. ML-only results did not improve; combined FP fell
from 55 to 51. **ML adds candidate findings but does not improve synthetic
incident-level F1.** Both approaches remain visible in the delivered reports.

Remaining limits:

- Auth-IP ML has insufficient training support; rules still evaluate it.
- Minimum-support guards can miss sparse error bursts. Unsupported windows
  receive no ML score; another service's model is never substituted.
- Separate scopes and related categories can yield extra incident predictions.
- The original dataset has no authoritative incident labels. These synthetic
  precision/recall/F1 values are not accuracy estimates for the original logs.
- Seven days cannot establish reliable weekly seasonality; weekend holdout is
  a distribution shift. Correlation is not root-cause proof.
- This is a local batch delivery, with no live alerting, dashboard, or production
  calibration. Batch runtime is not online detection delay.

Reproduce the frozen experiment without changing thresholds:

```bash
python scripts/evaluate_fixtures.py --output artifacts/evaluation-dev
python scripts/evaluate_fixtures.py --output artifacts/evaluation-final --final
python -m log_anomaly evaluate --input artifacts/demo-run/reports --labels artifacts/demo-input/labels.json --config configs/evaluation.json --output artifacts/demo-evaluation
```

The experiment writes per-seed stages/evaluations plus `comparison.json` and
`comparison.md`. Labels never enter the detectors. Do not tune on final seeds.

## Benchmark and verification

The original 100,000-row input was processed on an AMD Ryzen 5 5600U (6 cores,
12 threads), 14.4 GiB RAM, Ubuntu 26.04.1, Python 3.14.4, using the dependency
lock. Source SHA-256:
`79766aeb46aa0516ab4f54eaa1a9397c12ff1159d8ad2b968616d89dbc811238`.

| Measurement | Result | Acceptance target |
| --- | ---: | ---: |
| End-to-end process wall time | 40.43 s | ≤120 s |
| Peak process RSS (GNU time) | 1,019,508 KiB = 995.6 MiB | <1 GiB (1,048,576 KiB) |
| Accepted / rejected records | 100,000 / 0 | All source records accounted for |
| Source CSV size | 14,645,090 bytes | Original preserved |
| Windows (both resolutions/all scopes) | 75,744 | Traceable counts |
| Rule / ML signals | 201 / 18 | Independent detector evidence |
| Grouped incidents | 77 | JSON/CSV consistency |
| Available ML models | 10 | Both detectors operate |
| JSON / CSV / summary sizes | 1,086,415 / 430,268 / 57,531 bytes | Usable report formats |
| All artifacts excluding root manifest | 299,760,847 bytes | Recorded in manifest |

Incidents comprise 61 rule-only, 8 ML-only, and 8 corroborated candidates;
46 are in training, 21 validation, and 10 test. Nothing is labelled confirmed.
Repeated full runs produced identical non-manifest artifacts; manifests retain
runtime and absolute provenance paths. The final incident report also matches
Phase 6 byte-for-byte. Streaming artifact hashes removed unnecessary whole-file
hash buffers without changing results. Memory passes with limited headroom;
these measurements do not establish a budget for larger inputs or other hosts.
The clean-environment full-source repeat took 47.16 s and peaked at
1,025,176 KiB (1,001.1 MiB), also passing both targets with identical artifacts.

To measure your machine, choose a fresh output path:

```bash
/usr/bin/time -v python -m log_anomaly run --input logs/server_application_logs_100k.csv --config configs/default.json --output artifacts/benchmark-run
```

The final submission audit used a new Python 3.14.4 virtual environment and a
copy containing only Git-tracked submission files, excluding `docs/`, original
logs, existing artifacts, and the working environment. All locked dependencies
installed from locally cached wheels; installation, CLI help, tests, generation,
and the one-command demonstration passed. The full local suite passed **195
tests**. The clean-copy suite passed **194**, with the
private-source integration test intentionally skipped because its CSV was
absent; the full-source benchmark above covers that input separately. The CI
workflow in `.github/workflows/ci.yml` runs locked installation, CLI help, all
tests, a development-seed evaluation, and the full pipeline on generated input.
Hosted CI must run after publication; local verification does not claim a
GitHub Actions run.

The audit fixed the first-evidence timestamp for out-of-order CSV records and
added a regression test. All 42 saved full-run artifact hashes and all 100,000
original record references were verified. The fresh seed-11 demonstration's
incident JSON matched the saved evaluation report exactly; detector settings
and the reported Phase 6 metrics are unchanged.

Requirement evidence available in the repository:

| Requirement | Evidence |
| --- | --- |
| R01 ≥50k records | [Full-file benchmark](#benchmark-and-verification); [generator](scripts/generate_fixture_logs.py) |
| R02 parsing/quality | [Ingestion tests](tests/unit/test_ingest.py), [parser tests](tests/unit/test_parse.py) |
| R03 rules and ML | [Rules](src/log_anomaly/rules.py), [ML](src/log_anomaly/ml.py), [evaluation results](#evaluation-observed-failure-and-limitations) |
| R04 grouping | [Incident grouping and policy tests](tests/unit/test_incidents.py) |
| R05 explanations | [Report implementation](src/log_anomaly/incidents.py), [output examples](#outputs-and-interpretation) |
| R06 held-out evaluation | [Evaluator](src/log_anomaly/evaluate.py), [reproducible experiment](scripts/evaluate_fixtures.py) |
| R07 one-command run | [Pipeline](src/log_anomaly/pipeline.py), [end-to-end tests](tests/integration/test_run.py) |
| R08 tests | [Test suites](tests), [CI workflow](.github/workflows/ci.yml) |
| R09 no special anomaly identities | [Shifted-time/IP rule tests](tests/unit/test_rules.py), [ML leakage tests](tests/unit/test_ml.py) |
| R10 rules vs ML explanation | [Detection methods](#detection-and-chronological-separation) |
| R11 demonstrated improvement | [Before/after metrics](#evaluation-observed-failure-and-limitations), [recurrence regression](tests/unit/test_rules.py) |
| R12 reproducible delivery | [Setup](#setup), [fresh-clone demonstration](#input-and-a-fresh-clone-demonstration) |
| R13 EDA/enrichment | [Features](src/log_anomaly/features.py), [parsing](src/log_anomaly/parse.py), [charts](src/log_anomaly/report.py) |

## Individual commands

Existing stage commands remain available. Use new output directories:

```bash
python -m log_anomaly ingest --input artifacts/demo-input/logs.csv --config configs/ingest.json --output artifacts/ingest-only
python -m log_anomaly features --input artifacts/demo-input/logs.csv --config configs/features.json --output artifacts/features-only
python -m log_anomaly detect --input artifacts/features-only --config configs/detection.json --output artifacts/rules-only
python -m log_anomaly ml --input artifacts/features-only --rules artifacts/rules-only --config configs/ml.json --output artifacts/ml-only
python -m log_anomaly incidents --input artifacts/features-only --rules artifacts/rules-only --ml artifacts/ml-only --config configs/incidents.json --output artifacts/reports-only
```

Stage manifests deliberately retain `pipeline_complete: false`; each certifies
only that stage. `run` validates source hashes, stage compatibility, every
incident's evidence, and JSON/CSV content before publishing the root delivery.

## Troubleshooting and recovery

| Exit | Meaning / action |
| --- | --- |
| 0 | Complete. Check the root manifest and summary. |
| 2 | Invalid input/config. Check required columns, timestamps, split coverage, and strict config keys. Check quarantine if >5% of records are rejected or all records are unusable. |
| 3 | Required stage/dependency failure. Install the lock and inspect the error/staging manifest. No supported ML model leaves `status: partial` and diagnostics in `ml/ml_summary.json`; provide sufficient input/support before claiming full delivery. |
| 4 | Output failure. Check free space/permissions and choose a new output directory. Existing output or `.incomplete` paths are never overwritten. |

Stages emit structured progress on stderr; completion is printed on stdout.
Failures retain `<output>.incomplete` for diagnosis and never publish the final
output directory. Do not consume staging as a complete delivery or rename it
to bypass validation. Recover by rerunning the raw input into a **fresh** output
path. Keep source data stable during the run. Model bundles are trusted local
pickle artifacts and must not be loaded from untrusted sources. Reports contain
bounded original message samples; review sensitive data before sharing.
