# Server Log Anomaly Detection

Detect unusual server activity in CSV logs and group related alerts into incidents with severity, confidence, explanations, and source evidence.

## Detection strategies

Both detectors use 5- and 15-minute windows:

- **Rules:** Explicit thresholds catch HTTP errors, repeated login failures, and failed health checks. Historical median/MAD baselines identify unusual errors, latency, and traffic; recurring daily traffic patterns are suppressed.
- **ML:** Isolation Forest learns unusual combinations of traffic, error, latency, and time features. Preprocessing and models are fitted on training data; scores above the validation set's 99th percentile trigger alerts by default.

Rules describe specific conditions; ML finds unusual combinations. Inference reuses saved models, thresholds, and rule baselines without retraining. Alerts are candidates for investigation, not confirmed failures.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
```

## Run

CSV columns: `timestamp,service,log_level,http_status,response_time_ms,endpoint,source_ip,message`.
Timestamps use `YYYY-MM-DD HH:MM:SS`. Default training uses four days for fitting and one for validation, with sufficient supported windows.

Log files are local and excluded from GitHub. Supply your own CSVs and replace the input paths below. Locally, `heldout.csv` contains rows excluded from fitting and calibration.

```bash
# Train once; a verified matching bundle is reused on repeat runs.
python -m log_anomaly train \
  --input logs/server_application_logs_100k.csv \
  --config configs/default.json \
  --output models/server-logs

# Score observations at or after the saved calibration period ends.
python -m log_anomaly infer \
  --input logs/heldout.csv \
  --config configs/default.json \
  --model-bundle models/server-logs \
  --output outputs/incidents
```

Use a new inference output directory for each run.

- **Training saves:** `estimators.pkl` and `metadata.json` (including rule baseline state).
- **Inference saves:** `incidents.json`, `incidents.csv`, and `summary.md`.

Tests: `python -m pytest -q`. Models, logs, and generated outputs remain Git-ignored.
