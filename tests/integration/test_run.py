"""Phase 7 publication, recovery, reproducibility, and stage interoperability."""

import csv
import json
from pathlib import Path

import pytest

from log_anomaly.cli import main
from log_anomaly.config import RunConfig, load_config
from log_anomaly import pipeline

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inputs(tmp_path, csv_file, synthetic_rows):
    source = csv_file(synthetic_rows())
    config = json.loads((ROOT / 'configs/default.json').read_text())
    config['features'].update(train_days=1, validation_days=1)
    config['rules'].update(minimum_statuses=10, minimum_latencies=10, minimum_records=10)
    config['ml'].update(resolutions=[5], scopes=['service'], minimum_records=5,
                        minimum_training_windows=50, minimum_validation_windows=10,
                        n_estimators=10, stability_seeds=[])
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return source, path


def cli(inputs, output):
    source, config = inputs
    return main(['run', '--input', str(source), '--config', str(config), '--output', str(output)])


def test_run_reproducible_consistent_and_stage_compatible(inputs, tmp_path):
    source_hash = pipeline._file_hash(inputs[0])
    first, second = tmp_path / 'first', tmp_path / 'second'
    assert cli(inputs, first) == cli(inputs, second) == 0
    manifest = json.loads((first / 'manifest.json').read_text())
    other = json.loads((second / 'manifest.json').read_text())
    assert manifest['pipeline_complete'] and manifest['phase'] == 7
    assert manifest['run_id'] == other['run_id']
    assert manifest['counts']['parsed_count'] == manifest['counts']['accepted_count'] == 5184
    assert manifest['counts']['rejected_count'] == 0
    assert manifest['counts']['available_models'] == 1
    for name, digest in manifest['artifacts'].items():
        assert pipeline._file_hash(first / name) == digest
        # Only stage manifests contain runtime and absolute provenance paths.
        if not name.endswith('manifest.json'):
            assert digest == other['artifacts'][name], name
    assert pipeline._file_hash(inputs[0]) == source_hash
    assert not (tmp_path / 'first.incomplete').exists()
    report = json.loads((first / 'incidents.json').read_text())
    with (first / 'incidents.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == manifest['counts']['incidents'] > 0
    assert [r['incident_id'] for r in rows] == [i['incident_id'] for i in report['incidents']]
    assert len((first / 'signals.jsonl').read_text().splitlines()) == (
        manifest['counts']['rule_signals'] + manifest['counts']['ml_signals'])
    pipeline._validate_delivery(first, load_config(inputs[1], RunConfig))
    # Publication must relocate provenance so existing commands can consume these stages.
    for stage in ('rules', 'ml', 'reports'):
        child = json.loads((first / stage / 'manifest.json').read_text())
        assert Path(child['source_run']['path']) == first / 'features'
    labels = tmp_path / 'labels.json'
    labels.write_text('{"schema_version":"1.0","events":[]}')
    assert main(['evaluate', '--input', str(first / 'reports'), '--labels', str(labels),
                 '--config', str(ROOT / 'configs/evaluation.json'), '--output', str(tmp_path / 'evaluation')]) == 0
    assert pipeline.load_model_bundle(first / 'ml')[1]
    assert cli(inputs, first) == 4


@pytest.mark.parametrize('kind', ['missing', 'schema', 'empty', 'config', 'nested', 'reserved'])
def test_run_invalid_input(inputs, tmp_path, kind):
    source, path = inputs
    output = tmp_path / 'bad'
    if kind == 'missing':
        source = tmp_path / 'missing.csv'
    elif kind == 'schema':
        source.write_text('timestamp,message\nbad,x\n')
    elif kind == 'empty':
        source.write_text(source.read_text().splitlines()[0] + '\n')
    elif kind == 'config':
        path.write_text('{"schema_version":"1.0"}')
    elif kind == 'nested':
        config = json.loads(path.read_text())
        config['ml']['unknown'] = True
        path.write_text(json.dumps(config))
    else:
        output = tmp_path / 'bad.incomplete'
    assert cli((source, path), output) == 2
    assert not output.exists()


@pytest.mark.parametrize('failure,code', [('stage', 3), ('disk', 4), ('corruption', 3)])
def test_run_failure_and_fresh_recovery(inputs, tmp_path, monkeypatch, failure, code):
    original = pipeline.report_incidents

    def fail(*args):
        if failure == 'disk':
            raise OSError('simulated disk full')
        if failure == 'stage':
            raise RuntimeError('simulated grouping failure')
        manifest = original(*args)
        (args[2] / 'incidents.csv').write_text('corrupted export\n')
        return manifest

    monkeypatch.setattr(pipeline, 'report_incidents', fail)
    output = tmp_path / 'failed'
    source_hash = pipeline._file_hash(inputs[0])
    assert cli(inputs, output) == code
    assert not output.exists()
    state = json.loads((tmp_path / 'failed.incomplete/manifest.json').read_text())
    assert not state['pipeline_complete'] and state['status'] == 'incomplete'
    assert 'error' in state
    assert cli(inputs, output) == 4
    monkeypatch.setattr(pipeline, 'report_incidents', original)
    assert cli(inputs, tmp_path / 'recovered') == 0
    assert pipeline._file_hash(inputs[0]) == source_hash


def test_run_requires_ml_and_keeps_partial_diagnostics(inputs, tmp_path):
    config = json.loads(inputs[1].read_text())
    config['ml']['minimum_training_windows'] = 1000000
    inputs[1].write_text(json.dumps(config))
    output = tmp_path / 'unsupported'
    assert cli(inputs, output) == 3
    assert not output.exists()
    state = json.loads((tmp_path / 'unsupported.incomplete/manifest.json').read_text())
    assert state['status'] == 'partial' and not state['pipeline_complete']
    assert (tmp_path / 'unsupported.incomplete/ml/ml_summary.json').exists()


def test_default_reuses_frozen_stage_settings():
    config = load_config(ROOT / 'configs/default.json', RunConfig)
    for section, name in (('features', 'features'), ('rules', 'detection'), ('ml', 'ml'), ('incidents', 'incidents')):
        assert config.to_dict()[section] == json.loads((ROOT / f'configs/{name}.json').read_text())
