"""Phase 7 publication, recovery, reproducibility, and stage interoperability."""

import csv
import gzip
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
    return main(['run', '--debug-artifacts', '--input', str(source), '--config', str(config), '--output', str(output)])


def test_train_reuses_matching_bundle_without_fitting(inputs, tmp_path, monkeypatch, capsys):
    from log_anomaly import ml as detector
    from dataclasses import replace

    config = load_config(inputs[1], RunConfig)
    output = tmp_path / 'model'
    pipeline.train(inputs[0], config, output)
    before = {p.name: pipeline._file_hash(p) for p in output.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError('matching bundle must not retrain')

    monkeypatch.setattr(detector, 'fit_and_score', forbidden)
    args = ['train', '--input', str(inputs[0]), '--config', str(inputs[1]), '--output', str(output)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['reused'] is True
    assert before == {p.name: pipeline._file_hash(p) for p in output.iterdir()}
    with pytest.raises(FileExistsError, match='does not match input/config'):
        pipeline.train(inputs[0], replace(config, ml=replace(config.ml, n_estimators=11)), output)
    with inputs[0].open('a') as handle:
        handle.write('\n')
    with pytest.raises(FileExistsError, match='does not match input/config'):
        pipeline.train(inputs[0], config, output)
    (output / 'estimators.pkl').write_bytes(b'corrupted')
    from log_anomaly.config import InputError
    with pytest.raises(InputError, match='hash mismatch'):
        pipeline.train(inputs[0], config, output)


def test_empty_inference_input_explains_unset_variable(inputs, tmp_path, capsys):
    output = tmp_path / 'incidents'
    assert main(['infer', '--input', '', '--config', str(inputs[1]),
                 '--model-bundle', str(tmp_path / 'model'), '--output', str(output)]) == 2
    assert '--input is empty' in capsys.readouterr().err
    assert not output.exists()


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
    assert not (first / 'signals.jsonl').exists()  # No duplicate combined export.
    for name in ('normalized.jsonl', 'windows.jsonl'):
        assert not (first / 'features' / name).exists()
        compressed = first / 'features' / (name + '.gz')
        with gzip.open(compressed, 'rb') as handle:
            expanded = handle.read()
        assert compressed.stat().st_size < len(expanded) / 4
    for name in ('incidents.json', 'incidents.csv', 'summary.md'):
        assert (first / name).samefile(first / 'reports' / name)
    assert (first / 'charts/volume.png').samefile(first / 'features/charts/volume.png')
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


@pytest.mark.parametrize('legacy', [False, True])
def test_saved_run_inference_is_frozen_deterministic_and_traceable(inputs, tmp_path, monkeypatch, legacy):
    from log_anomaly import ml as detector
    from log_anomaly import baselines
    from log_anomaly.features import build_windows
    from log_anomaly.ingest import read_logs
    from log_anomaly.parse import parse_messages
    from sklearn.ensemble import IsolationForest

    trained = tmp_path / 'trained'
    assert (cli(inputs, trained) if legacy else main([
        'train', '--input', str(inputs[0]), '--config', str(inputs[1]), '--output', str(trained)])) == 0
    config = load_config(inputs[1], RunConfig)
    metadata, estimators = pipeline.load_model_bundle(trained / 'ml' if legacy else trained)
    records = read_logs(inputs[0]).records
    parse_messages(records)
    original = build_windows(records, config.features)
    test_windows = [w for w in original.windows if w['split'] == 'test']
    expected = detector.score_windows(metadata, estimators, test_windows)

    with inputs[0].open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [r for r in reader if '2025-02-05 11:45:00' <= r['timestamp'] <= '2025-02-05 12:30:00']
    # Unknown service and a sparse known-service window, with authentic source IDs assigned on ingestion.
    unknown = {**rows[0], 'service': 'new-service', 'timestamp': '2025-02-05 12:00:00'}
    rows = [r for r in rows if not '2025-02-05 12:20:00' <= r['timestamp'] < '2025-02-05 12:25:00']
    rows += [unknown, {**rows[0], 'timestamp': '2025-02-05 12:20:00'}]
    new = tmp_path / 'new.csv'
    with new.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reversed(rows))  # Event-time ordering, not file order.

    def forbidden(*args, **kwargs):
        raise AssertionError('inference attempted fitting/calibration')

    monkeypatch.setattr(IsolationForest, 'fit', forbidden)
    monkeypatch.setattr(detector, 'fit_preprocessing', forbidden)
    monkeypatch.setattr(detector, 'calibrate', forbidden)
    monkeypatch.setattr(baselines, 'fit_baselines', forbidden)
    monkeypatch.setattr('log_anomaly.rules.fit_baselines', forbidden)
    before = {p: pipeline._file_hash(p) for p in trained.rglob('*') if p.is_file()}
    captured = {}
    original_score = detector.score_windows
    from log_anomaly import rules
    original_rules = rules.detect_rules

    def capture_scores(metadata, estimators, windows):
        scores = original_score(metadata, estimators, windows)
        captured['scores'] = [{**w, **s} for w, s in zip(windows, scores)]
        return scores

    def capture_rules(*args, **kwargs):
        result = original_rules(*args, **kwargs)
        captured['rules'] = result.signals
        return result

    monkeypatch.setattr(detector, 'score_windows', capture_scores)
    monkeypatch.setattr(rules, 'detect_rules', capture_rules)

    def inference(output, source=new):
        return main(['infer', '--input', str(source), '--config', str(inputs[1]),
                     '--model-bundle', str(trained), '--output', str(output)])

    first, second = tmp_path / 'infer1', tmp_path / 'infer2'
    assert inference(first) == inference(second) == 0
    assert {p.name for p in first.iterdir()} == {'incidents.json', 'incidents.csv', 'summary.md'}
    assert all(p.read_bytes() == (second / p.name).read_bytes() for p in first.iterdir())
    scores = captured['scores']
    assert {'scored', 'low_support', 'ml_unavailable', 'incomplete_window'} <= {s['status'] for s in scores}
    assert all(s['score'] is None for s in scores if s['service'] == 'new-service')
    for score in scores:
        if score['status'] == 'scored':
            index = next(i for i, w in enumerate(test_windows) if all(w[k] == score[k] for k in
                         ('window_start', 'window_minutes', 'scope', 'service', 'entity_key')))
            assert score['score'] == expected[index]['score']
    report = json.loads((first / 'incidents.json').read_text())
    assert report['incidents']
    assert report['provenance']['input_sha256'] == pipeline._file_hash(new)
    assert report['coverage']['unavailable_models']
    references = {r.record_id: r for r in read_logs(new).records}
    for incident in report['incidents']:
        assert 'chart' not in incident['evidence']
        for rid in incident['evidence']['record_ids']:
            row = references[rid]
            assert incident['start_time'] <= row.timestamp.isoformat() < incident['end_time']
            assert row.service in incident['services']
        for sample in incident['evidence']['samples']:
            assert sample['message'] == references[sample['record_id']].message
        for signal in incident['evidence']['signals']:
            if signal['detector_id'] == 'isolation_forest':
                key = f"{signal['window_minutes']}:{incident['entity']['scope']}:{incident['services'][0]}"
                assert signal['threshold']['score_gt'] == metadata['models'][key]['threshold']['threshold']
    frozen = (json.loads((trained / 'rules/baselines.json').read_text()) if legacy
              else metadata['reuse']['baselines'])['references']
    for signal in captured['rules']:
        for ref in signal['baseline_metrics'].values():
            if isinstance(ref, dict) and ref.get('mode') == 'frozen_training':
                assert any(all(ref[k] == r[k] for k in ('median', 'mad', 'cutoff', 'daypart')) for r in frozen)
    assert before == {p: pipeline._file_hash(p) for p in before}
    singleton = tmp_path / 'singleton.csv'
    with singleton.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**rows[0], 'timestamp': '2026-02-05 12:00:00'})
    assert inference(tmp_path / 'singleton', singleton) == 0
    assert json.loads((tmp_path / 'singleton/incidents.json').read_text())['incidents'] == []
    assert all(s['status'] == 'incomplete_window' for s in captured['scores'])
    # Earlier observations would leak calibration information; configuration changes are incompatible.
    assert inference(tmp_path / 'early', inputs[0]) == 2
    changed = json.loads(inputs[1].read_text())
    changed['features']['health_endpoints'] = ['/other-health']
    inputs[1].write_text(json.dumps(changed))
    assert inference(tmp_path / 'bad-config') == 2
    inputs[1].write_text(json.dumps(config.to_dict()))
    estimator_path = trained / ('ml/model/estimators.pkl' if legacy else 'estimators.pkl')
    estimator_path.write_bytes(b'invalid')
    assert inference(tmp_path / 'bad-model') == 2
    assert not (tmp_path / 'bad-model').exists()


def test_training_minimal_bundle_matches_legacy_and_needs_no_test_day(inputs, tmp_path):
    import pickle
    from log_anomaly import ml as detector
    from log_anomaly.features import build_windows, build_training_windows
    from log_anomaly.ingest import read_logs
    from log_anomaly.parse import parse_messages
    from log_anomaly.config import InputError

    config = load_config(inputs[1], RunConfig)
    records = read_logs(inputs[0]).records
    parse_messages(records)
    features = build_windows(records, config.features)
    streamed = build_training_windows(lambda end: (r for r in records if r.timestamp.isoformat() < end),
                                      [records[0], records[-1]], sorted({r.service for r in records}), config.features)
    assert streamed.windows == [w for w in features.windows if w['split'] != 'test']
    assert streamed.template_reference == features.template_reference
    original = detector.fit_and_score(features.windows, records, features.split_ranges,
                                     config.features.health_endpoints, config.ml)
    first, second = tmp_path / 'model1', tmp_path / 'model2'
    for command, path in (('train', first), ('run', second)):
        assert main([command, '--input', str(inputs[0]), '--config', str(inputs[1]), '--output', str(path)]) == 0
        assert {p.name for p in path.iterdir()} == {'metadata.json', 'estimators.pkl'}
    assert all(p.read_bytes() == (second / p.name).read_bytes() for p in first.iterdir())
    saved, estimators = pipeline.load_model_bundle(first)
    assert all(saved[k] == v for k, v in original.metadata.items())
    import numpy as np
    for key, estimator in estimators.items():
        reference = original.estimators[key]
        assert estimator.get_params() == reference.get_params()
        for actual, expected in zip(estimator.estimators_, reference.estimators_):
            for field, value in actual.tree_.__getstate__().items():
                np.testing.assert_array_equal(value, expected.tree_.__getstate__()[field])
    assert detector.score_windows(saved, estimators, features.windows) == detector.score_windows(
        original.metadata, original.estimators, features.windows)
    # A real train/validation-only CSV must train; a test day is an evaluation requirement only.
    source = tmp_path / 'history.csv'
    with inputs[0].open(newline='') as handle, source.open('w', newline='') as target:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(r for r in reader if r['timestamp'] < '2025-02-05 00:00:00')
    short = tmp_path / 'short'
    pipeline.train(source, config, short)
    short_meta, short_models = pipeline.load_model_bundle(short)
    assert short_meta['models'] == saved['models']
    assert pickle.dumps(short_models) == pickle.dumps(estimators)
    for field, value in (('dependencies', {**saved['dependencies'], 'scikit-learn': 'incompatible'}),
                         ('reuse_version', 'future')):
        changed = {**saved, field: value}
        changed.pop('metadata_sha256')
        import hashlib
        changed['metadata_sha256'] = hashlib.sha256(pipeline.json_text(changed).encode()).hexdigest()
        (first / 'metadata.json').write_text(pipeline.json_text(changed))
        with pytest.raises(InputError):
            pipeline.load_model_bundle(first)
    (first / 'metadata.json').write_text(pipeline.json_text({**saved, 'seed': 99}))
    with pytest.raises(InputError, match='hash mismatch'):
        pipeline.load_model_bundle(first)


@pytest.mark.parametrize('operation', ['train', 'infer'])
@pytest.mark.parametrize('failure', ['disk', 'corruption'])
def test_minimal_publication_failure_cleans_temporary_state(inputs, tmp_path, monkeypatch, operation, failure):
    config = load_config(inputs[1], RunConfig)
    model = tmp_path / 'model'
    pipeline.train(inputs[0], config, model)
    source = inputs[0]
    if operation == 'infer':
        with source.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [r for r in reader if r['timestamp'] >= '2025-02-05 00:00:00']
        source = tmp_path / 'unseen.csv'
        with source.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    original = pipeline._write_json

    def fail(path, value):
        if failure == 'disk':
            raise OSError('simulated disk full')
        original(path, value)
        path.write_text('{}')

    monkeypatch.setattr(pipeline, '_write_json', fail)
    output = tmp_path / 'failed'
    args = [operation, '--input', str(source), '--config', str(inputs[1]), '--output', str(output)]
    if operation == 'infer':
        args += ['--model-bundle', str(model)]
    assert main(args) != 0
    assert not output.exists() and not (tmp_path / 'failed.incomplete').exists()
    assert not list(tmp_path.glob('log-anomaly-*'))
    monkeypatch.setattr(pipeline, '_write_json', original)
    assert main(args) == 0
