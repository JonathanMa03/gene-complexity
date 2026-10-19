"""Shared, fail-closed train/validation artifact checks. Never opens test.npz."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def default_config():
    from run_all_validation_coarse_to_fine import DICTIONARY_RUNS
    return dict(expression_dir='data/processed/baron_classical_v1',
                split_dir='data/processed/baron_splits_final',
                dictionary_dir='results/dict-learning/baron_grid_v1',
                dictionary_runs=DICTIONARY_RUNS,
                scgpt_path='code/scgpt/pretraining_handoff/outputs/baron/baron_scgpt_full.npz')


def checked_rows(path, key, manifest, split, reference_ids=None):
    with np.load(path, allow_pickle=False) as z:
        x, ids = z[key], z['cell_ids'].astype(str)
        if ids.ndim != 1 or x.ndim != 2 or len(ids) != len(x):
            raise ValueError(f'{path}: invalid dimensions')
        expected = manifest.index[manifest.split == split]
        if len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError(f'{path}: cell membership differs from manifest')
        for field, column in [('labels', 'cell_type'), ('donors', 'donor')]:
            if not np.array_equal(z[field].astype(str), manifest.loc[ids, column].to_numpy()):
                raise ValueError(f'{path}: {field} differs from manifest')
        if not np.isfinite(x).all():
            raise ValueError(f'{path}: nonfinite features')
        if reference_ids is not None:
            lookup = dict(zip(ids, range(len(ids))))
            x = x[[lookup[c] for c in reference_ids]]
            ids = reference_ids
        return x, ids


def load_inputs(config, methods=("baseline", "dictionary", "scgpt")):
    if not set(methods) <= {"baseline", "dictionary", "scgpt"} or not methods:
        raise ValueError("Unknown or empty methods")
    hashes = {}
    def record(path, expected=None):
        path = Path(path)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Hash mismatch: {path}')
        hashes[str(path)] = actual
        return path
    def read(path):
        return json.loads(record(path).read_text())
    def resolve(key):
        return ROOT / config[key]

    split_dir, exp, dictionary = map(resolve, ['split_dir', 'expression_dir', 'dictionary_dir'])
    settings = read(split_dir / 'split_settings.json')
    if settings['status'] != 'final' or not settings['scgpt_confirmed']:
        raise ValueError('Requires final, scGPT-confirmed splits')
    manifest_path = record(split_dir / 'baron_split_manifest.csv', settings['manifest_sha256'])
    manifest = pd.read_csv(manifest_path, dtype=str)
    required = ['cell_id', 'cell_type', 'donor', 'split']
    if manifest[required].isna().any().any() or manifest.cell_id.duplicated().any():
        raise ValueError('Missing or duplicate manifest metadata')
    if set(manifest.split) != {'train', 'validation', 'test'}:
        raise ValueError('Unexpected manifest splits')
    if manifest.groupby('split').size().to_dict() != settings['split_counts']:
        raise ValueError('Split counts differ from settings')
    manifest = manifest.set_index('cell_id')
    prep_path = exp / 'preprocessing.json'
    prep = read(prep_path)
    if prep['manifest_sha256'] != settings['manifest_sha256'] or prep['input_sha256'] != settings['input_sha256']:
        raise ValueError('Preprocessing provenance differs from finalized split')
    record(exp / 'split_manifest.csv', settings['manifest_sha256'])
    record(exp / 'preprocessing.npz', prep['parameters_sha256'])
    expression, ids, labels = {}, {}, {}
    for split in ['train', 'validation']:
        path = record(exp / f'{split}.npz', prep['artifacts'][f'{split}.npz']['sha256'])
        expression[split], ids[split] = checked_rows(path, 'X', manifest, split)
        labels[split] = manifest.loc[ids[split], 'cell_type'].to_numpy()
    if set(labels['train']) != set(labels['validation']):
        raise ValueError('Train/validation class coverage differs')
    datasets = [('Expression', expression['train'], expression['validation'])]
    if 'dictionary' in methods and (not config['dictionary_runs'] or {'Expression', 'scGPT'} & set(config['dictionary_runs'])):
        raise ValueError('Invalid dictionary representation names')
    for name, folder in (config['dictionary_runs'].items() if 'dictionary' in methods else []):
        directory = dictionary / folder
        report = read(directory / 'pilot_report.json')
        if report['status'] != 'pilot_complete' or report['test_accessed']:
            raise ValueError(f'{name}: dictionary is not a train/validation run')
        if report['manifest_sha256'] != settings['manifest_sha256'] or report['preprocessing_report_sha256'] != digest(prep_path):
            raise ValueError(f'{name}: dictionary provenance mismatch')
        record(directory / 'dictionary.npz', report['output_hashes']['dictionary.npz'])
        codes = {}
        for split in ['train', 'validation']:
            if report['input_artifact_hashes'][split] != prep['artifacts'][f'{split}.npz']['sha256']:
                raise ValueError(f'{name}: dictionary expression input mismatch')
            filename = f'{split}_codes.npz'
            path = record(directory / filename, report['output_hashes'][filename])
            codes[split], _ = checked_rows(path, 'vectors', manifest, split, ids[split])
            if codes[split].shape[1] != report['atoms']:
                raise ValueError(f'{name}: atom count mismatch')
        datasets.append((name, codes['train'], codes['validation']))
    meta = {}
    if 'scgpt' in methods:
        scpath = resolve('scgpt_path')
        meta = read(scpath.with_suffix('.metadata.json'))
        if meta['status'] != 'PASS':
            raise ValueError('scGPT metadata does not report PASS')
        record(scpath, meta['output_sha256'])
        with np.load(scpath, allow_pickle=False) as z:
            vectors, scids = z['vectors'], z['cell_ids'].astype(str)
        if vectors.ndim != 2 or scids.ndim != 1 or len(vectors) != len(scids) or len(set(scids)) != len(scids):
            raise ValueError('Invalid scGPT dimensions or duplicate IDs')
        if list(vectors.shape) != meta['shape'] or not np.isfinite(vectors).all():
            raise ValueError('Invalid scGPT shape or values')
        excluded = pd.read_csv(record(split_dir / 'baron_exclusions.csv'), dtype=str)
        if set(scids) != set(manifest.index) | set(excluded.cell_id):
            raise ValueError('scGPT cell coverage differs from shared cohort plus exclusions')
        lookup = dict(zip(scids, range(len(scids))))
        datasets.append(('scGPT', *[vectors[[lookup[c] for c in ids[s]]] for s in ['train', 'validation']]))
    return dict(datasets=datasets, y_train=labels['train'], y_val=labels['validation'],
                input_hashes=hashes, split_counts=settings['split_counts'],
                scgpt_model_id=meta.get('model_id'), ids=ids, manifest=manifest)
