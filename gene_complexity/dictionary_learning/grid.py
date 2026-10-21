"""Run configurable training/validation dictionaries; no test evaluation or selection."""
import argparse
import contextlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from gene_complexity.dictionary_learning.training import run, load_features
from gene_complexity.preprocessing.classical import ROOT, digest


def candidates(config):
    for key in ('atoms', 'seeds', 'alpha'):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'{key} must be a nonempty list of unique numbers')
    for value in config['atoms'] + [config['epochs']]:
        if type(value) is not int or value < 1:
            raise ValueError('Atom counts and epochs must be positive integers')
    for seed in config['seeds']:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Seeds must be integers in [0, 2**32)')
    for alpha in config['alpha']:
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Penalties must be finite and positive')
    return [dict(atoms=k, alpha=float(a), seed=s, epochs=config['epochs'])
            for k, a, s in itertools.product(config['atoms'], config['alpha'], config['seeds'])]


def verify_existing(folder, expected):
    report_path = folder / 'pilot_report.json'
    if not report_path.is_file():
        raise ValueError(f'Incomplete candidate: {folder}; preserve it and choose a new output root')
    report = json.loads(report_path.read_text())
    if report.get('status') != 'pilot_complete' or report.get('test_accessed') is not False:
        raise ValueError(f'Invalid completion report: {folder}')
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Existing candidate has different {key}: {folder}')
    for name in ('dictionary.npz', 'train_codes.npz', 'validation_codes.npz'):
        if not (folder/name).is_file() or digest(folder/name) != report['output_hashes'].get(name):
            raise ValueError(f'Existing candidate artifact changed: {folder/name}')
    return report


def run_grid(config_path, features, output):
    config_path, features, output = map(Path, (config_path, features, output))
    config = json.loads(config_path.read_text())
    planned = candidates(config)
    # Validate current source artifacts even when all candidates are cached.
    data, provenance = load_features(features)
    if max(c['atoms'] for c in planned) > min(data['train']['X'].shape):
        raise ValueError('Atom count exceeds training matrix dimensions')
    del data
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output/'grid_config.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError('Config changed: use a new versioned output root')
    snapshot.write_text(json.dumps(config, indent=2)+'\n')
    rows = []
    for i, candidate in enumerate(planned, 1):
        # Round-trip float representation avoids collisions between distinct penalties.
        name = f"k{candidate['atoms']}_a{repr(candidate['alpha'])}_s{candidate['seed']}_e{candidate['epochs']}"
        folder = output/name
        expected = dict(atoms=candidate['atoms'], alpha=candidate['alpha'], seed=candidate['seed'],
            epochs_requested=candidate['epochs'],
            preprocessing_report_sha256=digest(features/'preprocessing.json'),
            manifest_sha256=provenance['manifest_sha256'],
            script_sha256=digest(Path(__file__).with_name('training.py')),
            sklearn_version=sklearn.__version__, numpy_version=np.__version__)
        if folder.exists():
            report = verify_existing(folder, expected)
            print(f'[{i}/{len(planned)}] Verified existing {name}', flush=True)
        else:
            print(f'[{i}/{len(planned)}] Fitting {name}', flush=True)
            with (output/f'{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                report = run(features, folder, **candidate)
        row = dict(candidate=name, artifact_directory=str(folder.resolve()), **candidate,
                   warning_count=report['warning_count'], epochs_run=report['epochs_run'],
                   reached_epoch_budget=report['reached_epoch_budget'])
        for split in ('train', 'validation'):
            row.update({f'{split}_{key}':value for key,value in report['metrics'][split].items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output/'grid_summary.csv', index=False)
    (output/'grid_report.json').write_text(json.dumps(dict(
        status='complete', candidate_count=len(rows), test_accessed=False,
        config_sha256=digest(snapshot), runner_sha256=digest(__file__),
        preprocessing_report_sha256=digest(features/'preprocessing.json'),
        manifest_sha256=provenance['manifest_sha256'],
        selection='No winner selected. Await logistic validation and revise candidate grid in a new version.'
    ), indent=2)+'\n')
    print(table[['candidate','validation_relative_mse','validation_active_fraction','warning_count']].to_string(index=False))
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/dictionary_grid_v1.json')
    p.add_argument('--features', type=Path, default=ROOT/'data/processed/baron_classical_v1')
    p.add_argument('--output', type=Path, default=ROOT/'results/dict-learning/baron_grid_v1')
    a = p.parse_args(argv)
    run_grid(a.config, a.features, a.output)


if __name__ == "__main__":
    main()
