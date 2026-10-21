"""Fit and export a training dictionary with frozen validation encoding."""
import argparse
import json
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode
from threadpoolctl import threadpool_limits
from gene_complexity.preprocessing.classical import ROOT, digest


def load_features(root):
    root = Path(root)
    report = json.loads((root / 'preprocessing.json').read_text())
    if digest(root / 'split_manifest.csv') != report['manifest_sha256']:
        raise ValueError('Manifest hash mismatch')
    manifest = pd.read_csv(root / 'split_manifest.csv', dtype=str, keep_default_na=False)
    if not manifest.cell_id.is_unique:
        raise ValueError('Duplicate manifest cell IDs')
    if digest(root / 'preprocessing.npz') != report['parameters_sha256']:
        raise ValueError('Preprocessing parameter hash mismatch')
    loaded = {}
    for split in ('train', 'validation'):
        path = root / f'{split}.npz'
        if digest(path) != report['artifacts'][path.name]['sha256']:
            raise ValueError(f'{split} feature hash mismatch')
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        expected = manifest.loc[manifest.split == split]
        for key, column in [('cell_ids', 'cell_id'), ('labels', 'cell_type'), ('donors', 'donor')]:
            if not np.array_equal(data[key], expected[column]):
                raise ValueError(f'{split} {key} differ from manifest order')
        if data['X'].shape != (len(expected), len(data['gene_ids'])) or not np.isfinite(data['X']).all():
            raise ValueError('Invalid feature shape or values')
        loaded[split] = data
    if not np.array_equal(loaded['train']['gene_ids'], loaded['validation']['gene_ids']):
        raise ValueError('Gene order mismatch')
    with np.load(root / 'preprocessing.npz', allow_pickle=False) as params:
        if not np.array_equal(params['gene_ids'], loaded['train']['gene_ids']):
            raise ValueError('Preprocessing gene IDs mismatch')
        if not np.array_equal(params['training_cell_ids'], loaded['train']['cell_ids']):
            raise ValueError('Preprocessing training IDs mismatch')
    return loaded, report


def run(features, output, atoms=16, alpha=1.0, epochs=10, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    if atoms < 1 or epochs < 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Atoms, epochs, and alpha must be positive')
    data, provenance = load_features(features)
    # Double precision avoids coordinate-descent tolerances below float32 resolution.
    for arrays in data.values():
        arrays['X'] = arrays['X'].astype(np.float64)
    X = data['train']['X']
    if atoms > min(X.shape):
        raise ValueError('Pilot atoms must not exceed training matrix dimensions')
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model = MiniBatchDictionaryLearning(
            n_components=atoms, alpha=alpha, max_iter=epochs, batch_size=128,
            fit_algorithm='cd', transform_algorithm='lasso_cd', transform_alpha=alpha,
            transform_max_iter=3000, random_state=seed, n_jobs=1,
            tol=1e-3, max_no_improvement=10,
        ).fit(X)
        dictionary = model.components_.copy()
        if not np.isfinite(dictionary).all() or (np.linalg.norm(dictionary, axis=1) > 1.00001).any():
            raise ValueError('Invalid dictionary or atom norm')
        metrics, vectors = {}, {}
        for split, arrays in data.items():
            z = model.transform(arrays['X'])
            np.testing.assert_array_equal(dictionary, model.components_)
            if not np.isfinite(z).all():
                raise ValueError('Nonfinite codes')
            residual = arrays['X'].astype(np.float64) - z @ dictionary
            mse = float(np.mean(residual**2))
            baseline = float(np.mean(arrays['X'].astype(np.float64)**2))
            metrics[split] = dict(mse=mse, zero_reconstruction_mse=baseline,
                relative_mse=mse/baseline, active_fraction=float(np.mean(np.abs(z)>1e-6)),
                mean_active_atoms=float(np.mean(np.sum(np.abs(z)>1e-6,axis=1))),
                unused_atoms=int(np.sum(~np.any(np.abs(z)>1e-6,axis=0))))
            vectors[split] = z
        output.mkdir(parents=True)
        np.savez_compressed(output/'dictionary.npz', dictionary=dictionary,
                            gene_ids=data['train']['gene_ids'], alpha=np.array(alpha),
                            training_cell_ids=data['train']['cell_ids'])
        for split, z in vectors.items():
            np.savez_compressed(output/f'{split}_codes.npz', vectors=z,
                cell_ids=data[split]['cell_ids'], labels=data[split]['labels'], donors=data[split]['donors'])
        # Verify encoding from the exported dictionary, independent of the fitted estimator.
        with np.load(output/'dictionary.npz', allow_pickle=False) as saved:
            reencoded = sparse_encode(data['validation']['X'][:8], saved['dictionary'],
                                      algorithm='lasso_cd', alpha=float(saved['alpha']), max_iter=3000)
            np.testing.assert_allclose(reencoded, vectors['validation'][:8], atol=1e-4, rtol=1e-4)
    report = dict(status='pilot_complete', atoms=atoms, alpha=alpha, seed=seed,
        epochs_requested=epochs, epochs_run=int(model.n_iter_), steps=int(model.n_steps_),
        reached_epoch_budget=bool(model.n_iter_ >= epochs),
        convergence_note='Early stopping is a solver heuristic, not proof of an optimal dictionary.',
        warning_count=len(caught), warnings=list(dict.fromkeys(str(w.message) for w in caught))[:10],
        solver_precision='float64', seconds=time.perf_counter()-start,
        metrics=metrics, sklearn_version=sklearn.__version__, numpy_version=np.__version__,
        encoding='lasso_cd', transform_max_iter=3000, batch_size=128, threads=1,
        activity_tolerance=1e-6, test_accessed=False,
        preprocessing_report_sha256=digest(Path(features)/'preprocessing.json'),
        input_artifact_hashes={s:provenance['artifacts'][f'{s}.npz']['sha256'] for s in data},
        manifest_sha256=provenance['manifest_sha256'], script_sha256=digest(__file__),
        output_hashes={p.name:digest(p) for p in output.glob('*.npz')})
    (output/'pilot_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=ROOT/'data/processed/baron_classical_v1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--atoms', type=int, default=16)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    run(args.features, args.output, args.atoms, args.alpha, args.epochs, args.seed)


if __name__ == "__main__":
    main()
