"""Small signed dictionary-learning baseline; run from any working directory."""
import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import DictionaryLearning, sparse_encode


def generate_synthetic(seed=42):
    """Processed-expression-like data, not simulated raw sequencing counts."""
    rng = np.random.default_rng(seed)
    n, p, k = 240, 40, 8
    dictionary = rng.normal(size=(k, p))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    codes = np.zeros((n, k))
    for row in codes:
        active = rng.choice(k, size=2, replace=False)
        row[active] = rng.choice([-1, 1], size=2) * rng.uniform(1, 3, size=2)
    expression = codes @ dictionary + rng.normal(scale=0.03, size=(n, p))
    order = rng.permutation(n)
    splits = np.empty(n, dtype='<U10')
    splits[order[:160]] = 'train'
    splits[order[160:200]] = 'validation'
    splits[order[200:]] = 'test'
    return dict(X=expression, true_dictionary=dictionary, true_codes=codes,
                cell_ids=np.array([f'synthetic_cell_{i:04d}' for i in range(n)]),
                gene_ids=np.array([f'synthetic_gene_{j:03d}' for j in range(p)]),
                split=splits)


def fit_dictionary(X_train, n_components=8, alpha=0.1, seed=42):
    """Input must already be processed using training-fitted transforms."""
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2 or not X_train.size or not np.isfinite(X_train).all():
        raise ValueError('Expected a nonempty finite cells-by-genes matrix')
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('alpha must be finite and positive')
    model = DictionaryLearning(
        n_components=n_components, alpha=alpha, max_iter=300, tol=1e-6,
        fit_algorithm='cd', transform_algorithm='lasso_cd',
        transform_alpha=alpha, transform_max_iter=3000, random_state=seed,
    )
    return model.fit(X_train)


def run(output, seed=42):
    # Preserve prior experiments, including partially written outputs.
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    data = generate_synthetic(seed)
    train = data['split'] == 'train'
    # Center using training cells only; no variance scaling for this toy example.
    mean = data['X'][train].mean(axis=0)
    X = data['X'] - mean
    alpha = 0.1
    model = fit_dictionary(X[train], alpha=alpha, seed=seed)
    frozen = model.components_.copy()
    vectors = model.transform(X)
    np.testing.assert_array_equal(model.components_, frozen)
    assert vectors.shape == (240, 8) and np.isfinite(vectors).all()
    assert np.max(np.linalg.norm(frozen, axis=1)) <= 1 + 1e-6
    assert len(np.unique(data['cell_ids'])) == len(vectors)
    reconstruction = vectors @ frozen
    metrics = {}
    for split in ('train', 'validation', 'test'):
        mask = data['split'] == split
        mse = float(np.mean((X[mask] - reconstruction[mask]) ** 2))
        zero_mse = float(np.mean(X[mask] ** 2))
        metrics[split] = dict(n_cells=int(mask.sum()), mse=mse,
                              zero_reconstruction_mse=zero_mse,
                              active_fraction=float(np.mean(np.abs(vectors[mask]) > 1e-6)))
        assert mse < zero_mse, f'{split}: reconstruction did not beat zero baseline'
    output.mkdir(parents=True)
    np.savez_compressed(output / 'synthetic_data.npz', **data)
    np.savez_compressed(output / 'dictionary.npz', dictionary=frozen,
                        gene_ids=data['gene_ids'], train_mean=mean,
                        alpha=np.array(alpha), train_cell_ids=data['cell_ids'][train])
    np.savez_compressed(output / 'codes.npz', vectors=vectors,
                        cell_ids=data['cell_ids'], split=data['split'])
    # Reload without pickle and independently re-encode held-out cells.
    with np.load(output / 'dictionary.npz', allow_pickle=False) as saved:
        held_out = ~train
        reencoded = sparse_encode(data['X'][held_out] - saved['train_mean'],
                                  saved['dictionary'], algorithm='lasso_cd',
                                  alpha=float(saved['alpha']), max_iter=3000)
        np.testing.assert_allclose(reencoded, vectors[held_out], atol=1e-8)
        np.testing.assert_array_equal(saved['gene_ids'], data['gene_ids'])
    with np.load(output / 'codes.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['cell_ids'], data['cell_ids'])
        np.testing.assert_array_equal(saved['vectors'], vectors)
    report = dict(seed=seed, alpha=alpha, n_components=8,
                  sklearn_version=sklearn.__version__, numpy_version=np.__version__,
                  preprocessing='training-mean centering only',
                  objective='0.5 * ||X - ZD||_F^2 + alpha * ||Z||_1; atom norms <= 1',
                  encoding='lasso_cd', iterations=int(model.n_iter_),
                  iteration_budget=300, reached_iteration_budget=bool(model.n_iter_ >= 300),
                  training_objective_history=np.asarray(model.error_).tolist(),
                  checks='finite codes, atom norms, frozen transform, reconstruction, export/reload',
                  metrics=metrics,
                  note='Synthetic pipeline check only; no biological labels or hyperparameter tuning.')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    print(f'Checks passed. Artifacts: {output.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'results/dict-learning/synthetic_seed42')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.seed)
