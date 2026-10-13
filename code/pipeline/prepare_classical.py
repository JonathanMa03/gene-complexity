"""Validate the frozen Baron cohort and prepare shared classical features."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'validation', 'test')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_cohort(input_path, bundle):
    """Return raw counts in manifest order, with verified shared metadata."""
    bundle = Path(bundle)
    settings = json.loads((bundle / 'split_settings.json').read_text())
    if settings['status'] != 'final' or not settings['scgpt_confirmed']:
        raise ValueError('A finalized, scGPT-confirmed split bundle is required')
    manifest_path = bundle / 'baron_split_manifest.csv'
    if digest(input_path) != settings['input_sha256']:
        raise ValueError('Raw data hash differs from the finalized split input')
    if digest(manifest_path) != settings['manifest_sha256']:
        raise ValueError('Split manifest hash mismatch')
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    excluded = pd.read_csv(bundle / 'baron_exclusions.csv', dtype=str, keep_default_na=False)
    for frame in (manifest, excluded):
        if not frame.cell_id.is_unique or frame.cell_id.eq('').any():
            raise ValueError('Cell IDs must be unique and nonempty')
    if set(manifest.split) != set(SPLITS):
        raise ValueError('Expected train, validation, and test assignments')
    if manifest[['cell_type', 'donor']].eq('').any().any():
        raise ValueError('Missing labels or donors')
    if not pd.crosstab(manifest.cell_type, manifest.split).gt(0).all().all():
        raise ValueError('All retained classes must appear in each split')
    data = ad.read_h5ad(input_path)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError('Raw cell and gene IDs must be unique')
    kept, removed = set(manifest.cell_id), set(excluded.cell_id)
    if kept & removed or kept | removed != set(data.obs_names):
        raise ValueError('Manifest and exclusions must partition raw cells exactly')
    if len(manifest) != settings['retained_cells'] or len(excluded) != settings['excluded_cells']:
        raise ValueError('Cohort sizes differ from split settings')
    if manifest.split.value_counts().to_dict() != settings['split_counts']:
        raise ValueError('Split sizes differ from settings')
    aligned = data[manifest.cell_id.tolist()].copy()
    for key in ('cell_type', 'donor'):
        if not np.array_equal(manifest[key], aligned.obs[key].astype(str)):
            raise ValueError(f'{key} metadata differs from manifest')
    counts = sparse.csr_matrix(aligned.layers['counts'], dtype=np.float64)
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not np.allclose(counts.data, np.rint(counts.data), atol=1e-6, rtol=0):
        raise ValueError('Counts must be integer-valued')
    return counts, manifest, np.asarray(aligned.var_names, dtype=str), settings


def normalize_log(counts, target_sum):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if not np.isfinite(target_sum) or target_sum <= 0 or (totals <= 0).any():
        raise ValueError('Positive target sum and nonzero cell library sizes required')
    normalized = sparse.diags(target_sum / totals) @ counts
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def fit_preprocessing(train_counts, n_genes=2000, target_sum=10000.0):
    """Select training-variable genes and fit z-scoring on log-normalized counts."""
    if n_genes < 1 or train_counts.shape[0] < 2:
        raise ValueError('Need positive n_genes and at least two training cells')
    log_train = normalize_log(train_counts, target_sum)
    mean = np.asarray(log_train.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(log_train.power(2).mean(axis=0)).ravel() - mean**2, 0)
    eligible = np.flatnonzero(variance > 1e-12)
    if not len(eligible):
        raise ValueError('No variable training genes')
    # Stable tie resolution by original gene position; exported genes stay in source order.
    ranked = eligible[np.argsort(-variance[eligible], kind='stable')]
    selected = np.sort(ranked[:n_genes])
    return dict(selected=selected, mean=mean[selected], scale=np.sqrt(variance[selected]),
                variance=variance[selected], target_sum=float(target_sum))


def transform(counts, fitted):
    # Library totals use all original genes, not only the selected subset.
    log_data = normalize_log(counts, fitted['target_sum'])
    result = log_data[:, fitted['selected']].toarray()
    result = ((result - fitted['mean']) / fitted['scale']).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transformed features')
    return result


def prepare(input_path, bundle, output, n_genes=2000, target_sum=10000.0):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory; existing artifacts are preserved')
    counts, manifest, gene_ids, split_settings = load_cohort(input_path, bundle)
    train = manifest.split.eq('train').to_numpy()
    fitted = fit_preprocessing(counts[train], n_genes, target_sum)
    output.mkdir(parents=True)
    artifacts = {}
    for split in SPLITS:
        mask = manifest.split.eq(split).to_numpy()
        X = transform(counts[mask], fitted)
        path = output / f'{split}.npz'
        np.savez_compressed(path, X=X, cell_ids=manifest.loc[mask, 'cell_id'].to_numpy(dtype=str),
                            labels=manifest.loc[mask, 'cell_type'].to_numpy(dtype=str),
                            donors=manifest.loc[mask, 'donor'].to_numpy(dtype=str),
                            gene_ids=gene_ids[fitted['selected']])
        artifacts[path.name] = dict(shape=list(X.shape), sha256=digest(path))
        print(split, X.shape)
    params = output / 'preprocessing.npz'
    np.savez_compressed(params, **fitted, input_gene_ids=gene_ids,
                        gene_ids=gene_ids[fitted['selected']],
                        training_cell_ids=manifest.loc[train, 'cell_id'].to_numpy(dtype=str))
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    report = dict(input_sha256=split_settings['input_sha256'],
                  manifest_sha256=split_settings['manifest_sha256'],
                  script_sha256=digest(__file__), n_genes_requested=n_genes,
                  n_genes_selected=len(fitted['selected']), target_sum=target_sum,
                  method='all-gene library-size normalization; log1p; training variance ranking; training z-score (ddof=0)',
                  gene_selection='simple log-variance ranking, not a dispersion-adjusted HVG method',
                  scaling='signed centered features; no clipping; train parameters reused unchanged',
                  artifacts=artifacts, parameters_sha256=digest(params),
                  numpy_version=np.__version__, pandas_version=pd.__version__)
    (output / 'preprocessing.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'data/processed/baron_raw.h5ad')
    p.add_argument('--splits', type=Path, default=ROOT/'data/processed/baron_splits_final')
    p.add_argument('--output', type=Path, default=ROOT/'data/processed/baron_classical_v1')
    p.add_argument('--n-genes', type=int, default=2000)
    p.add_argument('--target-sum', type=float, default=10000.0)
    args = p.parse_args()
    prepare(args.input, args.splits, args.output, args.n_genes, args.target_sum)
