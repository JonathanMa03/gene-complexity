"""Create shared Baron splits without modifying counts or fitting preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from gene_complexity._paths import ROOT
SPLITS = ('train', 'validation', 'test')
FRACTIONS = np.array([0.70, 0.15, 0.15])


def make_splits(cells, exclusion_ids=(), seed=42):
    """Stratify within each class; drop classes below 30 after scGPT exclusions."""
    # AnnData obs may retain a named cell_id index alongside the exported column.
    cells = cells.loc[:, ['cell_id', 'cell_type', 'donor']].reset_index(drop=True).copy()
    if cells.empty or cells.isna().any().any():
        raise ValueError('Cell metadata must be nonempty and contain no missing values')
    for column in cells:
        cells[column] = cells[column].astype(str)
        if cells[column].str.strip().eq('').any():
            raise ValueError(f'Empty {column} values')
    if not cells.cell_id.is_unique:
        raise ValueError('Cell IDs must be unique')
    supplied = [str(x) for x in exclusion_ids]
    if len(set(supplied)) != len(supplied):
        raise ValueError('Duplicate scGPT exclusion IDs')
    unknown = set(supplied) - set(cells.cell_id)
    if unknown:
        raise ValueError(f'Unknown scGPT exclusion IDs: {sorted(unknown)[:5]}')
    cells = cells.sort_values('cell_id').reset_index(drop=True)
    scgpt_mask = cells.cell_id.isin(supplied)
    remaining_counts = cells.loc[~scgpt_mask, 'cell_type'].value_counts()
    rare_mask = ~scgpt_mask & cells.cell_type.map(remaining_counts).fillna(0).lt(30)
    excluded = cells.loc[scgpt_mask | rare_mask].copy()
    excluded['reason'] = np.where(scgpt_mask.loc[excluded.index],
                                  'scgpt_incompatible', 'cell_type_below_30_after_scgpt_exclusions')
    manifest = cells.loc[~(scgpt_mask | rare_mask)].copy()
    if manifest.empty:
        raise ValueError('No cells remain after exclusions')
    manifest['split'] = ''
    rng = np.random.default_rng(seed)
    for _, group in manifest.groupby('cell_type', sort=True):
        ideal = len(group) * FRACTIONS
        sizes = np.floor(ideal).astype(int)
        # Largest-remainder allocation; ties prefer train, validation, then test.
        priority = np.argsort(-(ideal - sizes), kind='stable')
        sizes[priority[:len(group) - sizes.sum()]] += 1
        positions = rng.permutation(group.index.to_numpy())
        for split, indices in zip(SPLITS, np.split(positions, np.cumsum(sizes)[:-1])):
            manifest.loc[indices, 'split'] = split
    counts = pd.crosstab(manifest.cell_type, manifest.split).reindex(columns=SPLITS, fill_value=0)
    if not counts.gt(0).all().all():
        raise ValueError('Every retained class must occur in every split')
    kept_ids, removed_ids = set(manifest.cell_id), set(excluded.cell_id)
    if kept_ids & removed_ids or kept_ids | removed_ids != set(cells.cell_id):
        raise ValueError('Retained/excluded IDs do not partition the input')
    if not manifest.split.isin(SPLITS).all() or not manifest.cell_id.is_unique:
        raise ValueError('Invalid split assignments')
    counts['total'] = counts.sum(axis=1)
    class_counts = cells.cell_type.value_counts().rename('original').to_frame()
    class_counts['after_scgpt'] = remaining_counts
    class_counts['retained'] = manifest.cell_type.value_counts()
    class_counts = class_counts.fillna(0).astype(int).sort_index()
    class_counts.index.name = 'cell_type'
    return manifest.reset_index(drop=True), excluded.reset_index(drop=True), counts, class_counts


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/baron_raw.h5ad')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/baron_splits_draft')
    parser.add_argument('--scgpt-exclusions', type=Path, help='CSV with a cell_id column')
    parser.add_argument('--scgpt-confirmed', action='store_true',
                        help='Set only after scGPT confirms this exclusion list, including none')
    parser.add_argument('--seed', type=int, default=42)
    args=parser.parse_args(argv)
    return run(args)


def run(args):
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing manifests are preserved')
    data = ad.read_h5ad(args.input, backed='r')
    try:
        cells = data.obs[['cell_type', 'donor']].copy()
        cells.insert(0, 'cell_id', data.obs_names.astype(str))
    finally:
        data.file.close()
    ids = ()
    if args.scgpt_exclusions:
        frame = pd.read_csv(args.scgpt_exclusions, dtype=str, keep_default_na=False)
        if 'cell_id' not in frame:
            raise ValueError('Exclusion CSV requires a cell_id column')
        ids = frame.cell_id.tolist()
    manifest, excluded, counts, class_counts = make_splits(cells, ids, args.seed)
    settings = dict(status='final' if args.scgpt_confirmed else 'draft_pending_scgpt',
                    seed=args.seed, fractions=dict(zip(SPLITS, FRACTIONS.tolist())),
                    strategy='within-cell-type random allocation; donors may cross splits',
                    rounding='largest remainder per class; ties train, validation, test',
                    ordering='sort cell IDs and class names before seeded permutation',
                    minimum_class_cells=30, rare_rule_applied_after_scgpt=True,
                    input_path=str(args.input.resolve()), input_sha256=sha256(args.input),
                    scgpt_exclusions_sha256=sha256(args.scgpt_exclusions) if args.scgpt_exclusions else None,
                    scgpt_confirmed=args.scgpt_confirmed,
                    input_cells=len(cells), retained_cells=len(manifest), excluded_cells=len(excluded),
                    split_counts=manifest.split.value_counts().to_dict(),
                    numpy_version=np.__version__, pandas_version=pd.__version__,
                    script_sha256=sha256(__file__))
    args.output_dir.mkdir(parents=True)
    manifest.to_csv(args.output_dir / 'baron_split_manifest.csv', index=False)
    excluded.to_csv(args.output_dir / 'baron_exclusions.csv', index=False)
    counts.to_csv(args.output_dir / 'baron_split_counts.csv')
    class_counts.to_csv(args.output_dir / 'baron_class_counts.csv')
    settings['manifest_sha256'] = sha256(args.output_dir / 'baron_split_manifest.csv')
    (args.output_dir / 'split_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(counts.to_string())
    print(f"Status: {settings['status']}; output: {args.output_dir}")


if __name__ == '__main__':
    main()
