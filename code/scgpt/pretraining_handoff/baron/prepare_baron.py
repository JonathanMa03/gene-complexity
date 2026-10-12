"""Validate the supplied Baron mapping and add panel Ensembl IDs without filtering."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HANDOFF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOFF))
from model_utils import panel_counts


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panel', type=Path, default=HANDOFF/'assets/sctab_hvg_2000.tsv')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory; existing results are preserved')
    hashes = {str(p.resolve()): digest(p) for p in [args.input, args.mapping, args.panel]}
    mapping = pd.read_csv(args.mapping, dtype=str, keep_default_na=False)
    panel = pd.read_csv(args.panel, sep='\t', dtype=str)
    if len(panel) != 2000 or not panel.feature_id.is_unique or not panel.feature_name.is_unique:
        raise ValueError('Expected the fixed 2,000-gene unique panel')
    if list(zip(mapping.ensembl_id, mapping.gene_symbol)) != list(zip(panel.feature_id, panel.feature_name)):
        raise ValueError('Mapping must match the fixed panel IDs, symbols and row order exactly')
    data = ad.read_h5ad(args.input)
    if not data.var_names.is_unique or not data.obs_names.is_unique:
        raise ValueError('Gene symbols and cell IDs must be unique; no implicit aggregation allowed')
    if 'counts' not in data.layers:
        raise ValueError('An explicit counts layer is required')
    genes = data.var_names.astype(str).tolist()
    positions = {gene: i for i, gene in enumerate(genes)}
    present = mapping.gene_symbol.isin(positions)
    if not mapping.present_in_baron.isin(['True', 'False']).all():
        raise ValueError('Unexpected presence flags')
    if not np.array_equal(present, mapping.present_in_baron.eq('True')):
        raise ValueError('Mapping presence flags do not match the input')
    expected_indices = np.array([positions.get(gene, -1) for gene in mapping.gene_symbol])
    if not np.array_equal(expected_indices, mapping.baron_gene_index.astype(int).to_numpy()):
        raise ValueError('Mapping column indices do not match the input')
    counts = sp.csr_matrix(data.layers['counts'])
    values = counts.data
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not np.allclose(values, np.rint(values), atol=1e-4, rtol=0):
        raise ValueError('Counts must be integer-valued; no rounding is performed')
    selected = counts[:, expected_indices[present.to_numpy()]]
    zero = np.asarray((selected != 0).sum(axis=1)).ravel() == 0
    zero_ids = data.obs_names[zero].astype(str).tolist()
    lookup = dict(zip(mapping.gene_symbol, mapping.ensembl_id))
    column = 'scgpt_panel_ensembl_id'
    if column in data.var:
        raise ValueError('Output annotation already exists; refusing to overwrite it')
    # These sentinels cannot match the Ensembl vocabulary. They are not inferred IDs.
    ids = [lookup.get(g, '__UNMAPPED_SYMBOL__:' + g) for g in genes]
    data.var[column] = ids
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir/'zero_panel_cell_ids.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cell_id'])
        writer.writerows([[cell] for cell in zero_ids])
    mapping.loc[~present].to_csv(args.output_dir/'missing_panel_genes.csv', index=False)
    report = dict(status='FAIL_ZERO_PANEL_CELLS' if zero_ids else 'PENDING_SERIALIZATION_VALIDATION',
        input_shape=list(data.shape), input_hashes=hashes,
        panel_genes=2000, present_panel_genes=int(present.sum()),
        missing_panel_genes=int((~present).sum()), coverage_fraction=float(present.mean()),
        zero_panel_cells=len(zero_ids), zero_panel_cell_ids=zero_ids,
        all_counts_finite=True, counts_nonnegative=True, counts_integer_valued=True,
        mapping_presence_and_column_indices_verified=True, mapping_matches_fixed_panel=True,
        x_equals_counts=bool((sp.csr_matrix(data.X) != counts).nnz == 0),
        gene_id_column=column, counts_layer='counts',
        input_cell_order_sha256=hashlib.sha256(json.dumps(data.obs_names.tolist()).encode()).hexdigest(),
        python=platform.python_version(), anndata=ad.__version__, numpy=np.__version__, pandas=pd.__version__,
        provenance_limit='Supplied file has no source metadata in uns. Integer checks do not prove '
                         'upstream processing history or absence of pretraining overlap.')
    report_path = args.output_dir/'preparation_report.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    if zero_ids:
        raise ValueError('Zero-panel cells reported; no cells were removed and no mapped data written')
    _, coverage = panel_counts(data, panel.feature_id.tolist(), layer='counts',
                               gene_id_column=column, allow_missing=True)
    output = args.output_dir/'baron_scgpt_mapped.h5ad'
    # The supplied file uses pandas nullable-string indices. Preserve them
    # using the same AnnData >=0.11 encoding rather than coercing cell metadata.
    ad.settings.allow_write_nullable_strings = True
    data.write_h5ad(output, compression='gzip')
    # Verify actual serialized data, not only the in-memory object.
    check = ad.read_h5ad(output)
    if not check.obs_names.equals(data.obs_names) or not check.var_names.equals(data.var_names):
        raise ValueError('Serialized cell/gene order changed')
    if not check.obs.equals(data.obs) or check.var[column].astype(str).tolist() != ids:
        raise ValueError('Serialized metadata changed')
    if (sp.csr_matrix(check.layers['counts']) != counts).nnz or (sp.csr_matrix(check.X) != sp.csr_matrix(data.X)).nnz:
        raise ValueError('Serialized expression values changed')
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('A source file changed during preparation')
    report.update(status='PASS', output_sha256=digest(output), output_filename=output.name,
                  extractor_panel_counts=coverage, serialized_expression_and_cell_order_preserved=True,
                  originals_unchanged=True)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
