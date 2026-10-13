"""Verify all Baron vectors and bind the delivery metadata to source hashes."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['npz', 'input', 'raw-input', 'mapping', 'preparation-report', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--selection', type=Path,
                        default=Path(__file__).with_name('checkpoint_selection.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.npz.with_suffix('.json').read_text())
    prep = json.loads(args.preparation_report.read_text())
    selection = json.loads(args.selection.read_text())
    if report['status'] != 'PASS' or report['limit'] is not None:
        raise ValueError('A successful full extraction without --limit is required')
    if prep['status'] != 'PASS' or prep['zero_panel_cells'] != 0:
        raise ValueError('Preparation must pass with no excluded cells')
    if report['model_id'] != selection['checkpoint_id'] or report['checkpoint_sha256'] != selection['checkpoint_sha256']:
        raise ValueError('Extraction did not use the prespecified checkpoint')
    expected_hashes = set(prep['input_hashes'].values())
    raw_hash, mapping_hash = sha(args.raw_input), sha(args.mapping)
    if raw_hash not in expected_hashes or mapping_hash not in expected_hashes:
        raise ValueError('Raw data or mapping do not match validated preparation')
    input_hash = sha(args.input)
    if input_hash != prep['output_sha256'] or input_hash != report['input_sha256']:
        raise ValueError('Mapped input is not the validated preparation output')
    output_hash = sha(args.npz)
    if output_hash != report['output_sha256']:
        raise ValueError('NPZ hash does not match the extraction report')
    data = ad.read_h5ad(args.input, backed='r')
    raw = ad.read_h5ad(args.raw_input, backed='r')
    try:
        if data.n_obs != 8569 or not data.obs_names.equals(raw.obs_names):
            raise ValueError('Expected the same 8,569 raw cells in original order')
        if not data.obs.equals(raw.obs) or not data.obs_names.is_unique:
            raise ValueError('Cell metadata changed or IDs are not unique')
        with np.load(args.npz, allow_pickle=False) as archive:
            if set(archive.files) != {'vectors', 'cell_ids'}:
                raise ValueError('NPZ must contain vectors and cell_ids')
            vectors, ids = archive['vectors'], archive['cell_ids']
            if vectors.shape != (8569, 128) or ids.shape != (8569,):
                raise ValueError('Invalid vector/ID shape')
            if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
                raise ValueError('Vectors must be finite float32')
            if ids.tolist() != data.obs_names.astype(str).tolist():
                raise ValueError('NPZ cell IDs do not match input order')
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1., atol=1e-5):
                raise ValueError('Invalid L2 normalization')
            coverage = report['coverage']
            if coverage['present_panel_genes'] != 1879 or coverage['missing_panel_genes'] != 121:
                raise ValueError('Unexpected panel coverage')
            if report['trainable_parameters'] != 0:
                raise ValueError('Expected a frozen model')
            report.update(
                input_raw_sha256=raw_hash, mapping_sha256=mapping_hash,
                preparation_report_sha256=sha(args.preparation_report),
                checkpoint_selection=selection, checkpoint_selection_sha256=sha(args.selection),
                zero_panel_cells=0, excluded_cells=0,
                all_cell_ids_unique=True, all_cell_ids_match_original_order=True,
                cell_metadata_preserved=True, npz_load_allow_pickle_false=True,
                validated_shape=list(vectors.shape), validated_all_finite=True,
                validated_norm_min=float(norms.min()), validated_norm_max=float(norms.max()),
                cell_ids_sha256=hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest(),
                note='Full frozen Baron extraction. No cell filtering, training, or split changes. '
                     'Compatibility and completeness verified; downstream performance and pretraining overlap not evaluated.')
    finally:
        data.file.close()
        raw.file.close()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: 8569 x 128 finite vectors; all cell IDs verified; metadata:', args.output)


if __name__ == '__main__':
    main()
