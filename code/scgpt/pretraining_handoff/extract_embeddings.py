"""Extract native 128-d embeddings; no adapter, LoRA, or downstream training."""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from model_utils import HERE, load_model, panel_counts, embed_counts, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel', type=Path, default=HERE/'assets/sctab_hvg_2000.tsv')
    p.add_argument('--counts-layer', default='X')
    p.add_argument('--gene-id-column')
    p.add_argument('--allow-missing-panel', action='store_true')
    p.add_argument('--strip-gene-versions', action='store_true')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026092500)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise FileExistsError('Use a new output path; existing results are preserved')
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    torch.set_num_threads(a.threads)
    backed = ad.read_h5ad(a.input, backed='r')
    try:
        data = backed[:a.limit].to_memory() if a.limit else backed.to_memory()
    finally:
        backed.file.close()
    panel = pd.read_csv(a.panel, sep='\t')['feature_id'].astype(str).tolist()
    if len(panel) != 2000 or len(set(panel)) != 2000:
        raise ValueError('Expected exactly 2,000 unique panel genes')
    matrix, coverage = panel_counts(data, panel, a.counts_layer, a.gene_id_column,
                                   a.allow_missing_panel, a.strip_gene_versions)
    model, vocab, config = load_model(a.checkpoint_dir, a.device)
    vectors = embed_counts(matrix, panel, model, vocab, config, a.batch_size, a.device, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('wb') as f:
        np.savez_compressed(f, vectors=vectors, cell_ids=np.asarray(data.obs_names, dtype=str))
    report = dict(status='PASS', shape=list(vectors.shape), dtype=str(vectors.dtype),
        all_finite=bool(np.isfinite(vectors).all()), min=float(vectors.min()),
        max=float(vectors.max()), norm_min=float(np.linalg.norm(vectors,axis=1).min()),
        norm_max=float(np.linalg.norm(vectors,axis=1).max()),
        minimum_between_cell_distance=float(np.linalg.norm(vectors[1:]-vectors[0],axis=1).min())
            if len(vectors)>1 else None,
        parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
        model_id=config['model_id'], checkpoint_sha256=config['checkpoint_sha256'],
        config_sha256=sha256(a.checkpoint_dir/'config.json'), panel_sha256=sha256(a.panel),
        input_path=str(a.input), input_cells_total=backed.n_obs, limit=a.limit,
        output_sha256=sha256(a.output), seed=a.seed, batch_size=a.batch_size,
        device=a.device, precision='float32', normalization='CLS L2', coverage=coverage,
        note='Matches native cell-to-text extraction; historical DOGMA extraction used BF16. '
             'No cross-device bitwise-equivalence claim. Full input hash not computed for a limited smoke run.')
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
