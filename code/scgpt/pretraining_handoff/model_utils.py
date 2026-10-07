"""Portable loading/preprocessing for the server's small, frozen scGPT models."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.data_collator import DataCollator


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_model(vocab, config):
    return TransformerModel(
        ntoken=len(vocab), d_model=config['embsize'], nhead=config['nheads'],
        d_hid=config['d_hid'], nlayers=config['nlayers'],
        nlayers_cls=config['n_layers_cls'], n_cls=1, vocab=vocab,
        dropout=config['dropout'], pad_token=config['pad_token'],
        pad_value=config['pad_value'], do_mvc=True, do_dab=False,
        use_batch_labels=False, domain_spec_batchnorm=False,
        input_emb_style='continuous', cell_emb_style='cls',
        explicit_zero_prob=False, use_fast_transformer=config['use_fast_transformer'],
        pre_norm=config['pre_norm'],
    )


def load_model(checkpoint_dir, device='cpu'):
    root = Path(checkpoint_dir)
    config = json.loads((root / 'config.json').read_text())
    vocab = GeneVocab.from_dict(json.loads((root / 'vocab.json').read_text()))
    vocab.set_default_index(vocab[config['pad_token']])
    digest = sha256(root / 'model.pt')
    if digest != config['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 does not match its recorded config')
    if config['embsize'] != 128:
        raise ValueError('This handoff requires 128-dimensional checkpoints')
    model = make_model(vocab, config)
    # Original metadata stores torch.__version__ as this string subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(root / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, vocab, config


def panel_counts(adata, panel_genes, layer='X', gene_id_column=None,
                 allow_missing=False, strip_gene_versions=False):
    """Map raw counts into panel order; preserve every cell, aggregate duplicate IDs."""
    source = adata.X if layer == 'X' else adata.layers[layer]
    matrix = sp.csr_matrix(source, dtype=np.float32)
    if not matrix.has_canonical_format:
        # Coalesce actual duplicate entries only. Do not reorder normal CSR rows:
        # their order controls which genes survive the native 1,200-token cap.
        unique_per_row = all(len(np.unique(matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]))
                             == matrix.indptr[i+1]-matrix.indptr[i]
                             for i in range(matrix.shape[0]))
        if not unique_per_row:
            matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not np.allclose(matrix.data, np.rint(matrix.data), atol=1e-4, rtol=0):
        raise ValueError('Provide raw integer counts, not normalized/log expression')
    genes = (adata.var_names.astype(str).to_numpy() if gene_id_column is None
             else adata.var[gene_id_column].astype(str).to_numpy())
    if strip_gene_versions:
        genes = np.array([g.split('.')[0] for g in genes])
    lookup = {g: i for i, g in enumerate(panel_genes)}
    columns, targets = [], []
    for col, gene in enumerate(genes):
        if gene in lookup:
            columns.append(col)
            targets.append(lookup[gene])
    present = len(set(targets))
    if present != len(panel_genes) and not allow_missing:
        raise ValueError(f'Only {present}/{len(panel_genes)} panel genes found; use Ensembl IDs '
                         'or explicitly --allow-missing-panel')
    if present == 0:
        raise ValueError('No panel genes matched; check gene identifiers')
    if present == len(panel_genes) and len(columns) == present:
        # Match the actual cell-to-text prepare.py: raw[:, panel_positions].
        positions = dict(zip((genes[c] for c in columns), columns))
        result = sp.csr_matrix(matrix[:, [positions[g] for g in panel_genes]],
                               dtype=np.float32)
        ordering = 'native CSR panel slicing; nonzero input order preserved'
    else:
        mapping = sp.csr_matrix((np.ones(len(columns), np.float32), (columns, targets)),
                               shape=(len(genes), len(panel_genes)))
        result = sp.csr_matrix(matrix @ mapping, dtype=np.float32)
        result.sum_duplicates()
        ordering = 'canonical panel order after missing-gene filling/duplicate aggregation'
    result.eliminate_zeros()
    empty = np.flatnonzero(np.diff(result.indptr) == 0)
    if len(empty):
        raise ValueError(f'{len(empty)} cells have no nonzero panel genes; filter explicitly '
                         'and retain the resulting cell-ID mapping')
    return result, dict(panel_genes=len(panel_genes), present_panel_genes=present,
                        missing_panel_genes=len(panel_genes)-present,
                        duplicate_matched_columns=len(columns)-present,
                        gene_ordering=ordering)


class SparseExpressionDataset(Dataset):
    def __init__(self, matrix, gene_ids, cls_id, pad_value):
        self.matrix, self.gene_ids = matrix, gene_ids
        self.cls_id, self.pad_value = cls_id, pad_value

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, index):
        start, end = self.matrix.indptr[index:index+2]
        columns = self.matrix.indices[start:end]
        values = self.matrix.data[start:end]
        genes = np.empty(len(columns)+1, dtype=np.int64)
        expressions = np.empty(len(columns)+1, dtype=np.float32)
        genes[0], expressions[0] = self.cls_id, self.pad_value
        genes[1:], expressions[1:] = self.gene_ids[columns], values
        return {'genes': torch.from_numpy(genes), 'expressions': torch.from_numpy(expressions)}


def embed_counts(matrix, panel_genes, model, vocab, config, batch_size=8,
                 device='cpu', seed=2026092500):
    """Frozen float32 CLS + L2, matching the cell-to-text feature extraction recipe."""
    if batch_size < 1 or matrix.shape[0] == 0:
        raise ValueError('Need a positive batch size and at least one cell')
    np.random.seed(seed)
    torch.manual_seed(seed)
    gene_ids = np.array([vocab[g] for g in panel_genes], dtype=np.int64)
    dataset = SparseExpressionDataset(matrix, gene_ids, vocab[config['cls_token']],
                                      config['pad_value'])
    collator = DataCollator(do_padding=True, pad_token_id=vocab[config['pad_token']],
        pad_value=config['pad_value'], do_mlm=False, do_binning=True,
        mlm_probability=config['mask_ratio'], mask_value=config['mask_value'],
        max_length=config['max_length'], sampling=False, keep_first_n_tokens=1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    vectors = []
    with torch.inference_mode():
        for batch in loader:
            genes = batch['gene'].to(device)
            expr = batch['expr'].to(device)
            encoded = model._encode(genes, expr, genes.eq(vocab[config['pad_token']]))
            vectors.append(torch.nn.functional.normalize(encoded[:, 0, :], dim=1)
                           .cpu().numpy())
    result = np.concatenate(vectors).astype(np.float32)
    if result.shape != (len(dataset), 128) or not np.isfinite(result).all():
        raise ValueError('Invalid embedding shape or nonfinite values')
    if not np.allclose(np.linalg.norm(result, axis=1), 1., atol=1e-5):
        raise ValueError('Embeddings are not L2 normalized')
    return result
