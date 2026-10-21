"""Snapshot saved experimental tables/figures for the self-contained LaTeX report.

Run from any directory. This reads existing results; it never fits models or
modifies experiment artifacts. Report assets are regenerated deterministically.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil

from gene_complexity._paths import ROOT
OUT = ROOT / 'docs/notes/assets'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def main():
    inputs = {}

    def rows(relative):
        path = ROOT / relative
        inputs[relative] = digest(path)
        with path.open(newline='') as f:
            return list(csv.DictReader(f))

    tables = []

    def table(caption, columns, data, spec):
        tables.append('\n'.join([
            r'\begin{table}[htbp]\centering\small',
            r'\caption{' + caption + '}',
            r'\begin{tabular}{' + spec + r'}\toprule',
            ' & '.join(columns) + r'\\\midrule',
            *[' & '.join(map(str, row)) + r'\\' for row in data],
            r'\bottomrule\end{tabular}\end{table}',
        ]))

    donor = 'external/report_artifacts/donor_robustness_v2/'
    counts = rows(donor + 'cell_type_by_donor.csv')
    donors = list(counts[0])[1:]
    table('Retained cell-type counts by donor. Headers abbreviate GSM2230757--GSM2230760.',
          ['Cell type', '0757', '0758', '0759', '0760'],
          [[escape(r['cell_type'])] + [r[d] for d in donors] for r in counts], 'lrrrr')

    grid = rows('external/report_artifacts/model_selection_v2/dictionary_ranking.csv')
    table('Refined dictionary grid: validation-selected classifier C, validation Macro-F1, '
          'active coefficient percentage, and MSE relative to zero reconstruction. '
          'These selection scores are not independent test estimates.',
          ['$K$', r'$\alpha$', '$C$', 'Macro-F1', r'Active (\%)', 'Relative MSE'],
          [[r['atoms'], r['alpha'], f"{float(r['C']):.4g}",
            f"{float(r['val_macro_f1']):.6f}", f"{100*float(r['validation_active_fraction']):.2f}",
            f"{float(r['validation_relative_mse']):.4f}"] for r in grid], 'rrrrrr')

    dm = rows(donor + 'donor_means.csv')
    sv = rows(donor + 'seed_variation.csv')
    table('Retrospective standardized outer-donor Macro-F1. Dictionary mean and '
          'sample SD are over its three initializations within that donor.',
          ['Donor', 'Expression', 'Dictionary', 'Seed SD', 'scGPT'],
          [[d] + [f"{float(next(r['macro_f1'] for r in dm if r['donor']==d and r['method']==m and r['scaling']=='standard')):.6f}"
                   for m in ['expression', 'dictionary']] +
           [f"{float(next(r['std'] for r in sv if r['donor']==d and r['method']=='dictionary' and r['scaling']=='standard')):.6f}"] +
           [f"{float(next(r['macro_f1'] for r in dm if r['donor']==d and r['method']=='scgpt' and r['scaling']=='standard')):.6f}"]
           for d in donors], 'lrrrr')

    perclass = {m: rows(f'external/report_artifacts/frozen_models_v2/test/{m}/per_class.csv')
                for m in ['baseline', 'dictionary', 'scgpt']}
    classes = [r['cell_type'] for r in counts]
    table('Original frozen test per-class F1 and cell counts. Small class support '
          'limits interpretation of perfect scores.',
          ['Cell type', 'Cells', 'Expression', 'Dictionary', 'scGPT'],
          [[escape(c), str(int(float(next(r['support'] for r in perclass['baseline'] if r['']==c))))] +
           [f"{float(next(r['f1-score'] for r in perclass[m] if r['']==c)):.4f}" for m in perclass]
           for c in classes], 'lrrrr')

    activity = rows('external/report_artifacts/model_selection_review_v2/dictionary/cell_type_activity.csv')
    table('Illustrative original-dictionary associations: leading positive genes and '
          'cell-type mean codes standardized using training statistics. '
          'Atom numbers are one-based and run-specific; no enrichment test was performed.',
          ['Atom', 'Positive genes', 'Cell type', 'Train', 'Validation'],
          [[str(a), genes, escape(c)] +
           [f"{float(next(r['standardized_mean'] for r in activity if int(r['atom'])==a and r['cell_type']==c and r['split']==split)):.3f}"
            for split in ['train','validation']]
           for a, genes, c in [(2,'CUZD1, CTRC, PNLIPRP1','acinar'),
                               (5,'COL1A2, COL3A1, COL1A1','activated_stellate'),
                               (11,'PLVAP, STC1, ENG','endothelial')]], 'rl lrr')

    figures = {
        'donor_performance.png': 'external/report_artifacts/donor_robustness_v2_analysis/donor_performance.png',
        'initialization_stability.png': 'external/report_artifacts/donor_robustness_v2_analysis/initialization_stability.png',
        'cell_type_activity.png': 'external/report_artifacts/model_selection_review_v2/dictionary/cell_type_activity.png',
    }
    # Validate all sources before changing the snapshot.
    for relative in figures.values():
        inputs[relative] = digest(ROOT / relative)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tables.tex').write_text('% Generated by gene_complexity/reporting/paper.py; see manifest.json.\n'
                                  + '\n\n'.join(tables) + '\n')
    for name, relative in figures.items():
        shutil.copyfile(ROOT / relative, OUT / name)
    outputs = {name: digest(OUT / name) for name in ['tables.tex', *figures]}
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        generator='gene_complexity/reporting/paper.py', generator_sha256=digest(Path(__file__)),
        source_sha256=inputs, snapshot_sha256=outputs), indent=2) + '\n')
    print(f'Wrote {len(tables)} tables and {len(figures)} figures to {OUT}')


if __name__ == '__main__':
    main()
