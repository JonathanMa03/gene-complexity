"""Summarize evaluation directories from this workflow, optionally drawing plots."""
import argparse
from pathlib import Path
from gene_complexity._common import fresh, read, write, digest
import pandas as pd

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations',type=Path,nargs='+',required=True,help='Directories containing metrics.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--plots',action='store_true')
    a=p.parse_args(argv)
    return run(a)


def run(a):
    rows=[]
    for directory in a.evaluations:
        row=read(directory/'metrics.json'); row['evaluation_dir']=str(directory.resolve()); rows.append(row)
    if len({r['split'] for r in rows})!=1: raise ValueError('Summarize validation and test separately')
    # Different models may be compared only on the same exact evaluation cohort.
    cohorts=[]
    import numpy as np
    for directory in a.evaluations:
        with np.load(directory/'predictions.npz',allow_pickle=False) as z:
            cohorts.append(sorted(zip(z['cell_ids'].astype(str),z['labels'].astype(str))))
    if any(c!=cohorts[0] for c in cohorts): raise ValueError('Evaluation cohorts differ')
    out=fresh(a.output,'summary')
    columns=['method','candidate','C','split','n_cells','macro_f1','accuracy','balanced_accuracy','log_loss','evaluation_dir']
    table=pd.DataFrame(rows)[columns]; table.to_csv(out/'summary.csv',index=False)
    write(out/'sources.json',{str(d.resolve()):digest(d/'metrics.json') for d in a.evaluations})
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        names=[f"{r['candidate'] or r['method']} (C={r['C']:g})" for r in rows]
        fig,ax=plt.subplots(figsize=(max(7,len(rows)*1.5),5))
        ax.bar(names,[r['macro_f1'] for r in rows]); ax.set(ylabel='Macro-F1',ylim=(0,1),title=f"{rows[0]['split'].title()} comparison")
        ax.tick_params(axis='x',labelrotation=30); fig.tight_layout(); fig.savefig(out/'macro_f1.png',dpi=160); plt.close(fig)
        for i,directory in enumerate(a.evaluations):
            cm=pd.read_csv(directory/'confusion_matrix.csv',index_col=0)
            fig,ax=plt.subplots(figsize=(10,10)); ConfusionMatrixDisplay(cm.to_numpy(),display_labels=cm.index).plot(ax=ax,xticks_rotation=90,colorbar=False)
            ax.set_title(names[i]); fig.tight_layout(); fig.savefig(out/f'confusion_{i+1}.png',dpi=160); plt.close(fig)
    print(table.to_string(index=False)); print(f'Summary: {out}')
    return out

if __name__ == '__main__': main()
