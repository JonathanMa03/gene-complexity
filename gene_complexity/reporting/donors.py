"""Validate saved donor-study predictions and summarize donor/seed effects without refitting."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from gene_complexity.evaluation.donor import metrics
from gene_complexity._common import fresh,read,write,digest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--plots',action='store_true')
    a=p.parse_args(argv)
    return run(a)


def run(a):
    protocol=read(a.study/'protocol.json')
    if protocol['status']!='complete': raise ValueError('Study incomplete')
    for name,value in protocol.get('output_hashes',{}).items():
        if digest(a.study/name)!=value: raise ValueError(f'Changed study output: {name}')
    cohort=pd.read_csv(a.study/'cohort.csv',dtype=str).set_index('cell_id')
    rows=pd.read_csv(a.study/'outer_metrics.csv');keys=['macro_f1','accuracy','balanced_accuracy','log_loss']
    for row in rows.itertuples():
        folder=a.study/row.donor
        with np.load(folder/'preprocessing.npz',allow_pickle=False) as z:
            if set(cohort.loc[z['train_ids'],'donor']) & set(cohort.loc[z['hold_ids'],'donor']): raise ValueError('Donor overlap')
            if set(z['hold_ids'])!=set(cohort.index[cohort.donor==row.donor]): raise ValueError('Held donor coverage mismatch')
        with np.load(folder/f'{row.method}_{row.scaling}_seed{row.seed}.npz',allow_pickle=False) as z:
            ids=z['cell_ids'];y=z['labels'];prob=z['probabilities']
            if len(set(ids))!=len(ids) or set(ids)!=set(cohort.index[cohort.donor==row.donor]): raise ValueError('Prediction cohort mismatch')
            np.testing.assert_array_equal(y,cohort.loc[ids,'cell_type'].to_numpy())
            np.testing.assert_allclose(prob.sum(axis=1),1.,atol=1e-6)
            computed=metrics(y,prob,z['classes'])
            for key in keys: np.testing.assert_allclose(computed[key],getattr(row,key),atol=1e-12)
    out=fresh(a.output,'donor_summary')
    donor=rows.groupby(['method','scaling','donor'])[keys].mean().reset_index()
    summary=donor.groupby(['method','scaling'])[keys].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'summary.csv')
    paired=donor.pivot(index='donor',columns=['method','scaling'],values='macro_f1')
    comparisons=pd.DataFrame(index=paired.index)
    for method in ['expression','dictionary','scgpt']:
        comparisons[method+'_standard_minus_none']=paired[method,'standard']-paired[method,'none']
    for method in ['dictionary','scgpt']:
        comparisons[method+'_minus_expression_standard']=paired[method,'standard']-paired['expression','standard']
    comparisons.to_csv(out/'paired_donor_differences.csv')
    stability=pd.read_csv(a.study/'stability.csv'); stability.groupby('donor').mean(numeric_only=True).to_csv(out/'stability_by_donor.csv')
    diagnostics=read(a.study/'dictionary_diagnostics.json')
    write(out/'audit.json',dict(status='PASS',verified_outer_predictions=len(rows),donors=cohort.donor.nunique(),
        inner_classifier_fits=len(pd.read_csv(a.study/'inner_search.csv')),dictionary_fits=len(diagnostics),
        dictionary_warnings=sum(len(d['warnings']) for d in diagnostics),
        max_coding_kkt_residual=max(max(d['train_kkt']['max'],d['hold_kkt']['max']) for d in diagnostics),
        min_gram_eigenvalue=min(d['gram_min_eigenvalue'] for d in diagnostics),
        projected_dictionary_gradient_range=[min(d['dictionary_projected_gradient_norm'] for d in diagnostics),max(d['dictionary_projected_gradient_norm'] for d in diagnostics)],
        protocol_sha256=digest(a.study/'protocol.json'),
        uncertainty='Donor SD is descriptive, not a CI. Folds overlap in training; seed pairs are not independent biological replicates.'))
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,scaling in zip(axes,['standard','none']):
            for method in ['expression','dictionary','scgpt']:
                values=donor[(donor.method==method)&(donor.scaling==scaling)].sort_values('donor')
                ax.plot(range(len(values)),values.macro_f1,'o-',label=method)
            ax.set(title=scaling,xlabel='Held-out donor',ylabel='Macro-F1',ylim=(0,1));ax.set_xticks(range(cohort.donor.nunique()),sorted(cohort.donor.unique()),rotation=30);ax.legend()
        fig.savefig(out/'donor_performance.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for i,(donor_name,group) in enumerate(rows.query("method=='dictionary' and scaling=='standard'").groupby('donor')):
            axes[0].scatter(np.repeat(i,len(group)),group.macro_f1)
        axes[0].set_xticks(range(cohort.donor.nunique()),sorted(cohort.donor.unique()),rotation=30);axes[0].set(ylabel='Macro-F1',title='All initialization seeds; standardized codes')
        for i,(donor_name,group) in enumerate(stability.groupby('donor')): axes[1].scatter(np.repeat(i,len(group)),group.mean_abs_cosine)
        axes[1].set_xticks(range(cohort.donor.nunique()),sorted(cohort.donor.unique()),rotation=30);axes[1].set(ylabel='Mean matched absolute cosine',title='Atom stability, all seed pairs',ylim=(0,1))
        fig.savefig(out/'initialization_stability.png',dpi=180);plt.close(fig)
    print(summary.to_string());print(f'Prediction audit and summary: {out}')

if __name__=='__main__': main()
