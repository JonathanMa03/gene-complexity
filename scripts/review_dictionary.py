"""Describe dictionary weights and cell-type activities without accessing test features."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from _common import ROOT, config_arg, read, write, fresh, selected, digest


def main():
    p=argparse.ArgumentParser(description=__doc__); config_arg(p)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--top-genes',type=int,default=10)
    p.add_argument('--display-atoms',type=int,default=8)
    p.add_argument('--plots',action='store_true')
    a=p.parse_args()
    if a.top_genes<1 or a.display_atoms<1: p.error('Counts must be positive')
    config,b,train,val=selected(read(a.config),'dictionary',a.candidate)
    path=ROOT/config['dictionary_dir']/config['dictionary_runs'][a.candidate]/'dictionary.npz'
    with np.load(path,allow_pickle=False) as z: D,genes=z['dictionary'],z['gene_ids'].astype(str)
    out=fresh(a.output,'dictionary_review')
    classes=np.unique(b['y_train']); mean=train.mean(axis=0); std=train.std(axis=0)
    scale=np.where(std>1e-12,std,1.)
    class_train=np.array([train[b['y_train']==c].mean(axis=0) for c in classes])
    contrast=(class_train-mean)/scale
    # Figure atoms selected solely by training cell-type activity contrast.
    rank=np.argsort(-np.max(np.abs(contrast),axis=0),kind='stable')
    shown=rank[:min(a.display_atoms,len(rank))]
    weights=[]; activity=[]; diagnostics=[]
    for k,atom in enumerate(D):
        for sign,indices in [('positive',np.argsort(-atom,kind='stable')),('negative',np.argsort(atom,kind='stable'))]:
            indices=[i for i in indices if (atom[i]>0 if sign=='positive' else atom[i]<0)][:a.top_genes]
            weights.extend(dict(atom=k+1,sign=sign,rank=j+1,gene=genes[i],weight=float(atom[i])) for j,i in enumerate(indices))
        diagnostics.append(dict(atom=k+1,norm=float(np.linalg.norm(atom)),
            train_active_fraction=float(np.mean(np.abs(train[:,k])>1e-6)),
            validation_active_fraction=float(np.mean(np.abs(val[:,k])>1e-6)),
            max_abs_training_class_contrast=float(np.max(np.abs(contrast[:,k])))))
        for split,codes,labels in [('train',train,b['y_train']),('validation',val,b['y_val'])]:
            for c in classes:
                values=codes[labels==c,k]
                activity.append(dict(atom=k+1,split=split,cell_type=c,n_cells=len(values),
                    mean_activity=float(values.mean()),mean_absolute_activity=float(np.abs(values).mean()),
                    active_fraction=float(np.mean(np.abs(values)>1e-6)),
                    standardized_mean=float((values.mean()-mean[k])/scale[k])))
    pd.DataFrame(weights).to_csv(out/'top_gene_weights.csv',index=False)
    pd.DataFrame(activity).to_csv(out/'cell_type_activity.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'atom_diagnostics.csv',index=False)
    norms=np.linalg.norm(D,axis=1); cosine=D@D.T/np.outer(norms,norms)
    np.fill_diagonal(cosine,0)
    report=dict(candidate=a.candidate,dictionary_sha256=digest(path),input_hashes=b['input_hashes'],
        atoms=len(D),genes=len(genes),unused_train_atoms=int(np.sum(~(np.abs(train)>1e-6).any(axis=0))),
        unused_validation_atoms=int(np.sum(~(np.abs(val)>1e-6).any(axis=0))),
        max_absolute_atom_cosine=float(np.max(np.abs(cosine))),
        train_active_fraction=float(np.mean(np.abs(train)>1e-6)),
        validation_active_fraction=float(np.mean(np.abs(val)>1e-6)),
        displayed_atoms=(shown+1).tolist(),display_selection='Largest absolute training-standardized cell-type mean contrast',
        test_accessed=False,limitations='Descriptive weights and activities only; no pathway enrichment, causal inference, or multi-seed stability assessment. Atom sign and ordering are arbitrary.')
    write(out/'review.json',report)
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures=[]
        for split,codes,labels in [('train',train,b['y_train']),('validation',val,b['y_val'])]:
            values=np.array([(codes[labels==c].mean(axis=0)-mean)/scale for c in classes])[:,shown]
            figures.append((split,values))
        limit=max(float(np.max(np.abs(v))) for _,v in figures)
        fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
        for ax,(split,values) in zip(axes,figures):
            im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_xticks(range(len(shown)),[str(i+1) for i in shown]); ax.set_yticks(range(len(classes)),classes)
            ax.set(title=split,xlabel='Atom (1-based)')
        fig.colorbar(im,ax=axes,label='Mean code activity, centered/scaled using training cells')
        fig.savefig(out/'cell_type_activity.png',dpi=180); plt.close(fig)
    print(f'Dictionary review: {out}; displayed atoms: {(shown+1).tolist()}')

if __name__=='__main__': main()
