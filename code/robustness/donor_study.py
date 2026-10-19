"""Retrospective nested donor evaluation, scale ablation, and initialization stability."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'code/pipeline')]
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from prepare_classical import load_cohort,normalize_log
from _common import fresh,read,write,digest,environment
from run_end_to_end import preflight


def donor_folds(donors):
    groups=sorted(set(donors))
    if len(groups)<4: raise ValueError('Need at least four donors for nested donor evaluation')
    return [(d,np.flatnonzero(donors!=d),np.flatnonzero(donors==d)) for d in groups]


def preprocessing(log_counts,train,hold,genes,panel_indices,n_genes):
    # All population statistics and variance selection are fitted on this training fold.
    mean=np.asarray(log_counts[train].mean(axis=0)).ravel()
    var=np.maximum(np.asarray(log_counts[train].power(2).mean(axis=0)).ravel()-mean**2,0)
    if panel_indices is None:
        eligible=np.flatnonzero(var>1e-12)
        selected=np.sort(eligible[np.argsort(-var[eligible],kind='stable')[:n_genes]])
    else: selected=np.asarray(panel_indices)
    scale=np.sqrt(var[selected]);scale[scale<1e-12]=1.
    x=(log_counts[train][:,selected].toarray()-mean[selected])/scale
    h=(log_counts[hold][:,selected].toarray()-mean[selected])/scale
    if not np.isfinite(x).all() or not np.isfinite(h).all(): raise ValueError('Nonfinite preprocessing')
    return x,h,dict(selected=selected,gene_ids=genes[selected],mean=mean[selected],scale=scale)


def encode(x,d,alpha):
    return sparse_encode(x,d,algorithm='lasso_cd',alpha=alpha,max_iter=3000,n_jobs=1)


def kkt_residual(x,z,d,alpha):
    gradient=(z@d-x)@d.T
    active=np.abs(z)>1e-6
    residual=np.where(active,np.abs(gradient+alpha*np.sign(z)),np.maximum(np.abs(gradient)-alpha,0))
    return dict(max=float(residual.max()),p95=float(np.quantile(residual,.95)),
                relative_max=float(residual.max()/max(1.,alpha)))


def fit_dictionary(x,h,atoms,alpha,seed,epochs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=MiniBatchDictionaryLearning(n_components=atoms,alpha=alpha,max_iter=epochs,batch_size=128,
            fit_algorithm='cd',transform_algorithm='lasso_cd',transform_alpha=alpha,transform_max_iter=3000,
            random_state=seed,n_jobs=1,tol=1e-3,max_no_improvement=10).fit(x)
        d=model.components_.copy();z=encode(x,d,alpha); zh=encode(h,d,alpha)
    if not all(np.isfinite(a).all() for a in [d,z,zh]) or np.max(np.linalg.norm(d,axis=1))>1.00001:
        raise ValueError('Invalid dictionary or codes')
    return z,zh,d,dict(seed=seed,epochs_run=int(model.n_iter_),warnings=[str(w.message) for w in caught],
        train_active_fraction=float(np.mean(np.abs(z)>1e-6)),hold_active_fraction=float(np.mean(np.abs(zh)>1e-6)),
        train_mse=float(np.mean((x-z@d)**2)),hold_mse=float(np.mean((h-zh@d)**2)),
        train_kkt=kkt_residual(x,z,d,alpha),hold_kkt=kkt_residual(h,zh,d,alpha))


def fit_classifier(x,h,y,regularization,scaling):
    if scaling=='standard':
        scaler=StandardScaler().fit(x);xt=scaler.transform(x);xh=scaler.transform(h)
        center,scale=scaler.mean_,scaler.scale_
    else: xt,xh=x,h;center=np.zeros(x.shape[1]);scale=np.ones(x.shape[1])
    # mean cross entropy + lambda/2 ||W||_F^2: sklearn C = 1/(n*lambda).
    c=1/(len(y)*regularization)
    model=LogisticRegression(C=c,solver='lbfgs',max_iter=5000,tol=1e-5,random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(xt,y)
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    return model.predict_proba(xh),model,center,scale,converged


def metrics(y,prob,classes):
    pred=classes[prob.argmax(axis=1)]
    return dict(macro_f1=f1_score(y,pred,labels=classes,average='macro',zero_division=0),
        accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),
        log_loss=log_loss(y,prob,labels=classes))


def choose_regularization(rows):
    table=pd.DataFrame(rows)
    # Every candidate must converge on every inner donor and seed, not just the easy ones.
    valid=table.groupby('regularization').converged.all()
    means=table.groupby('regularization').macro_f1.mean().loc[valid]
    if means.empty: raise ValueError('No regularization converged in every inner fit')
    return float(sorted(means.index,key=lambda v:(-means.loc[v],-v))[0])


def match_atoms(a,b,za,zb,top=20):
    similarity=a@b.T/np.outer(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1))
    i,j=linear_sum_assignment(-np.abs(similarity)); signs=np.where(similarity[i,j]>=0,1.,-1.)
    overlaps=[];correlations=[]
    for k,l,s in zip(i,j,signs):
        ga=set(np.argsort(np.abs(a[k]),kind='stable')[-top:]);gb=set(np.argsort(np.abs(b[l]),kind='stable')[-top:])
        overlaps.append(len(ga&gb)/len(ga|gb))
        u,v=za[:,k],zb[:,l]*s
        correlations.append(float(np.corrcoef(u,v)[0,1]) if np.std(u)>1e-12 and np.std(v)>1e-12 else np.nan)
    return pd.DataFrame(dict(atom_a=i+1,atom_b=j+1,sign=signs,abs_cosine=np.abs(similarity[i,j]),
                             top_gene_jaccard=overlaps,train_code_correlation=correlations))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'data/processed/baron_raw.h5ad')
    p.add_argument('--splits',type=Path,default=ROOT/'data/processed/baron_splits_final')
    p.add_argument('--scgpt-dir',type=Path,default=ROOT/'code/scgpt/pretraining_handoff/outputs/baron')
    p.add_argument('--panel',type=Path,default=ROOT/'code/scgpt/pretraining_handoff/assets/sctab_hvg_2000.tsv')
    p.add_argument('--gene-policy',choices=['scgpt-panel','train-variance'],default='scgpt-panel')
    p.add_argument('--n-genes',type=int,default=2000,help='Only for train-variance policy')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--atoms',type=int,default=64);p.add_argument('--alpha',type=float,default=2.)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--regularization',type=float,nargs='+',default=[1e-5,1e-4,1e-3,1e-2,1e-1])
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if len(set(a.seeds))<3 or len(set(a.seeds))!=len(a.seeds) or any(s<0 or s>=2**32 for s in a.seeds): p.error('At least three unique valid seeds required')
    if min(a.atoms,a.epochs,a.n_genes)<1 or not np.isfinite(a.alpha) or a.alpha<=0: p.error('Positive dictionary/preprocessing settings required')
    if not all(np.isfinite(v) and v>0 for v in a.regularization) or len(set(a.regularization))!=len(a.regularization): p.error('Unique positive regularization values required')
    hashes=preflight(a.raw,a.splits,a.scgpt_dir)
    counts,manifest,genes,_=load_cohort(a.raw,a.splits)
    # Original within-donor split labels are deliberately not used in the new grouped folds.
    donors=manifest.donor.to_numpy();y=manifest.cell_type.to_numpy();ids=manifest.cell_id.to_numpy(dtype=str);classes=np.unique(y)
    donor_counts=pd.crosstab(manifest.cell_type,manifest.donor)
    if not (donor_counts>0).all().all(): raise ValueError('All fixed retained classes must occur in every donor for this protocol')
    meta=read(a.scgpt_dir/'baron_scgpt_full.metadata.json')
    if digest(a.panel)!=meta['panel_sha256']: raise ValueError('Panel differs from scGPT metadata')
    panel=pd.read_csv(a.panel,sep='\t',dtype=str)
    if not panel.feature_name.is_unique: raise ValueError('Panel symbols must be unique')
    index={g:i for i,g in enumerate(genes)}
    selected=np.array([index[g] for g in panel.feature_name if g in index])
    if len(selected)!=meta['coverage']['present_panel_genes']: raise ValueError('Matched panel coverage mismatch')
    panel_indices=selected if a.gene_policy=='scgpt-panel' else None
    with np.load(a.scgpt_dir/'baron_scgpt_full.npz',allow_pickle=False) as z:
        scids=z['cell_ids'].astype(str)
        if len(set(scids))!=len(scids): raise ValueError('Duplicate scGPT IDs')
        lookup={c:i for i,c in enumerate(scids)};sc=z['vectors'][[lookup[c] for c in ids]].astype(np.float64)
    if not np.isfinite(sc).all(): raise ValueError('Nonfinite scGPT vectors')
    log_counts=normalize_log(counts,10000.);del counts
    out=fresh(a.output,'donor_robustness').resolve()
    write(out/'protocol.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        design='Nested leave-one-donor-out; primary scaling standard; no best-seed selection',
        status='running',retrospective=True,original_experiment_unchanged=True,
        input_hashes=hashes,panel_sha256=digest(a.panel),source_sha256=digest(__file__),environment=environment(),
        primary_metric='equal-donor mean Macro-F1',regularization_objective='mean CE + lambda/2 * ||W||^2',
        limitation='All donors previously inspected in original experiment; conditional on inherited cohort and frozen pretrained representation; overlap unaudited.'))
    donor_counts.to_csv(out/'cell_type_by_donor.csv')
    all_rows=[];all_inner=[];stability=[];diagnostics=[]
    start=time.monotonic()
    try:
      with threadpool_limits(limits=1):
        for donor,train,hold in donor_folds(donors):
            print(f'Outer donor {donor}: train={len(train)}, held out={len(hold)}',flush=True)
            folder=out/donor;folder.mkdir()
            inner_rows=[]
            # All inner preprocessing/dictionaries are re-fitted from the two training donors.
            for validation_donor in sorted(set(donors[train])):
                it=train[donors[train]!=validation_donor];iv=train[donors[train]==validation_donor]
                x,h,_=preprocessing(log_counts,it,iv,genes,panel_indices,a.n_genes)
                representations=[('expression',-1,x,h),('scgpt',-1,sc[it],sc[iv])]
                for seed in a.seeds:
                    z,zh,d,diag=fit_dictionary(x,h,a.atoms,a.alpha,seed,a.epochs)
                    diagnostics.append(dict(outer_donor=donor,inner_donor=validation_donor,**diag))
                    representations.append(('dictionary',seed,z,zh))
                for method,seed,xt,xv in representations:
                    for scaling in ['standard','none']:
                        for regularization in a.regularization:
                            prob,model,_,_,converged=fit_classifier(xt,xv,y[it],regularization,scaling)
                            inner_rows.append(dict(outer_donor=donor,inner_donor=validation_donor,method=method,seed=seed,
                                scaling=scaling,regularization=regularization,converged=converged,**metrics(y[iv],prob,classes)))
                print(f'  inner donor {validation_donor} complete',flush=True)
            all_inner.extend(inner_rows);pd.DataFrame(all_inner).to_csv(out/'inner_search.csv',index=False)
            chosen={}
            for method in ['expression','scgpt','dictionary']:
                for scaling in ['standard','none']:
                    chosen[method,scaling]=choose_regularization([r for r in inner_rows if r['method']==method and r['scaling']==scaling])
            write(folder/'chosen_regularization.json',{f'{m}/{s}':v for (m,s),v in chosen.items()})
            x,h,params=preprocessing(log_counts,train,hold,genes,panel_indices,a.n_genes)
            np.savez_compressed(folder/'preprocessing.npz',**params,train_ids=ids[train],hold_ids=ids[hold])
            representations=[('expression',-1,x,h),('scgpt',-1,sc[train],sc[hold])];dictionaries={}
            for seed in a.seeds:
                z,zh,d,diag=fit_dictionary(x,h,a.atoms,a.alpha,seed,a.epochs)
                diagnostics.append(dict(outer_donor=donor,inner_donor=None,**diag));dictionaries[seed]=(d,z)
                np.savez_compressed(folder/f'dictionary_seed{seed}.npz',dictionary=d,gene_ids=params['gene_ids'],train_ids=ids[train],alpha=a.alpha)
                representations.append(('dictionary',seed,z,zh))
            for sa,sb in combinations(a.seeds,2):
                da,za=dictionaries[sa];db,zb=dictionaries[sb];matched=match_atoms(da,db,za,zb)
                matched.to_csv(folder/f'atom_match_{sa}_{sb}.csv',index=False)
                stability.append(dict(donor=donor,seed_a=sa,seed_b=sb,mean_abs_cosine=matched.abs_cosine.mean(),
                    median_abs_cosine=matched.abs_cosine.median(),mean_top_gene_jaccard=matched.top_gene_jaccard.mean(),
                    mean_train_code_correlation=matched.train_code_correlation.mean()))
            for method,seed,xt,xh in representations:
                for scaling in ['standard','none']:
                    regularization=chosen[method,scaling]
                    prob,model,center,scale,converged=fit_classifier(xt,xh,y[train],regularization,scaling)
                    if not converged: raise RuntimeError('Outer classifier did not converge')
                    name=f'{method}_{scaling}_seed{seed}'
                    np.savez_compressed(folder/f'{name}.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_.astype(str),
                        center=center,scale=scale,probabilities=prob,cell_ids=ids[hold],labels=y[hold].astype(str))
                    all_rows.append(dict(donor=donor,method=method,seed=seed,scaling=scaling,regularization=regularization,
                        C=float(model.C),n_train=len(train),n_hold=len(hold),**metrics(y[hold],prob,classes)))
            pd.DataFrame(all_rows).to_csv(out/'outer_metrics.csv',index=False)
            pd.DataFrame(stability).to_csv(out/'stability.csv',index=False)
            write(out/'dictionary_diagnostics.json',diagnostics)
        table=pd.DataFrame(all_rows)
        # Average seeds within donor first. Donors, not cells or seeds, are replication units.
        metrics_cols=['macro_f1','accuracy','balanced_accuracy','log_loss']
        donor_means=table.groupby(['method','scaling','donor'])[metrics_cols].mean()
        donor_means.to_csv(out/'donor_means.csv')
        summary=donor_means.groupby(['method','scaling']).agg(['mean','std'])
        summary.to_csv(out/'summary.csv');print(summary.to_string())
        table.groupby(['method','scaling','donor']).macro_f1.agg(['mean','std','min','max']).to_csv(out/'seed_variation.csv')
        protocol=read(out/'protocol.json');protocol.update(status='complete',seconds=time.monotonic()-start)
        write(out/'protocol.json',protocol)
    except Exception as exc:
        protocol=read(out/'protocol.json');protocol.update(status='failed',error=str(exc));write(out/'protocol.json',protocol);raise
    print(f'Donor study complete: {out}')

if __name__=='__main__': main()
