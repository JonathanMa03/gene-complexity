"""Sparse encoding, controlled dictionary fitting, optimality and atom stability."""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode

def encode(x,d,alpha):
    return sparse_encode(x,d,algorithm='lasso_cd',alpha=alpha,max_iter=3000,n_jobs=1)


def kkt_residual(x,z,d,alpha):
    gradient=(z@d-x)@d.T
    active=z!=0
    residual=np.where(active,np.abs(gradient+alpha*np.sign(z)),np.maximum(np.abs(gradient)-alpha,0))
    return dict(max=float(residual.max()),p95=float(np.quantile(residual,.95)),
                relative_max=float(residual.max()/max(1.,alpha)))


def fit_dictionary(x,h,atoms,alpha,seed,epochs,initialization="train-cells",batch_seed=42):
    init=None
    random_state=seed
    if initialization=="train-cells":
        norms=np.linalg.norm(x,axis=1)
        eligible=np.flatnonzero(norms>1e-12)
        if len(eligible)<atoms: raise ValueError("Not enough nonzero training rows to initialize atoms")
        rows=np.random.default_rng(seed).choice(eligible,atoms,replace=False)
        init=x[rows]/norms[rows,None]
        random_state=batch_seed
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=MiniBatchDictionaryLearning(n_components=atoms,alpha=alpha,max_iter=epochs,batch_size=128,
            fit_algorithm='cd',transform_algorithm='lasso_cd',transform_alpha=alpha,transform_max_iter=3000,
            dict_init=init,random_state=random_state,n_jobs=1,tol=1e-3,max_no_improvement=10).fit(x)
        d=model.components_.copy();z=encode(x,d,alpha); zh=encode(h,d,alpha)
    if not all(np.isfinite(a).all() for a in [d,z,zh]) or np.max(np.linalg.norm(d,axis=1))>1.00001:
        raise ValueError('Invalid dictionary or codes')
    gradient=z.T@(z@d-x)/len(x)
    step=d-gradient;step=step/np.maximum(1.,np.linalg.norm(step,axis=1))[:,None]
    return z,zh,d,dict(seed=seed,initialization=initialization,batch_seed=random_state,
        gram_min_eigenvalue=float(np.linalg.eigvalsh(d@d.T).min()),
        dictionary_projected_gradient_norm=float(np.linalg.norm(d-step)),
        train_objective_per_cell=float((.5*np.sum((x-z@d)**2)+alpha*np.abs(z).sum())/len(x)),epochs_run=int(model.n_iter_),warnings=[str(w.message) for w in caught],
        train_active_fraction=float(np.mean(np.abs(z)>1e-6)),hold_active_fraction=float(np.mean(np.abs(zh)>1e-6)),
        train_mse=float(np.mean((x-z@d)**2)),hold_mse=float(np.mean((h-zh@d)**2)),
        train_kkt=kkt_residual(x,z,d,alpha),hold_kkt=kkt_residual(h,zh,d,alpha))


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


