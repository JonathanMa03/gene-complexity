"""Fit and load checked multinomial logistic classifiers."""
import argparse
from pathlib import Path
import warnings
import numpy as np
from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from gene_complexity._common import ROOT, config_arg, read, write, fresh, digest, environment
from gene_complexity.classification.inputs import load_inputs

def selected(config, method, candidate):
    config = dict(config)
    if method == 'dictionary':
        if candidate is None or candidate not in config['dictionary_runs']:
            raise ValueError('--candidate must name an entry in config dictionary_runs')
        config['dictionary_runs'] = {candidate:config['dictionary_runs'][candidate]}
    b = load_inputs(config, methods=(method,))
    name = {'baseline':'Expression','scgpt':'scGPT','dictionary':candidate}[method]
    _, train, val = next(row for row in b['datasets'] if row[0]==name)
    return config, b, train, val

def train(args):
    if args.C <= 0 or not np.isfinite(args.C) or args.max_iter < 1 or args.tol <= 0 or args.threads < 1:
        raise ValueError('C, max-iter, tol and threads must be positive')
    config,b,x,_ = selected(read(args.config),args.method,args.candidate)
    output = fresh(args.output,'train_'+args.method)
    with threadpool_limits(limits=args.threads), warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=args.C,solver='lbfgs',max_iter=args.max_iter,tol=args.tol,random_state=args.seed)
        model.fit(x,b['y_train'])
    np.savez_compressed(output/'model.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_.astype(str))
    report=dict(status='trained', method=args.method,candidate=args.candidate,config=config,
                C=args.C,max_iter=args.max_iter,tol=args.tol,seed=args.seed,threads=args.threads,
                solver='lbfgs',scaling='none',fitted_split='train',test_used=False,
                input_hashes=b['input_hashes'],model_sha256=digest(output/'model.npz'),
                n_features=x.shape[1],environment=environment())
    write(output/'training.json',report)
    print(f'Trained model: {output}')
    return output

def train_parser(description, baseline=False):
    p=argparse.ArgumentParser(description=description); config_arg(p)
    if baseline: p.set_defaults(method='baseline',candidate=None)
    else:
        p.add_argument('--method',choices=['baseline','dictionary','scgpt'],required=True)
        p.add_argument('--candidate',help='Dictionary candidate name from config (required for dictionary)')
    p.add_argument('--C',type=float,default=1.0,help='Inverse L2 regularization strength')
    p.add_argument('--max-iter',type=int,default=5000)
    p.add_argument('--tol',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--output',type=Path,help='New model directory; default timestamped results/workflow')
    return p

def load_model(directory):
    directory=Path(directory); r=read(directory/'training.json')
    if r['status']!='trained' or digest(directory/'model.npz')!=r['model_sha256']:
        raise ValueError('Invalid trained model or model hash')
    for path,h in r['input_hashes'].items():
        if digest(path)!=h: raise ValueError(f'Training input changed: {path}')
    with np.load(directory/'model.npz',allow_pickle=False) as z:
        model={k:z[k] for k in z.files}
    return r,model

def probabilities(model,x):
    scores=x@model['coef'].T+model['intercept']
    if len(model['classes'])==2:
        p=expit(scores.ravel()); return np.column_stack([1-p,p])
    return softmax(scores,axis=1)

