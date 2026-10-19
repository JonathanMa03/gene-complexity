"""Repository paths, immutable outputs, and checked classifier artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from importlib.metadata import version
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code/logit'), str(ROOT/'code/pipeline')]
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from validation_inputs import digest, load_inputs, checked_rows

CONFIG = ROOT/'code/pipeline/configs/validation_v1.json'

def config_arg(p):
    p.add_argument('--config', type=Path, default=CONFIG, help='Input-artifact JSON config')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')

def fresh(path, prefix):
    path = Path(path) if path else ROOT/'results/workflow'/f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}'
    path.mkdir(parents=True, exist_ok=False)
    return path

def environment():
    return dict(python=platform.python_version(), versions={p:version(p) for p in ['numpy','scipy','scikit-learn','pandas']})

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

def evaluate(model_dir, output=None, test=False, selection=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
    from sklearn.decomposition import sparse_encode
    directory=Path(model_dir)
    r,model=load_model(directory)
    if test:
        if selection is None: raise ValueError('Test evaluation requires --selection from explicitly frozen validation evaluation')
        freeze=read(selection)
        if freeze['model_sha256']!=r['model_sha256'] or freeze['training_sha256']!=digest(directory/'training.json'):
            raise ValueError('Selection does not match this trained model')
        valpath=Path(freeze['validation_metrics'])
        if digest(valpath)!=freeze['validation_sha256'] or freeze['status']!='frozen':
            raise ValueError('Invalid frozen validation selection')
    config,b,_,val=selected(r['config'],r['method'],r['candidate'])
    split='test' if test else 'validation'
    additional={}
    if not test:
        x=val; ids=b['ids']['validation']; y=b['y_val']
    else:
        exp=ROOT/config['expression_dir']; prep=read(exp/'preprocessing.json')
        path=exp/'test.npz'
        if digest(path)!=prep['artifacts']['test.npz']['sha256']: raise ValueError('Test feature hash mismatch')
        additional[str(path)]=digest(path)
        x,ids=checked_rows(path,'X',b['manifest'],'test')
        y=b['manifest'].loc[ids,'cell_type'].to_numpy()
        if r['method']=='dictionary':
            path=ROOT/config['dictionary_dir']/config['dictionary_runs'][r['candidate']]/'dictionary.npz'
            with np.load(path,allow_pickle=False) as z, np.load(exp/'test.npz',allow_pickle=False) as t:
                if not np.array_equal(z['gene_ids'],t['gene_ids']): raise ValueError('Test gene order mismatch')
                with threadpool_limits(limits=r['threads']), warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    x=sparse_encode(x.astype(np.float64),z['dictionary'],algorithm='lasso_cd',alpha=float(z['alpha']),max_iter=3000)
        elif r['method']=='scgpt':
            with np.load(ROOT/config['scgpt_path'],allow_pickle=False) as z:
                lookup={c:i for i,c in enumerate(z['cell_ids'].astype(str))}
                x=z['vectors'][[lookup[c] for c in ids]]
    if x.shape[1]!=r['n_features'] or not np.isfinite(x).all(): raise ValueError('Invalid evaluation features')
    prob=probabilities(model,x); classes=model['classes']; pred=classes[prob.argmax(axis=1)]
    output=fresh(output,split+'_'+r['method'])
    metrics=dict(split=split,method=r['method'],candidate=r['candidate'],C=r['C'],n_cells=len(y),
        accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro'),
        balanced_accuracy=balanced_accuracy_score(y,pred),log_loss=log_loss(y,prob,labels=classes),
        model_sha256=r['model_sha256'],training_sha256=digest(directory/'training.json'),
        model_dir=str(directory.resolve()),additional_input_hashes=additional,environment=environment())
    write(output/'metrics.json',metrics)
    pd.DataFrame(classification_report(y,pred,labels=classes,output_dict=True,zero_division=0)).T.to_csv(output/'per_class.csv')
    pd.DataFrame(confusion_matrix(y,pred,labels=classes),index=classes,columns=classes).to_csv(output/'confusion_matrix.csv')
    np.savez_compressed(output/'predictions.npz',cell_ids=np.asarray(ids,dtype=str),labels=np.asarray(y,dtype=str),predictions=np.asarray(pred,dtype=str),probabilities=prob,classes=classes)
    print(f'{split} Macro-F1: {metrics["macro_f1"]:.6f}; output: {output}')
    return output
