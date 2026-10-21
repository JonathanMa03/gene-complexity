"""Evaluate saved coefficients with frozen-selection and provenance checks."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from gene_complexity._common import ROOT, read, write, fresh, digest, environment
from gene_complexity.classification.models import load_model, selected, probabilities
from gene_complexity.classification.inputs import checked_rows

def evaluate(model_dir, output=None, test=False, selection=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
    from gene_complexity.dictionary_learning.solver import encode
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
                    x=encode(x.astype(np.float64),z['dictionary'],float(z['alpha']))
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
