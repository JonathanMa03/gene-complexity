"""Synthetic end-to-end tests; no project test cells are loaded."""
import argparse
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/pipeline'))
from test_validation_inputs import ValidationInputTests
from _common import train, evaluate, read, write, digest, probabilities

class WorkflowTests(ValidationInputTests):
    def test_saved_probability_equivalence(self):
        for y in [np.array(['a','b']*3),np.array(['a','b','c']*2)]:
            x=np.arange(18).reshape(6,3)
            model=LogisticRegression().fit(x,y)
            saved=dict(coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
            np.testing.assert_allclose(probabilities(saved,x),model.predict_proba(x),rtol=1e-12)

    def test_three_methods_and_test_gate(self):
        config=self.root/'config.json'; write(config,self.config)
        # A synthetic test artifact; upstream parameters remain fixed.
        testpath=self.exp/'test.npz'
        np.savez(testpath,X=[[1.,0.]],cell_ids=['e'],labels=['A'],donors=['d1'],gene_ids=['g1','g2'])
        np.savez(self.dic/'dictionary.npz',dictionary=np.eye(2),alpha=np.array(1.),gene_ids=['g1','g2'])
        r=read(self.exp/'preprocessing.json'); r['artifacts']['test.npz']={'sha256':digest(testpath)}; write(self.exp/'preprocessing.json',r)
        r=read(self.dic/'pilot_report.json'); r['preprocessing_report_sha256']=digest(self.exp/'preprocessing.json'); r['output_hashes']['dictionary.npz']=digest(self.dic/'dictionary.npz'); write(self.dic/'pilot_report.json',r)
        for method in ['baseline','dictionary','scgpt']:
            args=argparse.Namespace(config=config,method=method,candidate='Dictionary' if method=='dictionary' else None,
                C=1.,max_iter=5000,tol=1e-4,seed=42,threads=1,output=self.root/f'model_{method}')
            original=np.load
            def no_test(path,*a,**kw):
                self.assertNotEqual(Path(path),testpath)
                return original(path,*a,**kw)
            with patch('numpy.load',side_effect=no_test):
                model=train(args); val=evaluate(model,self.root/f'val_{method}')
            with np.load(val/'predictions.npz',allow_pickle=False) as predictions:
                self.assertEqual(len(predictions['labels']),2)
            with self.assertRaisesRegex(ValueError,'requires --selection'):
                evaluate(model,self.root/'blocked',test=True)
            r=read(model/'training.json'); selection=val/'selection.json'
            write(selection,dict(status='frozen',model_sha256=r['model_sha256'],training_sha256=digest(model/'training.json'),
                validation_metrics=str(val/'metrics.json'),validation_sha256=digest(val/'metrics.json')))
            test=evaluate(model,self.root/f'test_{method}',test=True,selection=selection)
            self.assertEqual(read(test/'metrics.json')['n_cells'],1)
            with self.assertRaises(FileExistsError): train(args)
            bad=read(selection); bad['model_sha256']='wrong'; write(selection,bad)
            with self.assertRaisesRegex(ValueError,'does not match'): evaluate(model,test=True,selection=selection)

if __name__=='__main__': unittest.main()
