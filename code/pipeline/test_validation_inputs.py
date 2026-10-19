import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'logit'))
from validation_inputs import load_inputs, digest


class ValidationInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exp, self.splits, self.dic = [self.root / p for p in ['expression', 'splits', 'dictionary/k2']]
        for p in [self.exp, self.splits, self.dic]:
            p.mkdir(parents=True)
        self.manifest = pd.DataFrame(dict(cell_id=['a','b','c','d','e'],
            cell_type=['A','B','A','B','A'], donor=['d1']*5,
            split=['train','train','validation','validation','test']))
        self.manifest.to_csv(self.splits/'baron_split_manifest.csv', index=False)
        self.manifest.to_csv(self.exp/'split_manifest.csv', index=False)
        pd.DataFrame({'cell_id':['excluded']}).to_csv(self.splits/'baron_exclusions.csv', index=False)
        mh = digest(self.exp/'split_manifest.csv')
        self.write(self.splits/'split_settings.json',dict(status='final',scgpt_confirmed=True,
            manifest_sha256=mh,input_sha256='raw',split_counts=dict(train=2,validation=2,test=1)))
        np.savez(self.exp/'preprocessing.npz', training_cell_ids=['a','b'])
        artifacts = {}
        for split, ids, values in [('train',['a','b'],[[1,0],[0,1]]),('validation',['c','d'],[[2,0],[0,2]])]:
            np.savez(self.exp/f'{split}.npz', X=values, cell_ids=ids, labels=['A','B'], donors=['d1','d1'])
            artifacts[f'{split}.npz']={'sha256':digest(self.exp/f'{split}.npz')}
            np.savez(self.dic/f'{split}_codes.npz', vectors=values[::-1], cell_ids=ids[::-1], labels=['B','A'], donors=['d1','d1'])
        np.savez(self.dic/'dictionary.npz', components=np.eye(2))
        self.write(self.exp/'preprocessing.json',dict(manifest_sha256=mh,input_sha256='raw',artifacts=artifacts,
            parameters_sha256=digest(self.exp/'preprocessing.npz')))
        self.write(self.dic/'pilot_report.json',dict(status='pilot_complete',test_accessed=False,atoms=2,
            manifest_sha256=mh,preprocessing_report_sha256=digest(self.exp/'preprocessing.json'),
            input_artifact_hashes={s:artifacts[f'{s}.npz']['sha256'] for s in ['train','validation']},
            output_hashes={p.name:digest(p) for p in self.dic.glob('*.npz')}))
        self.sc=self.root/'sc.npz'
        np.savez(self.sc,vectors=np.ones((6,2)),cell_ids=['a','b','c','d','e','excluded'])
        self.write(self.sc.with_suffix('.metadata.json'),dict(status='PASS',shape=[6,2],output_sha256=digest(self.sc),model_id='fixture'))
        self.config=dict(expression_dir=str(self.exp),split_dir=str(self.splits),dictionary_dir=str(self.dic.parent),
                         dictionary_runs={'Dictionary': 'k2'},scgpt_path=str(self.sc))

    def write(self,path,value):
        path.write_text(json.dumps(value))

    def test_alignment_and_no_test_file_access(self):
        original=np.load
        def guarded(path,*a,**kw):
            self.assertNotEqual(Path(path).name,'test.npz')
            return original(path,*a,**kw)
        with patch('numpy.load',side_effect=guarded):
            b=load_inputs(self.config)
        np.testing.assert_array_equal(b['datasets'][1][1], [[1,0],[0,1]])

    def test_reject_modified_artifact(self):
        with (self.dic/'train_codes.npz').open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'): load_inputs(self.config)

    def test_reject_wrong_labels_even_with_updated_hash(self):
        p=self.exp/'train.npz'
        np.savez(p,X=np.eye(2),cell_ids=['a','b'],labels=['B','A'],donors=['d1','d1'])
        r=json.loads((self.exp/'preprocessing.json').read_text())
        r['artifacts']['train.npz']['sha256']=digest(p)
        self.write(self.exp/'preprocessing.json',r)
        with self.assertRaisesRegex(ValueError,'labels differs'): load_inputs(self.config)

    def test_reject_wrong_scgpt_cohort(self):
        np.savez(self.sc,vectors=np.ones((6,2)),cell_ids=['a','b','c','d','e','wrong'])
        p=self.sc.with_suffix('.metadata.json'); r=json.loads(p.read_text()); r['output_sha256']=digest(self.sc); self.write(p,r)
        with self.assertRaisesRegex(ValueError,'coverage differs'): load_inputs(self.config)


if __name__ == '__main__':
    unittest.main()
