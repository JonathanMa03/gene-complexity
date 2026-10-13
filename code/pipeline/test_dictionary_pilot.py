import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from dictionary_pilot import load_features
from prepare_classical import digest


class LoaderTests(unittest.TestCase):
    def test_train_validation_only_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = pd.DataFrame({'cell_id':['a','b','c'], 'cell_type':['type']*3,
                              'donor':['d']*3, 'split':['train','validation','test']})
            m.to_csv(root/'split_manifest.csv', index=False)
            genes=np.array(['g'])
            for split, cell in [('train','a'),('validation','b')]:
                np.savez(root/f'{split}.npz', X=np.ones((1,1)), cell_ids=np.array([cell]),
                         labels=np.array(['type']),donors=np.array(['d']),gene_ids=genes)
            np.savez(root/'preprocessing.npz',gene_ids=genes,training_cell_ids=np.array(['a']))
            report={'manifest_sha256':digest(root/'split_manifest.csv'),
                    'parameters_sha256':digest(root/'preprocessing.npz'),
                    'artifacts':{f'{s}.npz':{'sha256':digest(root/f'{s}.npz')} for s in ('train','validation')}}
            (root/'preprocessing.json').write_text(json.dumps(report))
            # No test file exists: loader must succeed without it.
            data,_=load_features(root)
            self.assertEqual(set(data), {'train','validation'})
            (root/'validation.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'feature hash mismatch'):
                load_features(root)


if __name__=='__main__': unittest.main()
