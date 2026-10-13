import json
import tempfile
import unittest
from pathlib import Path
from dictionary_grid import candidates, verify_existing
from prepare_classical import digest


class GridTests(unittest.TestCase):
    def test_grid_and_validation(self):
        config=dict(atoms=[16,32], alpha=[1,5], seeds=[42], epochs=10)
        self.assertEqual(len(candidates(config)),4)
        for key,value in [('alpha',[0]),('atoms',[16,16]),('seeds',[-1]),('epochs',0)]:
            with self.assertRaises(ValueError): candidates(dict(config,**{key:value}))

    def test_skip_requires_identity_and_intact_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with self.assertRaisesRegex(ValueError,'Incomplete'): verify_existing(folder,{})
            names=['dictionary.npz','train_codes.npz','validation_codes.npz']
            for name in names: (folder/name).write_bytes(b'fixture')
            report=dict(status='pilot_complete',test_accessed=False,alpha=1.0,
                        output_hashes={n:digest(folder/n) for n in names})
            (folder/'pilot_report.json').write_text(json.dumps(report))
            verify_existing(folder,dict(alpha=1.0))
            with self.assertRaisesRegex(ValueError,'different alpha'):
                verify_existing(folder,dict(alpha=5.0))
            (folder/names[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                verify_existing(folder,dict(alpha=1.0))


if __name__=='__main__': unittest.main()
