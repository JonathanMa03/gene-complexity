import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from _common import digest, write, read
from run_end_to_end import main, preflight


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.raw=self.root/'raw.h5ad'; self.raw.write_bytes(b'raw fixture')
        self.splits=self.root/'splits'; self.sc=self.root/'scgpt'
        self.splits.mkdir(); self.sc.mkdir()
        (self.splits/'baron_split_manifest.csv').write_text('cell_id,split\na,train\n')
        (self.splits/'baron_exclusions.csv').write_text('cell_id\n')
        write(self.splits/'split_settings.json',dict(status='final',scgpt_confirmed=True,
            input_sha256=digest(self.raw),manifest_sha256=digest(self.splits/'baron_split_manifest.csv')))
        embedding=self.sc/'baron_scgpt_full.npz'; embedding.write_bytes(b'embedding fixture')
        write(self.sc/'baron_scgpt_full.metadata.json',dict(status='PASS',output_sha256=digest(embedding),
            model_id='fixture',checkpoint_sha256='checkpoint',input_sha256='mapped'))
        write(self.sc/'checkpoint_selection.json',dict(checkpoint_id='fixture',checkpoint_sha256='checkpoint'))
        write(self.sc/'preparation_report.json',dict(status='PASS',output_sha256='mapped',input_hashes={'raw':digest(self.raw)}))
        self.output=self.root/'run'
        self.argv=['run_end_to_end.py','--raw',str(self.raw),'--splits',str(self.splits),'--scgpt-dir',str(self.sc),'--output',str(self.output)]

    def test_check_only_writes_nothing(self):
        with patch('sys.argv',self.argv+['--check-only']), patch('run_end_to_end.subprocess.run') as run:
            main(); run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_file_and_changed_hash(self):
        self.raw.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Raw Baron hash'): preflight(self.raw,self.splits,self.sc)
        self.raw.unlink()
        with self.assertRaisesRegex(FileNotFoundError,'Missing retained inputs'): preflight(self.raw,self.splits,self.sc)

    def test_failed_stage_recorded_and_preserved(self):
        with patch('sys.argv',self.argv), patch('run_end_to_end.subprocess.run',return_value=argparse.Namespace(returncode=1)):
            with self.assertRaisesRegex(RuntimeError,'preprocessing failed'): main()
        r=read(self.output/'end_to_end_report.json')
        self.assertEqual(r['status'],'failed'); self.assertEqual(len(r['stages']),1)
        with patch('sys.argv',self.argv):
            with self.assertRaises(FileExistsError): main()


if __name__ == '__main__': unittest.main()
