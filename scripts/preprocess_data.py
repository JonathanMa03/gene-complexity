"""Convert Baron exports, create or reuse final splits, and fit classical preprocessing."""
import argparse
import subprocess
import sys
from pathlib import Path
from _common import ROOT, read, digest
from prepare_classical import prepare
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,default=ROOT/'data/processed/baron_splits_final')
    p.add_argument('--output',type=Path,default=ROOT/'data/processed/baron_classical_v1')
    p.add_argument('--seed',type=int,default=42,help='Used only when creating a new split bundle')
    p.add_argument('--scgpt-confirmed',action='store_true',help='Required for creating final splits after exclusions confirmed')
    p.add_argument('--scgpt-exclusions',type=Path)
    p.add_argument('--n-genes',type=int,default=2000)
    p.add_argument('--target-sum',type=float,default=10000)
    a=p.parse_args(); raw=ROOT/'data/processed/baron_raw.h5ad'
    if a.output.exists(): raise FileExistsError('Choose a new preprocessing --output; existing features preserved')
    if not a.splits.exists() and not a.scgpt_confirmed: p.error('New final splits require --scgpt-confirmed')
    if not raw.exists(): subprocess.run([sys.executable,str(ROOT/'code/preprocessing/build_baron_h5ad.py')],cwd=ROOT,check=True)
    if not a.splits.exists():
        cmd=[sys.executable,str(ROOT/'code/preprocessing/create_splits.py'),'--input',str(raw),'--output-dir',str(a.splits.resolve()),'--seed',str(a.seed),'--scgpt-confirmed']
        if a.scgpt_exclusions: cmd+=['--scgpt-exclusions',str(a.scgpt_exclusions.resolve())]
        subprocess.run(cmd,cwd=ROOT,check=True)
    else:
        settings=read(a.splits/'split_settings.json')
        if settings['seed']!=a.seed: raise ValueError('Existing split seed differs; choose another bundle for new splits')
        if a.scgpt_exclusions and settings['scgpt_exclusions_sha256']!=digest(a.scgpt_exclusions): raise ValueError('Existing exclusions differ')
    prepare(raw,a.splits,a.output,a.n_genes,a.target_sum)
