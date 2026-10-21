"""Public-data download and preparation; only the R exporter uses a subprocess."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
from gene_complexity._paths import ROOT, PACKAGE_ROOT
from gene_complexity._common import read, digest
from .classical import prepare
from .convert import build_baron
from .splits import run as split_run


def download_data(rscript='Rscript'):
    files=[ROOT/'data/processed'/f'baron_{n}' for n in ['counts.mtx','genes.csv','cells.csv']]
    if all(f.is_file() for f in files):
        print('Existing Baron exports found; preserved. Preprocessing will validate them.')
    elif any(f.exists() for f in files):
        raise FileExistsError('Partial exports found. Move them aside before downloading again.')
    else:
        if not shutil.which(rscript): raise FileNotFoundError('Rscript not found. Install R or supply its path.')
        subprocess.run([rscript,str(PACKAGE_ROOT/'preprocessing/export_baron.R')],cwd=ROOT,check=True)


def preprocess_data(*, splits=ROOT/'external/baron_splits_final', output=ROOT/'data/processed/baron_classical_v1',
                    seed=42, scgpt_confirmed=False, scgpt_exclusions=None, n_genes=2000, target_sum=10000):
    splits,output=Path(splits),Path(output);raw=ROOT/'data/processed/baron_raw.h5ad'
    scgpt_exclusions=Path(scgpt_exclusions) if scgpt_exclusions else None
    if output.exists(): raise FileExistsError('Choose a new output; existing features are preserved')
    if not splits.exists() and not scgpt_confirmed: raise ValueError('Final splits require confirmed scGPT compatibility')
    if not raw.exists(): build_baron()
    if not splits.exists():
        split_run(SimpleNamespace(input=raw,output_dir=splits,seed=seed,scgpt_confirmed=True,scgpt_exclusions=scgpt_exclusions))
    else:
        settings=read(splits/'split_settings.json')
        if settings['seed']!=seed: raise ValueError('Existing split seed differs')
        if scgpt_exclusions and settings['scgpt_exclusions_sha256']!=digest(scgpt_exclusions):
            raise ValueError('Existing exclusions differ')
    return prepare(raw,splits,output,n_genes,target_sum)
