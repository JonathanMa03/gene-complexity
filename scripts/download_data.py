"""Download/cache raw Baron counts through the existing R/Bioconductor exporter."""
import argparse
import shutil
import subprocess
from _common import ROOT
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rscript',default='Rscript',help='Rscript executable')
    a=p.parse_args()
    files=[ROOT/'data/processed'/f'baron_{n}' for n in ['counts.mtx','genes.csv','cells.csv']]
    if all(f.is_file() for f in files):
        print('Existing Baron exports found; preserved. Preprocessing will validate them.')
    elif any(f.exists() for f in files):
        raise FileExistsError('Partial exports found. Move them aside before downloading again.')
    else:
        if not shutil.which(a.rscript): raise FileNotFoundError('Rscript not found. Install R or specify --rscript.')
        subprocess.run([a.rscript,str(ROOT/'code/preprocessing/export_baron.R')],cwd=ROOT,check=True)
