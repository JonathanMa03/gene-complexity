"""Fit a dictionary on train and encode train/validation using adjustable settings."""
import argparse
from pathlib import Path
from _common import ROOT
from dictionary_pilot import run
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=ROOT/'data/processed/baron_classical_v1')
    p.add_argument('--output',type=Path,required=True,help='New dictionary candidate directory')
    p.add_argument('--atoms',type=int,default=16)
    p.add_argument('--alpha',type=float,default=1.0)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); run(a.features,a.output,a.atoms,a.alpha,a.epochs,a.seed)
