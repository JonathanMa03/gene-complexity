"""Evaluate a frozen classifier on test cells without refitting any model or preprocessing."""
import argparse
from pathlib import Path
from _common import evaluate
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True,help='selection.json from explicit --freeze')
    p.add_argument('--output',type=Path)
    a=p.parse_args(); evaluate(a.model,a.output,test=True,selection=a.selection)
