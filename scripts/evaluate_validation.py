"""Evaluate a saved classifier on validation; optionally freeze it for later test use."""
import argparse
from pathlib import Path
from _common import evaluate, read, write, digest
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--freeze',action='store_true',help='Explicitly freeze this model choice for final test evaluation')
    a=p.parse_args(); out=evaluate(a.model,a.output)
    if a.freeze:
        r=read(a.model/'training.json')
        write(out/'selection.json',dict(status='frozen',model_sha256=r['model_sha256'],
            training_sha256=digest(a.model/'training.json'),validation_metrics=str((out/'metrics.json').resolve()),
            validation_sha256=digest(out/'metrics.json')))
        print(f'Frozen selection: {out / "selection.json"}')
