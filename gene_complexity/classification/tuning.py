"""Run a fixed dictionary refinement and comparable logistic C searches; validation only."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from gene_complexity._common import ROOT, CONFIG, read, write, fresh, digest, environment
from gene_complexity.dictionary_learning.grid import candidates, run_grid, verify_existing
from gene_complexity.dictionary_learning.training import load_features
from threadpoolctl import threadpool_limits


def rank_dictionaries(best, grid, names):
    """Exact F1 ties: fewer atoms, then fewer mean active atoms, then candidate ID."""
    rows = best[best.feature_set.isin(names)].copy()
    rows['candidate'] = rows.feature_set.map(names)
    rows = rows.merge(grid, on='candidate', validate='one_to_one')
    if len(rows) != len(names):
        raise ValueError('Missing candidate classifier or dictionary diagnostics')
    return rows.sort_values(['val_macro_f1','atoms','validation_mean_active_atoms','candidate'],
                            ascending=[False,True,True,True],kind='stable').reset_index(drop=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=CONFIG,help='Input config; expression, splits, scGPT paths are reused')
    p.add_argument('--grid',type=Path,default=ROOT/'gene_complexity/dictionary_learning/configs/dictionary_grid_v2.json')
    p.add_argument('--reuse-grid',type=Path,help='Optional cache from the current source version; missing candidates are fitted')
    p.add_argument('--output',type=Path,help='New output root, default timestamped results/workflow')
    a=p.parse_args(argv)
    return run(a)


def run(a):
    planned=candidates(read(a.grid)); config=read(a.inputs)
    features=ROOT/config['expression_dir']; _,prep=load_features(features)
    out=fresh(a.output,'model_selection_v2').resolve()
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),environment=environment(),
                threads=1,test_evaluated=False,model_selection='validation Macro-F1',
                dictionary_ties='fewer atoms, then fewer validation mean active atoms, then candidate ID',
                classifier_ties='smaller C',reused=[],fitted=[],requires_review=True)
    report['source_sha256']=digest(__file__)
    write(out/'grid_config.json',read(a.grid))
    write(out/'inputs.json',config)
    # A record of the installed environment, not a cross-platform lockfile.
    with (out/'environment.txt').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    def save(): write(out/'selection_report.json',report)
    save()
    try:
        grid_dir=out/'dictionary_grid'; grid_dir.mkdir()
        names={}
        for c in planned:
            folder=f"k{c['atoms']}_a{repr(c['alpha'])}_s{c['seed']}_e{c['epochs']}"
            names['Dictionary_'+folder]=folder
            source=a.reuse_grid/folder if a.reuse_grid else None
            expected=dict(atoms=c['atoms'],alpha=c['alpha'],seed=c['seed'],epochs_requested=c['epochs'],
                preprocessing_report_sha256=digest(features/'preprocessing.json'),manifest_sha256=prep['manifest_sha256'],
                script_sha256=digest(ROOT/'gene_complexity/dictionary_learning/training.py'),
                sklearn_version=version('scikit-learn'),numpy_version=version('numpy'))
            if source is not None and source.exists():
                verify_existing(source,expected)
                shutil.copytree(source,grid_dir/folder)
                report['reused'].append(str(source.resolve()))
            else:
                report['fitted'].append(folder)
        save()
        with threadpool_limits(limits=1):
            grid=run_grid(out/'grid_config.json',features,grid_dir)
        config.update(dictionary_dir=str(grid_dir),dictionary_runs=names)
        write(out/'validation_config.json',config)
        print('Running identical coarse-to-fine C search for expression, dictionary candidates, and scGPT.',flush=True)
        from gene_complexity.classification.validation import run as run_validation
        from types import SimpleNamespace
        run_validation(SimpleNamespace(config=out/'validation_config.json', output=out/'validation', threads=1))
        best=pd.read_csv(out/'validation/classification/validation_best_C.csv')
        ranked=rank_dictionaries(best,grid,names)
        ranked.to_csv(out/'dictionary_ranking.csv',index=False)
        winner=ranked.iloc[0]
        chosen=best[best.feature_set.isin(['Expression','scGPT',winner.feature_set])]
        chosen.to_csv(out/'recommended_models.csv',index=False)
        report.update(status='complete',recommended_dictionary=winner.candidate,
                      recommended_dictionary_C=float(winner.C),validation_macro_f1=float(winner.val_macro_f1),
                      validation_active_fraction=float(winner.validation_active_fraction),
                      warning_count=int(winner.warning_count),
                      note='Recommendations only. Review diagnostics before freezing; no test evaluation or selection.json generated.')
        print(chosen[['feature_set','C','val_macro_f1']].to_string(index=False))
        print(f'Review rankings and diagnostics in {out}')
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat(); save()


if __name__=='__main__': main()
