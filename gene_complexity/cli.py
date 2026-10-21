"""Optional command-line adapters; the notebook uses the function API directly."""
import argparse
from importlib import import_module
from pathlib import Path
import sys
from gene_complexity._paths import ROOT, CONFIG

MODULES = {
    'prepare-classical':'preprocessing.classical', 'create-splits':'preprocessing.splits',
    'run-dictionary':'dictionary_learning.training', 'dictionary-grid':'dictionary_learning.grid',
    'review-dictionary':'dictionary_learning.interpretation',
    'load-scgpt':'scgpt.handoff', 'run-validation':'classification.validation',
    'tune-models':'classification.tuning', 'run-donor-study':'evaluation.donor',
    'summarize-donor-study':'reporting.donors', 'summarize-results':'reporting.summary',
}
OTHER = ['download-data','preprocess-data','train-logistic','run-baseline',
         'evaluate-validation','run-test','run-end-to-end','build-report-assets','build-baron','map-panel']


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ['-h','--help']:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('command',choices=sorted([*MODULES,*OTHER]))
        p.print_help();return
    command,*args=argv
    if command in MODULES:
        return import_module('gene_complexity.'+MODULES[command]).main(args)
    if command not in OTHER:
        raise SystemExit(f'Unknown command {command!r}; use --help')
    from gene_complexity import api
    p=argparse.ArgumentParser(prog='gene-complexity '+command)
    if command in ['train-logistic','run-baseline']:
        from gene_complexity.classification.models import train_parser, train
        return train(train_parser('Train a logistic classifier',baseline=command=='run-baseline').parse_args(args))
    if command=='download-data':
        p.add_argument('--rscript',default='Rscript');a=p.parse_args(args)
        from gene_complexity.preprocessing import download_data
        return download_data(a.rscript)
    if command=='preprocess-data':
        p.add_argument('--splits',type=Path,default=ROOT/'external/baron_splits_final')
        p.add_argument('--output',type=Path,default=ROOT/'data/processed/baron_classical_v1')
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--scgpt-confirmed',action='store_true')
        p.add_argument('--scgpt-exclusions',type=Path)
        p.add_argument('--n-genes',type=int,default=2000)
        p.add_argument('--target-sum',type=float,default=10000)
        from gene_complexity.preprocessing import preprocess_data
        return preprocess_data(**vars(p.parse_args(args)))
    if command in ['evaluate-validation','run-test']:
        p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path)
        if command=='run-test':
            p.add_argument('--selection',type=Path,required=True)
            return api.evaluate_test(**vars(p.parse_args(args)))
        p.add_argument('--freeze',action='store_true')
        return api.evaluate_validation(**vars(p.parse_args(args)))
    if command=='run-end-to-end':
        p.add_argument('--profile',choices=['smoke','standard'],default='smoke')
        p.add_argument('--raw',type=Path,default=ROOT/'data/processed/baron_raw.h5ad')
        p.add_argument('--splits',type=Path,default=ROOT/'external/baron_splits_final')
        p.add_argument('--scgpt-dir',type=Path,default=ROOT/'external/scgpt/pretraining_handoff/outputs/baron')
        p.add_argument('--output',type=Path)
        p.add_argument('--check-only',action='store_true')
        for flag in ['n-genes','atoms','epochs']: p.add_argument('--'+flag,type=int)
        p.add_argument('--alpha',type=float,default=1.0);p.add_argument('--C',type=float,default=1.0)
        p.add_argument('--threads',type=int,default=1);p.add_argument('--plots',action='store_true')
        a=p.parse_args(args)
        if a.check_only:
            from gene_complexity.preflight import preflight
            preflight(a.raw.resolve(),a.splits.resolve(),a.scgpt_dir.resolve())
            print('Retained input provenance verified. No outputs written.');return
        from datetime import datetime,timezone
        from gene_complexity import Workflow
        defaults=(256,8,2) if a.profile=='smoke' else (2000,32,10)
        n,k,e=[x if x is not None else d for x,d in zip([a.n_genes,a.atoms,a.epochs],defaults)]
        out=a.output or ROOT/'results/workflow'/datetime.now(timezone.utc).strftime('end_to_end_%Y%m%dT%H%M%S_%fZ')
        return Workflow(output=out,raw=a.raw,splits=a.splits,scgpt=a.scgpt_dir,n_genes=n,
                        atoms=k,epochs=e,alpha=a.alpha,C=a.C,threads=a.threads,plots=a.plots).run_all()
    p.parse_args(args)
    if command=='build-report-assets': return api.build_report_assets()
    if command=='build-baron':
        from gene_complexity.preprocessing.convert import build_baron
        return build_baron()
    if command=='map-panel':
        from gene_complexity.preprocessing.mapping import map_panel
        return map_panel()
