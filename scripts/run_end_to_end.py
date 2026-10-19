"""Rebuild representations and run all three validation classifiers from minimal retained inputs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from _common import ROOT, digest, environment, fresh, read, write

HANDOFF = ('baron_scgpt_full.npz', 'baron_scgpt_full.metadata.json',
           'checkpoint_selection.json', 'preparation_report.json')
SPLIT_FILES = ('baron_split_manifest.csv', 'split_settings.json', 'baron_exclusions.csv')


def preflight(raw, splits, scgpt):
    paths = [raw, *[splits/n for n in SPLIT_FILES], *[scgpt/n for n in HANDOFF]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing retained inputs:\n' + '\n'.join(missing))
    hashes = {str(p): digest(p) for p in paths}
    settings = read(splits/'split_settings.json')
    if settings['status'] != 'final' or not settings['scgpt_confirmed']:
        raise ValueError('Retain the finalized, scGPT-confirmed split bundle')
    if hashes[str(raw)] != settings['input_sha256']:
        raise ValueError('Raw Baron hash differs from final split settings')
    if hashes[str(splits/'baron_split_manifest.csv')] != settings['manifest_sha256']:
        raise ValueError('Final manifest hash mismatch')
    meta, choice, prep = [read(scgpt/n) for n in HANDOFF[1:]]
    if meta['status'] != 'PASS' or hashes[str(scgpt/HANDOFF[0])] != meta['output_sha256']:
        raise ValueError('Invalid scGPT embedding metadata/hash')
    if choice['checkpoint_id'] != meta['model_id'] or choice['checkpoint_sha256'] != meta['checkpoint_sha256']:
        raise ValueError('scGPT checkpoint selection mismatch')
    if prep['status'] != 'PASS' or prep['output_sha256'] != meta['input_sha256'] or hashes[str(raw)] not in prep['input_hashes'].values():
        raise ValueError('scGPT preparation does not match shared raw data')
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['smoke', 'standard'], default='smoke')
    p.add_argument('--raw', type=Path, default=ROOT/'data/processed/baron_raw.h5ad')
    p.add_argument('--splits', type=Path, default=ROOT/'data/processed/baron_splits_final')
    p.add_argument('--scgpt-dir', type=Path, default=ROOT/'code/scgpt/pretraining_handoff/outputs/baron')
    p.add_argument('--output', type=Path, help='New run directory; default timestamped results/workflow')
    p.add_argument('--check-only', action='store_true', help='Check retained input files/hashes without generating outputs or fitting')
    p.add_argument('--n-genes', type=int, help='Override profile feature count: smoke=256, standard=2000')
    p.add_argument('--atoms', type=int, help='Override profile atoms: smoke=8, standard=32')
    p.add_argument('--epochs', type=int, help='Override profile epoch ceiling: smoke=2, standard=10')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--C', type=float, default=1.0, help='Fixed classifier C for all methods; no tuning')
    p.add_argument('--threads', type=int, default=1, help='Classifier thread limit; dictionary remains one thread')
    p.add_argument('--plots', action='store_true')
    a = p.parse_args()
    a.raw, a.splits, a.scgpt_dir = [v.resolve() for v in [a.raw,a.splits,a.scgpt_dir]]
    defaults = (256,8,2) if a.profile == 'smoke' else (2000,32,10)
    genes, atoms, epochs = [v if v is not None else d for v,d in zip([a.n_genes,a.atoms,a.epochs],defaults)]
    import math
    if min(genes,atoms,epochs,a.threads) < 1 or atoms > genes or not all(math.isfinite(v) and v>0 for v in [a.alpha,a.C]):
        p.error('Positive settings required, with atoms <= n-genes')
    if a.output and a.output.exists():
        raise FileExistsError('Output already exists; choose a new run directory')
    hashes = preflight(a.raw,a.splits,a.scgpt_dir)
    print('Retained inputs and provenance checks passed.', flush=True)
    if a.check_only:
        print('Check only: no outputs written and no models fitted. Full matrix/cohort checks run during execution.')
        return
    out = fresh(a.output, 'end_to_end_'+a.profile).resolve()
    config = dict(expression_dir=str(out/'features'), split_dir=str(a.splits),
                  dictionary_dir=str(out/'dictionary'), dictionary_runs={'Dictionary':'candidate'},
                  scgpt_path=str(a.scgpt_dir/HANDOFF[0]))
    write(out/'config.json',config)
    report = dict(status='running',profile=a.profile,started_utc=datetime.now(timezone.utc).isoformat(),
                  settings=dict(n_genes=genes,atoms=atoms,epochs=epochs,alpha=a.alpha,C=a.C,threads=a.threads,
                                seed=42,target_sum=10000,max_iter=5000,tol=1e-4,plots=a.plots),
                  retained_input_hashes=hashes,environment=environment(),stages=[],
                  model_selection_performed=False,test_evaluated=False,
                  note='Preprocessing writes transformed test features; no test classifier evaluation or selection freezing.')
    report['source_hashes'] = {str(path.relative_to(ROOT)):digest(path) for folder in ['scripts','code/pipeline','code/logit'] for path in (ROOT/folder).glob('*.py')}
    def save(): write(out/'end_to_end_report.json',report)
    save()
    def stage(name, script, *args):
        cmd = [sys.executable,str(ROOT/script),*[str(v) for v in args]]
        entry = dict(name=name,command=cmd,status='running'); report['stages'].append(entry); save()
        print(f'[{len(report["stages"])}/10] {name}',flush=True)
        start=time.monotonic()
        with (out/f'{len(report["stages"]):02d}_{name}.log').open('w') as log:
            result = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(seconds=time.monotonic()-start,returncode=result.returncode,
                     status='complete' if result.returncode==0 else 'failed'); save()
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out / (str(len(report["stages"])).zfill(2)+"_"+name+".log")}')
    try:
        stage('preprocessing','code/pipeline/prepare_classical.py','--input',a.raw,'--splits',a.splits,'--output',out/'features','--n-genes',genes)
        stage('scgpt_check','scripts/load_scgpt.py','--config',out/'config.json','--output',out/'scgpt_check')
        stage('dictionary','scripts/run_dictionary.py','--features',out/'features','--output',out/'dictionary/candidate',
              '--atoms',atoms,'--epochs',epochs,'--alpha',a.alpha)
        evaluations=[]
        for method in ['baseline','dictionary','scgpt']:
            candidate=['--candidate','Dictionary'] if method=='dictionary' else []
            stage('train_'+method,'scripts/train_logistic.py','--config',out/'config.json','--method',method,
                  '--C',a.C,'--threads',a.threads,'--output',out/'models'/method,*candidate)
            dest=out/'validation'/method
            stage('validate_'+method,'scripts/evaluate_validation.py','--model',out/'models'/method,'--output',dest)
            evaluations.append(dest)
        stage('summary','scripts/summarize_results.py','--evaluations',*evaluations,'--output',out/'summary',*(['--plots'] if a.plots else []))
        report['status']='complete'
        report['summary']=str(out/'summary/summary.csv')
    except Exception as exc:
        report.update(status='failed',error=str(exc)); raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat(); save()
    print(f'End-to-end validation completed. Summary: {out / "summary/summary.csv"}')


if __name__ == '__main__': main()
