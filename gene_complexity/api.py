"""Function API for notebook stages. No argument parsing or Python subprocesses."""
from pathlib import Path
from types import SimpleNamespace
from gene_complexity._paths import ROOT, CONFIG, PACKAGE_ROOT
from gene_complexity._common import read, write, digest


def train_classifier(*, config=CONFIG, method, candidate=None, C=1.0, max_iter=5000,
                     tol=1e-4, seed=42, threads=1, output=None):
    from gene_complexity.classification.models import train
    return train(SimpleNamespace(config=Path(config),method=method,candidate=candidate,C=C,
                                 max_iter=max_iter,tol=tol,seed=seed,threads=threads,output=output))


def evaluate_validation(model, output=None, freeze=False):
    from gene_complexity.evaluation.predictions import evaluate
    model=Path(model);out=evaluate(model,output)
    if freeze:
        r=read(model/'training.json')
        write(out/'selection.json',dict(status='frozen',model_sha256=r['model_sha256'],
              training_sha256=digest(model/'training.json'),validation_metrics=str((out/'metrics.json').resolve()),
              validation_sha256=digest(out/'metrics.json')))
    return out


def evaluate_test(model, *, selection, output=None):
    from gene_complexity.evaluation.predictions import evaluate
    return evaluate(model,output,test=True,selection=selection)


def check_scgpt(*, config=CONFIG, source_dir=None, checkpoint=None, output=None):
    from gene_complexity.scgpt.handoff import run
    return run(SimpleNamespace(config=Path(config),source_dir=Path(source_dir) if source_dir else None,
                               checkpoint=Path(checkpoint) if checkpoint else None,output=output))


def summarize(evaluations, *, output=None, plots=False):
    from gene_complexity.reporting.summary import run
    return run(SimpleNamespace(evaluations=[Path(p) for p in evaluations],output=output,plots=plots))


def review_dictionary(*, config=CONFIG, candidate, top_genes=10, display_atoms=8, plots=False, output=None):
    from gene_complexity.dictionary_learning.interpretation import run
    return run(SimpleNamespace(config=Path(config),candidate=candidate,top_genes=top_genes,
                               display_atoms=display_atoms,plots=plots,output=output))


def tune_models(*, inputs=CONFIG, grid=PACKAGE_ROOT/'dictionary_learning/configs/dictionary_grid_v2.json',
                reuse_grid=None, output=None):
    from gene_complexity.classification.tuning import run
    return run(SimpleNamespace(inputs=Path(inputs),grid=Path(grid),reuse_grid=Path(reuse_grid) if reuse_grid else None,
                               output=Path(output) if output else None))


def donor_study(*, raw=ROOT/'data/processed/baron_raw.h5ad', splits=ROOT/'external/baron_splits_final',
                scgpt_dir=ROOT/'external/scgpt/pretraining_handoff/outputs/baron',
                panel=ROOT/'external/scgpt/pretraining_handoff/assets/sctab_hvg_2000.tsv',
                gene_policy='scgpt-panel', n_genes=2000, seeds=(0,1,2), atoms=64, alpha=2.0,
                epochs=10, initialization='train-cells', batch_seed=42,
                regularization=(1e-5,1e-4,1e-3,1e-2,.1), output=None):
    from gene_complexity.evaluation.donor import run
    return run(SimpleNamespace(raw=Path(raw),splits=Path(splits),scgpt_dir=Path(scgpt_dir),panel=Path(panel),
        gene_policy=gene_policy,n_genes=n_genes,seeds=list(seeds),atoms=atoms,alpha=alpha,epochs=epochs,
        initialization=initialization,batch_seed=batch_seed,regularization=list(regularization),output=output))


def summarize_donors(study, *, output=None, plots=False):
    from gene_complexity.reporting.donors import run
    return run(SimpleNamespace(study=Path(study),output=output,plots=plots))


def build_report_assets():
    from gene_complexity.reporting.paper import main
    return main()
