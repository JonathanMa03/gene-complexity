"""Run the existing classifier search and analysis with verified inputs."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/logit'))
from validation_inputs import default_config, load_inputs, digest
from threadpoolctl import threadpool_limits, threadpool_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON input config; paths relative to repository root')
    parser.add_argument('--output', type=Path, help='New output directory; existing directories are rejected')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    config = json.loads(args.config.read_text()) if args.config else default_config()
    output = args.output or ROOT / 'results/logit' / datetime.now(timezone.utc).strftime('validation_%Y%m%dT%H%M%S_%fZ')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  config=config, threads=args.threads, python=platform.python_version(),
                  platform=platform.platform(),
                  versions={p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'threadpoolctl']},
                  test_used_for_fit_or_score=False, test_npz_opened=False,
                  full_scgpt_bundle_loaded=True)
    report['source_hashes'] = {str(p.relative_to(ROOT)): digest(p) for p in [
        Path(__file__), ROOT/'code/logit/validation_inputs.py',
        ROOT/'code/logit/run_all_validation_coarse_to_fine.py', ROOT/'code/logit/analyze_validation.py']}
    report['git_commit'] = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report['git_status'] = subprocess.run(['git','status','--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    def save():
        (output/'run_report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    print(f'Validation run: {output}\nProgress log: {output / "run.log"}', flush=True)
    try:
        with (output/'run.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=args.threads):
            import matplotlib
            matplotlib.use('Agg')
            import run_all_validation_coarse_to_fine as search
            import analyze_validation as analysis
            report['threadpools'] = threadpool_info()
            report['classifier'] = dict(solver='lbfgs', max_iter=5000, random_state=42,
                                        scaling='none', selection='validation Macro-F1; smaller C breaks ties',
                                        coarse_C=search.COARSE_C_VALUES.tolist(), fine_points=search.FINE_POINTS)
            bundle = load_inputs(config)
            report.update({k: bundle[k] for k in ['input_hashes', 'split_counts', 'scgpt_model_id']})
            save()
            search.main(bundle=bundle, output=output/'classification')
            analysis.main(bundle=bundle, results_dir=output/'classification')
        report['status'] = 'complete'
        report['output_hashes'] = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file() and p.name != 'run_report.json'}
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(f'Completed validation search and analysis: {output}')


if __name__ == '__main__':
    main()
