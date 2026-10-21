"""Checked, resumable notebook orchestration over the reusable workflow entry points."""
from pathlib import Path
import json
import os

from contextlib import redirect_stdout, redirect_stderr
import traceback
from gene_complexity._paths import ROOT, PACKAGE_ROOT
from gene_complexity._common import digest, environment
from gene_complexity.preflight import preflight, HANDOFF
from gene_complexity import api
from gene_complexity.preprocessing.classical import prepare
from gene_complexity.dictionary_learning.training import run as fit_dictionary


def serializable(value):
    if isinstance(value, Path): return str(value.resolve())
    if isinstance(value, dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serializable(v) for v in value]
    return value


def path_arguments(value):
    if isinstance(value, dict):
        for v in value.values(): yield from path_arguments(v)
    elif isinstance(value, (list,tuple)):
        for v in value: yield from path_arguments(v)
    elif isinstance(value, Path): yield value



class Workflow:
    def __init__(self, *, output, raw, splits, scgpt, n_genes=256, atoms=8,
                 epochs=2, alpha=1.0, C=1.0, threads=1,
                 methods=('baseline', 'dictionary', 'scgpt'), plots=True):
        import math
        if not methods or len(set(methods)) != len(methods) or set(methods) - {'baseline','dictionary','scgpt'}:
            raise ValueError('Select unique methods from baseline, dictionary, scgpt')
        if any(type(v) is not int or v < 1 for v in [n_genes,atoms,epochs,threads]) or atoms > n_genes:
            raise ValueError('Positive integer sizes required, with atoms <= n_genes')
        if not all(math.isfinite(v) and v > 0 for v in [alpha,C]):
            raise ValueError('alpha and C must be finite and positive')
        self.out = Path(output).resolve()
        self.raw, self.splits, self.scgpt = map(lambda p: Path(p).resolve(), (raw,splits,scgpt))
        self.methods = list(methods)
        self.settings = dict(n_genes=n_genes,atoms=atoms,epochs=epochs,alpha=alpha,
                             C=C,threads=threads,methods=self.methods,plots=plots)
        # Fail before writing anything if inputs are missing or their provenance disagrees.
        hashes = preflight(self.raw,self.splits,self.scgpt)
        sources = {str(p.resolve()):digest(p) for p in PACKAGE_ROOT.rglob('*.py')}
        identity = dict(settings=self.settings,input_hashes=hashes,source_hashes=sources,
                        environment=environment())
        self.state_path = self.out/'notebook_run.json'
        if self.out.exists():
            if not self.state_path.is_file():
                raise FileExistsError(f'{self.out} is not a notebook run. Choose a new RUN_NAME.')
            self.state=json.loads(self.state_path.read_text())
            if self.state['identity'] != identity:
                raise ValueError('Settings, inputs, source, or environment changed. Choose a new RUN_NAME; existing runs are preserved.')
        else:
            self.out.mkdir(parents=True)
            self.state=dict(identity=identity,stages={})
            self._save()
        config=dict(expression_dir=str(self.out/'features'),split_dir=str(self.splits),
                    dictionary_dir=str(self.out/'dictionary'),dictionary_runs={'Dictionary':'candidate'},
                    scgpt_path=str(self.scgpt/HANDOFF[0]))
        self.config = self.out/'config.json'
        text=json.dumps(config,indent=2)+'\n'
        if self.config.exists() and self.config.read_text()!=text:
            raise ValueError(f'Changed notebook config: {self.config}')
        self.config.write_text(text)

    def _save(self):
        tmp=self.state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state,indent=2)+'\n');tmp.replace(self.state_path)

    def check(self):
        current=preflight(self.raw,self.splits,self.scgpt)
        if current != self.state['identity']['input_hashes']:
            raise ValueError('Inputs changed since notebook setup; choose a new run.')
        if environment()!=self.state['identity']['environment']:
            raise ValueError('Environment changed since notebook setup; restart with a new run.')
        for name,value in self.state['identity']['source_hashes'].items():
            if digest(ROOT/name)!=value: raise ValueError(f'Source changed since setup: {name}')
        if not self.config.is_file(): raise FileNotFoundError(self.config)
        expected=dict(expression_dir=str(self.out/'features'),split_dir=str(self.splits),
                      dictionary_dir=str(self.out/'dictionary'),dictionary_runs={'Dictionary':'candidate'},
                      scgpt_path=str(self.scgpt/HANDOFF[0]))
        if json.loads(self.config.read_text())!=expected: raise ValueError('Notebook config was modified')

    def _verify_stage(self, name):
        item=self.state['stages'].get(name)
        if not item or item['status']!='complete':
            raise RuntimeError(f'Run prerequisite stage {name!r} first; incomplete stages cannot be resumed in place.')
        for dep in item.get('requires', []):
            self._verify_stage(dep)
        for path,value in item.get('input_hashes', {}).items():
            if not Path(path).is_file() or digest(path)!=value:
                raise ValueError(f'Missing/changed input to {name}: {path}')
        for path,value in item['output_hashes'].items():
            if not Path(path).is_file() or digest(path)!=value:
                raise ValueError(f'Missing/changed output from {name}: {path}')

    def run_step(self, name, function, kwargs, destination, requires=()):
        """Call a Python function once, or verify/reuse its recorded outputs."""
        self.check()
        for dep in requires: self._verify_stage(dep)
        destination=Path(destination).resolve()
        call=dict(function=function.__module__+'.'+function.__qualname__,kwargs=serializable(kwargs))
        if name in self.state['stages']:
            item=self.state['stages'][name]
            if item['call']!=call: raise ValueError(f'Arguments changed for {name}; choose a new run.')
            self._verify_stage(name)
            print(f'{name}: verified existing outputs')
            return destination
        if destination.exists(): raise FileExistsError(f'Unregistered output already exists: {destination}')
        log=self.out/f'{name}.log'
        inputs={}
        for path in path_arguments(kwargs):
            if path.resolve()==destination or not path.exists(): continue
            files=[path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
            inputs.update({str(p.resolve()):digest(p) for p in files})
        item=dict(call=call,status='running',log=str(log),requires=list(requires),input_hashes=inputs)
        self.state['stages'][name]=item;self._save()
        print(f'Running {name} …',flush=True)
        try:
            with log.open('w') as f, redirect_stdout(f), redirect_stderr(f):
                try: function(**kwargs)
                except BaseException:
                    traceback.print_exc();raise
            files=sorted(p for p in destination.rglob('*') if p.is_file())
            if not files: raise RuntimeError(f'{name} created no artifacts at {destination}')
            item.update(status='complete',output_hashes={str(p):digest(p) for p in files})
        except BaseException as exc:
            item.update(status='failed',error=str(exc))
            print(f'{name} failed; see {log}')
            raise
        finally: self._save()
        print(f'{name}: complete → {destination}',flush=True)
        return destination

    def preprocess(self):
        dest=self.out/'features'
        return self.run_step('preprocess',prepare,dict(input_path=self.raw,bundle=self.splits,
            output=dest,n_genes=self.settings['n_genes']),dest)

    def check_scgpt(self):
        dest=self.out/'scgpt_check'
        return self.run_step('scgpt_check',api.check_scgpt,dict(config=self.config,output=dest),dest,['preprocess'])

    def fit_dictionary(self):
        dest=self.out/'dictionary/candidate';s=self.settings
        return self.run_step('dictionary',fit_dictionary,dict(features=self.out/'features',output=dest,
            atoms=s['atoms'],epochs=s['epochs'],alpha=s['alpha']),dest,['preprocess'])

    def train_classifier(self, method):
        if method not in self.methods: raise ValueError(f'{method} is not enabled in METHODS')
        dest=self.out/'models'/method
        deps=['preprocess']+(['dictionary'] if method=='dictionary' else ['scgpt_check'] if method=='scgpt' else [])
        return self.run_step('train_'+method,api.train_classifier,dict(config=self.config,method=method,
            candidate='Dictionary' if method=='dictionary' else None,C=self.settings['C'],
            threads=self.settings['threads'],output=dest),dest,deps)

    def evaluate_validation(self, method):
        if method not in self.methods: raise ValueError(f'{method} is not enabled in METHODS')
        dest=self.out/'validation'/method
        return self.run_step('validate_'+method,api.evaluate_validation,
            dict(model=self.out/'models'/method,output=dest),dest,['train_'+method])

    def generate_figures(self):
        dest=self.out/'summary'
        return self.run_step('summary',api.summarize,dict(evaluations=[self.out/'validation'/m for m in self.methods],
            output=dest,plots=self.settings['plots']),dest,['validate_'+m for m in self.methods])

    def interpret_dictionary(self):
        dest=self.out/'dictionary_review'
        return self.run_step('dictionary_review',api.review_dictionary,dict(config=self.config,
            candidate='Dictionary',output=dest,plots=self.settings['plots']),dest,['dictionary'])

    def run_all(self):
        self.preprocess()
        if 'scgpt' in self.methods: self.check_scgpt()
        if 'dictionary' in self.methods: self.fit_dictionary()
        for method in self.methods:
            self.train_classifier(method);self.evaluate_validation(method)
        self.generate_figures()
        if 'dictionary' in self.methods: self.interpret_dictionary()
        print('Validation workflow complete. No model selection or test evaluation was performed.')
