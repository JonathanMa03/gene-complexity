"""Validate/import supplied full embeddings and provenance; optionally verify checkpoint weights."""
import argparse
from pathlib import Path
import shutil
from _common import ROOT, config_arg, read, digest, fresh, write
from validation_inputs import load_inputs

def main():
    p=argparse.ArgumentParser(description=__doc__); config_arg(p)
    p.add_argument('--source-dir',type=Path,help='Directory containing the four teammate-supplied files; copy into config destination')
    p.add_argument('--checkpoint',type=Path,help='Optional extracted checkpoint file to verify against metadata hash')
    p.add_argument('--output',type=Path,help='New check-report directory (default timestamped)')
    a=p.parse_args(); config=read(a.config); destination=ROOT/config['scgpt_path']
    source=a.source_dir if a.source_dir else destination.parent
    names=[destination.name,destination.with_suffix('.metadata.json').name,'checkpoint_selection.json','preparation_report.json']
    for name in names:
        if not (source/name).is_file(): raise FileNotFoundError(f'Missing scGPT handoff: {source/name}. Supply all four full-extraction files.')
    meta=read(source/names[1]); selection=read(source/names[2]); prep=read(source/names[3])
    if selection['checkpoint_id']!=meta['model_id'] or selection['checkpoint_sha256']!=meta['checkpoint_sha256']:
        raise ValueError('Checkpoint selection and embedding metadata disagree')
    if prep['status']!='PASS' or prep['output_sha256']!=meta['input_sha256']:
        raise ValueError('Preparation report does not match extracted input')
    settings=read(ROOT/config['split_dir']/'split_settings.json')
    if settings['input_sha256'] not in prep['input_hashes'].values(): raise ValueError('scGPT preparation does not reference shared raw data')
    if a.checkpoint and digest(a.checkpoint)!=meta['checkpoint_sha256']: raise ValueError('Checkpoint weights hash mismatch')
    checked=dict(config,scgpt_path=str((source/destination.name).resolve()))
    b=load_inputs(checked,methods=('scgpt',))
    if a.source_dir and source.resolve()!=destination.parent.resolve():
        if any((destination.parent/n).exists() for n in names): raise FileExistsError('Destination handoff files already exist; preserved')
        destination.parent.mkdir(parents=True,exist_ok=True)
        for name in names: shutil.copy2(source/name,destination.parent/name)
    out=fresh(a.output,'scgpt_check')
    write(out/'scgpt_check.json',dict(status='PASS',model_id=meta['model_id'],shape=meta['shape'],
        handoff_hashes={n:digest(source/n) for n in names},input_hashes=b['input_hashes'],
        checkpoint_verified=bool(a.checkpoint)))
    print(f'scGPT handoff verified: {out}')

if __name__ == '__main__': main()
