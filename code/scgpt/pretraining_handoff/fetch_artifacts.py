"""Retrieve a selected checkpoint or complete archives from the lab handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', help='model_id in checkpoint_manifest.json')
    p.add_argument('--archive', choices=['weights','sampling','weights_seed0','weights_seed1','weights_seed2'])
    p.add_argument('--source-root', type=Path)
    p.add_argument('--ssh', help='Your authorized USER@FASRC_LOGIN; no passwords/keys in Git')
    p.add_argument('--destination', type=Path, default=HERE/'artifacts')
    p.add_argument('--verify-file', type=Path, help='Verify a browser-downloaded archive by SHA256')
    a = p.parse_args()
    if a.verify_file:
        manifest = json.loads((HERE/'artifact_manifest.json').read_text())
        digest = sha(a.verify_file)
        matches = [r for r in manifest['archives'] if r['sha256'] == digest]
        if len(matches) != 1: raise ValueError('Archive SHA256 does not match the manifest')
        print('Verified:', matches[0]['name'], digest)
        return
    if bool(a.checkpoint) == bool(a.archive):
        p.error('Choose exactly one of --checkpoint or --archive')
    manifest = json.loads((HERE/'artifact_manifest.json').read_text())
    source = a.source_root or Path(manifest['shared_root'])
    if a.checkpoint:
        rows = json.loads((HERE/'checkpoint_manifest.json').read_text())['checkpoints']
        matches = [r for r in rows if r['model_id']==a.checkpoint]
        if len(matches)!=1: p.error('Unknown checkpoint ID')
        r = matches[0]
        files = [(Path(r['archive_relative_path']),r['checkpoint_sha256'])]
        # Small configs/vocab already travel in Git, with independent hash checks.
        target = a.destination/'weights'/r['model_id']
        target.mkdir(parents=True,exist_ok=True)
        for src,dst in [(HERE/r['config_path'],target/'config.json'),
                        (HERE/r['vocab_path'],target/'vocab.json')]:
            if dst.exists() and sha(dst)!=sha(src): raise FileExistsError(str(dst))
            if not dst.exists(): shutil.copyfile(src,dst)
    else:
        r = next(r for r in manifest['archives'] if r['name']=='scgpt_'+a.archive+'.tar.gz')
        files = [(Path(r['name']),r['sha256'])]
    for rel,expected in files:
        dst = a.destination/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha(dst)!=expected: raise FileExistsError(str(dst))
            print('Already verified:', dst)
            continue
        tmp = dst.with_name(dst.name+'.partial')
        if tmp.exists(): raise FileExistsError(str(tmp))
        if a.ssh:
            subprocess.run(['scp',a.ssh+':'+str(source/rel),str(tmp)],check=True)
        else:
            shutil.copyfile(source/rel,tmp)
        if sha(tmp)!=expected:
            raise ValueError('Downloaded SHA256 mismatch; partial file retained for diagnosis')
        tmp.rename(dst)
        print('Verified:',dst)


if __name__=='__main__':
    main()
