"""Compare downstream Census IDs and dataset/donor metadata with pretraining lists."""
import argparse
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--sampling-dir',type=Path,required=True)
    p.add_argument('--n-cells',type=int,required=True)
    p.add_argument('--sampling-seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--census-version',choices=['2023-05-15'],required=True)
    p.add_argument('--id-column',default='soma_joinid')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 0<a.n_cells<=1600000: p.error('n-cells outside provided maximum manifest')
    if a.output.exists(): raise FileExistsError(str(a.output))
    data=ad.read_h5ad(a.input,backed='r')
    try: obs=data.obs.copy()
    finally: data.file.close()
    if a.id_column not in obs:
        raise ValueError('Missing release-specific Census ID column; absence of IDs cannot establish disjointness')
    ids=pd.to_numeric(obs[a.id_column],errors='raise').to_numpy()
    if not np.isfinite(ids).all() or not np.equal(ids,np.rint(ids)).all():
        raise ValueError('Census IDs must be nonmissing integers')
    path=a.sampling_dir/f'sctab_seed{a.sampling_seed}_n1600000_cells.parquet'
    provenance=json.loads((Path(__file__).resolve().parent/'provenance/sampling_artifacts.json').read_text())
    entry=next(r for r in provenance['records'] if r['sampling_seed']==a.sampling_seed)
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): digest.update(chunk)
    if digest.hexdigest()!=entry['sha256']: raise ValueError('Sampling manifest SHA256 mismatch')
    train=pd.read_parquet(path)
    if len(train)!=1600000: raise ValueError('Unexpected maximum-manifest row count')
    if 'sampling_rank' not in train: raise ValueError('Missing sample ordering')
    train=train.sort_values('sampling_rank',kind='stable').iloc[:a.n_cells]
    hit=np.isin(ids.astype(np.int64),train.soma_joinid.to_numpy(dtype=np.int64))
    result=dict(census_version=a.census_version,sampling_seed=a.sampling_seed,
        n_pretraining_candidates=len(train),test_cells=len(obs),overlapping_cell_ids=int(hit.sum()),
        overlapping_test_cell_names=obs.index[hit].astype(str).tolist(),
        limitations=['ID comparison requires that target IDs actually belong to the declared release; '
                     'the CLI cannot independently prove that provenance.',
                     'Zero ID overlap does not exclude duplicate expression profiles or study/donor overlap.',
                     'Depth conditions can remove zero-panel cells; this audits the conservative candidate prefix.'])
    for col in ['dataset_id','donor_id']:
        if col in obs and col in train:
            result['overlapping_'+col]=sorted(set(obs[col].dropna().astype(str)) &
                                            set(train[col].dropna().astype(str)))
        else: result['overlapping_'+col]=None
    if all(c in obs and c in train for c in ['dataset_id','donor_id']):
        target=set(zip(obs.dataset_id.astype(str),obs.donor_id.astype(str)))
        source=set(zip(train.dataset_id.astype(str),train.donor_id.astype(str)))
        result['overlapping_dataset_donor_pairs']=[list(v) for v in sorted(target & source)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Wrote',a.output,'overlapping cell IDs:',int(hit.sum()))


if __name__=='__main__': main()
