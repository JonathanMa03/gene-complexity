#!/usr/bin/env python
"""Append cells to frozen 222k prefixes, then save validated raw-count H5ADs."""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import cellxgene_census
ROOT=Path('/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling')
MAN=ROOT/'manifests/cell_extension_888k'
SIZES=(444000,888000)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def write_json(path,value):
 temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(temp,path)

def manifests():
 MAN.mkdir(exist_ok=True)
 spec=importlib.util.spec_from_file_location('original_sampler',ROOT/'scripts/01_build_sampling_manifests.py')
 sampler=importlib.util.module_from_spec(spec); spec.loader.exec_module(sampler)
 eligible,source=sampler.eligible_sctab_obs()
 if len(eligible)!=sampler.EXPECTED_ELIGIBLE_CELLS: raise RuntimeError('Eligible universe changed')
 ids=eligible.soma_joinid.to_numpy(dtype=np.int64)
 for seed in range(3):
  oldpath=ROOT/f'manifests/sctab_seed{seed}_n222000_cells.parquet'
  old=pd.read_parquet(oldpath).sort_values('sampling_rank')
  assert len(old)==222000 and old.soma_joinid.is_unique
  assert np.isin(old.soma_joinid,ids).all()
  remaining=np.flatnonzero(~np.isin(ids,old.soma_joinid.to_numpy()))
  extension_seed=202609260+seed
  positions=np.random.default_rng(extension_seed).choice(remaining,size=666000,replace=False)
  extra=eligible.iloc[positions].copy()
  extra.insert(0,'sampling_rank',np.arange(222000,888000,dtype=np.int64)); extra.insert(0,'seed',seed)
  combined=pd.concat([old,extra],ignore_index=True)
  assert combined.soma_joinid.is_unique and np.array_equal(combined.soma_joinid[:222000],old.soma_joinid)
  path=MAN/f'sctab_seed{seed}_n888000_cells.parquet'
  if path.exists():
   prior=pd.read_parquet(path)
   if not np.array_equal(prior.soma_joinid,combined.soma_joinid): raise RuntimeError('Existing manifest differs')
  else:
   tmp=path.with_suffix('.parquet.tmp'); combined.to_parquet(tmp,index=False); os.replace(tmp,path)
  write_json(MAN/f'sampling_seed{seed}.json',dict(source=source,sampling_seed=seed,extension_rng_seed=extension_seed,original_manifest_sha256=sha(oldpath),manifest_sha256=sha(path),sizes=list(SIZES),original_prefix_cells=222000,model_seed=20260821))
  print('MANIFEST',seed,len(combined),flush=True)
 (MAN/'_MANIFESTS_SUCCESS').write_text('complete\n')

def matrices(seed):
 if not (MAN/'_MANIFESTS_SUCCESS').exists(): raise RuntimeError('Manifests incomplete')
 manifest= pd.read_parquet(MAN/f'sctab_seed{seed}_n888000_cells.parquet').sort_values('sampling_rank')
 seed_dir=ROOT/f'data/nested_raw/seed_{seed}'
 oldpath=seed_dir/f'sctab_seed{seed}_n222000_raw.h5ad'; old_hash=sha(oldpath)
 old=ad.read_h5ad(oldpath)
 assert old.shape==(222000,19331) and np.array_equal(old.obs.soma_joinid,manifest.soma_joinid[:222000])
 chunk_dir=ROOT/f'data/cell_extension_888k/seed_{seed}'; chunk_dir.mkdir(parents=True,exist_ok=True)
 chunks=[]
 # Bound remote query sizes; persisted chunks allow safe restart after network failures.
 with cellxgene_census.open_soma(census_version='2023-05-15') as census:
  for start in range(222000,888000,50000):
   end=min(start+50000,888000); selected=manifest.iloc[start:end]
   file=chunk_dir/f'raw_{start}_{end}.h5ad'
   if file.exists(): chunk=ad.read_h5ad(file)
   else:
    chunk=cellxgene_census.get_anndata(census=census,organism='Homo sapiens',X_name='raw',obs_coords=selected.soma_joinid.astype('int64').tolist(),var_value_filter=f'feature_id in {old.var_names.tolist()!r}',column_names={'obs':['soma_joinid','is_primary_data','dataset_id','donor_id','assay','cell_type','development_stage','disease','tissue','tissue_general','sex','self_reported_ethnicity','suspension_type'],'var':['feature_id','feature_name','feature_length']})
    chunk.obs_names=chunk.obs.soma_joinid.map(lambda x:f'sctab_{int(x)}')
    chunk.var_names=chunk.var.feature_id.astype(str)
    # Series assignment carries its name onto the index; keep axis names distinct from metadata.
    chunk.obs_names.name=None
    chunk.var_names.name=None
    names=pd.Index([f'sctab_{int(i)}' for i in selected.soma_joinid])
    chunk=chunk[names,old.var_names].copy()
    chunk.obs['sampling_rank']=np.arange(start,end,dtype=np.int64)
    chunk.X=sparse.csr_matrix(chunk.X,dtype=np.float32)
    tmp=file.with_suffix('.h5ad.tmp'); chunk.write_h5ad(tmp,compression='gzip'); os.replace(tmp,file)
   if not np.array_equal(chunk.obs.soma_joinid,selected.soma_joinid) or not chunk.var_names.equals(old.var_names): raise RuntimeError('Chunk IDs or genes mismatch')
   if not sparse.isspmatrix_csr(chunk.X) or not np.isfinite(chunk.X.data).all() or np.any(chunk.X.data<0) or np.any(chunk.X.data!=np.floor(chunk.X.data)): raise RuntimeError('Invalid raw counts')
   chunks.append(chunk); print('CHUNK',seed,start,end,flush=True)
 full=ad.AnnData(X=sparse.vstack([old.X]+[c.X for c in chunks],format='csr',dtype=np.float32),obs=pd.concat([old.obs]+[c.obs for c in chunks]),var=old.var.copy(),uns=dict(old.uns))
 del chunks
 full.obs_names.name=None
 full.var_names.name=None
 assert full.obs_names.is_unique and full.shape==(888000,19331)
 assert np.array_equal(full.obs.soma_joinid,manifest.soma_joinid)
 assert (full.X[:222000]!=old.X).nnz==0
 full.uns['nested_sizes']=[2220,4440,8880,17760,35520,71040,142080,222000,444000,888000]
 full.uns['extension_manifest_sha256']=sha(MAN/f'sctab_seed{seed}_n888000_cells.parquet')
 inventory=[]
 for size in SIZES:
  path=seed_dir/f'sctab_seed{seed}_n{size}_raw.h5ad'
  if path.exists():
   check=ad.read_h5ad(path)
   if check.shape!=(size,19331) or not check.obs_names.equals(full.obs_names[:size]) or not check.var_names.equals(full.var_names) or (check.X!=full.X[:size]).nnz: raise RuntimeError('Existing expanded matrix differs')
   del check
  else:
   subset=full[:size].copy(); subset.uns['n_pretraining_cells']=size
   tmp=path.with_suffix('.h5ad.tmp'); subset.write_h5ad(tmp,compression='gzip'); os.replace(tmp,path); del subset
  # Verify the persisted file, including original counts, in bounded blocks.
  check=ad.read_h5ad(path,backed='r')
  assert check.shape==(size,19331) and check.obs_names.equals(full.obs_names[:size]) and check.var_names.equals(full.var_names)
  for start in range(0,size,50000):
   end=min(start+50000,size)
   if (check.X[start:end]!=full.X[start:end]).nnz: raise RuntimeError('Saved counts mismatch')
  check.file.close()
  inventory.append(dict(seed=seed,n_cells=size,n_genes=19331,path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
  print('VALIDATED',path,flush=True)
 if sha(oldpath)!=old_hash: raise RuntimeError('Original matrix changed')
 write_json(MAN/f'matrix_inventory_seed{seed}.json',inventory)
 pd.DataFrame(inventory).to_csv(MAN/f'matrix_inventory_seed{seed}.tsv',sep='\t',index=False)
 write_json(MAN/f'validation_seed{seed}.json',dict(original_sha256=old_hash,original_prefix_equal=True,nesting_equal=True,unique_ids=True,gene_order_equal=True,raw_counts_verified=True,model_seed=20260821))
 (MAN/f'_SUCCESS_seed{seed}').write_text('validated\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['manifests','matrices']); p.add_argument('--seed',type=int,choices=range(3)); a=p.parse_args()
 if a.stage=='manifests': manifests()
 else:
  if a.seed is None: p.error('--seed required')
  matrices(a.seed)
