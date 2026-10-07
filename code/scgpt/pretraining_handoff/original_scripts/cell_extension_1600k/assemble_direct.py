from pathlib import Path
import argparse,hashlib,json,os
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse
R=Path('/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling');M=R/'manifests/cell_extension_1600k';EX=R/'data/cell_extension_1600k/direct_extracted'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(seed):
 manifest=pd.read_parquet(M/f'sctab_seed{seed}_n1600000_cells.parquet').sort_values('sampling_rank')
 oldpath=R/f'data/nested_raw/seed_{seed}/sctab_seed{seed}_n888000_raw.h5ad';digest=sha(oldpath);old=ad.read_h5ad(oldpath)
 assert len(manifest)==1600000 and manifest.soma_joinid.is_unique
 assert np.array_equal(old.obs.soma_joinid,manifest.soma_joinid[:888000])
 blocks=[];ids=[]
 for dataset,rows in manifest.iloc[888000:].groupby('dataset_id',observed=True):
  path=EX/f'{dataset}.h5ad'; report=json.loads((EX/f'{dataset}.json').read_text())
  assert report['exact_reference_counts'] and report['metadata_order_verified']
  a=ad.read_h5ad(path);assert a.var_names.equals(old.var_names)
  select=pd.Index(a.obs.soma_joinid).get_indexer(rows.soma_joinid)
  assert (select>=0).all()
  blocks.append(a.X[select]);ids.extend(rows.soma_joinid.tolist())
 extras=sparse.vstack(blocks,format='csr');del blocks
 order=pd.Index(ids).get_indexer(manifest.soma_joinid.iloc[888000:]);assert (order>=0).all()
 full=ad.AnnData(X=sparse.vstack([old.X,extras[order]],format='csr'),obs=pd.concat([old.obs,manifest.iloc[888000:].set_index(pd.Index([f'sctab_{i}' for i in manifest.soma_joinid.iloc[888000:]]))]),var=old.var.copy(),uns=dict(old.uns))
 del extras
 full.uns['extension_manifest_sha256']=sha(M/f'sctab_seed{seed}_n1600000_cells.parquet')
 full.obs_names.name=None;full.var_names.name=None;full.uns['n_pretraining_cells']=1600000;full.uns['nested_sizes']=[2220,4440,8880,17760,35520,71040,142080,222000,444000,888000,1600000];full.uns['retrieval']='direct versioned source H5AD files; verified Census row order and reference raw counts'
 assert full.shape==(1600000,19331) and full.obs_names.is_unique and np.array_equal(full.obs.soma_joinid,manifest.soma_joinid)
 output=Path('/n/netscratch/rongma_lab/Lab/jingyuan/pretraining_cell_scaling/nested_raw')/f'seed_{seed}'/f'sctab_seed{seed}_n1600000_raw.h5ad'
 output.parent.mkdir(parents=True,exist_ok=True)
 if not output.exists():
  tmp=output.with_suffix('.h5ad.tmp');full.write_h5ad(tmp,compression='gzip');os.replace(tmp,output)
 check=ad.read_h5ad(output,backed='r')
 assert check.shape==full.shape and check.obs_names.equals(full.obs_names) and check.var_names.equals(full.var_names)
 for start in range(0,1600000,50000):assert (check.X[start:start+50000]!=full.X[start:start+50000]).nnz==0
 check.file.close();assert sha(oldpath)==digest
 record=dict(seed=seed,n_cells=1600000,n_genes=19331,path=str(output),bytes=output.stat().st_size,sha256=sha(output))
 (M/f'matrix_inventory_seed{seed}.json').write_text(json.dumps([record],indent=2)+'\n');pd.DataFrame([record]).to_csv(M/f'matrix_inventory_seed{seed}.tsv',sep='\t',index=False)
 (M/f'validation_seed{seed}.json').write_text(json.dumps(dict(original_sha256=digest,original_prefix_equal=True,nesting_equal=True,unique_ids=True,gene_order_equal=True,raw_counts_verified=True,source='direct H5AD'),indent=2)+'\n')
 link=oldpath.parent/output.name
 if link.is_symlink():
  assert link.resolve()==output.resolve()
 elif link.exists():
  raise RuntimeError('Refusing to replace an existing matrix')
 else: link.symlink_to(output)
 (M/f'_SUCCESS_seed{seed}').write_text('validated direct source assembly\n');print(json.dumps(record),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=range(3));main(p.parse_args().seed)
