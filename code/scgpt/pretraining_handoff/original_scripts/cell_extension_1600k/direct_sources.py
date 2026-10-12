#!/usr/bin/env python
"""Download immutable source H5ADs; recover original Census row IDs with audited filters."""
from pathlib import Path
import argparse,ast,json,os,subprocess,time
import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import sparse
from anndata._io.specs import read_elem
from anndata._core.sparse_dataset import sparse_dataset
from resumable_http import download
import cellxgene_census
R=Path('/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling')
M=R/'manifests/cell_extension_1600k/direct_source'; CACHE=Path('/n/netscratch/rongma_lab/Lab/jingyuan/pretraining_cell_scaling/source_h5ad_2023-05-15'); EX=Path('/n/netscratch/rongma_lab/Lab/jingyuan/pretraining_cell_scaling/direct_extracted')
for d in [CACHE,EX]:d.mkdir(exist_ok=True)

def log(**kw):print(json.dumps(kw),flush=True)
def atomic_json(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def get_assays():
 tree=ast.parse((M/'builder_globals.py').read_text())
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RNA_SEQ' for t in n.targets))

def main(worker,workers,limit):
 log(status='loading_inventory')
 inventory=pd.read_csv(M/'download_inventory.tsv',sep='\t').sort_values('bytes').reset_index(drop=True)
 all_obs=pd.concat([pd.read_parquet(R/f'manifests/cell_extension_1600k/sctab_seed{s}_n1600000_cells.parquet',columns=['soma_joinid','dataset_id']).iloc[888000:] for s in range(3)]).drop_duplicates('soma_joinid')
 log(status='sample_ids_loaded',cells=len(all_obs))
 reference_path=R/('data/cell_extension_1600k/seed_0/raw_888000_938000.h5ad' if limit else 'data/nested_raw/seed_0/sctab_seed0_n888000_raw.h5ad')
 ref=ad.read_h5ad(reference_path,backed='r')
 log(status='reference_loaded')
 target_genes=ref.var_names; assays=get_assays(); census=None
 try:
  rows=inventory.iloc[worker::workers]
  if limit: rows=rows.iloc[:limit]
  for row in rows.itertuples():
   dataset=row.dataset_id; start=time.monotonic(); output=EX/f'{dataset}.h5ad'; done=EX/f'{dataset}.json'
   if output.exists() and done.exists(): log(dataset=dataset,status='reuse');continue
   path=CACHE/f'{dataset}.h5ad'; tmp=path.with_suffix('.h5ad.part')
   if not path.exists() and tmp.exists() and tmp.stat().st_size==row.bytes: os.replace(tmp,path)
   if not path.exists():
    download(row.url,tmp,int(row.bytes),connections=4)
    if tmp.stat().st_size!=row.bytes:raise RuntimeError('Download size mismatch')
    os.replace(tmp,path)
   if path.stat().st_size!=row.bytes:raise RuntimeError('Cached source size mismatch')
   transfer=time.monotonic()-start;log(dataset=dataset,status='downloaded',GB=row.bytes/1e9,seconds=transfer)
   # Only metadata is read from Census; no remote expression queries.
   meta_file=M/f'{dataset}_obs.parquet'
   fields=['soma_joinid','assay_ontology_term_id','cell_type_ontology_term_id','tissue_ontology_term_id','donor_id','is_primary_data']
   if meta_file.exists():meta=pd.read_parquet(meta_file)
   else:
    if census is None:census=cellxgene_census.open_soma(census_version='2023-05-15')
    meta=census['census_data']['homo_sapiens'].obs.read(value_filter=f"dataset_id == '{dataset}'",column_names=fields).concat().to_pandas().sort_values('soma_joinid')
    meta.to_parquet(meta_file,index=False)
   with h5py.File(path,'r') as h:
    obs=read_elem(h['obs']); var=read_elem(h['var']); raw=h['raw'] if 'raw' in h else h
    raw_var=read_elem(raw['var']); x=sparse_dataset(raw['X']) if isinstance(raw['X'],h5py.Group) else raw['X']
    # May 2023 Census builder keeps filtered source rows in their original order.
    mask=(obs.organism_ontology_term_id.astype(str)=='NCBITaxon:9606') & obs.assay_ontology_term_id.isin(assays)
    mask &= ~obs.tissue_ontology_term_id.astype(str).str.endswith((' (organoid)',' (cell culture)'))
    refs=set(var.feature_reference.astype(str).unique())
    if refs!={'NCBITaxon:9606'}:raise RuntimeError(f'Unverified feature reference: {refs}')
    positions=np.flatnonzero(mask.to_numpy()); filtered=obs.iloc[positions]
    if len(filtered)!=len(meta) or not np.array_equal(meta.soma_joinid,np.arange(meta.soma_joinid.iloc[0],meta.soma_joinid.iloc[0]+len(meta))):raise RuntimeError('Census/source row-count or contiguity mismatch')
    for key in fields[1:]:
     if not np.array_equal(filtered[key].astype(str).to_numpy(),meta[key].astype(str).to_numpy()):raise RuntimeError(f'Row identity metadata mismatch: {key}')
    target=all_obs[all_obs.dataset_id==dataset].sort_values('soma_joinid').copy()
    ref_positions=np.flatnonzero(ref.obs.dataset_id.astype(str).to_numpy()==dataset)[:32]
    check_ids=ref.obs.soma_joinid.iloc[ref_positions].to_numpy(dtype=np.int64)
    if len(check_ids)==0:raise RuntimeError('No saved reference cells for verification')
    request=np.unique(np.concatenate([target.soma_joinid.to_numpy(dtype=np.int64),check_ids]))
    local_rows=request-int(meta.soma_joinid.iloc[0])
    if np.any(local_rows<0) or np.any(local_rows>=len(positions)):raise RuntimeError('Requested cell outside source dataset')
    gene_dst=pd.Index(target_genes).get_indexer(raw_var.index)
    gene_mask=(gene_dst>=0)&(raw_var.feature_biotype.astype(str).to_numpy()=='gene')
    source_cols=np.flatnonzero(gene_mask); dest_cols=gene_dst[gene_mask]
    mapper=sparse.csr_matrix((np.ones(len(dest_cols),dtype=np.float32),(np.arange(len(dest_cols)),dest_cols)),shape=(len(dest_cols),len(target_genes)))
    blocks=[]
    for offset in range(0,len(request),5000):
     block=sparse.csr_matrix(x[positions[local_rows[offset:offset+5000]],:])[:,source_cols].astype(np.float32)@mapper
     if not np.isfinite(block.data).all() or np.any(block.data<0) or np.any(block.data!=np.floor(block.data)):raise RuntimeError('Source values are not raw counts')
     blocks.append(block)
    counts=sparse.vstack(blocks,format='csr')
    reference=sparse.csr_matrix(ref.X[ref_positions,:])
    if (counts[np.searchsorted(request,check_ids)]!=reference).nnz:raise RuntimeError('Counts disagree with saved Census reference')
    result=ad.AnnData(X=counts[np.searchsorted(request,target.soma_joinid)],obs=target.set_index('soma_joinid',drop=False),var=ref.var.copy())
    result.obs_names=[f'sctab_{int(i)}' for i in target.soma_joinid];result.obs_names.name=None;result.var_names.name=None
    result.uns.update(source_url=row.url,census_version='2023-05-15',values='raw counts',verified_reference_cells=len(check_ids))
    temp=output.with_suffix('.h5ad.tmp');result.write_h5ad(temp,compression='gzip');os.replace(temp,output)
    atomic_json(done,dict(dataset_id=dataset,cells=len(target),source_bytes=int(row.bytes),download_seconds=transfer,total_seconds=time.monotonic()-start,reference_cells=len(check_ids),exact_reference_counts=True,metadata_order_verified=True,source_url=row.url))
    log(dataset=dataset,status='VERIFIED',cells=len(target),seconds=time.monotonic()-start)
 finally:
  ref.file.close()
  if census is not None:census.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int,default=0);a=p.parse_args();main(a.worker,a.workers,a.limit)
