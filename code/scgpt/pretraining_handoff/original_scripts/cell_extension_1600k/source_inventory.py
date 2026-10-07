from pathlib import Path
import pandas as pd, json,cellxgene_census as c
R=Path('/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling'); O=R/'manifests/cell_extension_1600k/direct_source'; O.mkdir(exist_ok=True)
m=pd.concat([pd.read_parquet(R/f'manifests/cell_extension_1600k/sctab_seed{s}_n1600000_cells.parquet').iloc[888000:] for s in range(3)])
counts=m.groupby('dataset_id',observed=True).soma_joinid.nunique().sort_values(ascending=False)
print('datasets',len(counts),'unique needed',m.soma_joinid.nunique(),flush=True)
with c.open_soma(census_version='2023-05-15') as census:
 print('OBS SCHEMA',census['census_data']['homo_sapiens'].obs.schema,flush=True)
 datasets=census['census_info']['datasets'].read().concat().to_pandas()
 print('DATASET COLUMNS',datasets.columns.tolist(),flush=True)
 datasets=datasets[datasets.dataset_id.isin(counts.index)].copy(); datasets['needed_cells']=datasets.dataset_id.map(counts)
 datasets.to_parquet(O/'datasets.parquet',index=False)
 print(datasets.sort_values('dataset_total_cell_count').head(3).to_string(index=False),flush=True)
 print(json.dumps(c.get_census_version_description('2023-05-15'),default=str),flush=True)
