"""Portable entry point to the archived, unmodified training algorithm (CUDA)."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
from model_utils import HERE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--model-id',required=True)
    p.add_argument('--n-cells',type=int,required=True)
    p.add_argument('--sampling-seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--num-workers',type=int,default=4)
    a=p.parse_args()
    if (a.output_root/a.model_id).exists():
        raise FileExistsError('Choose a fresh output directory; original checkpoints are preserved')
    source=HERE/'original_scripts/08_train_scgpt.py'
    spec=importlib.util.spec_from_file_location('archived_scgpt_train',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.PANEL_FILE=HERE/'assets/sctab_hvg_2000.tsv'
    for key in ['INITIAL_CHECKPOINT','TRAINING_EPOCHS','TRAINING_LEARNING_RATE']:
        os.environ.pop(key,None)
    os.environ.update(TRAINING_SEED=str(a.sampling_seed),TRAINING_N_CELLS=str(a.n_cells),
        PRETRAINING_UMI_MODEL_ID=a.model_id,PRETRAINING_UMI_INPUT=str(a.input.resolve()),
        PRETRAINING_UMI_OUTPUT_ROOT=str(a.output_root.resolve()))
    sys.argv=[str(source),'--model-index','0','--num-workers',str(a.num_workers)]
    module.main()


if __name__=='__main__': main()
