#!/usr/bin/env bash
#SBATCH --job-name=sctab-umi-scgpt
#SBATCH --account=rongma_lab
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/n/holylabs/rongma_lab/Lab/scaling_law/jingyuan/pretraining_cell_scaling/logs/sctab-umi-scgpt-%j.log
#SBATCH --requeue
set -euo pipefail

LAB_ROOT=/n/holylabs/rongma_lab/Lab/scaling_law
ROOT="$LAB_ROOT/jingyuan/pretraining_cell_scaling"
INPUT_ROOT="$ROOT/data/pretraining_umi_scaling"
CHECKPOINT_ROOT="$ROOT/outputs/pretraining_umi_scaling/scgpt_checkpoints"
REP_ROOT="$ROOT/outputs/pretraining_umi_scaling/dogma_scgpt_representations"
LAYER="$LAB_ROOT/foundation_panelB_gpu_transfer_bundle/Application/Layer 1A"
module load python/3.10.13-fasrc01
export PYTHONPATH="$LAYER/models/scGPT_repo:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

for seed in 0 1 2; do
  test -f "$INPUT_ROOT/seed_${seed}/_SUCCESS"
  model_index=$((seed * 8 + 7))
  for label in rho0250 rho0500 rho1000 rho2000 rho4000 rho7000; do
    export PRETRAINING_UMI_INPUT="$INPUT_ROOT/seed_${seed}/sctab_seed${seed}_n222000_${label}_panel2000.h5ad"
    export PRETRAINING_UMI_MODEL_ID="sctab_seed${seed}_n222000_${label}_scgpt"
    export PRETRAINING_UMI_OUTPUT_ROOT="$CHECKPOINT_ROOT"
    "$LAYER/.venv-gpu/bin/python" "$ROOT/scripts/08_train_scgpt.py" --model-index "$model_index" --num-workers 4
    export PRETRAINING_UMI_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
    export PRETRAINING_UMI_OUTPUT_ROOT="$REP_ROOT"
    "$LAYER/.venv-gpu/bin/python" "$ROOT/scripts/09_embed_dogma_scgpt.py" --model-index "$model_index" --num-workers 4
  done
done

