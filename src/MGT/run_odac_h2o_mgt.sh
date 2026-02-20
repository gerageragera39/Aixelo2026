#!/bin/bash -l
#SBATCH --job-name=mgt_oda_h2o
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --clusters=alex
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
CODE_DIR="$PROJECT_ROOT/code"
DATASET_ARCHIVE="$HPCVAULT/datasets/odac_h2o.tar"
RESULTS_DIR="$PROJECT_ROOT/results/mgt/job_${SLURM_JOB_ID}"
CONTAINER="$PROJECT_ROOT/../container/cont.sif"

TMP_DATA="$TMPDIR/odac_h2o_data"

mkdir -p "$TMP_DATA" "$RESULTS_DIR"

echo "Getting data from $DATASET_ARCHIVE to $TMP_DATA..."
tar xf "$DATASET_ARCHIVE" -C "$TMP_DATA"

if [ -d "$TMP_DATA/dataset" ]; then
    DATA_ROOT="$TMP_DATA/dataset"
else
    DATA_ROOT="$TMP_DATA"
fi

echo "Data root: $DATA_ROOT"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun apptainer exec --nv \
    -B "$CODE_DIR:/workspace/code" \
    -B "$DATA_ROOT:/workspace/dataset" \
    -B "$RESULTS_DIR:/workspace/results" \
    --pwd /workspace/results \
    "$CONTAINER" \
    python -u /workspace/code/training.py \
        --root /workspace/dataset \
        --model_path /workspace/results/saved_models \
        --save_dir /workspace/results \
        --accelerator cuda \
        --n_devices 1 \
        --num_atom_fea 92 \
        --batch_size 1 \
        --n_cum 16 \
        --out_dims 1 \
        --epochs 3 \
        --target-column "target"
