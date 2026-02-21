#!/bin/bash -l
#SBATCH --job-name=mof_tranformer_qmof
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --clusters=alex
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
CODE_DIR="$PROJECT_ROOT/code"
DATASET_ARCHIVE="$HPCVAULT/datasets/qmof_full_processed_outputs.pbe.bandgap.tar"
RESULTS_DIR="$PROJECT_ROOT/my_results/job_${SLURM_JOB_ID}"
CONTAINER="$PROJECT_ROOT/container/moftransformer_container.sif"


CKPT_JOB_ID=3355686
CKPT_PATH="/workspace/my_results/job_${CKPT_JOB_ID}/training/version_0/checkpoints/epoch93-r20.0000.ckpt"


TMP_DATA="$TMPDIR/qmof_data_transformer"

mkdir -p "$TMP_DATA" "$RESULTS_DIR" "logs"

echo "Getting data from $DATASET_ARCHIVE to $TMP_DATA..."
tar xf "$DATASET_ARCHIVE" -C "$TMP_DATA"

if [ -d "$TMP_DATA/dataset" ]; then
    DATA_ROOT="$TMP_DATA/dataset/processed"
else
    DATA_ROOT="$TMP_DATA"
fi

echo "Data root: $DATA_ROOT"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun apptainer exec --nv \
    -B "$PROJECT_ROOT:/workspace" \
    -B "$DATA_ROOT:/dataset:rw" \
    -B "$RESULTS_DIR:/output:rw" \
    --pwd /workspace \
    "$CONTAINER" \
    python -u code/trainer.py \
        --data-dir /dataset \
        --log-dir /output \
        --per-gpu-batchsize 128 \
        --batch-size 128 \
        --max-epochs 100 \
        --target-column outputs.pbe.bandgap \
        --load-path "$CKPT_PATH" \
        --learning-rate 1e-5
