#!/bin/bash -l
#SBATCH --job-name=mgt_oda_co2
#SBATCH --output=/home/vault/b192aa/b192aa41/projects/mgt_odac_co2/logs/%x_%j.out
#SBATCH --error=/home/vault/b192aa/b192aa41/projects/mgt_odac_co2/logs/%x_%j.err
#SBATCH --clusters=alex
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail


# -------------------------------
# Пути
# -------------------------------
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
CODE_DIR="$PROJECT_ROOT/code"
# Это путь к АРХИВУ
DATASET_ARCHIVE="/home/vault/b192aa/b192aa41/datasets/odac_co2.tar"
RESULTS_DIR="$PROJECT_ROOT/results/mgt/job_${SLURM_JOB_ID}"
CONTAINER="$PROJECT_ROOT/../container/cont.sif"

# Это путь куда распакуем
TMP_DATA="$TMPDIR/odac_co2_data"

mkdir -p "$TMP_DATA" "$RESULTS_DIR"

# -------------------------------
# Распаковка
# -------------------------------
echo "Распаковка данных из $DATASET_ARCHIVE в $TMP_DATA..."
tar xf "$DATASET_ARCHIVE" -C "$TMP_DATA"

# ПРОВЕРКА СТРУКТУРЫ
# Если архив содержит папку 'dataset', то путь правильный.
# Если файлы лежат сразу в корне архива, то DATA_ROOT=$TMP_DATA
if [ -d "$TMP_DATA/dataset" ]; then
    DATA_ROOT="$TMP_DATA/dataset"
else
    # Предполагаем, что распаковалось прямо в TMP_DATA
    DATA_ROOT="$TMP_DATA"
fi

echo "Корневая папка данных: $DATA_ROOT"

# -------------------------------
# Запуск
# -------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ИСПРАВЛЕНИЕ: Монтируем DATA_ROOT, а не архив
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
        --epochs 3
