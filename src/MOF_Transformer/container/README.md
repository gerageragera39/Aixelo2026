# MOFTransformer Container

This directory contains containerization files for deploying MOFTransformer on HPC clusters and cloud environments.

## Overview

The container provides a reproducible environment with all dependencies pre-installed for training and inference with MOFTransformer. It includes:

- **CUDA 12.1** with cuDNN 8
- **Python 3** with pip
- **PyTorch 2.4.0** (CUDA 12.1 build)
- **DGL 2.1.0** (CUDA 12.1 build)
- **Chemistry stack**: RDKit, DGL-LifeSci
- **MOFTransformer** and all required dependencies

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image definition for building the container |
| `Singularity.def` | Singularity/Apptainer definition file for HPC deployment |
| `requirements.txt` | Python dependencies for MOFTransformer |

## Building the Container

### Docker

To build the Docker image:

```bash
cd container/
docker build -t moftransformer:latest .
```

### Singularity/Apptainer

To build a Singularity image from the Docker image:

```bash
# Using Apptainer (recommended for HPC)
apptainer build moftransformer.sif docker://<your-registry>/moftransformer:latest

# Or build from Dockerfile directly
apptainer build moftransformer.sif Dockerfile
```

To build from the definition file:

```bash
sudo apptainer build moftransformer.sif Singularity.def
```

## Usage

### Running with Docker

```bash
docker run --gpus all --rm \
    -v /path/to/data:/dataset \
    -v /path/to/results:/output \
    moftransformer:latest \
    python code/trainer.py \
        --data-dir /dataset \
        --target-column energy_per_atom \
        --log-dir /output
```

### Running with Singularity/Apptainer

```bash
apptainer exec --nv \
    -B /path/to/data:/dataset \
    -B /path/to/results:/output \
    moftransformer.sif \
    python code/trainer.py \
        --data-dir /dataset \
        --target-column energy_per_atom \
        --log-dir /output
```

## SLURM Integration

For HPC clusters using SLURM, see the example job script at `../run_mof_transformer.sh`. Key considerations:

1. **GPU Access**: Request GPUs with `--gres=gpu:a100:1` or appropriate partition
2. **Data Staging**: Copy data to local scratch (`$TMPDIR`) before processing
3. **Bind Mounts**: Use `-B` to mount host directories into the container
4. **Environment**: Set `OMP_NUM_THREADS` to match `--cpus-per-task`

Example SLURM script snippet:

```bash
#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00

TMP_DATA="$TMPDIR/mof_data"
mkdir -p "$TMP_DATA"

# Extract data
tar xf "$DATASET_ARCHIVE" -C "$TMP_DATA"

# Run container
srun apptainer exec --nv \
    -B "$PROJECT_ROOT:/workspace" \
    -B "$TMP_DATA:/dataset:rw" \
    -B "$RESULTS_DIR:/output:rw" \
    --pwd /workspace \
    moftransformer.sif \
    python code/trainer.py --data-dir /dataset --log-dir /output
```

## Environment Variables

The container sets the following environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `DEBIAN_FRONTEND` | `noninteractive` | Non-interactive apt operations |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |
| `OMP_NUM_THREADS` | `1` | Default OpenMP threads (override for SLURM) |
| `LC_ALL` | `C` | Locale setting |

## Version Information

| Component | Version |
|-----------|---------|
| CUDA | 12.1 |
| cuDNN | 8 |
| Python | 3.x (Ubuntu 22.04) |
| PyTorch | 2.4.0+cu121 |
| TorchVision | 0.19.0+cu121 |
| DGL | 2.1.0+cu121 |
| Numpy | < 2.0 |
| Pydantic | < 2.0 |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use gradient accumulation:

```bash
python code/trainer.py \
    --batch-size 8 \
    --per-gpu-batchsize 4 \
    ...
```

### Import Errors

If you encounter import errors with `transformers` or `lightning`, ensure the container was built successfully:

```bash
docker run --rm moftransformer:latest python -c "import moftransformer; print(moftransformer.__version__)"
```

### Permission Issues with Singularity

If you encounter permission errors, ensure proper bind mount permissions:

```bash
apptainer exec --nv -B /path:/path:rw moftransformer.sif ...
```

## License

This container is part of the MOFTransformer project. See the main repository for license information.
