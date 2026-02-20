# MOF Transformer

Universal transfer learning for Metal-Organic Frameworks (MOFs) property prediction using transformer-based models.

## Overview

This project provides a complete pipeline for training and evaluating MOFTransformer models on various MOF property prediction tasks. The codebase is split into modular components for preprocessing and training, with comprehensive metrics tracking and HPC cluster support.

## Project Structure

```
MOF_Transformer/
├── code/
│   ├── preprocessor.py      # Dataset preprocessing script
│   ├── trainer.py           # Model training and evaluation
│   ├── metrics_logger.py    # Metrics collection and CSV export
│   ├── README.md            # Detailed usage documentation
│   └── MOFTransformer/      # Core MOFTransformer package
├── container/
│   ├── Dockerfile           # Docker image definition
│   ├── Singularity.def      # Singularity/Apptainer definition
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Container usage guide
├── run_mof_transformer.sh   # Example SLURM job script
└── README.md                # This file
```

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
cd MOF_Transformer

# Install the MOFTransformer package
cd code/MOFTransformer
pip install -e .

# Install additional dependencies
pip install pandas numpy scikit-learn pytorch-lightning tensorboard
```

### 2. Prepare Your Dataset

Organize your dataset with the following structure:

```
dataset/
├── raw/
│   ├── *.cif files
│   └── id_prop.csv
```

The `id_prop.csv` should contain CIF filenames (without extension) in the first column and target properties in subsequent columns.

### 3. Preprocess Data (Local Machine Recommended)

```bash
cd code/
python preprocessor.py \
    --data-dir /path/to/dataset/ \
    --target-column energy_per_atom \
    --output-name qmof_preprocessed
```

> **HPC Tip:** Run preprocessing locally and upload the generated `.tar` archive to the cluster.

### 4. Train the Model

```bash
# Extract the archive
tar -xvf qmof_preprocessed_energy_per_atom.tar -C ./data/

# Train
python trainer.py \
    --data-dir ./data/dataset/ \
    --target-column energy_per_atom \
    --log-dir ./results/ \
    --max-epochs 50
```

## Features

### Preprocessor
- Flexible `id_prop.csv` format with any header name for the first column
- Support for multiple target columns
- Automatic tar archive creation
- Data filtering and validation

### Trainer
- Validation after each epoch with R² metric display
- Separate CSV files for train, validation, and test metrics
- Model checkpointing based on best validation R²
- Mixed precision training support (FP16, FP32, FP64)
- Multi-GPU training with DDP

### Containerization
- Docker and Singularity/Apptainer support
- Pre-configured environment with all dependencies
- Optimized for HPC clusters with SLURM integration

## Documentation

- **[Code Documentation](code/README.md)** - Detailed usage guide for preprocessor and trainer
- **[Container Guide](container/README.md)** - Containerization and deployment instructions
- **[MOFTransformer Docs](code/MOFTransformer/docs/)** - Original MOFTransformer documentation

## Requirements

### System Requirements
- Linux operating system (tested on Ubuntu 22.04)
- CUDA 12.1 compatible GPU (for GPU training)
- Python 3.8 or newer

### Python Dependencies
- PyTorch >= 2.4.0
- PyTorch Lightning == 1.9.5
- DGL == 2.1.0
- Numpy < 2.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0

See [`container/requirements.txt`](container/requirements.txt) for the complete list.

## HPC Cluster Usage

### SLURM Job Submission

An example SLURM script is provided at [`run_mof_transformer.sh`](run_mof_transformer.sh):

```bash
sbatch run_mof_transformer.sh
```

### Container Deployment

For HPC clusters with Singularity/Apptainer:

```bash
# Build the container
cd container/
apptainer build moftransformer.sif Singularity.def

# Run training
apptainer exec --nv \
    -B /path/to/data:/dataset \
    -B /path/to/results:/output \
    moftransformer.sif \
    python code/trainer.py --data-dir /dataset --log-dir /output
```

## Examples

### Basic Training

```bash
# Preprocess
python preprocessor.py --data-dir ./qmof_cif/ --target-column bandgap --output-name qmof_bandgap

# Extract
tar -xvf qmof_bandgap_bandgap.tar -C ./data/

# Train
python trainer.py --data-dir ./data/dataset/ --target-column bandgap --log-dir ./results/
```

### Custom Hyperparameters

```bash
python trainer.py \
    --data-dir ./data/dataset/ \
    --target-column formation_energy \
    --log-dir ./results/ \
    --max-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --weight-decay 1e-3
```

### Resume from Checkpoint

```bash
python trainer.py \
    --data-dir ./data/dataset/ \
    --target-column energy_per_atom \
    --log-dir ./results/ \
    --load-path ./results/checkpoints/best.ckpt
```

## Metrics Output

The training pipeline generates separate CSV files for each phase:

### train_metrics.csv
| epoch | loss | mae | learning_rate |
|-------|------|-----|---------------|
| 0 | 0.620595 | 289.28 | 1.4e-06 |

### val_metrics.csv
| epoch | loss | mae | r2 |
|-------|------|-----|-----|
| 0 | 0.804781 | 309.97 | -0.407748 |

### test_metrics.csv
| loss | mae | r2 |
|------|-----|-----|
| 0.795226 | 309.04 | 0.654321 |

## Troubleshooting

### GRIDAY Installation

If GRIDAY is not found, install it manually:

```bash
moftransformer install-griday
```

### Missing CIF Files

Ensure all CIF IDs in `id_prop.csv` have corresponding `.cif` files in the `raw/` directory.

### Out of Memory

Reduce batch size or use gradient accumulation:

```bash
python trainer.py \
    --batch-size 8 \
    --per-gpu-batchsize 4 \
    ...
```

### Low R² Score

- Increase `--max-epochs`
- Adjust `--learning-rate`
- Ensure data quality and sufficient training samples

## Citation

If you use MOFTransformer in your research, please cite:

1. **MOFTransformer**: A multi-modal pre-training transformer for universal transfer learning in metal–organic frameworks, Nature Machine Intelligence, 5, 2023. [DOI](https://www.nature.com/articles/s42256-023-00628-2)

2. **PMTransformer**: Enhancing Structure–Property Relationships in Porous Materials through Transfer Learning and Cross-Material Few-Shot Learning, ACS Appl. Mater. Interfaces 2023, 15, 48, 56375–56385. [DOI](https://doi.org/10.1021/acsami.3c10323)

## License

This project is licensed under the MIT License. See the [LICENSE](code/MOFTransformer/LICENSE) file for more information.

## Contributing

Contributions are welcome! If you have any suggestions or find any issues, please open an issue or a pull request.

## Acknowledgments

- Original MOFTransformer project: [GitHub](https://github.com/hspark1212/MOFTransformer)
- GRIDAY for energy grid calculations: [GitHub](https://github.com/Sangwon91/GRIDAY)
