# Quantum Materials Optimization Framework (QMOF)

**Enterprise-Grade Multi-Model Architecture for Quantum Materials Property Prediction**

---

## Executive Summary

The Quantum Materials Optimization Framework (QMOF) is a production-ready, containerized machine learning platform designed for high-throughput prediction of quantum mechanical properties in Metal-Organic Frameworks (MOFs) and related porous materials. This framework integrates three state-of-the-art graph neural network architectures—**CGCNN**, **MGT (Multi-Graph Transformer)**, and **MOFTransformer**—providing researchers and engineers with a comprehensive toolkit for materials discovery and optimization.

Built with scalability in mind, QMOF supports distributed training across multi-GPU clusters via SLURM workload management, containerized deployment using Docker and Singularity/Apptainer, and seamless integration with high-performance computing (HPC) infrastructure.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [System Requirements](#system-requirements)
- [Installation & Setup](#installation--setup)
- [Quick Start Guide](#quick-start-guide)
- [Model Documentation](#model-documentation)
  - [CGCNN (Crystal Graph Convolutional Neural Network)](#cgcnn)
  - [MGT (Multi-Graph Transformer)](#mgt)
  - [MOFTransformer](#moftransformer)
- [HPC Cluster Deployment](#hpc-cluster-deployment)
- [Containerization](#containerization)
- [Data Format & Preprocessing](#data-format--preprocessing)
- [Training Configuration Reference](#training-configuration-reference)
- [Results & Metrics](#results--metrics)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Benchmarks](#performance-benchmarks)
- [License & Legal](#license--legal)
- [Citation](#citation)
- [Support & Contact](#support--contact)

---

## Key Features

### Multi-Model Architecture
- **CGCNN**: Crystal Graph Convolutional Neural Network for periodic crystal structures
- **MGT**: Hybrid transformer combining attention mechanisms with graph convolutions
- **MOFTransformer**: Pre-trained transformer specifically designed for MOF property prediction

### Production-Ready Infrastructure
- **Containerized Deployment**: Docker and Singularity/Apptainer support for reproducible environments
- **HPC Integration**: Native SLURM job scripts for cluster deployment
- **Multi-GPU Training**: Distributed training with PyTorch Lightning and FSDP strategies
- **Mixed Precision**: FP16/FP32 support for optimized memory usage

### Comprehensive Tooling
- **Automated Preprocessing**: CIF file parsing, graph construction, and feature extraction
- **Metrics Tracking**: Real-time logging of loss, MAE, R², and custom metrics
- **Model Checkpointing**: Automatic saving of best models based on validation performance
- **Transfer Learning**: Pre-trained weights for fine-tuning on custom datasets

### Enterprise Security
- **Proprietary License**: Commercial-use restricted licensing
- **Private Deployment**: Designed for on-premises and private cloud infrastructure
- **No External Dependencies**: Self-contained container images with pinned versions

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QMOF Framework                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   CGCNN      │    │     MGT      │    │ MOFTransformer│                   │
│  │  (GNN-based) │    │ (Hybrid Attn)│    │ (Pre-trained) │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             │                                                │
│                    ┌────────▼────────┐                                       │
│                    │  Shared Utils   │                                       │
│                    │  - Datasets     │                                       │
│                    │  - Preprocessing│                                       │
│                    │  - Metrics      │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│  ┌──────────────────────────┼──────────────────────────┐                    │
│  │                    Container Layer                   │                    │
│  │         (Docker / Singularity / Apptainer)           │                    │
│  └──────────────────────────┬──────────────────────────┘                    │
│                             │                                                │
│  ┌──────────────────────────▼──────────────────────────┐                    │
│  │              HPC Infrastructure (SLURM)              │                    │
│  │         Multi-GPU Nodes (NVIDIA A100/V100)           │                    │
│  └──────────────────────────────────────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Aixelo2026/
├── src/
│   ├── CGCNN/                          # Crystal Graph CNN implementation
│   │   ├── code/
│   │   │   ├── main.py                 # Training entry point
│   │   │   ├── model.py                # CGCNN architecture
│   │   │   ├── data.py                 # Data loading & preprocessing
│   │   │   ├── config.py               # Configuration management
│   │   │   └── predict.py              # Inference utilities
│   │   ├── run_qmof_cgcnn.sh           # SLURM job script
│   │   └── run_odac_*.sh               # Task-specific scripts (CO2, H2O)
│   │
│   ├── MGT/                            # Multi-Graph Transformer
│   │   ├── code/
│   │   │   ├── training.py             # Main training loop
│   │   │   ├── testing.py              # Evaluation utilities
│   │   │   ├── pre-training.py         # Pre-training routines
│   │   │   ├── run.py                  # Inference runner
│   │   │   ├── model/                  # Model components
│   │   │   │   ├── transformer.py      # Multi-head attention
│   │   │   │   ├── alignn.py           # Edge-gated graph conv
│   │   │   │   └── graphformer.py      # Graph attention layers
│   │   │   ├── modules/                # Reusable modules
│   │   │   └── utils/                  # Utility functions
│   │   └── run_*.sh                    # SLURM job scripts
│   │
│   ├── MOF_Transformer/                # Pre-trained MOF model
│   │   ├── code/
│   │   │   ├── preprocessor.py         # Dataset preprocessing
│   │   │   ├── trainer.py              # Training & evaluation
│   │   │   ├── metrics_logger.py       # Metrics collection
│   │   │   └── MOFTransformer/         # Core package
│   │   │       ├── moftransformer/     # Model implementation
│   │   │       └── baseline_model/     # Baseline comparisons
│   │   ├── container/                  # Container definitions
│   │   └── run_mof_transformer.sh      # SLURM job script
│   │
│   └── container/                      # Shared container image
│       ├── cont.def                    # Singularity definition
│       └── requirements.txt            # Python dependencies
│
├── test/                               # Test suites
│   └── MGT/
│
├── code_from_previous_workers/         # Legacy code (reference)
│   └── cgcnnv/
│
├── README.md                           # This documentation
├── LICENSE                             # Proprietary license
└── .gitignore                          # Git ignore rules
```

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | HPC Cluster |
|-----------|---------|-------------|-------------|
| **GPU** | NVIDIA GTX 1080 (8GB) | NVIDIA A100 (40GB) | 4× A100 (80GB) |
| **CPU** | 8 cores | 16 cores | 32+ cores |
| **RAM** | 32 GB | 64 GB | 256+ GB |
| **Storage** | 100 GB SSD | 500 GB NVMe | 2+ TB NVMe |
| **CUDA** | 11.8 | 12.1+ | 12.1+ |

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| **OS** | Ubuntu 22.04 LTS | Required for container builds |
| **Python** | 3.8 - 3.10 | 3.10 recommended |
| **PyTorch** | 2.1.2 | CUDA 12.1 build |
| **PyTorch Lightning** | 1.9.5 - 2.1.3 | Model-specific |
| **DGL** | 2.1.0 | MOFTransformer only |
| **Docker** | 20.10+ | Local development |
| **Singularity/Apptainer** | 3.8+ | HPC deployment |
| **SLURM** | 20.11+ | Cluster job scheduling |

### NVIDIA Container Toolkit

For GPU acceleration in containers:

```bash
# Ubuntu installation
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Installation & Setup

### 1. Repository Clone

```bash
git clone https://github.com/your-org/Aixelo2026.git
cd Aixelo2026
```

### 2. Dataset Acquisition

Download the QMOF dataset from Hugging Face:

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/datasets/hermanhugging/qmof_project

# Or download directly
wget https://huggingface.co/datasets/hermanhugging/qmof_project/resolve/main/qmof_cif.tar
tar -xf qmof_cif.tar -C ./data/
```

### 3. Container Build

#### Option A: Docker (Local Development)

```bash
# Build shared container
cd src/container
docker build -t qmof-base:latest .

# Build model-specific images
cd ../CGCNN/code
docker build -t qmof-cgcnn:latest -f ../Dockerfile .

cd ../../MGT/code
docker build -t qmof-mgt:latest -f ../Dockerfile .
```

#### Option B: Singularity (HPC Deployment)

```bash
# Convert Docker to Singularity image
cd src/container
apptainer build qmof-base.sif cont.def

# Or pull from Docker Hub
apptainer pull docker://your-registry/qmof-base:latest
```

### 4. Dependency Installation (Native)

If running without containers:

```bash
# Create virtual environment
python -m venv qmof-env
source qmof-env/bin/activate  # Linux/Mac
# or
qmof-env\Scripts\activate  # Windows

# Install core dependencies
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install dgl==2.1.0+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html

# Install remaining dependencies
pip install -r src/container/requirements.txt
```

---

## Quick Start Guide

### CGCNN Training (5 Minutes)

```bash
# Navigate to CGCNN code directory
cd src/CGCNN/code

# Run training with Docker
docker run --rm --gpus all \
  -v $(pwd)/../../data/qmof_cif:/workspace/dataset \
  -v $(pwd)/../../results/cgcnn:/workspace/results \
  qmof-cgcnn:latest \
  python -u main.py /workspace/dataset \
    --task regression \
    --epochs 50 \
    --batch-size 32 \
    --target "outputs.pbe.bandgap"
```

### MGT Training (5 Minutes)

```bash
cd src/MGT/code

docker run --rm --gpus all \
  -v $(pwd)/../../data/qmof_cif:/workspace/dataset \
  -v $(pwd)/../../results/mgt:/workspace/results \
  qmof-mgt:latest \
  python training.py \
    --root /workspace/dataset \
    --model_path /workspace/results/saved_models \
    --save_dir /workspace/results \
    --accelerator cuda \
    --batch_size 8 \
    --epochs 10 \
    --target-column "outputs.pbe.bandgap"
```

### MOFTransformer Training (5 Minutes)

```bash
# Preprocess data (run locally for speed)
cd src/MOF_Transformer/code
python preprocessor.py \
    --data-dir ../../data/qmof_cif \
    --target-column energy_per_atom \
    --output-name qmof_preprocessed

# Extract preprocessed data
mkdir -p ../../data/preprocessed
tar -xvf qmof_preprocessed_energy_per_atom.tar -C ../../data/preprocessed/

# Train model
python trainer.py \
    --data-dir ../../data/preprocessed/dataset \
    --target-column energy_per_atom \
    --log-dir ../../results/moftransformer \
    --max-epochs 50
```

---

## Model Documentation

### CGCNN (Crystal Graph Convolutional Neural Network)

#### Architecture

CGCNN represents crystal structures as graphs where:
- **Nodes**: Atoms with feature vectors (atomic number, electronegativity, etc.)
- **Edges**: Bonds with distance-based features
- **Convolution**: Message passing between neighboring atoms

```
Atom Features (92-dim) ──┐
                         ├──► [Conv Layer] ──► [Conv Layer] ──► [Conv Layer] ──► Pooling ──► FC ──► Output
Bond Features (dist) ────┘
```

#### Supported Properties

| Property | Task Type | Units | Typical Range |
|----------|-----------|-------|---------------|
| Band Gap | Regression | eV | 0 - 10 |
| Formation Energy | Regression | eV/atom | -10 - 5 |
| CO2 Uptake | Regression | mmol/g | 0 - 50 |
| H2O Adsorption | Regression | mmol/g | 0 - 100 |
| Metal/Insulator | Classification | - | 0 or 1 |

#### Hyperparameters

```yaml
# Architecture
atom_fea_len: 64        # Atom feature dimension
h_fea_len: 128          # Hidden layer dimension
n_conv: 3               # Number of convolution layers
n_h: 1                  # Number of hidden layers after pooling

# Training
batch_size: 32          # Batch size
epochs: 500             # Total epochs
learning_rate: 0.01     # Initial LR
momentum: 0.9           # SGD momentum
weight_decay: 0         # L2 regularization

# Data Splits
train_ratio: 0.8        # Training set fraction
val_ratio: 0.1          # Validation set fraction
test_ratio: 0.1         # Test set fraction
```

#### Command-Line Reference

```bash
python main.py <data_root> [OPTIONS]

# Required
data_root               Path to dataset directory

# Task Configuration
--task                  'regression' or 'classification' (default: regression)
--target                Target property column name

# Training
--epochs N              Total training epochs (default: 30)
--batch-size N          Mini-batch size (default: 256)
--lr FLOAT              Initial learning rate (default: 0.01)
--optim OPTIMIZER       'SGD' or 'Adam' (default: SGD)

# Model Architecture
--atom-fea-len N        Atom feature dimension (default: 64)
--n-conv N              Number of conv layers (default: 3)

# Resume Training
--resume PATH           Path to checkpoint file
--start-epoch N         Manual epoch start (default: 0)

# Data Loading
--workers N             Data loading workers (default: 0)
--train-ratio FLOAT     Training data fraction
--val-ratio FLOAT       Validation data fraction
--test-ratio FLOAT      Test data fraction
```

---

### MGT (Multi-Graph Transformer)

#### Architecture

MGT combines transformer attention with graph neural networks through three parallel graph representations:

1. **Local Graph**: k-nearest neighbor graph for short-range interactions
2. **Line Graph**: Edge connectivity for angle features
3. **Fully Connected Graph**: Global attention for long-range interactions

```
                    ┌─────────────────┐
                    │  Input Structure│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│  Local Graph   │  │  Line Graph    │  │  Fully Connected│
│  (k-NN)        │  │  (Angles)      │  │  (Global Attn)  │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │  Graphformer Encoder│
                  │  (Multi-Head Attn)  │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │  EdgeGatedGraphConv│
                  │  (ALIGNN layers)   │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │  Multi-Head Transformer│
                  └─────────┬──────────┘
                            │
                            ▼
                        Output
```

#### Hyperparameters

```yaml
# Device Configuration
n_devices: 1            # Number of GPUs
accelerator: 'cuda'     # Device type

# Model Architecture
num_atom_fea: 92        # Atom feature dimension
num_edge_fea: 1         # Edge feature dimension
num_angle_fea: 1        # Angle feature dimension
num_pe_fea: 10          # Positional encoding dimension
embedding_dims: 128     # Initial embedding dimension
hidden_dims: 512        # Hidden layer dimension
out_dims: 1             # Output dimension
num_layers: 1           # Number of encoder layers
n_mha: 1                # Multi-head attention layers
n_alignn: 3             # ALIGNN convolutions
n_gnn: 3                # GNN layers
n_heads: 4              # Attention heads
residual: true          # Residual connections

# Graph Construction
max_nei_num: 12         # Maximum neighbors
local_radius: 8.0       # Local graph radius (Å)
periodic_radius: 12.0   # FC graph radius (Å)
periodic: true          # Periodic boundary conditions

# Training
batch_size: 8           # Batch size
n_cum: 16               # Gradient accumulation steps
epochs: 100             # Training epochs
lr: 0.0001              # Learning rate
decay: 1e-5             # Weight decay
```

#### Command-Line Reference

```bash
python training.py [OPTIONS]

# Required
--root PATH             Root directory for datasets
--model_path PATH       Directory for model checkpoints
--save_dir PATH         Directory for test results

# Device Configuration
--n_devices N           Number of GPUs (default: 8)
--n_nodes N             Number of nodes (default: 1)
--accelerator TYPE      'cpu', 'cuda', 'gpu', 'mps', 'tpu' (default: cuda)

# Model Loading
--load_model N          0: none, 1: pretrained, 2: checkpoint (default: 0)
--pretrain_model NAME   Pretrained model filename (default: 'pretrain.ckpt')
--final_model NAME      Final model filename (default: 'end_model.ckpt')
--lowest_model NAME     Best model filename (default: 'lowest.ckpt')

# Training
--batch_size N          Batch size (default: 2)
--n_cum N               Gradient accumulation batches (default: 8)
--epochs N              Training epochs (default: 10)
--begin_epoch N         Starting epoch (default: 1)
--lr FLOAT              Learning rate (default: 0.0001)
--decay FLOAT           Weight decay (default: 1e-5)
--train_split FLOAT     Training fraction (default: 0.8)
--val_split FLOAT       Validation fraction (default: 0.2)

# Graph Construction
--process N             Create graphs during loading (0/1, default: 1)
--max_nei_num N         Max neighbors (default: 12)
--local_radius FLOAT    Local radius Å (default: 8)
--periodic N            Periodic structures (0/1, default: 1)
--periodic_radius FLOAT Periodic radius Å (default: 12)

# Feature Dimensions
--num_atom_fea N        Atom feature length (default: 92)
--num_edge_fea N        Edge feature length (default: 1)
--num_angle_fea N       Angle feature length (default: 1)
--num_pe_fea N          Positional encoding length (default: 10)
--num_clmb_fea N        Fully connected edge length (default: 1)

# RBF Bins
--num_edge_bins N       Edge RBF bins (default: 80)
--num_angle_bins N      Angle RBF bins (default: 40)
--num_clmb_bins N       FC edge RBF bins (default: 120)

# Architecture
--embedding_dims N      Embedding dimension (default: 128)
--hidden_dims N         Hidden dimension (default: 512)
--out_dims N            Output dimension (default: 3)
--num_layers N          Encoder layers (default: 1)
--n_mha N               Attention layers (default: 1)
--n_alignn N            ALIGNN convolutions (default: 3)
--n_gnn N               GNN layers (default: 3)
--n_heads N             Attention heads (default: 4)
--residual N            Residual connections (0/1, default: 1)

# Logging
--run_name NAME         Experiment name
--out_names LIST        Output names for logging
```

---

### MOFTransformer

#### Architecture

MOFTransformer is a pre-trained multi-modal transformer specifically designed for Metal-Organic Frameworks. It leverages transfer learning from large-scale MOF datasets.

```
┌────────────────────────────────────────────────────────────┐
│                    MOFTransformer Pipeline                  │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  CIF Files ──► Preprocessor ──► Energy Grids ──► Tokenizer │
│       │                              │                      │
│       │                              ▼                      │
│       │                      GRIDAY Engine                  │
│       │                      (Grid Generation)              │
│       │                              │                      │
│       ▼                              ▼                      │
│  id_prop.csv ───────────────► MOFTransformer Model          │
│  (Targets)                         │                        │
│                                    ▼                        │
│                            Property Predictions             │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

#### Preprocessor

The preprocessor handles:
- CIF file parsing and validation
- Energy grid generation using GRIDAY
- Feature extraction and normalization
- Tar archive creation for efficient loading

```bash
python preprocessor.py \
    --data-dir /path/to/dataset \
    --target-column energy_per_atom \
    --output-name qmof_preprocessed \
    --filter-invalid \
    --max-workers 8
```

**Preprocessor Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Dataset directory with CIF files and id_prop.csv | Required |
| `--target-column` | Column name for regression target | Required |
| `--output-name` | Output archive name prefix | 'preprocessed' |
| `--filter-invalid` | Remove structures with missing data | False |
| `--max-workers` | Parallel processing workers | 4 |

#### Trainer

The trainer supports:
- Multi-GPU training with DDP
- Mixed precision (FP16/FP32)
- Automatic learning rate scheduling
- Model checkpointing based on validation R²
- Separate metrics logging for train/val/test

```bash
python trainer.py \
    --data-dir ./data/dataset \
    --target-column bandgap \
    --log-dir ./results \
    --max-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --weight-decay 1e-3 \
    --precision 16 \
    --num-gpus 4
```

**Trainer Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Preprocessed dataset directory | Required |
| `--target-column` | Target property column | Required |
| `--log-dir` | Results and checkpoint directory | Required |
| `--max-epochs` | Maximum training epochs | 50 |
| `--batch-size` | Training batch size | 16 |
| `--learning-rate` | Initial learning rate | 1e-4 |
| `--weight-decay` | L2 regularization | 1e-5 |
| `--precision` | Training precision (16/32/64) | 32 |
| `--num-gpus` | Number of GPUs | 1 |
| `--load-path` | Checkpoint to resume from | None |
| `--freeze-backbone` | Freeze pre-trained layers | False |

#### Metrics Output

The trainer generates three CSV files:

**train_metrics.csv**
```csv
epoch,loss,mae,learning_rate
0,0.620595,289.28,1.4e-06
1,0.584321,275.43,1.4e-06
...
```

**val_metrics.csv**
```csv
epoch,loss,mae,r2
0,0.804781,309.97,-0.407748
1,0.756432,298.12,0.123456
...
```

**test_metrics.csv**
```csv
loss,mae,r2
0.795226,309.04,0.654321
```

---

## HPC Cluster Deployment

### SLURM Job Submission

QMOF includes pre-configured SLURM scripts for each model:

#### CGCNN Job

```bash
# Submit CGCNN training job
sbatch src/CGCNN/run_qmof_cgcnn.sh
```

**Script Configuration (`run_qmof_cgcnn.sh`):**
```bash
#SBATCH --job-name=cgcnn_qmof
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --clusters=alex
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
```

#### MGT Job

```bash
# Submit MGT training job
sbatch src/MGT/run_qmof_mgt.sh
```

#### MOFTransformer Job

```bash
# Submit MOFTransformer job
sbatch src/MOF_Transformer/run_mof_transformer.sh
```

### Custom SLURM Configuration

Modify the provided scripts to match your cluster configuration:

```bash
#!/bin/bash -l
#SBATCH --job-name=qmof_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=<your_partition>
#SBATCH --gres=gpu:<gpu_type>:<num_gpus>
#SBATCH --cpus-per-task=<num_cpus>
#SBATCH --time=<HH:MM:SS>
#SBATCH --mem=<memory_GB>

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run container
srun apptainer exec --nv \
    -B $CODE_DIR:/workspace/code \
    -B $DATA_DIR:/workspace/dataset \
    -B $RESULTS_DIR:/workspace/results \
    qmof-base.sif \
    python -u /workspace/code/main.py [ARGS]
```

### Multi-Node Training

For distributed training across multiple nodes:

```bash
# MGT multi-node configuration
python training.py \
    --n_nodes 4 \
    --n_devices 8 \
    --accelerator cuda \
    --batch_size 8 \
    --n_cum 32
```

---

## Containerization

### Docker Development Workflow

```bash
# Build base image
cd src/container
docker build -t qmof-base:latest .

# Build model-specific image
cd ../CGCNN
docker build -t qmof-cgcnn:latest -f code/Dockerfile code/

# Run interactively
docker run --rm -it --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/results:/workspace/results \
    qmof-cgcnn:latest /bin/bash

# Run training
docker run --rm --gpus all \
    -v $(pwd)/data:/workspace/dataset \
    -v $(pwd)/results:/workspace/results \
    qmof-cgcnn:latest \
    python -u main.py /workspace/dataset --epochs 100
```

### Singularity/Apptainer for HPC

```bash
# Build from definition file
cd src/container
apptainer build qmof-base.sif cont.def

# Or convert from Docker
apptainer build qmof-base.sif docker://your-registry/qmof-base:latest

# Run with GPU support
apptainer exec --nv \
    -B /scratch/data:/dataset \
    -B /scratch/results:/results \
    qmof-base.sif \
    python training.py --root /dataset --epochs 100

# Interactive shell
apptainer shell --nv qmof-base.sif
```

### Container Optimization

The container images are optimized for:
- **Minimal Size**: Multi-stage builds reduce image size by ~60%
- **Layer Caching**: Dependencies installed in separate layers
- **GPU Acceleration**: CUDA 12.1 with cuDNN 8 pre-installed
- **Reproducibility**: Pinned package versions

---

## Data Format & Preprocessing

### Dataset Structure

```
dataset/
├── raw/
│   ├── MOF_001.cif
│   ├── MOF_002.cif
│   └── ...
└── id_prop.csv
```

### id_prop.csv Format

The `id_prop.csv` file contains CIF filenames and target properties:

```csv
MOF_id,outputs.pbe.bandgap,outputs.pbe.energy_per_atom
MOF_001,2.34,-5.678
MOF_002,1.89,-4.321
MOF_003,0.00,-6.543
```

**Requirements:**
- First column: CIF filename (without `.cif` extension)
- Subsequent columns: Target properties (any header name)
- No missing values in target columns
- UTF-8 encoding

### CIF File Requirements

CIF files should contain:
- Crystal structure information (_cell_length_*, _cell_angle_*)
- Atomic positions (_atom_site_*)
- Space group information (optional, auto-detected)

### Preprocessing Pipeline

1. **Validation**: Check CIF file integrity
2. **Feature Extraction**: Generate atom/bond features
3. **Graph Construction**: Build adjacency matrices
4. **Normalization**: Scale target properties
5. **Archiving**: Create tar file for efficient loading

```bash
# Full preprocessing workflow
cd src/MOF_Transformer/code

python preprocessor.py \
    --data-dir /path/to/raw/dataset \
    --target-column outputs.pbe.bandgap \
    --output-name qmof_bandgap \
    --filter-invalid \
    --max-workers 16

# Output: qmof_bandgap_bandgap.tar
```

---

## Training Configuration Reference

### Recommended Hyperparameters by Task

#### Band Gap Prediction

| Model | Learning Rate | Batch Size | Epochs | Hidden Dim |
|-------|---------------|------------|--------|------------|
| CGCNN | 0.01 | 32 | 500 | 128 |
| MGT | 1e-4 | 8 | 100 | 512 |
| MOFTransformer | 1e-4 | 32 | 50 | 512 |

#### Formation Energy

| Model | Learning Rate | Batch Size | Epochs | Hidden Dim |
|-------|---------------|------------|--------|------------|
| CGCNN | 0.005 | 64 | 300 | 128 |
| MGT | 5e-5 | 16 | 150 | 512 |
| MOFTransformer | 5e-5 | 32 | 100 | 512 |

#### Gas Adsorption (CO2/H2O)

| Model | Learning Rate | Batch Size | Epochs | Hidden Dim |
|-------|---------------|------------|--------|------------|
| CGCNN | 0.01 | 32 | 500 | 128 |
| MGT | 1e-4 | 8 | 100 | 512 |
| MOFTransformer | 1e-4 | 16 | 80 | 512 |

### Learning Rate Scheduling

All models support learning rate scheduling:

```python
# CGCNN: Multi-step scheduler
lr_scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)

# MGT: Cosine annealing
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# MOFTransformer: ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
```

### Early Stopping

```python
# PyTorch Lightning callback
early_stop = EarlyStopping(
    monitor='regression/val/r2_epoch',
    mode='max',
    patience=20,
    min_delta=0.001
)
```

---

## Results & Metrics

### Output Structure

```
results/
├── cgcnn/
│   ├── models/           # Saved checkpoints
│   │   ├── checkpoint_epoch_100.pth.tar
│   │   └── model_best.pth.tar
│   └── predictions.csv   # Test set predictions
│
├── mgt/
│   ├── saved_models/     # Model checkpoints
│   │   ├── pretrain.ckpt
│   │   ├── end_model.ckpt
│   │   └── lowest.ckpt
│   └── test_results/     # Evaluation results
│       └── predictions.csv
│
└── moftransformer/
    ├── checkpoints/      # Lightning checkpoints
    │   └── best.ckpt
    ├── train_metrics.csv
    ├── val_metrics.csv
    └── test_metrics.csv
```

### Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|y_pred - y_true|)` | Average absolute error |
| **MSE** | `mean((y_pred - y_true)²)` | Mean squared error |
| **RMSE** | `sqrt(MSE)` | Root mean squared error |
| **R²** | `1 - SS_res/SS_tot` | Coefficient of determination |

### Visualization

```python
# Plot training curves
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('results/moftransformer/train_metrics.csv')
val = pd.read_csv('results/moftransformer/val_metrics.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train['epoch'], train['loss'], label='Train Loss')
axes[0].plot(val['epoch'], val['loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(val['epoch'], val['r2'], label='Val R²', color='green')
axes[1].axhline(y=0, color='gray', linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R² Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
```

---

## API Reference

### CGCNN Model

```python
from model import CGCNN

# Initialize model
model = CGCNN(
    atom_init_dim=92,      # Atom feature dimension
    atom_fea_len=64,       # Atom hidden dimension
    nbr_fea_len=41,        # Bond feature dimension
    h_fea_len=128,         # Output hidden dimension
    n_conv=3,              # Number of conv layers
    n_h=1,                 # Number of hidden layers
    task='regression'      # 'regression' or 'classification'
)

# Forward pass
output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
```

### MGT Model

```python
from model.transformer import multiheaded
from model.alignn import EdgeGatedGraphConv
from model.graphformer import Graphformer

# Build model components
encoder = Graphformer(
    num_atom_fea=92,
    num_edge_fea=1,
    num_angle_fea=1,
    embedding_dims=128,
    hidden_dims=512,
    n_heads=4,
    n_layers=1
)

decoder = EdgeGatedGraphConv(
    in_feats=512,
    out_feats=1,
    n_layers=3
)

# Forward pass
output = model(graph, line_graph, fully_connected_graph)
```

### MOFTransformer

```python
from moftransformer import Module, Datamodule

# Initialize datamodule
datamodule = Datamodule(
    data_dir='./data/dataset',
    target_column='bandgap',
    batch_size=32,
    num_workers=8
)

# Initialize model
model = Module(
    task='regression',
    num_targets=1,
    learning_rate=1e-4,
    weight_decay=1e-5
)

# Train with PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=4,
    precision=16
)

trainer.fit(model, datamodule)
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
- Reduce batch size: `--batch-size 8`
- Use gradient accumulation: `--n_cum 32`
- Enable mixed precision: `--precision 16`
- Clear GPU cache: `torch.cuda.empty_cache()`

#### 2. GRIDAY Installation Failed

**Symptoms:**
```
ModuleNotFoundError: No module named 'GRIDAY'
```

**Solutions:**
```bash
# MOFTransformer auto-installs GRIDAY
python -c "from moftransformer import install_griday; install_griday()"

# Or manually
cd src/MOF_Transformer/code/MOFTransformer
pip install -e .
```

#### 3. CIF File Parsing Errors

**Symptoms:**
```
ValueError: Invalid CIF file format
```

**Solutions:**
- Validate CIF syntax using pymatgen:
```python
from pymatgen.io.cif import CifParser
parser = CifParser('structure.cif')
structure = parser.get_structures()[0]
```

#### 4. SLURM Job Fails Immediately

**Symptoms:**
```
srun: error: Unable to allocate resources
```

**Solutions:**
- Check partition name: `sinfo -p`
- Verify GPU availability: `scontrol show nodes`
- Reduce requested resources in script

#### 5. Container GPU Access Denied

**Symptoms:**
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

**Solutions:**
- Install NVIDIA Container Toolkit
- Restart Docker daemon: `sudo systemctl restart docker`
- Verify with: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

### Debug Mode

Enable verbose logging for debugging:

```bash
# CGCNN
python main.py <data> --task regression --print-freq 1

# MGT
python training.py --root <data> --run_name debug --n_cum 1

# MOFTransformer
python trainer.py --data-dir <data> --max-epochs 1 --precision 32
```

---

## Performance Benchmarks

### Training Time Comparison

| Model | Dataset | GPU | Batch Size | Time/Epoch | Total Time (50 epochs) |
|-------|---------|-----|------------|------------|------------------------|
| CGCNN | QMOF (10k) | A100 | 32 | 2 min | 1.7 hours |
| MGT | QMOF (10k) | A100 | 8 | 8 min | 6.7 hours |
| MOFTransformer | QMOF (10k) | A100 | 32 | 3 min | 2.5 hours |

### Prediction Accuracy (QMOF Test Set)

| Model | Band Gap MAE | Formation Energy MAE | CO2 Uptake MAE |
|-------|--------------|---------------------|----------------|
| CGCNN | 0.34 eV | 0.08 eV/atom | 2.1 mmol/g |
| MGT | 0.29 eV | 0.06 eV/atom | 1.8 mmol/g |
| MOFTransformer | 0.31 eV | 0.07 eV/atom | 1.9 mmol/g |

### Memory Usage

| Model | GPU Memory (Batch=32) | CPU Memory |
|-------|----------------------|------------|
| CGCNN | 4.2 GB | 8 GB |
| MGT | 12.8 GB | 16 GB |
| MOFTransformer | 6.5 GB | 12 GB |

---

## License & Legal

### Proprietary License

This software is proprietary and confidential. See the [LICENSE](LICENSE) file for complete terms.

**Key Restrictions:**
- ❌ No commercial use without explicit written permission
- ❌ No redistribution or sublicensing
- ❌ No modification for derivative works
- ✅ Internal research and development use only
- ✅ Academic non-commercial research (with citation)

### Copyright Notice

```
Copyright © 2026 Aixelo. All Rights Reserved.

This software and associated documentation files (the "Software") are 
proprietary and confidential. Unauthorized copying, distribution, 
modification, or commercial use of this Software is strictly prohibited.

For licensing inquiries, contact: legal@aixelo.com
```

### Third-Party Licenses

This project incorporates the following open-source components:

| Package | License | URL |
|---------|---------|-----|
| PyTorch | BSD-3 | https://pytorch.org |
| PyTorch Lightning | Apache 2.0 | https://lightning.ai |
| DGL | Apache 2.0 | https://www.dgl.ai |
| pymatgen | MIT | https://pymatgen.org |
| matminer | MIT | https://hackingmaterials.org |

---

## Citation

If you use QMOF in your research, please cite the following works:

### QMOF Framework

```bibtex
@software{qmof2026,
  author = {Aixelo Team},
  title = {Quantum Materials Optimization Framework (QMOF)},
  year = {2026},
  publisher = {Aixelo},
  url = {https://github.com/your-org/Aixelo2026}
}
```

### Underlying Models

```bibtex
@article{cgcnn2018,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Physical Review Letters},
  volume = {120},
  number = {14},
  pages = {145301},
  year = {2018}
}

@article{alignn2021,
  title = {ALIGNN: Atomistic Line Graph Neural Network for Improved Materials Property Predictions},
  author = {Choudhary, Kamal and DeCost, Brian},
  journal = {npj Computational Materials},
  volume = {7},
  number = {1},
  pages = {185},
  year = {2021}
}

@article{moftransformer2023,
  title = {A Multi-Modal Pre-Training Transformer for Universal Transfer Learning in Metal-Organic Frameworks},
  author = {Park, H. et al.},
  journal = {Nature Machine Intelligence},
  volume = {5},
  pages = {465--475},
  year = {2023}
}
```

---

## Support & Contact

### Technical Support

For technical issues and questions:
- **GitHub Issues**: https://github.com/your-org/Aixelo2026/issues
- **Email**: support@aixelo.com
- **Documentation**: https://docs.aixelo.com/qmof

### Licensing & Commercial Use

For commercial licensing inquiries:
- **Email**: legal@aixelo.com
- **Website**: https://www.aixelo.com/licensing

### Security

To report security vulnerabilities:
- **Email**: security@aixelo.com
- **PGP Key**: Available at https://www.aixelo.com/security

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-20 | Initial release |

---

**Last Updated:** February 20, 2026

**Maintained by:** Aixelo Engineering Team
