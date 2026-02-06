# Advanced Materials Science Neural Networks for HPC

This repository contains implementations of state-of-the-art neural networks for materials science applications, specifically designed for High Performance Computing (HPC) environments. The project includes Crystal Graph Convolutional Neural Networks (CGCNN) and Molecular Graph Transformers (MGT) for predicting material properties.

## Table of Contents
- [Overview](#overview)
- [Neural Networks](#neural-networks)
- [Datasets](#datasets)
- [Container Setup](#container-setup)
- [Running the Models](#running-the-models)
- [HPC Environment Configuration](#hpc-environment-configuration)
- [Parameter Configuration](#parameter-configuration)
- [Results and Evaluation](#results-and-evaluation)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements two advanced neural network architectures for materials science applications:

1. **Crystal Graph Convolutional Neural Network (CGCNN)**: A convolutional neural network designed for crystal structure property prediction using graph representations of crystals.
2. **Molecular Graph Transformer (MGT)**: A transformer-based architecture for molecular property prediction using graph neural networks.

Both networks are optimized for HPC environments and utilize GPU acceleration for efficient training and inference.

## Neural Networks

### Crystal Graph Convolutional Neural Network (CGCNN)

CGCNN is a deep learning model that represents crystalline materials as graphs, where atoms are nodes and bonds are edges. The model learns material properties by processing these graph representations through convolutional layers.

#### Architecture Features:
- Graph convolutional layers for processing atomic structures
- Crystal graph representation with atom features and bond distances
- Multiple convolutional layers with pooling operations
- Regression and classification capabilities

#### Key Components:
- `model.py`: Contains the CrystalGraphConvNet implementation
- `data.py`: Handles CIF data loading and preprocessing
- `main.py`: Main training and evaluation pipeline

#### Parameters:
- `--task`: Task type (`regression` or `classification`)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Mini-batch size (default: 256)
- `--lr`: Learning rate (default: 0.01)
- `--atom-fea-len`: Hidden atom features in conv layers (default: 64)
- `--h-fea-len`: Hidden features after pooling (default: 128)
- `--n-conv`: Number of conv layers (default: 3)
- `--n-h`: Number of hidden layers after pooling (default: 1)
- `--optim`: Optimizer choice (`SGD` or `Adam`)

### Molecular Graph Transformer (MGT)

MGT is a transformer-based architecture that applies attention mechanisms to molecular graph structures. It combines graph neural networks with transformer architectures for enhanced molecular property prediction.

#### Architecture Features:
- Transformer encoder layers with multi-head attention
- Graph neural network components (ALIGNN layers)
- Multi-scale graph representations (local and global graphs)
- Positional encoding for spatial relationships

#### Key Components:
- `model/transformer.py`: Transformer implementation
- `model/graphformer.py`: Graph transformer architecture
- `model/alignn.py`: Edge-gated graph convolution layers
- `training.py`: Training pipeline with Lightning Fabric
- `utils/datasets.py`: Dataset utilities

#### Parameters:
- `--root`: Root directory for datasets
- `--model_path`: Directory to save trained models
- `--save_dir`: Directory to save test results
- `--accelerator`: Device type (`cuda`, `gpu`, `cpu`, etc.)
- `--n_devices`: Number of GPUs/CPUs (default: 8)
- `--batch_size`: Batch size for training (default: 2)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.0001)
- `--num_atom_fea`: Length of atom feature vector (default: 90)
- `--num_edge_fea`: Length of edge feature vector (default: 1)
- `--num_angle_fea`: Length of angle feature vector (default: 1)
- `--num_pe_fea`: Length of positional encoding vector (default: 10)
- `--embedding_dims`: Dimension of embedding layer (default: 128)
- `--hidden_dims`: Dimensions of hidden layers (default: 512)
- `--out_dims`: Length of output vector (default: 3)
- `--num_layers`: Number of encoder layers (default: 1)
- `--n_mha`: Number of attention layers per encoder (default: 1)
- `--n_alignn`: Number of ALIGNN convolutions per encoder (default: 3)
- `--n_gnn`: Number of GNN convolutions per encoder (default: 3)

## Datasets

The project utilizes several materials science datasets:

### QMOF Dataset
- **Format**: CIF files for crystal structures
- **Content**: Quantum mechanical properties of metal-organic frameworks
- **Usage**: Used by both CGCNN and MGT models
- **Location**: `/home/vault/b192aa/b192aa41/datasets/qmof_cif.tar` (for MGT)
- **Location**: `/home/vault/b192aa/b192aa41/datasets/qmof_pkl.tar` (for CGCNN)

### ODAC CO₂ Dataset
- **Format**: Archive containing processed molecular data
- **Content**: Properties related to CO₂ adsorption
- **Usage**: Used by both CGCNN and MGT models
- **Location**: `/home/vault/b192aa/b192aa41/datasets/odac_co2.tar`

### ODAC H₂O Dataset
- **Format**: Archive containing processed molecular data
- **Content**: Properties related to water adsorption
- **Usage**: Used by both CGCNN and MGT models
- **Location**: `/home/vault/b192aa/b192aa41/datasets/odac_h2o.tar`

## Container Setup

The project uses Apptainer/Singularity containers for consistent deployment across HPC systems.

### Building the Container

The container definition file is located at `container/cont.def`:

```apptainer
Bootstrap: docker
From: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

To build the container:

```bash
# Navigate to the container directory
cd container/

# Build the container (requires sudo privileges)
sudo apptainer build cont.sif cont.def
```

### Container Dependencies

The container includes the following key dependencies:

- **Python 3.x** with scientific computing packages
- **PyTorch 2.1.2** with CUDA 12.1 support
- **DGL 2.1.0** with CUDA 12.1 support
- **RDKit** for cheminformatics
- **PyTorch Lightning 2.1.3** for distributed training
- **Materials science libraries**: pymatgen, ase, matminer, spglib
- **Data processing libraries**: pandas, numpy, scipy

## Running the Models

### Prerequisites

Before running the models, ensure:
1. The container is built (`container/cont.sif`)
2. Required datasets are accessible
3. SLURM job scheduler is available (for HPC)
4. NVIDIA GPUs are available (for GPU acceleration)

### CGCNN Scripts

#### Running CGCNN on QMOF Dataset
```bash
cd CGCNN/
sbatch run_cgcnn.sh
```

#### Running CGCNN on ODAC CO₂ Dataset
```bash
cd CGCNN/
sbatch run_odac_co2.sh
```

#### Running CGCNN on ODAC H₂O Dataset
```bash
cd CGCNN/
sbatch run_odac_h2o.sh
```

### MGT Scripts

#### Running MGT on QMOF Dataset
```bash
cd MGT/
sbatch run_mgt.sh
```

#### Running MGT on ODAC CO₂ Dataset
```bash
cd MGT/
sbatch run_odac_co2.sh
```

#### Running MGT on ODAC H₂O Dataset
```bash
cd MGT/
sbatch run_odac_h2o.sh
```

### Script Parameters

Each script is configured for the Alex HPC cluster with the following parameters:

- **Cluster**: Alex
- **Partition**: a100 (A100 GPU partition)
- **GPU**: 1× A100 GPU
- **CPU cores**: 16 cores per task
- **Memory**: Automatically allocated based on system configuration
- **Time limits**: Vary by dataset (1-3 hours)

## HPC Environment Configuration

### SLURM Job Configuration

All scripts use SLURM directives optimized for the Alex HPC cluster:

```bash
#SBATCH --job-name=<model_name>_<dataset>
#SBATCH --output=/path/to/logs/%x_%j.out
#SBATCH --error=/path/to/logs/%x_%j.err
#SBATCH --clusters=alex
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=<time_limit>
#SBATCH --export=NONE
```

### Apptainer Execution

Models run inside the container using Apptainer with the following bindings:

- Code directory: `/workspace/code`
- Dataset directory: `/workspace/dataset`
- Results directory: `/workspace/results`

Example execution command:
```bash
srun apptainer exec --nv \
    -B "$CODE_DIR:/workspace/code" \
    -B "$DATA_ROOT:/workspace/dataset" \
    -B "$RESULTS_DIR:/workspace/results" \
    --pwd /workspace/results \
    "$CONTAINER" \
    python -u /workspace/code/main.py \
        /workspace/dataset \
        --task regression \
        --epochs 500
```

## Parameter Configuration

### CGCNN Parameter Tuning

For optimal performance, consider adjusting these parameters based on your dataset:

- `--epochs`: Increase for larger datasets (e.g., 500 for QMOF)
- `--batch-size`: Adjust based on GPU memory (reduce for larger models)
- `--lr`: Lower values (e.g., 0.001) for fine-tuning
- `--n-conv`: Increase for more complex structural features
- `--atom-fea-len`: Increase for richer atomic representations

### MGT Parameter Tuning

For optimal performance, consider adjusting these parameters:

- `--epochs`: Increase for larger datasets (e.g., 100+ for better convergence)
- `--batch_size`: Adjust based on GPU memory (typically 1-8)
- `--lr`: Start with 0.0001, adjust based on convergence
- `--num_layers`: Increase for deeper representations
- `--n_alignn`: Adjust number of ALIGNN convolutions
- `--n_gnn`: Adjust number of GNN convolutions
- `--embedding_dims`: Increase for richer embeddings
- `--hidden_dims`: Increase for more capacity

## Results and Evaluation

### Output Files

After execution, models generate the following outputs:

- **Model checkpoints**: Saved in `results/<model>/job_<job_id>/saved_models/`
- **Training logs**: Available in `results/<model>/job_<job_id>/`
- **Predictions**: Stored in `results/<model>/job_<job_id>/predictions.csv`
- **Evaluation metrics**: Included in training logs and separate result files

### Metrics

- **CGCNN**: Mean Absolute Error (MAE) for regression tasks
- **MGT**: Mean Absolute Error (MAE) and MSE loss for regression tasks
- **Classification**: Accuracy, Precision, Recall, F1-score, and AUC for classification tasks

## Performance Optimization

### Memory Management

- Monitor GPU memory usage with `nvidia-smi`
- Adjust batch sizes to maximize GPU utilization without exceeding memory
- Use gradient accumulation (`--n_cum` in MGT) for larger effective batch sizes

### Distributed Training

- MGT supports multi-GPU training through Lightning Fabric
- Configure `--n_devices` to utilize multiple GPUs
- Adjust learning rates when scaling to multiple devices

### Data Pipeline Optimization

- Pre-process datasets to `.pkl` format for faster loading (CGCNN)
- Use appropriate data loaders with multiprocessing
- Optimize data transfer between CPU and GPU

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory Errors**:
   - Reduce batch size
   - Clear GPU cache: `torch.cuda.empty_cache()`
   - Use gradient accumulation instead of large batches

2. **Container Build Failures**:
   - Ensure Docker daemon is running
   - Check internet connectivity for downloading base images
   - Verify sufficient disk space

3. **Dataset Loading Issues**:
   - Verify dataset paths in scripts
   - Check dataset integrity and format
   - Ensure proper permissions for dataset files

4. **SLURM Submission Errors**:
   - Check partition availability
   - Verify resource requests are within limits
   - Confirm account allocation and quotas

### Debugging Tips

- Enable verbose logging by modifying scripts
- Test locally with small datasets before HPC submission
- Monitor job progress through SLURM logs
- Use `squeue` to monitor job status

## Contributing

For contributions to this project, please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make changes following the existing code style
4. Test changes on a small dataset
5. Submit a pull request with detailed description

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

This work builds upon the foundational research in materials science machine learning, particularly the CGCNN and Graph Transformer architectures. Special thanks to the HPC resources provided by the Alex cluster for enabling large-scale experiments.