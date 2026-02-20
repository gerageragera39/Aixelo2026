# Quantum Materials Optimization Framework (QMOF) Neural Network Training Guide

This folder contains Docker configurations and datasets for training two neural networks: CGCNN and MGT, designed for quantum materials property prediction.

## Table of Contents
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Dataset Setup](#dataset-setup)
- [CGCNN Training](#cgcnn-training)
- [MGT Training](#mgt-training)

## Folder Structure

Your current folder contains the following files:

```
Docker_networks\
├───qmof_dataset\         # DATASET 
│
├───CGCNN\                # CGCNN model directory
│   ├───Dockerfile        # Docker configuration for CGCNN
│   ├───requirements.txt  # Python dependencies for CGCNN
│   ├───main.py           # Main training script for CGCNN
│   ├───model.py          # Model definition for CGCNN
│   ├───data.py           # Data processing for CGCNN
│   └───...               # Extra
│
└───MGT\                  # MGT model directory
    ├───Dockerfile        # Docker configuration for MGT
    ├───requirements.txt  # Python dependencies for MGT
    ├───training.py       # Main training script for MGT
    ├───run.py            # Script to run MGT model
    ├───model\            # Model components for MGT
    ├───modules\          # Module definitions for MGT
    └───utils\            # Utility functions for MGT
```

### Download Links

All files in this repository can be downloaded from the Hugging Face dataset repository:
https://huggingface.co/datasets/hermanhugging/qmof_project/tree/main

## Prerequisites

Before starting the training, you need to install Docker Desktop:

1. Download and install Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Make sure Docker is running before proceeding with the training
3. Ensure you have NVIDIA Container Toolkit installed if using GPU acceleration

## Dataset Setup

Unzip the dataset archive to make it available for training:

```bash
unzip qmof_dataset.zip
```

This will create a `qmof_dataset` directory containing the training data.

## CGCNN Training

### Building the Docker Image

Navigate to the CGCNN directory and build the Docker image:

```bash
cd CGCNN
docker build -t cgcnn-app .
```

### Saving the Docker Image (Optional)

After building, you can save the Docker image to a .tar file:

```bash
docker save -o cgcnn_image.tar cgcnn-app
```

### Running CGCNN Training

Execute the training with the following command:

```bash
docker run --rm --gpus all \
  -v "../qmof_dataset/dataset:/dataset" \
  -v "$(pwd)/cgcnn/models:/app/models" \
  -v "$(pwd)/cgcnn/results:/app/results" \
  cgcnn-app \
  python -u main.py /dataset --task regression
```

### CGCNN Training Parameters

The CGCNN training script accepts the following parameters:

#### Basic Options
- `data_options`: Dataset options, started with the path to root directory, then other options
- `--task`: Choose between 'regression' or 'classification' tasks (default: regression)
- `--disable-cuda`: Disable CUDA (GPU acceleration)
- `-j, --workers`: Number of data loading workers (default: 0)
- `--epochs`: Number of total epochs to run (default: 30)
- `--start-epoch`: Manual epoch number (useful on restarts) (default: 0)
- `-b, --batch-size`: Mini-batch size (default: 256)
- `--lr, --learning-rate`: Initial learning rate (default: 0.01)
- `--lr-milestones`: Milestones for scheduler (default: [100])
- `--momentum`: Momentum value (default: 0.9)
- `--weight-decay, --wd`: Weight decay (default: 0)
- `--print-freq, -p`: Print frequency (default: 10)
- `--resume`: Path to latest checkpoint (default: none)

#### Training Data Options
- `--train-ratio`: Fraction of training data to be loaded (mutually exclusive with --train-size)
- `--train-size`: Number of training data samples to be loaded (mutually exclusive with --train-ratio)

#### Validation Data Options
- `--val-ratio`: Percentage of validation data to be loaded (default: 0.1) (mutually exclusive with --val-size)
- `--val-size`: Number of validation data samples to be loaded (default: 1000) (mutually exclusive with --val-ratio)

#### Test Data Options
- `--test-ratio`: Percentage of test data to be loaded (default: 0.1) (mutually exclusive with --test-size)
- `--test-size`: Number of test data samples to be loaded (default: 1000) (mutually exclusive with --test-size)

#### Model Architecture Options
- `--optim`: Choose an optimizer ('SGD' or 'Adam') (default: SGD)
- `--atom-fea-len`: Number of hidden atom features in conv layers (default: 64)
- `--h-fea-len`: Number of hidden features after pooling (default: 128)
- `--n-conv`: Number of conv layers (default: 3)
- `--n-h`: Number of hidden layers after pooling (default: 1)

#### Additional Options
- `--disable-save-torch`: Do not save CIF PyTorch data as .pkl files
- `--clean-torch`: Clean CIF PyTorch data .pkl files

## MGT Training

### Building the Docker Image

Navigate to the MGT directory and build the Docker image:

```bash
cd MGT
docker build -t mgt-project .
```

### Saving the Docker Image (Optional)

After building, you can save the Docker image to a .tar file:

```bash
docker save -o mgt_image.tar mgt-project
```

### Running MGT Training

Execute the training with the following command:

```bash
docker run --gpus all -it --rm \
  -v "../qmof_dataset/dataset":/data \
  -v "$(pwd)/mgt/saved_models":/app/saved_models \
  -v "$(pwd)/mgt/results":/app/results \
  mgt-project \
  python training.py \
  --root /data \
  --model_path /app/saved_models \
  --save_dir /app/results \
  --accelerator cuda \
  --n_devices 1 \
  --num_atom_fea 92 \
  --batch_size 1 \
  --n_cum 16 \
  --out_dims 1 \
  --epochs 10
```

### MGT Training Parameters

The MGT training script accepts the following parameters:

#### Device Configuration
- `--n_devices`: Number of GPUs/CPUs that the code has access to (default: 8)
- `--n_nodes`: Number of nodes/computers on which the model is being trained (default: 1)
- `--accelerator`: Device type for training ('cpu', 'gpu', 'mps', 'cuda', 'tpu') (default: cuda)

#### Save and Load Arguments
- `--root`: Root directory for all datasets (required)
- `--model_path`: Directory in which to save the trained model (required)
- `--run_name`: Name of run for logging purposes (default: None)
- `--save_dir`: Directory in which to save the test results (default: None)
- `--pretrain_model`: Name with which to save the pretrained model (default: 'pretrain.ckpt')
- `--final_model`: Name with which to save the final model (default: 'end_model.ckpt')
- `--lowest_model`: Name with which to save the model with the best performance (default: 'lowest.ckpt')
- `--load_model`: Whether there is a model to be loaded (0: no model loading, 1: load pretrained, 2: load checkpoint) (default: 0)
- `--out_names`: Names of the outputs [for logging purposes only]

#### Training Arguments
- `--n_cum`: Number of batches to accumulate the error for (default: 8)
- `--batch_size`: Batch size for training (default: 2)
- `--train_split`: Fraction of dataset to be used for training (default: 0.8)
- `--val_split`: Fraction of dataset to be used for validation (default: 0.2)
- `--epochs`: Number of epochs to train for (default: 10)
- `--begin_epoch`: Set to restart training from a specific epoch (default: 1)
- `--lr`: Learning rate (default: 0.0001)
- `--decay`: Weight decay for the optimizers (default: 1e-5)

#### Model and Dataset Arguments
- `--process`: Whether the graphs for the structures/molecules need to be created during dataset loading (0: False, 1: True) (default: 1)
- `--max_nei_num`: Maximum number of neighbors allowed for each atom in the local graph (default: 12)
- `--local_radius`: Radius used to form the local graph (default: 8)
- `--periodic`: Whether the input structure is a periodic structure or not (0: False, 1: True) (default: 1)
- `--periodic_radius`: Radius used to form the fully connected graph (default: 12)
- `--num_atom_fea`: Length of feature vector for atoms (default: 90)
- `--num_edge_fea`: Length of feature vector for edges in local graph (default: 1)
- `--num_angle_fea`: Length of feature vector for edges in line graph (default: 1)
- `--num_pe_fea`: Length of feature vector for atom's positional encoding (default: 10)
- `--num_clmb_fea`: Length of feature vector for edges in fully connected graph (default: 1)
- `--num_edge_bins`: Number of bins for RBF expansion of edges in local graph (default: 80)
- `--num_angle_bins`: Number of bins for RBF expansion of edges in line graph (default: 40)
- `--num_clmb_bins`: Number of bins for RBF expansion of edges in fully connected graph (default: 120)
- `--embedding_dims`: Dimension of embedding layer (default: 128)
- `--hidden_dims`: Dimensions of each hidden layer (default: 512)
- `--out_dims`: Length of output vector of the network (default: 3)
- `--num_layers`: Number of encoders in the network (default: 1)
- `--n_mha`: Number of attention layers in each encoder (default: 1)
- `--n_alignn`: Number of graph convolutions in each encoder (default: 3)
- `--n_gnn`: Number of graph convolutions in each encoder (default: 3)
- `--n_heads`: Number of attention heads (default: 4)
- `--residual`: Whether to add residuality to the network or not (0: False, 1: True) (default: 1)