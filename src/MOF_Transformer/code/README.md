# MOFTransformer - Preprocessor and Trainer

This directory contains the refactored MOFTransformer training pipeline, split into separate preprocessor and trainer modules with improved metrics tracking.

## Overview

The code has been restructured to:
1. **Preprocessor** (`preprocessor.py`): Prepares raw MOF datasets for training
2. **Trainer** (`trainer.py`): Trains and evaluates models on preprocessed data
3. **Metrics Logger** (`metrics_logger.py`): Handles separate CSV files for train/val/test metrics

## Key Features

### Preprocessor
- Accepts flexible `id_prop.csv` format with any header name for the first column
- Supports multiple target columns - you specify which one to use
- Creates tar archive with preprocessed data
- Filters `id_prop.csv` to contain only CIF filenames and the selected target column

### Trainer
- Validation after each epoch with R² metric display
- Separate CSV files for train, validation, and test metrics
- Fixed metrics calculation and proper logging
- Model checkpointing based on best validation R²

## Installation

Ensure you have the required dependencies installed:

```bash
cd MOFTransformer
pip install -e .
cd ..
pip install pandas numpy scikit-learn pytorch-lightning tensorboard
```

## Usage

### Step 1: Preprocess the Dataset

> **Note for HPC Cluster Users:** It is recommended to run the preprocessor **locally** on your machine. After preprocessing, upload the generated `.tar` archive to the HPC cluster and use only the `trainer.py` there. This approach saves cluster resources and avoids unnecessary preprocessing overhead on shared systems.

```bash
python preprocessor.py \
    --data-dir /path/to/dataset/ \
    --target-column energy_per_atom \
    --output-name qmof_preprocessed
```

**Arguments:**
- `--data-dir`: Path to dataset directory containing `id_prop.csv` and `raw/` folder
- `--target-column`: Name of the target column in `id_prop.csv` to predict
- `--output-name`: Base name for output tar archive
- `--train-fraction`: Fraction of data for training (default: 0.8)
- `--test-fraction`: Fraction of data for testing (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)

**Output:** Creates `{output-name}_{target}.tar` archive containing:
```
dataset/
├── raw/
│   ├── *.cif files
│   ├── raw_{target}.json
│   └── id_prop.csv (filtered: 2 columns only)
└── processed/
    ├── train/, val/, test/ directories
    └── train_{target}.json, val_{target}.json, test_{target}.json
```

### Step 2: Extract the Archive

```bash
tar -xvf qmof_preprocessed_energy_per_atom.tar -C ./extracted_data/
```

### Step 3: Train the Model

```bash
python trainer.py \
    --data-dir ./extracted_data/dataset/ \
    --target-column energy_per_atom \
    --log-dir ./results/ \
    --max-epochs 50
```

**Arguments:**
- `--data-dir`: Path to extracted preprocessed dataset directory
- `--target-column`: Name of the target column (must match preprocessed data)
- `--log-dir`: Directory to save logs, checkpoints, and metrics
- `--max-epochs`: Maximum number of training epochs (default: 10)
- `--batch-size`: Total batch size (default: 16)
- `--per-gpu-batchsize`: Batch size per GPU (default: 4)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-2)
- `--load-path`: Path to pretrained model checkpoint (optional)
- `--seed`: Random seed (default: 0)
- `--precision`: Training precision - 16, 32, or 64 (default: 32)
- `--accelerator`: Accelerator type - 'auto', 'cpu', 'gpu' (default: 'auto')

**Output:**
```
results/
├── pretrained_mof_seed0_from_/
│   ├── checkpoints/
│   │   └── best.ckpt
│   └── events.out.tfevents.* (TensorBoard logs)
├── train_metrics.csv
├── val_metrics.csv
└── test_metrics.csv
```

## Metrics CSV Format

### train_metrics.csv
| epoch | loss | mae | learning_rate |
|-------|------|-----|---------------|
| 0 | 0.620595 | 289.28 | 1.4e-06 |
| 1 | 0.984702 | 342.87 | 2.9e-06 |
| ... | ... | ... | ... |

### val_metrics.csv
| epoch | loss | mae | r2 |
|-------|------|-----|-----|
| 0 | 0.804781 | 309.97 | -0.407748 |
| 1 | 0.762935 | 271.38 | -0.123456 |
| ... | ... | ... | ... |

### test_metrics.csv
| loss | mae | r2 |
|------|-----|-----|
| 0.795226 | 309.04 | 0.654321 |

## Validation Output

During training, you will see validation metrics after each epoch:

```
============================================================
Validation Epoch 5:
  Loss: 0.762935
  MAE:  271.3800
  R²:   0.654321
============================================================
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

## Changes from Original Code

### module_utils.py
- Added `R2Score` class for R² metric calculation
- Modified `set_metrics()` to initialize R² metric for regression tasks
- Updated `epoch_wrapup()` to:
  - Compute and log R² score at the end of each validation epoch
  - Print R² score to console
  - Use R² as the main metric for model selection

### objectives.py
- Modified `compute_regression()` to update R² score metric during training/validation

### module.py
- Added `val_logits` and `val_labels` for validation predictions storage
- Updated `validation_step()` to collect predictions and labels
- Updated `on_validation_start()` to reset validation storage

### New Files
- `preprocessor.py`: Dataset preprocessing script
- `trainer.py`: Training script with improved metrics
- `metrics_logger.py`: Utility for saving separate metric CSV files

## Troubleshooting

### GRIDAY Installation
If GRIDAY is not found, the preprocessor will attempt to install it automatically. You can also install it manually:

```bash
moftransformer install-griday
```

### Missing CIF Files
Ensure all CIF IDs in `id_prop.csv` have corresponding `.cif` files in the `raw/` directory.

### Out of Memory
Reduce `--batch-size` or `--per-gpu-batchsize` if you encounter CUDA out of memory errors.

### Low R² Score
- Increase `--max-epochs`
- Adjust `--learning-rate`
- Ensure data quality and sufficient training samples

## License

This code is part of the MOFTransformer project. See the original repository for license information.
