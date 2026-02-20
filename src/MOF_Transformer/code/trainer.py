"""
MOFTransformer Trainer

Trains and evaluates MOFTransformer models on preprocessed datasets.
Supports validation after each epoch with R² metric display and saves
metrics to separate CSV files.

Author: MOFTransformer Team
Date: 2026-02-19
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add MOFTransformer to path
sys.path.insert(0, str(Path(__file__).parent / "MOFTransformer"))

from moftransformer.datamodules.datamodule import Datamodule
from moftransformer.modules.module import Module
from moftransformer.modules.module_utils import Normalizer
from moftransformer.utils.validation import get_valid_config, get_num_devices

# Import metrics logger
sys.path.insert(0, str(Path(__file__).parent))
from metrics_logger import MetricsLogger, calculate_r2_score, calculate_mae, calculate_mse


class MetricsCollectionCallback(pl.Callback):
    """
    Callback to collect metrics during training for later saving to CSV.
    
    This callback collects training and validation metrics at the end of each epoch
    and stores them for later export to separate CSV files.
    """
    
    def __init__(self):
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_r2 = float('-inf')
        self.best_epoch = 0
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Collect training metrics at the end of each epoch."""
        metrics = {
            'epoch': pl_module.current_epoch,
            'loss': trainer.callback_metrics.get('regression/train/loss_epoch', None),
            'mae': trainer.callback_metrics.get('regression/train/mae_epoch', None),
            'learning_rate': trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None
        }
        
        # Extract scalar values from tensors
        for key in ['loss', 'mae']:
            if metrics[key] is not None and hasattr(metrics[key], 'item'):
                metrics[key] = metrics[key].item()
        
        self.train_metrics.append(metrics)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Collect validation metrics and display R² score."""
        metrics = {
            'epoch': pl_module.current_epoch,
            'loss': trainer.callback_metrics.get('regression/val/loss_epoch', None),
            'mae': trainer.callback_metrics.get('regression/val/mae_epoch', None),
            'r2': trainer.callback_metrics.get('regression/val/r2_epoch', None)
        }
        
        # Extract scalar values from tensors
        for key in ['loss', 'mae', 'r2']:
            if metrics[key] is not None and hasattr(metrics[key], 'item'):
                metrics[key] = metrics[key].item()
        
        self.val_metrics.append(metrics)
        
        # Display R² score prominently
        if metrics['r2'] is not None:
            print(f"\n{'='*60}")
            print(f"Validation Epoch {pl_module.current_epoch}:")
            print(f"  Loss: {metrics['loss']:.6f}" if metrics['loss'] else "  Loss: N/A")
            print(f"  MAE:  {metrics['mae']:.4f}" if metrics['mae'] else "  MAE:  N/A")
            print(f"  R²:   {metrics['r2']:.6f}")
            print(f"{'='*60}\n")
            
            # Track best epoch
            if metrics['r2'] > self.best_val_r2:
                self.best_val_r2 = metrics['r2']
                self.best_epoch = pl_module.current_epoch
    
    def save_metrics(self, log_dir: Path) -> Dict[str, Path]:
        """
        Save collected metrics to CSV files.
        
        Parameters
        ----------
        log_dir : Path
            Directory to save CSV files.
            
        Returns
        -------
        dict
            Dictionary mapping phase names to CSV file paths.
        """
        saved_files = {}
        
        # Save training metrics
        if self.train_metrics:
            train_df = pd.DataFrame(self.train_metrics)
            train_path = log_dir / "train_metrics.csv"
            train_df.to_csv(train_path, index=False)
            saved_files['train'] = train_path
            print(f"Saved training metrics to {train_path}")
        
        # Save validation metrics
        if self.val_metrics:
            val_df = pd.DataFrame(self.val_metrics)
            val_path = log_dir / "val_metrics.csv"
            val_df.to_csv(val_path, index=False)
            saved_files['val'] = val_path
            print(f"Saved validation metrics to {val_path}")
        
        return saved_files


def load_preprocessed_config(data_dir: Path, target_column: str) -> Dict[str, Any]:
    """
    Load configuration from preprocessed dataset.

    Parameters
    ----------
    data_dir : Path
        Path to preprocessed dataset directory.
    target_column : str
        Name of the target column.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    # Check for required files
    required_files = [
        data_dir / f"train_{target_column}.json",
        data_dir / f"val_{target_column}.json",
        data_dir / f"test_{target_column}.json"
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required JSON files: {missing_files}. "
            "Please ensure the dataset was preprocessed correctly."
        )

    # Load target statistics from train data
    mean = 0.0
    std = 1.0

    # First try to load from id_prop.csv
    id_prop_path = data_dir / "id_prop.csv"
    if id_prop_path.exists():
        try:
            df = pd.read_csv(id_prop_path)
            if df.shape[1] >= 2:
                target_values = df.iloc[:, 1].values
                mean = float(np.mean(target_values))
                std = float(np.std(target_values))
                print(f"Loaded target statistics from id_prop.csv:")
                print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
        except Exception as e:
            print(f"Warning: Could not load from id_prop.csv: {e}")

    # Load from train JSON to compute accurate statistics
    train_json_path = data_dir / f"train_{target_column}.json"
    try:
        import json
        with open(train_json_path) as f:
            train_data = json.load(f)
        train_values = list(train_data.values())
        mean = float(np.mean(train_values))
        std = float(np.std(train_values))
        print(f"\nComputed target statistics from train data:")
        print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
        print(f"  Samples: {len(train_values)}")
    except Exception as e:
        print(f"Warning: Could not compute statistics from train data: {e}")
    
    return {
        'mean': mean,
        'std': std,
        'data_dir': data_dir
    }


def train_model(
    data_dir: str,
    target_column: str,
    log_dir: str,
    max_epochs: int = 10,
    batch_size: int = 16,
    per_gpu_batchsize: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    load_path: Optional[str] = None,
    seed: int = 0,
    num_workers: int = 16,
    precision: int = 32,
    accelerator: str = "auto"
) -> Dict[str, Any]:
    """
    Train MOFTransformer model on preprocessed dataset.
    
    Parameters
    ----------
    data_dir : str
        Path to preprocessed dataset directory.
    target_column : str
        Name of the target column for regression.
    log_dir : str
        Directory to save logs and checkpoints.
    max_epochs : int, optional
        Maximum number of training epochs (default: 10).
    batch_size : int, optional
        Total batch size (default: 16).
    per_gpu_batchsize : int, optional
        Batch size per GPU (default: 4).
    learning_rate : float, optional
        Learning rate (default: 1e-4).
    weight_decay : float, optional
        Weight decay (default: 1e-2).
    load_path : str, optional
        Path to pretrained model checkpoint.
    seed : int, optional
        Random seed (default: 0).
    num_workers : int, optional
        Number of data loading workers (default: 16).
    precision : int, optional
        Training precision - 16, 32, or 64 (default: 32).
    accelerator : str, optional
        Accelerator type - 'cpu', 'gpu', 'auto' (default: 'auto').
        
    Returns
    -------
    dict
        Dictionary containing training results and paths to output files.
    """
    print("=" * 60)
    print("MOFTransformer Trainer")
    print("=" * 60)
    
    # Set random seed
    pl.seed_everything(seed)
    
    # Convert to Path objects
    data_dir = Path(data_dir).resolve()
    log_dir = Path(log_dir).resolve()
    
    # Validate data directory
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load configuration
    print(f"\nLoading preprocessed dataset from: {data_dir}")
    config = load_preprocessed_config(data_dir, target_column)

    # Warn about small dataset
    try:
        import json
        with open(data_dir / f"train_{target_column}.json") as f:
            train_size = len(json.load(f))
        with open(data_dir / f"val_{target_column}.json") as f:
            val_size = len(json.load(f))
        with open(data_dir / f"test_{target_column}.json") as f:
            test_size = len(json.load(f))
        print(f"\nDataset sizes: train={train_size}, val={val_size}, test={test_size}")
        if train_size < 100:
            print(f"WARNING: Small dataset ({train_size} samples). Consider increasing max_epochs.")
    except Exception as e:
        print(f"Warning: Could not check dataset size: {e}")
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved to: {log_dir}")
    
    # Build configuration dictionary for MOFTransformer
    mof_config = {
        "root_dataset": str(data_dir),
        "downstream": target_column,
        "log_dir": str(log_dir),
        "test_only": False,
        
        # Model architecture
        "hid_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "mlp_ratio": 4,
        "drop_rate": 0.1,
        "mpp_ratio": 0.15,
        
        # Graph parameters
        "atom_fea_len": 64,
        "nbr_fea_len": 64,
        "max_graph_len": 300,
        "max_nbr_atoms": 12,
        
        # Grid parameters
        "img_size": 30,
        "patch_size": 5,
        "in_chans": 1,
        "max_grid_len": -1,
        "draw_false_grid": False,
        
        # Training parameters
        "loss_names": {"regression": 1},
        "n_classes": 0,
        "n_targets": 1,
        "batch_size": batch_size,
        "per_gpu_batchsize": per_gpu_batchsize,
        "num_workers": num_workers,
        "precision": precision,
        "max_epochs": max_epochs,
        "seed": seed,
        
        # Optimizer parameters
        "optim_type": "adamw",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "decay_power": 1,
        "max_steps": -1,
        "warmup_steps": 0.05,
        "end_lr": 0,
        "lr_mult": 1,
        
        # Normalization
        "mean": config['mean'],
        "std": config['std'],
        
        # Device parameters
        "accelerator": accelerator,
        "devices": "auto",
        "num_nodes": 1,
        
        # Other
        "load_path": load_path if load_path else "",
        "resume_from": None,
        "val_check_interval": 1.0,
        "exp_name": "pretrained_mof",
        "visualize": False,
    }
    
    # Validate configuration
    mof_config = get_valid_config(mof_config)
    
    # Create datamodule
    print("\nInitializing datamodule...")
    datamodule = Datamodule(mof_config)
    
    # Create model
    print("Initializing model...")
    model = Module(mof_config)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        filename="epoch{epoch:02d}-r2{val/the_metric:.4f}",
        auto_insert_metric_name=False
    )
    
    lr_callback = LearningRateMonitor(logging_interval="step")
    
    # Metrics collection callback
    metrics_callback = MetricsCollectionCallback()
    
    callbacks = [checkpoint_callback, lr_callback, metrics_callback]
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name="training",
    )
    
    # Calculate gradient accumulation
    num_devices = get_num_devices(mof_config)
    print(f"Number of devices: {num_devices}")
    
    if num_devices == 0:
        accumulate_grad_batches = batch_size // (per_gpu_batchsize * mof_config["num_nodes"])
    else:
        accumulate_grad_batches = batch_size // (per_gpu_batchsize * num_devices * mof_config["num_nodes"])
    
    accumulate_grad_batches = max(1, accumulate_grad_batches)
    print(f"Gradient accumulation batches: {accumulate_grad_batches}")
    
    # Create trainer
    print("\nInitializing trainer...")
    
    # Determine strategy based on number of devices
    if num_devices > 1:
        strategy = "ddp_find_unused_parameters_false"
    else:
        strategy = "ddp"  # Works for single GPU too
    
    trainer = pl.Trainer(
        accelerator=mof_config["accelerator"],
        devices=mof_config["devices"],
        num_nodes=mof_config["num_nodes"],
        precision=mof_config["precision"],
        strategy=strategy,
        benchmark=True,
        max_epochs=max_epochs,
        max_steps=mof_config["max_steps"] if mof_config["max_steps"] is not None else None,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        val_check_interval=mof_config["val_check_interval"],
        deterministic=True,
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.fit(model, datamodule=datamodule)
    
    # Test on best model
    print("\n" + "=" * 60)
    print("Testing best model...")
    print("=" * 60 + "\n")
    
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
    # Save metrics to CSV
    print("\n" + "=" * 60)
    print("Saving metrics...")
    print("=" * 60)
    
    saved_files = metrics_callback.save_metrics(log_dir)
    
    # Save test metrics separately
    test_metrics = {
        'loss': [],
        'mae': [],
        'r2': []
    }
    
    if hasattr(model, 'test_logits') and hasattr(model, 'test_labels'):
        if len(model.test_logits) > 0 and len(model.test_labels) > 0:
            predictions = np.array(model.test_logits)
            labels = np.array(model.test_labels)
            
            test_metrics['loss'].append(float(calculate_mse(predictions, labels)))
            test_metrics['mae'].append(float(calculate_mae(predictions, labels)))
            test_metrics['r2'].append(float(calculate_r2_score(predictions, labels)))
            
            test_df = pd.DataFrame(test_metrics)
            test_path = log_dir / "test_metrics.csv"
            test_df.to_csv(test_path, index=False)
            saved_files['test'] = test_path
            print(f"Saved test metrics to {test_path}")
            
            print(f"\nTest Results:")
            print(f"  MSE Loss: {test_metrics['loss'][0]:.6f}")
            print(f"  MAE:      {test_metrics['mae'][0]:.4f}")
            print(f"  R²:       {test_metrics['r2'][0]:.6f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation R²: {metrics_callback.best_val_r2:.6f} (Epoch {metrics_callback.best_epoch})")
    print(f"\nOutput files:")
    for phase, path in saved_files.items():
        print(f"  {phase.capitalize()} metrics: {path}")
    print(f"  TensorBoard logs: {logger.log_dir}")
    print(f"  Checkpoints: {checkpoint_callback.dirpath}")
    print("=" * 60)
    
    return {
        'log_dir': log_dir,
        'best_model_path': checkpoint_callback.best_model_path,
        'best_val_r2': metrics_callback.best_val_r2,
        'best_epoch': metrics_callback.best_epoch,
        'metrics_files': saved_files
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MOFTransformer model on preprocessed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python trainer.py --data-dir ./extracted_data/dataset/ --target-column energy_per_atom --log-dir ./results/
  
  # Custom training parameters
  python trainer.py --data-dir ./extracted_data/dataset/ --target-column bandgap --log-dir ./results/ --max-epochs 50 --batch-size 32 --learning-rate 1e-5
  
  # Resume from checkpoint
  python trainer.py --data-dir ./extracted_data/dataset/ --target-column formation_energy --log-dir ./results/ --load-path ./results/checkpoints/best.ckpt
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to extracted preprocessed dataset directory (containing train_*.json, val_*.json, test_*.json)"
    )
    
    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Name of the target column (must match the preprocessed data)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory to save logs, checkpoints, and metrics (default: ./logs)"
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Total batch size for training (default: 16)"
    )
    
    parser.add_argument(
        "--per-gpu-batchsize",
        type=int,
        default=4,
        help="Batch size per GPU (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay for optimizer (default: 1e-2)"
    )
    
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (optional)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of data loading workers (default: 16)"
    )
    
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32, 64],
        help="Training precision - 16 (mixed), 32 (float), or 64 (double) (default: 32)"
    )
    
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Accelerator type (default: auto)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        results = train_model(
            data_dir=args.data_dir,
            target_column=args.target_column,
            log_dir=args.log_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            per_gpu_batchsize=args.per_gpu_batchsize,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            load_path=args.load_path,
            seed=args.seed,
            num_workers=args.num_workers,
            precision=args.precision,
            accelerator=args.accelerator
        )
        
        print("\nTraining pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
