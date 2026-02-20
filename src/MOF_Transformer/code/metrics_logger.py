"""
Metrics Logger for MOFTransformer

Handles collection and saving of training, validation, and test metrics
to separate CSV files with proper formatting.

Author: MOFTransformer Team
Date: 2026-02-19
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np


class MetricsLogger:
    """
    Logger for collecting and saving training metrics to separate CSV files.
    
    This class handles:
    - Collection of metrics per epoch for train, validation, and test phases
    - Saving metrics to separate CSV files (train_metrics.csv, val_metrics.csv, test_metrics.csv)
    - Proper formatting and organization of metric data
    
    Attributes
    ----------
    log_dir : Path
        Directory to save metric CSV files.
    metrics : dict
        Dictionary storing metrics for each phase.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the metrics logger.
        
        Parameters
        ----------
        log_dir : str
            Directory to save metric CSV files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics per phase
        self.metrics: Dict[str, Dict[str, List[Any]]] = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }
        
        # Track current epoch for each phase
        self._current_epoch: Dict[str, int] = {
            "train": 0,
            "val": 0,
            "test": 0
        }
    
    def log_train_metrics(
        self,
        epoch: int,
        loss: float,
        mae: float,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log training metrics for an epoch.
        
        Parameters
        ----------
        epoch : int
            Epoch number.
        loss : float
            Training loss (MSE).
        mae : float
            Mean Absolute Error.
        learning_rate : float, optional
            Learning rate at this epoch.
        **kwargs
            Additional metrics to log.
        """
        self._current_epoch["train"] = epoch
        
        self.metrics["train"]["epoch"].append(epoch)
        self.metrics["train"]["loss"].append(loss)
        self.metrics["train"]["mae"].append(mae)
        
        if learning_rate is not None:
            self.metrics["train"]["learning_rate"].append(learning_rate)
        
        # Add any additional metrics
        for key, value in kwargs.items():
            self.metrics["train"][key].append(value)
    
    def log_val_metrics(
        self,
        epoch: int,
        loss: float,
        mae: float,
        r2: float,
        **kwargs
    ) -> None:
        """
        Log validation metrics for an epoch.
        
        Parameters
        ----------
        epoch : int
            Epoch number.
        loss : float
            Validation loss (MSE).
        mae : float
            Mean Absolute Error.
        r2 : float
            R² score.
        **kwargs
            Additional metrics to log.
        """
        self._current_epoch["val"] = epoch
        
        self.metrics["val"]["epoch"].append(epoch)
        self.metrics["val"]["loss"].append(loss)
        self.metrics["val"]["mae"].append(mae)
        self.metrics["val"]["r2"].append(r2)
        
        # Add any additional metrics
        for key, value in kwargs.items():
            self.metrics["val"][key].append(value)
    
    def log_test_metrics(
        self,
        loss: float,
        mae: float,
        r2: float,
        predictions: Optional[List[float]] = None,
        labels: Optional[List[float]] = None,
        cif_ids: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Log test metrics.
        
        Parameters
        ----------
        loss : float
            Test loss (MSE).
        mae : float
            Mean Absolute Error.
        r2 : float
            R² score.
        predictions : list, optional
            List of predicted values.
        labels : list, optional
            List of true values.
        cif_ids : list, optional
            List of CIF IDs for each prediction.
        **kwargs
            Additional metrics to log.
        """
        self.metrics["test"]["loss"].append(loss)
        self.metrics["test"]["mae"].append(mae)
        self.metrics["test"]["r2"].append(r2)
        
        if predictions is not None:
            self.metrics["test"]["predictions"] = predictions
        if labels is not None:
            self.metrics["test"]["labels"] = labels
        if cif_ids is not None:
            self.metrics["test"]["cif_ids"] = cif_ids
        
        # Add any additional metrics
        for key, value in kwargs.items():
            self.metrics["test"][key].append(value)
    
    def save_all_metrics(self) -> Dict[str, Path]:
        """
        Save all collected metrics to CSV files.
        
        Returns
        -------
        dict
            Dictionary mapping phase names to their CSV file paths.
        """
        saved_files = {}
        
        # Save train metrics
        if self.metrics["train"]["epoch"]:
            train_path = self.log_dir / "train_metrics.csv"
            self._save_csv("train", train_path)
            saved_files["train"] = train_path
            print(f"Saved training metrics to {train_path}")
        
        # Save validation metrics
        if self.metrics["val"]["epoch"]:
            val_path = self.log_dir / "val_metrics.csv"
            self._save_csv("val", val_path)
            saved_files["val"] = val_path
            print(f"Saved validation metrics to {val_path}")
        
        # Save test metrics
        if self.metrics["test"]["loss"]:
            test_path = self.log_dir / "test_metrics.csv"
            self._save_csv("test", test_path)
            saved_files["test"] = test_path
            print(f"Saved test metrics to {test_path}")
        
        return saved_files
    
    def _save_csv(self, phase: str, path: Path) -> None:
        """
        Save metrics for a specific phase to CSV.
        
        Parameters
        ----------
        phase : str
            Phase name ('train', 'val', or 'test').
        path : Path
            Path to save the CSV file.
        """
        metrics_data = self.metrics[phase]
        
        if not metrics_data:
            return
        
        # Determine the order of columns
        column_order = self._get_column_order(phase)
        
        # Get available columns (those with data)
        available_columns = [col for col in column_order if col in metrics_data and metrics_data[col]]
        
        # Add any additional columns not in the predefined order
        for key in metrics_data.keys():
            if key not in available_columns:
                available_columns.append(key)
        
        # Determine the maximum length
        max_len = max(len(v) for v in metrics_data.values() if v)
        
        # Pad shorter lists with None
        for key in available_columns:
            if len(metrics_data[key]) < max_len:
                metrics_data[key].extend([None] * (max_len - len(metrics_data[key])))
        
        # Write to CSV
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=available_columns)
            writer.writeheader()
            
            for i in range(max_len):
                row = {col: metrics_data[col][i] for col in available_columns}
                writer.writerow(row)
    
    def _get_column_order(self, phase: str) -> List[str]:
        """
        Get predefined column order for each phase.
        
        Parameters
        ----------
        phase : str
            Phase name ('train', 'val', or 'test').
            
        Returns
        -------
        list
            List of column names in preferred order.
        """
        if phase == "train":
            return ["epoch", "loss", "mae", "learning_rate"]
        elif phase == "val":
            return ["epoch", "loss", "mae", "r2"]
        elif phase == "test":
            return ["loss", "mae", "r2", "cif_ids", "predictions", "labels"]
        else:
            return []
    
    def get_best_epoch(self, metric: str = "r2", mode: str = "max") -> int:
        """
        Get the epoch with the best value for a given metric.
        
        Parameters
        ----------
        metric : str
            Name of the metric to optimize.
        mode : str
            Either 'max' (higher is better) or 'min' (lower is better).
            
        Returns
        -------
        int
            Epoch number with the best metric value.
        """
        if metric not in self.metrics["val"] or not self.metrics["val"][metric]:
            return -1
        
        values = self.metrics["val"][metric]
        epochs = self.metrics["val"]["epoch"]
        
        if mode == "max":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return epochs[best_idx]
    
    def print_summary(self) -> None:
        """Print a summary of the collected metrics."""
        print("\n" + "=" * 60)
        print("Metrics Summary")
        print("=" * 60)
        
        # Training summary
        if self.metrics["train"]["epoch"]:
            print("\nTraining:")
            print(f"  Epochs: {len(self.metrics['train']['epoch'])}")
            if self.metrics["train"]["loss"]:
                print(f"  Final Loss: {self.metrics['train']['loss'][-1]:.6f}")
                print(f"  Best Loss: {min(self.metrics['train']['loss']):.6f}")
            if self.metrics["train"]["mae"]:
                print(f"  Final MAE: {self.metrics['train']['mae'][-1]:.4f}")
        
        # Validation summary
        if self.metrics["val"]["epoch"]:
            print("\nValidation:")
            print(f"  Epochs: {len(self.metrics['val']['epoch'])}")
            if self.metrics["val"]["loss"]:
                print(f"  Final Loss: {self.metrics['val']['loss'][-1]:.6f}")
                print(f"  Best Loss: {min(self.metrics['val']['loss']):.6f}")
            if self.metrics["val"]["mae"]:
                print(f"  Final MAE: {self.metrics['val']['mae'][-1]:.4f}")
                print(f"  Best MAE: {min(self.metrics['val']['mae']):.4f}")
            if self.metrics["val"]["r2"]:
                print(f"  Final R²: {self.metrics['val']['r2'][-1]:.6f}")
                print(f"  Best R²: {max(self.metrics['val']['r2']):.6f}")
        
        # Test summary
        if self.metrics["test"]["loss"]:
            print("\nTest:")
            print(f"  Loss: {self.metrics['test']['loss'][-1]:.6f}")
            print(f"  MAE: {self.metrics['test']['mae'][-1]:.4f}")
            if "r2" in self.metrics["test"]:
                print(f"  R²: {self.metrics['test']['r2'][-1]:.6f}")
        
        print("\n" + "=" * 60)


def calculate_r2_score(predictions: List[float], labels: List[float]) -> float:
    """
    Calculate R² (coefficient of determination) score.
    
    Parameters
    ----------
    predictions : list
        List of predicted values.
    labels : list
        List of true values.
        
    Returns
    -------
    float
        R² score.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions and labels must have same length. "
            f"Got {len(predictions)} predictions and {len(labels)} labels."
        )
    
    if len(predictions) == 0:
        return 0.0
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate mean of labels
    mean_labels = np.mean(labels)
    
    # Calculate sum of squared residuals
    ss_res = np.sum((labels - predictions) ** 2)
    
    # Calculate total sum of squares
    ss_tot = np.sum((labels - mean_labels) ** 2)
    
    # Calculate R²
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    
    return float(r2)


def calculate_mae(predictions: List[float], labels: List[float]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Parameters
    ----------
    predictions : list
        List of predicted values.
    labels : list
        List of true values.
        
    Returns
    -------
    float
        MAE value.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions and labels must have same length. "
            f"Got {len(predictions)} predictions and {len(labels)} labels."
        )
    
    if len(predictions) == 0:
        return 0.0
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    mae = np.mean(np.abs(predictions - labels))
    
    return float(mae)


def calculate_mse(predictions: List[float], labels: List[float]) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Parameters
    ----------
    predictions : list
        List of predicted values.
    labels : list
        List of true values.
        
    Returns
    -------
    float
        MSE value.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Predictions and labels must have same length. "
            f"Got {len(predictions)} predictions and {len(labels)} labels."
        )
    
    if len(predictions) == 0:
        return 0.0
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    mse = np.mean((predictions - labels) ** 2)
    
    return float(mse)
