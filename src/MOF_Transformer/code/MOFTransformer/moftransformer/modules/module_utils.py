# MOFTransformer version 2.2.0
import torch
import numpy as np

from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)

from moftransformer.gadgets.my_metrics import Accuracy, Scalar


class R2Score:
    """
    R² Score metric for regression tasks.
    
    Accumulates predictions and labels to compute R² at the end of an epoch.
    """
    
    def __init__(self):
        self.predictions = []
        self.labels = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the metric with new predictions and labels.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted values.
        labels : torch.Tensor
            True values.
        """
        self.predictions.extend(predictions.cpu().flatten().tolist())
        self.labels.extend(labels.cpu().flatten().tolist())
    
    def compute(self) -> float:
        """
        Compute the R² score.
        
        Returns
        -------
        float
            R² score value.
        """
        if len(self.predictions) == 0 or len(self.labels) == 0:
            return 0.0
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        mean_labels = np.mean(labels)
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - mean_labels) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
    
    def reset(self) -> None:
        """Reset the metric state."""
        self.predictions = []
        self.labels = []


def set_metrics(pl_module):
    """
    Initialize metrics for training and validation phases.
    
    For regression tasks, initializes loss, MAE, and R² metrics.
    For classification tasks, initializes loss and accuracy metrics.
    
    Parameters
    ----------
    pl_module : pl.LightningModule
        The PyTorch Lightning module.
    """
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "regression" or k == "vfp":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_mae", Scalar())
                # Add R² score metric for regression tasks
                setattr(pl_module, f"{split}_{k}_r2", R2Score())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def epoch_wrapup(pl_module):
    """
    Wrap up an epoch by logging aggregated metrics and resetting metric states.
    
    This function is called at the end of each training and validation epoch.
    It computes final metrics, logs them, and resets the metric accumulators.
    For regression tasks, it also computes and logs the R² score.
    
    Parameters
    ----------
    pl_module : pl.LightningModule
        The PyTorch Lightning module.
    """
    phase = "train" if pl_module.training else "val"

    the_metric = 0

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        if loss_name == "regression" or loss_name == "vfp":
            # MSE loss
            loss_value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                loss_value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            # MAE loss
            mae_value = getattr(pl_module, f"{phase}_{loss_name}_mae").compute()
            pl_module.log(
                f"{loss_name}/{phase}/mae_epoch",
                mae_value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_mae").reset()

            # R² score - only for validation phase
            if phase == "val":
                r2_value = getattr(pl_module, f"{phase}_{loss_name}_r2").compute()
                pl_module.log(
                    f"{loss_name}/{phase}/r2_epoch",
                    r2_value,
                    batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                    sync_dist=True,
                )
                getattr(pl_module, f"{phase}_{loss_name}_r2").reset()
                
                # Use R² as the main metric for model selection (higher is better)
                value = r2_value
                
                # Print R² score to console at the end of each validation epoch
                print(f"\nValidation Epoch {pl_module.current_epoch}: "
                      f"Loss={loss_value:.6f}, MAE={mae_value:.4f}, R²={r2_value:.6f}")
            else:
                # For training, use negative MAE as the metric (for consistency)
                value = -mae_value
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(
                f"{loss_name}/{phase}/accuracy_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric, sync_dist=True)


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["regression_head", "classification_head"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)  # within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and any(bb in n for bb in head_names)  # within head_names
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                # within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps == -1:
        max_steps = pl_module.trainer.estimated_stepping_batches
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean, std, device):
        if mean and std:
            if isinstance(mean, list):
                mean = torch.tensor(mean).to(device)
            if isinstance(std, list):
                std = torch.tensor(std).to(device)
            self.mean = mean
            self.std = std
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

    def encode(self, tensor):
        return self._norm_func(tensor)

    def decode(self, tensor):
        return self._denorm_func(tensor)
