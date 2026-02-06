import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os


# Constants
LR_DECAY_FACTOR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_optimizer(model_parameters, lr=0.1):
    return optim.SGD(model_parameters, lr=lr,
                     momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY)


def adjust_learning_rate(optimizer, epoch, num_epochs, lr=0.1):
    decay_epoch1 = int(num_epochs * 0.5)
    decay_epoch2 = int(num_epochs * 0.75)

    if epoch >= decay_epoch2:
        lr *= LR_DECAY_FACTOR ** 2
    elif epoch >= decay_epoch1:
        lr *= LR_DECAY_FACTOR

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def plot_histories(log_dfs, labels, times, avg_batch_times, output_dir):
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 3, 1)
    for log_df, label in zip(log_dfs, labels):
        plt.plot(log_df['loss'], label=f"{label} Train")
        plt.plot(log_df['val_loss'], '--', label=f"{label} Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    for log_df, label in zip(log_dfs, labels):
        plt.plot(log_df['accuracy'] * 100, label=f"{label} Train")
        plt.plot(log_df['val_accuracy'] * 100, '--', label=f"{label} Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Training time
    plt.subplot(1, 3, 3)
    bars = plt.bar(labels, times, color=['blue', 'orange'])
    plt.title("Total Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)

    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()

