import os
import time
import torch
import torch.nn as nn
import pandas as pd
from utils import set_seed, adjust_learning_rate, create_cifar_optimizer
from data_loader import load_cifar100_dataset, make_dataloaders
from model import ResNetCIFAR
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def train_single_model_cifar100(config, data_dir='./cifar100'):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainset, testset = load_cifar100_dataset(data_dir=data_dir)
    trainloader, testloader = make_dataloaders(trainset, testset, config.batch_size)
    print(f"\nCIFAR-100 Classes: 100")
    print(f"Train: {len(trainset)}, Test: {len(testset)}")

    # Build model
    model = ResNetCIFAR(config, num_classes=100).to(device)

    model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = create_cifar_optimizer(model.parameters())
    criterion = nn.CrossEntropyLoss()

    out_dir = config.output_dir + f'/seed_{config.seed}'

    # logging
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "timing_history.csv")
    log_df = pd.DataFrame(columns=["epoch", "epoch_time", "total_elapsed", "loss", "val_loss", "accuracy", "val_accuracy"])

    print(f"Starting training {'with' if config.sd_on else 'without'} Stochastic Depth...")
    start_train_time = time.time()
    epoch_times = []
    total_elapsed = 0

    scaler = torch.amp.GradScaler()

    for epoch in range(config.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch)

        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        epoch_start_time = time.time()
        for batch_idx, (data, target) in enumerate(trainloader):
        # pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch", leave=False)
        # for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            total_train += target.size(0)

            # pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for data, target in testloader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_val += pred.eq(target.view_as(pred)).sum().item()
                total_val += target.size(0)

        # Calculate metrics
        train_loss /= len(trainloader)
        val_loss /= len(testloader)
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        total_elapsed = time.time() - start_train_time

        # Logging
        log_entry = {
            "epoch": epoch + 1,
            "epoch_time": epoch_time,
            "total_elapsed": total_elapsed,
            "loss": train_loss,
            "val_loss": val_loss,
            "accuracy": train_acc / 100.0,
            "val_accuracy": val_acc / 100.0
        }
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        print(f'Epoch {epoch+1}/{config.epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Val Acc: {val_acc:.2f}% | '
              f'LR: {current_lr:.6f} | '
              f'Time: {epoch_time:.2f}s')

    total_train_time = time.time() - start_train_time

    # Final evaluation
    model.eval()
    final_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for data, target in testloader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            final_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    final_loss /= len(testloader)
    final_acc = 100. * correct / total
    print(f"Final Test Accuracy: {final_acc:.4f}%, Loss: {final_loss:.4f}")
    print(f"Total training time: {total_train_time:.2f} seconds")

    if len(epoch_times) > 0:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f"Average epoch time: {avg_epoch_time:.4f} seconds")
    log_df.to_csv(csv_file, index=False)
    model_path = os.path.join(out_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.epochs,
        'loss': final_loss,
    }, model_path)
    print(f"Model saved: {model_path}")
    num_batches = len(trainloader) * config.epochs
    avg_batch_time = total_train_time / num_batches if num_batches > 0 else 0

    return log_df, total_train_time, avg_batch_time

