import os
import time
import torch
import torch.nn as nn
import pandas as pd
from utils import set_seed, create_optimizer, adjust_learning_rate
from resnet_bottleneck_model import ResNetBottleneck, BottleneckModelConfig
from tqdm import tqdm


torch.set_float32_matmul_precision('high')


def train_bottleneck_optimized(config: BottleneckModelConfig, trainloader, testloader, num_classes, learning_rate: float = 0.1):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    start_epoch = 0
    model = None

    model = ResNetBottleneck(config, num_classes=num_classes).to(device)
    optimizer = create_optimizer(model.parameters(), lr=learning_rate)


    model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet-{config.depth} Total parameters: {total_params:,}")
    print(f"ResNet-{config.depth} Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    # Output directory setup
    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Logging setup
    csv_file = os.path.join(out_dir, f"resnet_{config.depth}_timing_history.csv")
    log_df = pd.DataFrame(columns=["epoch", "epoch_time", "total_elapsed", "loss", "val_loss", "accuracy", "val_accuracy"])

    print(f"Starting training ResNet-{config.depth} {'with' if config.sd_on else 'without'} Stochastic Depth...")
    start_train_time = time.time()
    epoch_times = []
    total_elapsed = 0

    scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, config.epochs, learning_rate)

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
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
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

    if os.path.exists(csv_file): 
        log_df.to_csv(csv_file, mode='a', index=False, header=False)
    else:
        log_df.to_csv(csv_file, index=False)
    
    # Save model
    model_path = os.path.join(out_dir, f"resnet_{config.depth}_model_sd_{config.sd_on}_epoch_{config.epochs}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, model_path)
    print(f"Model saved: {model_path}")
    
    num_batches = len(trainloader) * config.epochs
    avg_batch_time = total_train_time / num_batches if num_batches > 0 else 0

    return log_df, total_train_time, avg_batch_time


