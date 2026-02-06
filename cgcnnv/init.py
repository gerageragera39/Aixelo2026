# init.py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import psutil
import torch
from config import EXTERNAL_DATA_DIR, MODEL_PATH, TEST_DIR
from sklearn.metrics import mean_absolute_error, r2_score

# Function to print system memory and GPU usage
def print_system_usage():
    # CPU / RAM
    mem = psutil.virtual_memory()
    print(f"💾 RAM Usage: {mem.percent}% ({mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB)")

    # GPU (CUDA)
    if torch.cuda.is_available():
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        print(f"   Memory Cached   : {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
        print(f"   CUDA Available  : {torch.cuda.is_available()}")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Step 0: Print initial system usage
print_system_usage()

# === Step 1: Train the model ===
print("🔧 Running main.py (training the model)...")
subprocess.run([
    "python", "main.py",
    EXTERNAL_DATA_DIR,
    "--task", "regression"
])

# Print system usage after training
print_system_usage()

# === Step 2: Run prediction ===
print("🧪 Running predict.py (generating predictions)...")
subprocess.run([
    "python", "predict.py",
    MODEL_PATH,
    EXTERNAL_DATA_DIR
])

# Print system usage after prediction
print_system_usage()

# === Step 3: Calculate metrics ===
predictions_df = pd.read_csv("results/predictions.csv", header=None)
y_true = predictions_df.iloc[:, 1].values
y_pred = predictions_df.iloc[:, 2].values

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
rho = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")

# === Step 4: Plot results ===
plt.figure(figsize=(12, 5))

# Plot 1: Predictions vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred, alpha=0.6, label="Predictions")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", label="Ideal")
plt.xlabel("Actual Band Gap")
plt.ylabel("Predicted Band Gap")
plt.title("Predicted vs Actual")
plt.legend()
plt.text(min(y_true), max(y_pred) * 0.9, f"MAE: {mae:.3f}", fontsize=10)
plt.text(min(y_true), max(y_pred) * 0.8, f"R²: {r2:.3f}", fontsize=10)
plt.text(min(y_true), max(y_pred) * 0.7, f"ρ: {rho:.3f}", fontsize=10)

# Plot 2: Training Loss over Epochs
try:
    train_loss_df = pd.read_csv("results/train_loss.csv")
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_df.iloc[:, 0], train_loss_df.iloc[:, 1], label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
except FileNotFoundError:
    print("❌ train_loss.csv not found. Skipping loss plot.")

plt.tight_layout()
plt.show()
