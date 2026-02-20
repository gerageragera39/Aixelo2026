import os

EXTERNAL_DATA_DIR = '/workspace/dataset'
TEST_DIR = EXTERNAL_DATA_DIR

RESULTS_DIR = '/workspace/results'

MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "model_best.pth.tar")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Data dir: {EXTERNAL_DATA_DIR}")
print(f"Results dir: {RESULTS_DIR}")

