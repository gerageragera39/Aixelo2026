import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Dataset and result paths
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'dataset')
TEST_DIR = EXTERNAL_DATA_DIR  # <-- updated this line
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_PATH = os.path.join(MODEL_DIR, "model_best.pth.tar")

# Create folders if missing
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)