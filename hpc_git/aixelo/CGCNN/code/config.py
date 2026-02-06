import os

# ПУТИ ВНУТРИ КОНТЕЙНЕРА (как мы указали в run_cgcnn.sh через флаг -B)

# Данные примонтированы сюда: -B "$DATA_ROOT:/workspace/dataset"
EXTERNAL_DATA_DIR = '/workspace/dataset'
TEST_DIR = EXTERNAL_DATA_DIR

# Результаты примонтированы сюда: -B "$RESULTS_DIR:/workspace/results"
# Это корень твоей папки job_ID на хосте
RESULTS_DIR = '/workspace/results'

# Папка для моделей будет ВНУТРИ результатов
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "model_best.pth.tar")

# Создаем папки (они создадутся сразу в твоем vault на Alex)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Data dir: {EXTERNAL_DATA_DIR}")
print(f"Results dir: {RESULTS_DIR}")

