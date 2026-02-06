import os
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.site import CrystalNNFingerprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set path to dataset folder and CSV file
DATASET_DIR = "dataset"
CSV_FILE = os.path.join(DATASET_DIR, "id_prop.csv")

# Load CSV containing filenames and target values
df = pd.read_csv(CSV_FILE)

# Initialize featurizer
featurizer = SiteStatsFingerprint(CrystalNNFingerprint.from_preset("ops"))

# Prepare feature matrix and target list
X = []
y = []

print("Featurizing structures...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        cif_path = os.path.join(DATASET_DIR, row["filename"] + ".cif")
        structure = Structure.from_file(cif_path)
        features = featurizer.featurize(structure)
        X.append(features)
        y.append(row["target"])
    except Exception as e:
        print(f"Failed to read {row['filename']}: {e}")

# Convert to DataFrame
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model trained!")
print(f"🔢 Mean Squared Error: {mse:.4f}")
print(f"📈 R² Score: {r2:.4f}")
