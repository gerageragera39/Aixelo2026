
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.rcParams['figure.dpi'] = 300

from config import BASE_DIR

# Path to the directory containing ALL data files
DATA_DIR = os.path.join(BASE_DIR, "data_fingerprints")

# Load the datasets from the data directory
df_45 = pd.read_csv(os.path.join(DATA_DIR, "stoich45_fingerprints.csv"))
df_120 = pd.read_csv(os.path.join(DATA_DIR, "stoich120_fingerprints.csv"))
df_sine = pd.read_csv(os.path.join(DATA_DIR, "sine_matrix_fingerprints.csv"))
df_qmof_refcodes = pd.read_csv(os.path.join(DATA_DIR, "qmof-refcodes.csv"))
df_ofm_fp = pd.read_csv(os.path.join(DATA_DIR, "ofm_fingerprints.csv"))
df_bandgaps = pd.read_csv(os.path.join(DATA_DIR, "qmof-bandgaps.csv"))
df_bandgaps = df_bandgaps.rename(columns={"refcode": "MOF"})

# Compatibility checks (originally for Google Colab issues)
if 'MOF' in df_120.columns:
    columns = ['MOF'] + [col for col in df_120.columns if col != 'MOF']
    df_120 = df_120[columns]
else:
    raise KeyError("The column 'MOF' does not exist in df_120.")

if 'MOF' in df_45.columns:
    columns = ['MOF'] + [col for col in df_120.columns if col != 'MOF']
    df_120 = df_120[columns]  # Note: This again modifies df_120, not df_45
else:
    raise KeyError("The column 'MOF' does not exist in df_45.")

if 'MOF' in df_bandgaps.columns:
    columns = ['MOF'] + [col for col in df_120.columns if col != 'MOF']
    df_120 = df_120[columns]
else:
    raise KeyError("The column 'MOF' does not exist in df_bandgaps.")

# Create the joined DataFrame between fingerprints and bandgaps
df_45_bandgap_joined = df_45.set_index('MOF').join(df_bandgaps.set_index('MOF'), how='outer', rsuffix='_bandgaps')

# Remove rows with missing values
df_45_bandgap_joined_clean = df_45_bandgap_joined.dropna()

# Reset index and drop unnecessary columns for further analysis
df_45_bandgap_joined_clean = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Check the difference in features between 45 and 120 datasets
features_45 = set(df_45.columns)
features_120 = set(df_120.columns)
unique_features_45 = features_45 - features_120

print("Features in df_45 that are not in df_120:")
print(unique_features_45)
print(len(unique_features_45))


"""# **Training models**

**1. Training on the 45_bandgap_clean dataframe.*
"""

# Splitting the 45_bandgap_clean df into train, validation and test
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

X = df.drop(columns=['BG_PBE'])
y = df['BG_PBE']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

"""#neural network"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and preprocess the data
data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data_clean = data.dropna()

# Split features and target variable
X = data_clean.drop(columns=['BG_PBE'])
y = data_clean['BG_PBE']

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Applying PCA for dimensionality reduction (optional)
#pca = PCA(n_components=20)  # Increase the number of components for more variance retention
#X_train_pca = pca.fit_transform(X_train)
#X_valid_pca = pca.transform(X_valid)
#X_test_pca = pca.transform(X_test)

# Define the neural network model with additional layers and dropout
model = Sequential([
    Dense(256, input_dim=X_train_pca.shape[1], activation='relu'),  # Increase number of neurons
    Dropout(0.3),  # Increase dropout rate
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')  # Adjust learning rate

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increase patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)  # Adjust patience and min_lr

# Train the model with increased epochs
history = model.fit(X_train_pca, y_train, validation_data=(X_valid_pca, y_valid), epochs=500, batch_size=32, verbose=1,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
loss = model.evaluate(X_test_pca, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test_pca)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()

# Applying PCA to the full scaled data for the cumulative explained variance plot
pca_full = PCA().fit(X_train)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()

"""# without pca

"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and preprocess the data
data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data_clean = data.dropna()

# Split features and target variable
X = data_clean.drop(columns=['BG_PBE'])
y = data_clean['BG_PBE']

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Define the neural network model with additional layers and dropout
model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),  # Increase number of neurons
    Dropout(0.3),  # Increase dropout rate
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')  # Adjust learning rate

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increase patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)  # Adjust patience and min_lr

# Train the model with increased epochs
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=500, batch_size=32, verbose=1,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()
#

"""# automated hypertuning"""




import kerastuner
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner import RandomSearch

# Load and preprocess the data
data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data_clean = data.dropna()

# Split features and target variable
X = data_clean.drop(columns=['BG_PBE'])
y = data_clean['BG_PBE']

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Define the neural network model with additional layers, batch normalization, and dropout
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_1', min_value=64, max_value=512, step=32), input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))

    model.add(Dense(hp.Int('units_2', min_value=64, max_value=512, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))

    model.add(Dense(hp.Int('units_3', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_3', 0.2, 0.5, step=0.1)))

    model.add(Dense(1))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])),
                  loss='mean_squared_error')
    return model

# Hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=2,
    directory='my_dir',
    project_name='helloworld')

tuner.search_space_summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)

tuner.search(X_train, y_train, epochs=500, validation_data=(X_valid, y_valid), callbacks=[early_stopping, reduce_lr])

best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the model on the test set
loss = best_model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(tuner.oracle.get_best_trials(num_trials=1)[0].metrics.get_history('loss'), label='Train Loss')
plt.plot(tuner.oracle.get_best_trials(num_trials=1)[0].metrics.get_history('val_loss'), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()

"""#after plot  fixing"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from kerastuner.tuners import RandomSearch

# Load and preprocess the data
data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data_clean = data.dropna()

# Split features and target variable
X = data_clean.drop(columns=['BG_PBE'])
y = data_clean['BG_PBE']

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Define the neural network model with additional layers, batch normalization, and dropout
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_1', min_value=64, max_value=512, step=32), input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))

    model.add(Dense(hp.Int('units_2', min_value=64, max_value=512, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))

    model.add(Dense(hp.Int('units_3', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_3', 0.2, 0.5, step=0.1)))

    model.add(Dense(1))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])),
                  loss='mean_squared_error')
    return model

# Hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=2,
    directory='my_dir',
    project_name='helloworld')

tuner.search_space_summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)

tuner.search(X_train, y_train, epochs=500, validation_data=(X_valid, y_valid), callbacks=[early_stopping, reduce_lr])

best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the model on the test set
loss = best_model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
history = best_trial.metrics.get_history()

train_loss = [metric.get('value') for metric in history['loss']]
val_loss = [metric.get('value') for metric in history['val_loss']]

plt.figure(figsize=(10, 7))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Number of rows and columns before dropping NaN values
initial_rows = data.shape[0]
initial_columns = data.shape[1]

# Drop rows with NaN values
data_clean = data.dropna()

# Number of rows and columns after dropping NaN values
final_rows = data_clean.shape[0]
final_columns = data_clean.shape[1]

# Calculate and print the number of dropped rows and columns
dropped_rows = initial_rows - final_rows
dropped_columns = initial_columns - final_columns
print(f'Number of rows with NaN values dropped: {dropped_rows}')
print(f'Number of columns with NaN values dropped: {dropped_columns}')

# Check if essential columns exist after cleaning
#essential_columns = ['BG_PBE', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE']
#missing_columns = [col for col in essential_columns if col not in data_clean.columns]
#if missing_columns:
#    raise ValueError(f"The following essential columns are missing after cleaning: {missing_columns}")

# Splitting features and target variable
#X = data_clean.drop(essential_columns, axis=1)
#y = data_clean['BG_PBE']

# Standardizing the features
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Splitting data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(
#    X_norm, y, random_state=13, test_size=0.25, shuffle=True
#)

# Applying PCA for dimensionality reduction (optional)
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train_pca.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_pca, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test_pca, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test_pca)

# Plotting the PCA results
plt.figure(figsize=(10, 7))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Training Data')
plt.colorbar(label='BG_PBE')
plt.show()

# Applying PCA to the full scaled data for the cumulative explained variance plot
pca_full = PCA().fit(X_norm)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()



mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()



#mse = mean_squared_error(y_valid, y_pred)
#r2 = r2_score(y_valid, y_pred)

#print(f'Mean Squared Error: {mse}')
#print(f'R^2 Score: {r2}')
#from sklearn.metrics import precision_score
#precision = precision_score(y_valid, y_pred)
#print(f"Precision: {precision:.2f}")
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_valid, y_pred)
#print(f"Accuracy: {accuracy:.2f}")

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and preprocess the data
data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data_clean = data.dropna()

# Split features and target variable
X = data_clean.drop(columns=['BG_PBE'])
y = data_clean['BG_PBE']

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Applying PCA for dimensionality reduction (optional)
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=X_train_pca.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

# Train the model
history = model.fit(X_train_pca, y_train, validation_data=(X_valid_pca, y_valid), epochs=200, batch_size=32, verbose=1,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
loss = model.evaluate(X_test_pca, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test_pca)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()

# Applying PCA to the full scaled data for the cumulative explained variance plot
pca_full = PCA().fit(X_norm)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()

"""Convolutional neural network"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]


# Drop rows with NaN values
data_clean = data.dropna()

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Reshape the data for CNN input
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_valid = X_valid.reshape(-1, X_valid.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

# Create a neural network model for regression
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load and preprocess the data
data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Drop rows with NaN values
data_clean = data.dropna()

# Separate features and target variable
X = data_clean.drop(columns=['BG_PBE'])
y = data_clean['BG_PBE']

# Split the data into train, validation, and test sets
#X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

X = X.sample(frac=1, axis=1, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Reshape the data for CNN input
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_valid = X_valid.reshape(-1, X_valid.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

# Create a neural network model for regression
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Define a learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=64, callbacks=[lr_scheduler, early_stopping], verbose=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()

"""**Selection of Models** : Random Forest, Gaussian Process Regressor, XGB Regressor, Multilayer perceptron NN, Ridge Regression, Lasso."""

'''
Shortlist the best performing models. We aim to choose 3 models to proceed further.
'''
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", None)
                    ])

param_grid = {'regressor': [
                             RandomForestRegressor(n_estimators=100),
                             GaussianProcessRegressor(),
                             Ridge(),
                             Lasso(),
                             XGBRegressor(),
                             MLPRegressor()
                             ]
              }

grid_search = GridSearchCV(pipeline,
                           param_grid,
                           cv=5,
                           scoring='r2',
                           return_train_score=True
                           )
grid_search.fit(X_train_val, y_train_val)

grid_search.best_estimator_

grid_search.cv_results_

"""'mean_test_score': array([ 6.94196553e-01, -3.36830726e+03,  5.12576164e-01, -1.84752611e-04,
         6.84328776e-01,  6.55321024e-01]),

Performance of models on the 45_bandgap:

1.   Random Forest: 0,69
2.   XGB Regressor: 0,68
3.   MLP Regressor: 0,65
4.   Ridge: 0,51
5.   Lasso: -0,0001
6.   Gaussian Process Regressor: -3000

Training on Dimension reduction
"""

from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

pca = PCA(n_components=10)

pipe_pca = Pipeline([('PCA', pca),
                 ('regressor', XGBRegressor())])

cv_xgb_pca = cross_validate(pipe_pca, X_train_val, y_train_val, cv=5, n_jobs=-1)
cv_xgb_pca['test_score']

#XGB Regressor with PCA 45 Hyperparameter optimisation

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

param_grid = {
    #"regressor__n_estimators":[100, 300, 500] ,
    #"regressor__max_depth": [6, 8, 10],
    #"regressor__min_samples_split": 5,
    #"regressor__learning_rate": [0.1, 0.05, 0.01],
    #"regressor__loss": "squared_error",
}
pca = PCA(n_components=10)

reg = Pipeline([('PCA', pca),
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05)) #GradientBoostingRegressor())
                    ])

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_val, y_train_val)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

"""**Perfomance of the XGB Regressor with PCA on the 45df with optimal hyperparameters on the validation set is: 0.62145388**

**Optimal hyperparameters are: n_estimators=500, max_depth=8, learning_rate=0.05**

Evalutaion metric for hyperparameter optimisation used is r^2.
"""

# Do this at the end for XGB
reg = Pipeline([('PCA', pca),
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05)) # insert your best parameters here
                    ])

reg.fit(X_train_val, y_train_val)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

x = y_test
y = y_pred

plt.scatter(x, y, alpha=0.4, label=f"r^2 = {round(r2, 2)}, RMSE = {round(np.sqrt(mse), 2)}")
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('XGB with PCA on the test set for 45_bandgap')
plt.legend()
plt.show()

"""1. **Random Forest**"""

assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == df.shape[0]

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)

mse = mean_squared_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

plt.scatter(y_valid, y_pred)

plt.ylabel('Predicted')
plt.xlabel('Measured')
plt.show()

# Random Forest 45 Hyperparameter optimisation

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define the parameter grid for GridSearchCV
param_grid = {
    #'regressor__n_estimators': [1000],
    #'regressor__max_features': [1.0, 'sqrt', 'log2']
    'regressor__max_depth': [8, 10, 12],
    'regressor__min_samples_split': [4, 6, 8],
    'regressor__min_samples_leaf': [3, 5, 7]
}

rf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", RandomForestRegressor(n_estimators=1000, max_features=1.0))
                    ])

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1,  return_train_score=True)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_val, y_train_val)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

# Results for all parameter combinations, lookibg fo the best performance
grid_search.cv_results_

"""Extra trees on 45"""

# Extra Trees 45 Hyperparameter optimisation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Define the parameter grid for GridSearchCV
param_grid = {
    #'regressor__n_estimators': [1000],
    #'regressor__max_features': [1.0, 'sqrt', 'log2']
    #'regressor__max_depth': [18, 23, 28],
    #'regressor__min_samples_split': [4, 6],
    #'regressor__min_samples_leaf': [3, 5]
}

rf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", ExtraTreesRegressor(n_estimators=1000, max_features=1.0, min_samples_leaf=3, min_samples_split=4, max_depth=18))
                    ])

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1,  return_train_score=True)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_val, y_train_val)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

grid_search.cv_results_

"""**Perfomance of the Extra Trees Regressor on the 45df with optimal hyperparameters on the validation set is mean_test_score: 0.699859**

**Optimal hyperparameters are: n_estimators=1000, max_features=1.0, min_samples_leaf=3, min_samples_split=4, max_depth=18**
"""

# Do this at the end for ET
rf = Pipeline([ ("scaler", StandardScaler()),
                    ("regressor", ExtraTreesRegressor(n_estimators=1000, max_features=1.0, min_samples_leaf=3, min_samples_split=4, max_depth=18)) # insert your best parameters here
                    ])

rf.fit(X_train_val, y_train_val)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

x = y_test
y = y_pred

plt.scatter(x, y, alpha=0.4, label=f"r^2 = {round(r2, 2)}, RMSE = {round(np.sqrt(mse), 2)}")
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Extra Trees on the test set for 45_bandgap')
plt.legend()
plt.show()

"""Extra Trees with PCA on 45"""

#Extra Trees with PCA 45 Hyperparameter optimisation

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

param_grid = {
     'regressor__n_estimators': [300, 500, 1000],
    'regressor__max_features': [1.0, 'sqrt', 'log2'],
    'regressor__max_depth': [16, 18, 23],
    'regressor__min_samples_split': [2, 4, 6],
    'regressor__min_samples_leaf': [3, 5]
}
pca = PCA(n_components=10)

et = Pipeline([("scaler", StandardScaler()),
               ('PCA', pca),
               ("regressor", ExtraTreesRegressor())
            ])

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=et, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, return_train_score=True)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_val, y_train_val)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

"""2. **Gaussian process regression**

"""

from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    #"regressor__kernel": [RBF(), None],
    # "regressor__alpha": [1e-5, 1e-4, 1e-3, 0.01, 0.1],
    "regressor__alpha": [0.1, 1, 10],
    #"regressor__optimizer": ["fmin_l_bfgs_b", "fmin_ncg"],
    #"regressor__n_restarts_optimizer": [0, 1, 2]
}

# Initialize GaussianProcessRegressor
gpr = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", GaussianProcessRegressor())
                    ])

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=gpr, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_val, y_train_val)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

# Results for all parameter combinations, lookibg fo the best performance
grid_search.cv_results_

# Do this at the end for Gaussian
gpr = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0)) # insert your best parameters here
                    ])

gpr.fit(X_train_val, y_train_val)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

"""3. **XGB Regressor**"""

#XGB Regressor 45 Hyperparameter optimisation

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
#from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    "regressor__n_estimators":[100, 300, 500] ,
    "regressor__max_depth": [2, 4, 6],
    #"regressor__min_samples_split": 5,
    "regressor__learning_rate": [0.1, 0.01, 0.001, 0.0001],
    #"regressor__loss": "squared_error",
}


reg = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor()) #GradientBoostingRegressor())
                    ])

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_val, y_train_val)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

"""First time run results: Best parameters found: {'regressor__learning_rate': 0.1, 'regressor__max_depth': 6, 'regressor__n_estimators': 300}"""

# Results for all parameter combinations, lookibg fo the best performance
grid_search.cv_results_

param_grid = {
    #"regressor__n_estimators":[100, 300, 500] ,
    #"regressor__max_depth": [6, 8, 10],
    #"regressor__min_samples_split": 5,
    "regressor__learning_rate": [0.5, 0.1, 0.05, 0.01],
    #"regressor__loss": "squared_error",
}


reg = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor(n_estimators=300, max_depth=8)) #GradientBoostingRegressor())
                    ])

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_val, y_train_val)

# Retrieve the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

# Results for all parameter combinations, lookibg fo the best performance
grid_search.cv_results_

"""**Perfomance of the XGB Regressor on the 45df with optimal hyperparameters on the validation set is: 0.70288743**

**Optimal hyperparameters are: n_estimators=300, max_depth=8, learning_rate=0.05**
"""

# Do this at the end for XGB
gpr = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05)) # insert your best parameters here
                    ])

gpr.fit(X_train_val, y_train_val)

y_pred = gpr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

x = y_test
y = y_pred

plt.scatter(x, y, alpha=0.4, label=f"r^2 = {round(r2, 2)}, RMSE = {round(np.sqrt(mse), 2)}")
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('XGBoost on the test set')
plt.legend()
plt.show()

"""4. **MLP Regressor**"""

#We dont need this right now, only later when modelling.
#Joining data sets into one data frame

df_120 = df_120.set_index(['MOF'])
df_sine = df_sine.set_index(['MOF'])
#df_qmof_refcodes = df_qmof_refcodes.set_index(['MOF'])
df_ofm_fp = df_ofm_fp.set_index(['MOF'])


df_joinn = df_45.join(df_120, how='outer', rsuffix='_120').join(df_sine, how='outer', rsuffix='_sine').join(df_ofm_fp, how='outer', rsuffix='ofm_fp')