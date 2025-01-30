import h5py
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



file_path = "dataset_channel3_final_filtered.h5"
with h5py.File(file_path, "r") as f:
    dataset = f["Channel3"][:] 
    num_samples, total_columns = dataset.shape
    num_features = 1024

X_data = dataset[:, :num_features][:, ::20].astype(np.float32)
Y_data = dataset[:, num_features:].astype(np.float32)

new_num_features = X_data.shape[1]

train_split = int(num_samples * 0.7)
val_split = int(num_samples * 0.9)

X_train = X_data[:train_split]
Y_train = Y_data[:train_split]

X_val = X_data[train_split:val_split]
Y_val = Y_data[train_split:val_split]

X_test = X_data[val_split:]
Y_test = Y_data[val_split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

scaler_Y_0 = MinMaxScaler()
scaler_Y_1 = MinMaxScaler()

Y_train_scaled = np.zeros_like(Y_train)
Y_val_scaled = np.zeros_like(Y_val)

Y_train_scaled[:, 0] = scaler_Y_0.fit_transform(Y_train[:, 0].reshape(-1, 1)).flatten()
Y_train_scaled[:, 1] = scaler_Y_1.fit_transform(Y_train[:, 1].reshape(-1, 1)).flatten()

Y_val_scaled[:, 0] = scaler_Y_0.transform(Y_val[:, 0].reshape(-1, 1)).flatten()
Y_val_scaled[:, 1] = scaler_Y_1.transform(Y_val[:, 1].reshape(-1, 1)).flatten()


model = models.Sequential([
    layers.Input(shape=(52,)),
    
    # Layer 1: 32 neurons
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    # Layer 2: 16 neurons
    layers.Dense(16),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    # Layer 3: 8 neurons
    layers.Dense(8),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    # Output layer: 2 neurons for the regression targets
    layers.Dense(2, activation='relu')
])

# --------------------------------------------------------------
# 3. Compile the model
# --------------------------------------------------------------
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

epochs = 200
batch_size = 32

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=20, min_lr=1e-8)

history = model.fit(
    X_train_scaled, 
    Y_train_scaled,
    validation_data=(X_val_scaled, Y_val_scaled),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr]
)

# --------------------------------------------------------------
# 5. Evaluate the model
# --------------------------------------------------------------
val_loss, val_mae = model.evaluate(X_val_scaled, Y_val_scaled, verbose=0)
print(f"Validation MSE: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

current_lr = model.optimizer.learning_rate.numpy()
print(f"Current learning rate: {current_lr}")


predictions = model.predict(X_test_scaled)
predictions_original = np.zeros_like(predictions)
predictions_original[:, 0] = scaler_Y_0.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
predictions_original[:, 1] = scaler_Y_1.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()

print("Predictions shape:", predictions.shape)
print("First 5 predictions:\n", predictions[:5])
print("First 5 original predictions:\n", predictions_original[:5])
print("First 5 true values (original scale):\n", Y_test[:5])
