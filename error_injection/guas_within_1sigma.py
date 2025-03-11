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


file_path = "./filtered_dataset_no_nan.h5"
with h5py.File(file_path, "r") as f:
    dataset = f["Channel3"][:] 
    num_features = 1024

X_data_full = dataset[:, :num_features][:, ::20].astype(np.float32)
Y_data_full_b = dataset[:, num_features:].astype(np.float32)
Y_data_full = Y_data_full_b[:, 1:3]
Y_data_full_error = Y_data_full_b[:, 4:6]

row_sums = np.sum(Y_data_full, axis=1)
valid_indices = row_sums > 2
X_data_b = X_data_full[valid_indices]
Y_data_b = Y_data_full[valid_indices]
Y_data_error = Y_data_full_error[valid_indices]

C_data_b = Y_data_b[:, 0]
S_data_b = Y_data_b[:, 1]
C_data_error_b = Y_data_error[:, 0]
S_data_error_b = Y_data_error[:, 1]


def generate_within_sigma(loc, scale):
    result = np.random.normal(loc=loc, scale=scale)
    lower_bound = loc - scale
    upper_bound = loc + scale
    while ((result < lower_bound) | (result > upper_bound)).any():
        result = np.where((result < lower_bound) | (result > upper_bound), np.random.normal(loc=loc, scale=scale), result)
    return result

Y_data_a = np.zeros_like(Y_data_b)
Y_data_a[:, 0] = generate_within_sigma(C_data_b, C_data_error_b)
Y_data_a[:, 1] = generate_within_sigma(S_data_b, S_data_error_b)


print(f"X_data shape: {X_data_b.shape}, dtype: {X_data_b.dtype}")
print(f"Y_data shape: {Y_data_a.shape}, dtype: {Y_data_a.dtype}")

non_negative_indices = (Y_data_a >= 0).all(axis=1)
Y_data = Y_data_a[non_negative_indices]
X_data = X_data_b[non_negative_indices]

# 输出清理后的数据形状
print(f"X_data_clean shape: {X_data.shape}, dtype: {X_data.dtype}")
print(f"Y_data_clean shape: {Y_data.shape}, dtype: {Y_data.dtype}")


num_samples, total_columns = Y_data.shape

train_split = int(num_samples * 0.7)
val_split = int(num_samples * 0.9)

X_train = X_data[:train_split]
Y_train = Y_data[:train_split]

X_val = X_data[train_split:val_split]
Y_val = Y_data[train_split:val_split]

X_test = X_data[val_split:]
Y_test = Y_data[val_split:]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

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
    
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(16),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Dense(8),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(2, activation='relu')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

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
