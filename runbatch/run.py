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


file_path = "filtered_dataset_channel3_candsleq10.h5"
with h5py.File(file_path, "r") as f:
    dataset = f["Channel3"][:] 
    num_samples, total_columns = dataset.shape
    num_features = 52

X_data = dataset[:, :num_features].astype(np.float32)
Y_data = dataset[:, num_features:].astype(np.float32)

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

predictions = model.predict(X_test_scaled)
predictions_original = np.zeros_like(predictions)
predictions_original[:, 0] = scaler_Y_0.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
predictions_original[:, 1] = scaler_Y_1.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()

true_C = Y_test[:, 0]
true_S = Y_test[:, 1]
pred_C = predictions_original[:, 0]
pred_S = predictions_original[:, 1]



relative_error_C = (pred_C - true_C) / true_C
relative_error_S = (pred_S - true_S) / true_S

mean_C = np.mean(relative_error_C)
std_C = np.std(relative_error_C)
mean_S = np.mean(relative_error_S)
std_S = np.std(relative_error_S)

result_df = pd.DataFrame([[mean_C, std_C, mean_S, std_S]], columns=["mean_C", "std_C", "mean_S", "std_S"])
result_df.to_csv("32_16_8_model_100_results.csv", mode='a', header=False, index=False)

print(f"Experiment completed: mean_C={mean_C:.4f}, std_C={std_C:.4f}, mean_S={mean_S:.4f}, std_S={std_S:.4f}")
