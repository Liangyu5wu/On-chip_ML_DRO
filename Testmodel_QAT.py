import h5py
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.linear_model import LinearRegression
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import qkeras
from qkeras import QDense, QActivation, quantizers

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


file_path = "./../dataset_channel3_final_filtered.h5"
with h5py.File(file_path, "r") as f:
    dataset = f["Channel3"][:] 
    num_samples, total_columns = dataset.shape
    num_features = 1024

X_data = dataset[:, :num_features][:, ::20].astype(np.float32)
# X_data = dataset[:, :num_features].astype(np.float32)
Y_data = dataset[:, num_features:].astype(np.float32)

new_num_features = X_data.shape[1]
print(f"num_samples: {num_samples}, Original num_features: {num_features}, New num_features: {new_num_features}")
print(f"X_data shape: {X_data.shape}, dtype: {X_data.dtype}")
print(f"Y_data shape: {Y_data.shape}, dtype: {Y_data.dtype}")

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

    QDense(32, kernel_quantizer=quantizers.quantized_bits(8, 3), 
           bias_quantizer=quantizers.quantized_bits(8, 3)),
    layers.BatchNormalization(),
    QActivation(activation=quantizers.quantized_relu(8)),

    QDense(16, kernel_quantizer=quantizers.quantized_bits(8, 3), 
           bias_quantizer=quantizers.quantized_bits(8, 3)),
    layers.BatchNormalization(),
    QActivation(activation=quantizers.quantized_relu(8)),

    QDense(8, kernel_quantizer=quantizers.quantized_bits(8, 3), 
           bias_quantizer=quantizers.quantized_bits(8, 3)),
    layers.BatchNormalization(),
    QActivation(activation=quantizers.quantized_relu(8)),

    QDense(2, kernel_quantizer=quantizers.quantized_bits(8, 3), 
           bias_quantizer=quantizers.quantized_bits(8, 3)),
    QActivation(activation=quantizers.quantized_relu(8))
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the model
epochs = 300
batch_size = 32

early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=20, min_lr=1e-8)

history = model.fit(
    X_train_scaled, 
    Y_train_scaled,
    validation_data=(X_val_scaled, Y_val_scaled),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate the model
val_loss, val_mae = model.evaluate(X_val_scaled, Y_val_scaled, verbose=0)
print(f"Validation MSE: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

current_lr = model.optimizer.learning_rate.numpy()
print(f"Current learning rate: {current_lr}")

# Predictions
predictions = model.predict(X_test_scaled)

predictions_original = np.zeros_like(predictions)
predictions_original[:, 0] = scaler_Y_0.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
predictions_original[:, 1] = scaler_Y_1.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()

print("Predictions shape:", predictions.shape)
print("First 5 predictions:\n", predictions[:5])
print("First 5 original predictions:\n", predictions_original[:5])
print("First 5 true values (original scale):\n", Y_test[:5])

true_C = Y_test[:, 0]
true_S = Y_test[:, 1]

pred_C = predictions_original[:, 0]
pred_S = predictions_original[:, 1]

absolute_error_C = pred_C - true_C
absolute_error_S = pred_S - true_S

mean_C = np.mean(absolute_error_C)
std_C = np.std(absolute_error_C)
mean_S = np.mean(absolute_error_S)
std_S = np.std(absolute_error_S)

plt.figure(figsize=(8, 6))
plt.scatter(true_S, true_C , alpha=0.5, s=10, c='blue')
plt.scatter(pred_S, pred_C , alpha=0.5, s=10, c='yellow')
plt.xlabel("S")
plt.ylabel("C")
plt.grid(True)
plt.tight_layout()
plt.show()

error_s = (true_S - pred_S) / true_S * 100
error_s = error_s[np.abs(error_s) <= 80]

mean_s = np.mean(error_s)
rms_s = np.sqrt(np.mean(error_s**2))
std_s = np.std(error_s)

within_25_percent_s = np.sum(np.abs(error_s) <= 25) / len(error_s) * 100
within_20_percent_s = np.sum(np.abs(error_s) <= 20) / len(error_s) * 100
within_10_percent_s = np.sum(np.abs(error_s) <= 10) / len(error_s) * 100
within_5_percent_s = np.sum(np.abs(error_s) <= 5) / len(error_s) * 100



error_c = (true_C - pred_C) / true_C * 100
error_c = error_c[np.abs(error_c) <= 80]

mean_c = np.mean(error_c)
rms_c = np.sqrt(np.mean(error_c**2))
std_c = np.std(error_c)

within_25_percent_c = np.sum(np.abs(error_c) <= 25) / len(error_c) * 100
within_20_percent_c = np.sum(np.abs(error_c) <= 20) / len(error_c) * 100
within_10_percent_c = np.sum(np.abs(error_c) <= 10) / len(error_c) * 100
within_5_percent_c = np.sum(np.abs(error_c) <= 5) / len(error_c) * 100

print(f'S - Mean: {mean_s:.2f}')
print(f'S - RMS: {rms_s:.2f}')
print(f'S - Standard Deviation: {std_s:.2f}')
print(f'S - Within 25% Error: {within_25_percent_s:.2f}%')
print(f'S - Within 20% Error: {within_20_percent_s:.2f}%')
print(f'S - Within 10% Error: {within_10_percent_s:.2f}%')
print(f'S - Within 5% Error: {within_5_percent_s:.2f}%')

print(f'C - Mean: {mean_c:.2f}')
print(f'C - RMS: {rms_c:.2f}')
print(f'C - Standard Deviation: {std_c:.2f}')
print(f'C - Within 25% Error: {within_25_percent_c:.2f}%')
print(f'C - Within 20% Error: {within_20_percent_c:.2f}%')
print(f'C - Within 10% Error: {within_10_percent_c:.2f}%')
print(f'C - Within 5% Error: {within_5_percent_c:.2f}%')

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].hist(error_s, bins=150, edgecolor='black')
axs[0].set_title('Histogram of (true_S - pred_S) / true_S')
axs[0].set_xlabel('(true_S - pred_S) / true_S (%)')
axs[0].set_ylabel('Frequency')
axs[0].set_xlim(-100, 100)
axs[0].grid(True)

info_text_s = (f'Mean: {mean_s:.2f}\n'
               f'RMS: {rms_s:.2f}\n'
               f'StD: {std_s:.2f}\n'
               f'Within 25% Error: {within_25_percent_s:.2f}%\n'
               f'Within 20% Error: {within_20_percent_s:.2f}%\n'
               f'Within 10% Error: {within_10_percent_s:.2f}%\n'
               f'Within 5%   Error: {within_5_percent_s:.2f}%')

axs[0].text(0.55, 0.95, info_text_s, transform=axs[0].transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

axs[1].hist(error_c, bins=150, edgecolor='black')
axs[1].set_title('Histogram of (true_C - pred_C) / true_C')
axs[1].set_xlabel('(true_C - pred_C) / true_C (%)')
axs[1].set_ylabel('Frequency')
axs[1].set_xlim(-100, 100)
axs[1].grid(True)

info_text_c = (f'Mean: {mean_c:.2f}\n'
               f'RMS: {rms_c:.2f}\n'
               f'StD: {std_c:.2f}\n'
               f'Within 25% Error: {within_25_percent_c:.2f}%\n'
               f'Within 20% Error: {within_20_percent_c:.2f}%\n'
               f'Within 10% Error: {within_10_percent_c:.2f}%\n'
               f'Within 5%   Error: {within_5_percent_c:.2f}%')

axs[1].text(0.55, 0.95, info_text_c, transform=axs[1].transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
# plt.savefig("stas_cs_full_1sigma.png", dpi=300, bbox_inches='tight')
plt.show()
