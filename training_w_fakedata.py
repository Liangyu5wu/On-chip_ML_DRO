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



real_file_path = "filtered_dataset_channel3_candsleq10.h5"
with h5py.File(real_file_path, "r") as f:
    real_dataset = f["Channel3"][:] 
    real_num_samples, real_total_columns = real_dataset.shape
    real_num_features = 52

real_X_data = real_dataset[:, :real_num_features].astype(np.float32)
real_Y_data = real_dataset[:, real_num_features:].astype(np.float32)

print(f"real_X_data shape: {real_X_data.shape}, dtype: {real_X_data.dtype}")
print(f"real_Y_data shape: {real_Y_data.shape}, dtype: {real_Y_data.dtype}")

fake_file_path_a = "waveform_data_10000.h5"
with h5py.File(fake_file_path_a, "r") as f:
    fake_dataset_a = f["waveforms"][:] 
    fake_num_features_a = 1024

fake_X_data_a = fake_dataset_a[:, :fake_num_features_a][:, ::20].astype(np.float32)
fake_Y_data_a = fake_dataset_a[:, fake_num_features_a:].astype(np.float32)


print(f"fake_X_data_a shape: {fake_X_data_a.shape}, dtype: {fake_X_data_a.dtype}")
print(f"fake_Y_data_a shape: {fake_Y_data_a.shape}, dtype: {fake_Y_data_a.dtype}")

fake_file_path_b = "waveform_data_5000_c_0_15_s_0_50.h5"
with h5py.File(fake_file_path_b, "r") as f:
    fake_dataset_b = f["waveforms"][:] 
    fake_num_features_b = 1024

fake_X_data_b = fake_dataset_b[:, :fake_num_features_b][:, ::20].astype(np.float32)
fake_Y_data_b = fake_dataset_b[:, fake_num_features_b:].astype(np.float32)


print(f"fake_X_data_b shape: {fake_X_data_b.shape}, dtype: {fake_X_data_b.dtype}")
print(f"fake_Y_data_b shape: {fake_Y_data_b.shape}, dtype: {fake_Y_data_b.dtype}")


fake_file_path_c = "waveform_data_5000_c_0_15_s_0_50.h5"
with h5py.File(fake_file_path_c, "r") as f:
    fake_dataset_c = f["waveforms"][:] 
    fake_num_features_c = 1024

fake_X_data_c = fake_dataset_c[:, :fake_num_features_c][:, ::20].astype(np.float32)
fake_Y_data_c = fake_dataset_c[:, fake_num_features_c:].astype(np.float32)


print(f"fake_X_data_c shape: {fake_X_data_c.shape}, dtype: {fake_X_data_c.dtype}")
print(f"fake_Y_data_c shape: {fake_Y_data_c.shape}, dtype: {fake_Y_data_c.dtype}")


fake_file_path_e = "waveform_data_10000_c_0_10_s_0_10.h5"
with h5py.File(fake_file_path_e, "r") as f:
    fake_dataset_e = f["waveforms"][:] 
    fake_num_features_e = 1024

fake_X_data_e = fake_dataset_e[:, :fake_num_features_e][:, ::20].astype(np.float32)
fake_Y_data_e = fake_dataset_e[:, fake_num_features_e:].astype(np.float32)


print(f"fake_X_data_e shape: {fake_X_data_e.shape}, dtype: {fake_X_data_e.dtype}")
print(f"fake_Y_data_e shape: {fake_Y_data_e.shape}, dtype: {fake_Y_data_e.dtype}")



train_split = int(real_num_samples * 0.5)
val_split = int(real_num_samples * 0.8)

X_train = real_X_data[:train_split]
Y_train = real_Y_data[:train_split]

X_val = real_X_data[train_split:val_split]
Y_val = real_Y_data[train_split:val_split]

X_test = real_X_data[val_split:]
Y_test = real_Y_data[val_split:]


print(f"X_train before adding fake data: {X_train.shape}, Y_train before adding fake data: {Y_train.shape}")

X_train = np.concatenate((X_train, fake_X_data_a), axis=0)
Y_train = np.concatenate((Y_train, fake_Y_data_a), axis=0)

X_train = np.concatenate((X_train, fake_X_data_b), axis=0)
Y_train = np.concatenate((Y_train, fake_Y_data_b), axis=0)

X_train = np.concatenate((X_train, fake_X_data_c), axis=0)
Y_train = np.concatenate((Y_train, fake_Y_data_c), axis=0)

X_train = np.concatenate((X_train, fake_X_data_e), axis=0)
Y_train = np.concatenate((Y_train, fake_Y_data_e), axis=0)

print(f"X_train after adding fake data: {X_train.shape}, Y_train after adding fake data: {Y_train.shape}")

train_indices = np.arange(X_train.shape[0])
np.random.shuffle(train_indices)
X_train = X_train[train_indices]
Y_train = Y_train[train_indices]

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
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
model.summary()

# --------------------------------------------------------------
# 4. Train the model
# --------------------------------------------------------------
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

true_C = Y_test[:, 0]
true_S = Y_test[:, 1]

pred_C = predictions_original[:, 0]
pred_S = predictions_original[:, 1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(14, 6),
    sharex='col',
    gridspec_kw={'height_ratios': [3, 1]}
)

bins = np.linspace(min(min(true_S), min(pred_S)), max(max(true_S), max(pred_S)), 50)
bin_centers_S = (bins[:-1] + bins[1:]) / 2

bins_C = np.linspace(min(min(true_C), min(pred_C)), max(max(true_C), max(pred_C)), 50)
bin_centers_C = (bins_C[:-1] + bins_C[1:]) / 2

counts_true_S, _, _ = ax1.hist(true_S, bins=bins, edgecolor='black', alpha=0.5, color='blue', label="True S")
counts_pred_S, _, _ = ax1.hist(pred_S, bins=bins, edgecolor='black', alpha=0.5, color='red', label="Pred S")
ax1.set_ylabel("Counts")
ax1.set_title("Distribution of True S and Pred S")
ax1.grid(True)
ax1.legend()

ratio_S = np.divide(counts_pred_S, counts_true_S, out=np.zeros_like(counts_pred_S), where=counts_true_S!=0)
ax3.scatter(bin_centers_S, ratio_S, color='purple', label='Pred/True Ratio')
ax3.plot(bin_centers_S, ratio_S, linestyle='--', color='purple')
ax3.axhline(y=1, color='blue', linestyle='--', linewidth=1, label='Ideal Ratio')
ax3.set_xlabel("S Value")
ax3.set_ylabel("Pred/True Ratio")
ax3.set_ylim(0, np.max(ratio_S) * 1.2)
ax3.grid(True)
ax3.legend()

counts_true_C, _, _ = ax2.hist(true_C, bins=bins_C, edgecolor='black', alpha=0.5, color='blue', label="True C")
counts_pred_C, _, _ = ax2.hist(pred_C, bins=bins_C, edgecolor='black', alpha=0.5, color='red', label="Pred C")
ax2.set_ylabel("Counts")
ax2.set_title("Distribution of True C and Pred C")
ax2.grid(True)
ax2.legend()

ratio_C = np.divide(counts_pred_C, counts_true_C, out=np.zeros_like(counts_pred_C), where=counts_true_C!=0)
ax4.scatter(bin_centers_C, ratio_C, color='green', label='Pred/True Ratio')
ax4.plot(bin_centers_C, ratio_C, linestyle='--', color='green')
ax4.axhline(y=1, color='blue', linestyle='--', linewidth=1, label='Ideal Ratio')
ax4.set_xlabel("C Value")
ax4.set_ylabel("Pred/True Ratio")
ax4.set_ylim(0, np.max(ratio_C) * 1.2)
ax4.grid(True)
ax4.legend()

plt.tight_layout()

plt.savefig("TrueS_TrueC_with_Ratio_32168_testA_fulldata.png", dpi=300, bbox_inches='tight')
plt.show()
