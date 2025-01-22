import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
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

print(f"num_samples: {num_samples}, Original num_features: {num_features}, New num_features: {new_num_features}")
print(f"X_data shape: {X_data.shape}, dtype: {X_data.dtype}")
print(f"Y_data shape: {Y_data.shape}, dtype: {Y_data.dtype}")

print("X_data sample:\n", X_data[:5])
print("Y_data sample:\n", Y_data[:5])


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
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

scaler_Y = MinMaxScaler()
Y_train = scaler_Y.fit_transform(Y_train)
Y_val = scaler_Y.transform(Y_val)



model = models.Sequential([
    # Layer 1: 32 neurons
    layers.Dense(32, input_shape=(52,)),
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
    layers.Dense(2, activation='softplus')
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
epochs = 100
batch_size = 32

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=7, min_lr=1e-7)

history = model.fit(
    X_train, 
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr]
    # callbacks=[reduce_lr]
)

# --------------------------------------------------------------
# 5. Evaluate the model
# --------------------------------------------------------------
val_loss, val_mae = model.evaluate(X_val, Y_val, verbose=0)
print(f"Validation MSE: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

current_lr = model.optimizer.learning_rate.numpy()
print(f"Current learning rate: {current_lr}")


predictions = model.predict(X_test)
predictions_original = scaler_Y.inverse_transform(predictions)

print("Predictions shape:", predictions.shape)
print("First 5 predictions:\n", predictions[:5])
print("First 5 predictions:\n", predictions_original[:5])
print("First 5 predictions answers:\n", Y_test[:5])
