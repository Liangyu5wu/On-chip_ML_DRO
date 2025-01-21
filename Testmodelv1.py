import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = "dataset_channel3_final.h5"
with h5py.File(file_path, "r") as f:
    dataset = f["Channel3"][:] 

    num_samples, total_columns = dataset.shape
    num_features = 1024
    
num_samples = 32501

X_data = dataset[:, :num_features].astype(np.float32)
Y_data = dataset[:, [1025, 1026]].astype(np.float32)

print(f"num_samples: {num_samples}, num_features: {num_features}")
print(f"X_data shape: {X_data.shape}, dtype: {X_data.dtype}")
print(f"Y_data shape: {Y_data.shape}, dtype: {Y_data.dtype}")

print("X_data sample:\n", X_data[:5])
print("Y_data sample:\n", Y_data[:5])


train_split = int(num_samples * 0.8)
X_train = X_data[:train_split]
Y_train = Y_data[:train_split]
X_val = X_data[train_split:]
Y_val = Y_data[train_split:]



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# X_test = X_data[:train_split]
# Y_test = Y_data[:train_split]



model = models.Sequential([
    # Layer 1: 32 neurons
    layers.Dense(32, input_shape=(1024,)),
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
    layers.Dense(2)  
])

# --------------------------------------------------------------
# 3. Compile the model
# --------------------------------------------------------------
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
model.summary()

# --------------------------------------------------------------
# 4. Train the model
# --------------------------------------------------------------
epochs = 100
batch_size = 32

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, 
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr]
)

# --------------------------------------------------------------
# 5. Evaluate the model
# --------------------------------------------------------------
val_loss, val_mae = model.evaluate(X_val, Y_val, verbose=0)
print(f"Validation MSE: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

current_lr = model.optimizer.learning_rate.numpy()
print(f"Current learning rate: {current_lr}")


predictions = model.predict(X_val)
print("Predictions shape:", predictions.shape)
print("First 5 predictions:\n", predictions[:5])
