import h5py
import numpy as np
import os

file_path = "./../csfakewf/dataset_channel3_final.h5"

with h5py.File(file_path, "r") as f:
    dataset = f["Channel3"][:] 
    num_samples, total_columns = dataset.shape
    num_features = 1024

last_three_columns = dataset[:, -3:]

nan_rows = np.any(np.isnan(last_three_columns), axis=1)

filtered_dataset = dataset[~nan_rows]

print(f"Original dataset shape: {dataset.shape}")
print(f"Filtered dataset shape: {filtered_dataset.shape}")

filtered_file_path = "filtered_dataset_no_nan.h5"
with h5py.File(filtered_file_path, "w") as f:
    f.create_dataset("Channel3", data=filtered_dataset)

print(f"Filtered dataset saved to {filtered_file_path}")
