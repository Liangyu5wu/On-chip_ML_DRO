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

dfs = pd.read_csv('true_pred_S_1ns.csv')

true_S = dfs['True_S']
pred_S = dfs['Pred_S']

dfc = pd.read_csv('true_pred_C_1ns.csv')

true_C = dfc['True_C']
pred_C = dfc['Pred_C']

dfsb = pd.read_csv('true_pred_S_both_1ns.csv')

true_S_both = dfsb['True_S']
pred_S_both = dfsb['Pred_S']

dfcb = pd.read_csv('true_pred_C_both_1ns.csv')

true_C_both = dfcb['True_C']
pred_C_both = dfcb['Pred_C']


plt.figure(figsize=(8, 6))
plt.scatter(pred_S, pred_C , alpha=0.5, s=10, c='blue')
plt.xlabel("S")
plt.ylabel("C")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(true_S, true_C , alpha=0.5, s=10, c='blue')
plt.xlabel("S")
plt.ylabel("C")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(pred_S_both, pred_C_both , alpha=0.5, s=10, c='blue')
plt.xlabel("S")
plt.ylabel("C")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(true_S_both, true_C_both , alpha=0.5, s=10, c='blue')
plt.xlabel("S")
plt.ylabel("C")
plt.grid(True)
plt.tight_layout()
plt.show()


splice = 150

pred_C_down = pred_C[::splice]
pred_S_down = pred_S[::splice]


true_C_both_down = true_C_both[::splice]
true_S_both_down = true_S_both[::splice]
pred_C_both_down = pred_C_both[::splice]
pred_S_both_down = pred_S_both[::splice]


true_C_both_down = np.array(true_C_both_down)
true_S_both_down = np.array(true_S_both_down)
pred_C_both_down = np.array(pred_C_both_down)
pred_S_both_down = np.array(pred_S_both_down)
pred_C_down = np.array(pred_C_down)
pred_S_down = np.array(pred_S_down)


colors = np.random.rand(len(true_C_both_down), 3)

plt.figure(figsize=(8, 6))


for i in range(len(pred_S_both_down)):
    plt.scatter(pred_S_both_down[i], pred_C_both_down[i], alpha=0.5, s=50, c=[colors[i]], marker='x')

for i in range(len(true_S_both_down)):
    plt.scatter(true_S_both_down[i], true_C_both_down[i], alpha=0.5, s=50, c=[colors[i]], marker='o')

for i in range(len(pred_C_down)):
    plt.scatter(pred_S_down[i], pred_C_down[i], alpha=0.5, s=50, c=[colors[i]], marker='v')

coefficients = np.polyfit(true_S_both, true_C_both, 1)
polynomial = np.poly1d(coefficients)
print(f"coefficients: {coefficients}")

x_fit = np.linspace(min(true_S_both), max(true_S_both), 100)
y_fit = polynomial(x_fit)

plt.plot(x_fit, y_fit, 'g--', label='Fit Line for true value')


plt.scatter([], [], c='red', marker='o', label='True')
plt.scatter([], [], c='blue', marker='x', label='Pred - both')
plt.scatter([], [], c='green', marker='v', label='Pred - single')

plt.legend()

plt.xlabel("S")
plt.ylabel("C")

plt.grid(True)

plt.tight_layout()

plt.show()
