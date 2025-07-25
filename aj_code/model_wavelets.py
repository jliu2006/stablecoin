# -*- coding: utf-8 -*-
"""Model_Wavelets.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1W14MXMyyvb2SFqe-F-GtF1fzdS2KY8c4
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import yfinance as yf

def pad_rows_to_multiple_of_M(X, M):
    n_rows, n_cols = X.shape
    remainder = n_rows % M
    if remainder == 0:
        return X
    pad_n = M - remainder
    padding = np.zeros((pad_n, n_cols))
    return np.vstack([X, padding])

# Modified build_dwt_matrix to use NumPy
def build_dwt_matrix(signal_length, M, filters):
        k = signal_length // M
        Lf = len(filters[0])
        W = np.zeros((signal_length, signal_length), dtype=np.float32) # Use NumPy
        for band in range(M):
            h = np.asarray(filters[band], dtype=np.float32) # Use NumPy
            for i in range(k):
                row = band * k + i
                cols = [(i * M + j) % signal_length for j in range(Lf)]
                W[row, cols] = h
        Q, _ = np.linalg.qr(W.T) # Use NumPy
        return Q.T

all_filters = {
    (3, 2): [  # From Lin et al. (Table 1)
        np.array([ 0.33838609728386, 0.53083618701374, 0.72328627674361, 0.23896417190576, 0.04651408217589, -0.14593600755399 ]),
        np.array([-0.11737701613483, 0.54433105395181, -0.01870574735313, -0.69911956479289, -0.13608276348796,  0.42695403781698 ]),
        np.array([ 0.40363686892892, -0.62853936105471, 0.46060475252131, -0.40363686892892, -0.07856742013185,  0.24650202866523 ])
    ],
    (4, 2): [  # Your default 4-band 2-reg filters
        np.array([-0.067371764, 0.094195111, 0.40580489, 0.567371764, 0.567371764, 0.40580489, 0.094195111, -0.067371764]),
        np.array([-0.094195111, 0.067371764, 0.567371764, 0.40580489, -0.40580489, -0.567371764, -0.067371764, 0.094195111]),
        np.array([-0.094195111, -0.067371764, 0.567371764, -0.40580489, -0.40580489, 0.567371764, -0.067371764, -0.094195111]),
        np.array([-0.067371764, -0.094195111, 0.40580489, -0.567371764, 0.567371764, -0.40580489, 0.094195111, 0.067371764])
    ],
    (4, 4): [  # From Lin et al. Table 2
        np.array([0.08571302, 0.1931394393, 0.3491805097, 0.5616494215, 0.4955029828, 0.4145647737, 0.2190308939, -0.1145361261,
                  -0.0952930728, -0.1306948909, -0.0827496793, 0.0719795354, 0.0140770701, 0.0229906779, 0.0145382757, -0.0190928308]),
        np.array([-0.1045086525, 0.1183282069, -0.1011065044, -0.0115563891, 0.6005913823, -0.2550401616, -0.4264277361, -0.0827398180,
                  0.0722022649, 0.2684936992, 0.1691549718, -0.4437039320, 0.0849964877, 0.1388163056, 0.0877812188, -0.1152813433]),
        np.array([0.2560950163, -0.2048089157, -0.2503433230, -0.2484277272, 0.4477496752, 0.0010274000, -0.0621881917, 0.5562313118,
                  -0.2245618041, -0.3300536827, -0.2088643503, 0.2202951830, 0.0207171125, 0.0338351983, 0.0213958651, -0.0280987676]),
        np.array([0.1839986022, -0.6622893130, 0.6880085746, -0.1379502447, 0.0446493766, -0.0823301969, -0.0923899104, -0.0233349758,
                  0.0290655661, 0.0702950474, 0.0443561794, -0.0918374833, 0.0128845052, 0.0210429802, 0.0133066389, -0.0174753464])
    ]
}
def run_wavelet_pipeline(X, M, L):
  X = pad_rows_to_multiple_of_M(X, M)
  n_rows = X.shape[0]
  filters = all_filters[(M, L)]

  X_cpu = np.asarray(X, dtype=np.float32) # Use NumPy
  W_cpu = build_dwt_matrix(n_rows, M, filters) # build_dwt_matrix now uses NumPy
  wavelet_cpu = W_cpu @ X_cpu # Use NumPy

  W_check = W_cpu @ W_cpu.T # Use NumPy
  I_check = np.eye(W_cpu.shape[0], dtype=np.float32) # Use NumPy
  max_diff1 = np.max(np.abs(W_check - I_check)) # Use NumPy
  print("W @ W.T max diff from I:", float(max_diff1))

  W1_check = W_cpu.T @ W_cpu # Use NumPy
  I_check = np.eye(W_cpu.shape[1], dtype=np.float32) # Use NumPy
  max_diff2 = np.max(np.abs(W1_check - I_check)) # Use NumPy
  print("W.T @ W max diff from I:", float(max_diff2))

  k = wavelet_cpu.shape[0] // M
  bands_cpu = [wavelet_cpu[i * k:(i + 1) * k, :] for i in range(M)] # Use NumPy
  max_rows = X.shape[0]

  reconstructions = [
      W_cpu[i * k:(i + 1) * k, :].T @ bands_cpu[i][:max_rows, :] # Use NumPy
      for i in range(M)
  ]

  return reconstructions  # e.g., [A1, D1, D2, D3]

!pip install torchsde
import torchsde

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import yfinance as yf
import torchsde

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torchsde
import torch.optim as optim

# import pywt # Need to keep pywt for potential future use or if CWT is still needed elsewhere.


def create_lookback_sequence(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def preprocess(ticker, lookback=14, M=4, L=2): # Added M and L parameters for wavelet pipeline
    data = yf.download(ticker).Close.values.reshape(-1, 1) # Get values and reshape for pipeline

    reconstructions = run_wavelet_pipeline(data, M, L)

    # For simplicity, let's use the first reconstruction (approximation) as the input data.
    # You might want to experiment with different reconstructions or combine information from multiple bands.
    input_data = reconstructions[0] # Use the first reconstruction

    # Create lookback sequences
    X, y = create_lookback_sequence(input_data, lookback)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_scaler = MinMaxScaler(feature_range=(1,10))
    y_scaler = MinMaxScaler(feature_range=(1,10))

    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1]) # Reshape based on the last dimension
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1]) # Reshape based on the last dimension


    # Scale the data
    X_train_scaled_reshaped = X_scaler.fit_transform(X_train_reshaped)
    X_test_scaled_reshaped = X_scaler.transform(X_test_reshaped)

    # Reshape back to sequence format
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)

    # Print min/max of y_test before scaling
    print(f"Min of y_test before scaling: {np.min(y_test)}")
    print(f"Max of y_test before scaling: {np.max(y_test)}")

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    # Print min/max of y_test after scaling
    print(f"Min of y_test after scaling: {np.min(y_test_scaled)}")
    print(f"Max of y_test after scaling: {np.max(y_test_scaled)}")


    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32) # Removed reshape(-1, lookback, 1)
    # Ensure X_test_scaled has the correct shape before creating tensor
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32) # Removed reshape(-1, lookback, 1)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_scaler, y_scaler, input_data

# Call preprocess with desired parameters
# You might need to adjust M and L based on the available filters in all_filters
train_loader, test_loader, X_scaler, y_scaler, input_data = preprocess('USDT-USD', lookback=14, M=4, L=2)

import torch.nn as nn
import torchsde
import torch

class NeuralSDE(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(NeuralSDE, self).__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"


        self.drift_nn = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.diffusion_nn = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def f(self, t, y):
        return self.drift_nn(y)

    def g(self, t, y):
        return self.diffusion_nn(y)


    def forward(self, ts, y0):
        return torchsde.sdeint(self, y0, ts, dt=0.01,method = 'euler')

hidden_size = 1024
output_size = 1

model = NeuralSDE(hidden_size, output_size)


print("Neural SDE model defined with f and g methods calling internal networks.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()

        y0 = inputs[:, 0, :]

        ts = torch.linspace(0, 1, inputs.shape[1])

        solutions = model(ts, y0)

        predictions = solutions[-1]
        loss = criterion(predictions, targets)
        print(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

print("Training finished.")

model.eval()

actual_values = []
predicted_values = []

with torch.no_grad():
    for inputs, targets in test_loader:

        y0 = inputs[:, 0, :]
        ts = torch.linspace(0, 1, inputs.shape[1])

        solutions = model(ts, y0)

        predictions = solutions[-1]

        actual_values.append(targets.squeeze().numpy())
        predicted_values.append(predictions.squeeze().numpy())

actual_values = np.concatenate(actual_values)
predicted_values = np.concatenate(predicted_values)

print("Prediction generation complete.")

import matplotlib.pyplot as plt
import numpy as np

actual_values_original_scale = y_scaler.inverse_transform(actual_values.reshape(-1, 1))
predicted_values_original_scale = y_scaler.inverse_transform(predicted_values.reshape(-1, 1))

shifted_predicted_values = predicted_values_original_scale[14:]
corresponding_actual_values = actual_values_original_scale[:-14]

plt.figure(figsize=(12, 6))
plt.plot(corresponding_actual_values[:, 0], label='Actual Values (Shifted)')
plt.plot(shifted_predicted_values[:, 0], label='Predicted Values (Shifted Left by 14 Days)')
plt.title('Actual vs Predicted Values (Test Set, Predicted Shifted Left)')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()
