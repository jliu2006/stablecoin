import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import torchsde
import matplotlib.pyplot as plt

# --- Configuration ---
ASSET_1 = 'USDT-USD'
ASSET_2 = 'COP=X'
START_DATE = '2020-09-01'
END_DATE = '2025-07-02'
LOOKBACK = 45
TIME_STEP = 1
BATCH_SIZE = 64
TRAIN_SPLIT = 0.65
EPOCHS = 5
HIDDEN_SIZE = 1024
INPUT_FEATURES = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-5

# --- Data Preprocessing ---
def preprocess_data(df, lookback, time_step, batch_size=64, split_ratio=0.65):
    features = df.values
    targets = df.values[:, 1]
    features_scaler = MinMaxScaler()
    targets_scaler = MinMaxScaler()
    features_scaled = features_scaler.fit_transform(features)
    targets_scaled = targets_scaler.fit_transform(targets.reshape(-1, 1))

    X, y = [], []
    for i in range(len(features_scaled) - lookback - time_step):
        X.append(features_scaled[i:i+lookback])
        y.append(targets_scaled[i+lookback+time_step])

    X, y = np.array(X), np.array(y)
    split_idx = int(split_ratio * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, features_scaler, targets_scaler

def load_combined_data(asset1, asset2, start, end):
    data1 = yf.download(asset1, start=start, end=end)['Close'].rolling(window=30).mean()
    data2 = yf.download(asset2, start=start, end=end)['Close']
    combined = pd.concat([data1, data2], axis=1, join='outer').dropna()
    combined.columns = ['Asset1', 'Asset2']
    return combined

# --- Model Definition ---
class NeuralSDE(nn.Module):
    def __init__(self, hidden_size, input_features, output_size, lookback):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.lookback = lookback
        self.drift_lstm = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.diffusion_lstm = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.drift_linear = nn.Linear(hidden_size + output_size, output_size)
        self.diffusion_linear = nn.Linear(hidden_size + output_size, output_size)
        self._lstm_hidden_buffer = None

    def f(self, t, y):
        if self._lstm_hidden_buffer is None:
            raise RuntimeError("LSTM hidden state buffer is not set. Call forward with input_sequence first.")
        combined = torch.cat((self._lstm_hidden_buffer, y), dim=1)
        return self.drift_linear(combined)

    def g(self, t, y):
        if self._lstm_hidden_buffer is None:
            raise RuntimeError("LSTM hidden state buffer is not set. Call forward with input_sequence first.")
        combined = torch.cat((self._lstm_hidden_buffer, y), dim=1)
        return self.diffusion_linear(combined)

    def forward(self, ts, y0, input_seq):
        lstm_output, (hidden, _) = self.drift_lstm(input_seq)
        self._lstm_hidden_buffer = hidden.squeeze(0)
        solution = torchsde.sdeint(self, y0, ts, dt=0.01, method='euler')
        self._lstm_hidden_buffer = None
        return solution

# --- Training Routine ---
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            y0 = inputs[:, 0, 1].unsqueeze(1)
            ts = torch.linspace(0, 1, inputs.size(1))
            solutions = model(ts, y0, inputs)
            preds = solutions[-1]
            loss = criterion(preds, targets)
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# --- Evaluation Routine ---
def evaluate_model(model, test_loader, target_scaler, device, lookback):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            y0 = inputs[:, 0, 1].unsqueeze(1)
            ts = torch.linspace(0, 1, inputs.size(1))
            solutions = model(ts, y0, inputs)
            preds = solutions[-1]
            actual.append(targets.squeeze().numpy())
            predicted.append(preds.squeeze().numpy())
    actual = np.concatenate(actual)
    predicted = np.concatenate(predicted)
    actual_orig = target_scaler.inverse_transform(actual.reshape(-1, 1))
    predicted_orig = target_scaler.inverse_transform(predicted.reshape(-1, 1))
    # Align predictions for plotting
    shifted_pred = predicted_orig[:-lookback]
    actual_aligned = actual_orig[lookback:]
    return actual_aligned, shifted_pred

def plot_results(actual, predicted, lookback):
    plt.figure(figsize=(14, 7))
    plt.plot(actual, label='Actual Asset 2 Price')
    plt.plot(predicted, label=f'Predicted Asset 2 Price (Shifted by {lookback} days)')
    plt.title('Actual vs Predicted Prices (Test Set, Shifted)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    combined_data = load_combined_data(ASSET_1, ASSET_2, START_DATE, END_DATE)
    train_loader, test_loader, feat_scaler, tgt_scaler = preprocess_data(
        combined_data, LOOKBACK, TIME_STEP, BATCH_SIZE, TRAIN_SPLIT
    )
    model = NeuralSDE(HIDDEN_SIZE, INPUT_FEATURES, OUTPUT_SIZE, LOOKBACK).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, train_loader, criterion, optimizer, device, EPOCHS)
    actual, predicted = evaluate_model(model, test_loader, tgt_scaler, device, lookback=14)
    plot_results(actual, predicted, lookback=14)
    print("Processing complete.")

if __name__ == "__main__":
    main()
