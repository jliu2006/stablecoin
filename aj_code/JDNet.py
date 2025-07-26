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
seed = 42
import torch
import numpy as np
import random
# Python built-in random module
random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)

# --- Configuration ---
ASSET_1 = 'ILS=X'
ASSET_2 = 'USDT-USD'

START_DATE = '2017-07-01'
END_DATE = '2025-07-24'
LOOKBACK = 44
TIME_STEP = 1
BATCH_SIZE = 64
TRAIN_SPLIT = 0.65
EPOCHS = 6
HIDDEN_SIZE = 1024
INPUT_FEATURES = 2
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-5


# --- Data Preprocessing ---
def preprocess_data(df, lookback, time_step, batch_size=64, split_ratio=0.65):
    features = df[['Asset1', 'Asset2']].values  # Use only Asset1 and Asset2 as features
    targets = df['Asset2'].values  # Use Asset2 as target

    X_raw, y_raw = [], []
    for i in range(len(features) - lookback - time_step):
        X_raw.append(features[i:i+lookback])
        y_raw.append(targets[i+lookback+time_step])

    X_raw, y_raw = np.array(X_raw), np.array(y_raw).reshape(-1, 1)
    split_idx = int(split_ratio * len(X_raw))
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

    feat_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()

    # Scale features and targets separately
    X_train = feat_scaler.fit_transform(X_train_raw.reshape(-1, features.shape[1])).reshape(X_train_raw.shape)
    X_test = feat_scaler.transform(X_test_raw.reshape(-1, features.shape[1])).reshape(X_test_raw.shape)

    y_train = tgt_scaler.fit_transform(y_train_raw)
    y_test = tgt_scaler.transform(y_test_raw)

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, feat_scaler, tgt_scaler, X_test


def load_combined_data(asset1, asset2, start, end):
    data1 = yf.download(asset1, start=start, end=end)['Close']
    data2 = yf.download(asset2, start=start, end=end)['Close']
    combined = pd.concat([data1, data2], axis=1, join='outer').dropna()
    combined.columns = ['Asset1', 'Asset2']
    return combined


# --- Neural SDE with Jump Diffusion ---
class NeuralSDEWithJump(nn.Module):
    def __init__(self, hidden_size, input_features, output_size, lookback):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.lookback = lookback

        self.drift_lstm = nn.LSTM(input_features, hidden_size, batch_first=True)
        self.diffusion_lstm = nn.LSTM(input_features, hidden_size, batch_first=True)

        self.drift_linear = nn.Linear(hidden_size + output_size, output_size)
        self.diffusion_linear = nn.Linear(hidden_size + output_size, output_size)

        # Jump parameters (you can tune these)
        self.jump_intensity = 0.000005  # Approx. expected jump rate per dt
        self.jump_mean = 0.0       # Mean jump size
        self.jump_std = 0.05       # Stddev of jump size

        self._lstm_hidden_buffer = None

    def f(self, t, y):
        if self._lstm_hidden_buffer is None:
            raise RuntimeError("LSTM hidden state buffer not set. Call forward with input_sequence first.")
        combined = torch.cat((self._lstm_hidden_buffer, y), dim=1)
        return self.drift_linear(combined)

    def g(self, t, y):
        if self._lstm_hidden_buffer is None:
            raise RuntimeError("LSTM hidden state buffer not set. Call forward with input_sequence first.")
        combined = torch.cat((self._lstm_hidden_buffer, y), dim=1)
        return self.diffusion_linear(combined)

    def jump(self, t, y):
        # Generate jump occurrence using Bernoulli with small prob = jump_intensity * dt (assume dt=0.01)
        jump_prob = self.jump_intensity * 0.01
        jump_occurred = torch.bernoulli(torch.full_like(y, jump_prob))
        jump_noise = torch.randn_like(y) * self.jump_std + self.jump_mean
        return jump_occurred * jump_noise

    def forward(self, ts, y0, input_seq):
        # Prepare LSTM hidden state buffer
        lstm_output, (hidden, _) = self.drift_lstm(input_seq)
        self._lstm_hidden_buffer = hidden.squeeze(0)

        y = y0
        ys = [y0]
        dt = ts[1] - ts[0]

        for t in ts[1:]:
            drift = self.f(t, y)
            diffusion = self.g(t, y)
            brownian = torch.randn_like(y) * torch.sqrt(dt)
            jump = self.jump(t, y)

            y = y + drift * dt + diffusion * brownian + jump
            ys.append(y)

        ys = torch.stack(ys)
        self._lstm_hidden_buffer = None
        return ys


# --- Training Routine ---
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            y0 = inputs[:, 0, 1].unsqueeze(1)
            ts = torch.linspace(0, 1, inputs.size(1)).to(device)
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
            inputs, targets = inputs.to(device), targets.to(device)
            y0 = inputs[:, 0, 1].unsqueeze(1)
            ts = torch.linspace(0, 1, inputs.size(1)).to(device)
            solutions = model(ts, y0, inputs)
            preds = solutions[-1]
            actual.append(targets.squeeze().cpu().numpy())
            predicted.append(preds.squeeze().cpu().numpy())
    actual = np.concatenate(actual)
    predicted = np.concatenate(predicted)
    actual_orig = target_scaler.inverse_transform(actual.reshape(-1, 1))
    predicted_orig = target_scaler.inverse_transform(predicted.reshape(-1, 1))
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


# --- Shock Simulation ---
def simulate_shock_and_predict(model, feat_scaler, tgt_scaler, combined_data, lookback, device, std_multiplier=-2):
    model.eval()
    # Take the last lookback window
    window = combined_data[['Asset1', 'Asset2']].values[-lookback:].copy()

    # Calculate std of Asset1 (RUB here)
    asset1_std = combined_data['Asset1'].std()

    # Apply shock to Asset1 with given std multiplier
    shocked_value = window[-1, 0] + std_multiplier * asset1_std
    print(f"Original Asset1 Value: {window[-1, 0]:.2f}, Shocked Value: {shocked_value:.2f}")

    # Replace last value with shocked value
    window[-1, 0] = shocked_value

    # Scale window
    window_scaled = feat_scaler.transform(window)
    input_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Initial condition for Asset2
    y0 = input_tensor[:, 0, 1].unsqueeze(1)
    ts = torch.linspace(0, 1, lookback).to(device)

    with torch.no_grad():
        output = model(ts, y0, input_tensor)
        predicted_scaled = output[-1].cpu().numpy().reshape(-1, 1)
        predicted = tgt_scaler.inverse_transform(predicted_scaled)

    print(f"Predicted Asset2 value (after shock): {predicted[0][0]:.4f}")
    return predicted[0][0]


# --- Main Execution ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    combined_data = load_combined_data(ASSET_1, ASSET_2, START_DATE, END_DATE)
    train_loader, test_loader, feat_scaler, tgt_scaler, X_test = preprocess_data(
        combined_data, LOOKBACK, TIME_STEP, BATCH_SIZE, TRAIN_SPLIT
    )

    model = NeuralSDEWithJump(HIDDEN_SIZE, INPUT_FEATURES, OUTPUT_SIZE, LOOKBACK).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, criterion, optimizer, device, EPOCHS)
    actual, predicted = evaluate_model(model, test_loader, tgt_scaler, device, lookback=1)
    plot_results(actual, predicted, lookback=1)

    print("Simulating shock and predicting...")
    simulate_shock_and_predict(
        model=model,
        feat_scaler=feat_scaler,
        tgt_scaler=tgt_scaler,
        combined_data=combined_data,
        lookback=LOOKBACK,
        device=device,
        std_multiplier=-2
    )

    print("Processing complete.")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

combined_data = load_combined_data(ASSET_1, ASSET_2, START_DATE, END_DATE)
train_loader, test_loader, feat_scaler, tgt_scaler, X_test = preprocess_data(
    combined_data, LOOKBACK, TIME_STEP, BATCH_SIZE, TRAIN_SPLIT
)

model = NeuralSDEWithJump(HIDDEN_SIZE, INPUT_FEATURES, OUTPUT_SIZE, LOOKBACK).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_model(model, train_loader, criterion, optimizer, device, EPOCHS)
actual, predicted = evaluate_model(model, test_loader, tgt_scaler, device, lookback=1)
plot_results(actual, predicted, lookback=1)

print("Simulating shock and predicting...")
simulate_shock_and_predict(
    model=model,
    feat_scaler=feat_scaler,
    tgt_scaler=tgt_scaler,
    combined_data=combined_data,
    lookback=LOOKBACK,
    device=device,
    std_multiplier=-2
)

print("Processing complete.")
simulate_shock_and_predict(
    model=model,
    feat_scaler=feat_scaler,
    tgt_scaler=tgt_scaler,
    combined_data=combined_data,
    lookback=LOOKBACK,
    device=device,
    std_multiplier=4
)
