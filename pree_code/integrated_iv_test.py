# ---------------------------- Imports ----------------------------
import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aj_code.sde import NeuralSDE

# ---------------------------- Step 1: Load Financial Data ----------------------------
start_date = '2020-01-01'
end_date = '2025-01-01'

usdt_usd = yf.download('USDT-USD', start=start_date, end=end_date)['Close'].squeeze()
rub_usd = yf.download('RUB=X', start=start_date, end=end_date)['Close'].squeeze()
usdt_cad = yf.download('USDCAD=X', start=start_date, end=end_date)['Close'].squeeze()
btc_usd = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].squeeze()
oil_price = yf.download('CL=F', start=start_date, end=end_date)['Close'].squeeze()
gas_price = yf.download('NG=F', start=start_date, end=end_date)['Close'].squeeze()
interest_rate = yf.download('^IRX', start=start_date, end=end_date)['Close'].squeeze()
moex_index = yf.download('IMOEX.ME', start=start_date, end=end_date)['Close'].squeeze()

# ---------------------------- Step 2: Build DataFrame ----------------------------
data = pd.DataFrame({
    'USDT_USD': usdt_usd,
    'RUB_USD': rub_usd,
    'USDT_CAD_price': usdt_cad,
    'BTC_Price': btc_usd,
    'Oil_Price': oil_price,
    'Gas_Price': gas_price,
    'Interest_Rate': interest_rate,
    'MOEX_Index': moex_index
})

# Derived variables (customize as needed)
data['USDT_RUB_price'] = data['RUB_USD'] / data['USDT_USD']
data['BTC_USDT_ratio'] = data['BTC_Price'] / data['USDT_USD']

# ---------------------------- Step 3: Define Regression Variables ----------------------------
# ---- KEY VARIABLES TO UPDATE FOR MODEL SPECIFICATION ----
# To change endogenous, exogenous, or IVs, update these variable lists:
endogenous_var = 'USDT_RUB_price'          # <-- endogenous variable
exogenous_vars = ['Oil_Price', 'Interest_Rate']     # <-- exogenous variables
instrument_vars = ['BTC_USDT_ratio', 'USDT_CAD_price', 'Gas_Price']  # <-- instruments
outcome_var = 'MOEX_Index'                 # <-- outcome variable (what you want to predict)

# ---------------------------- Step 4: Data Preparation ----------------------------
# Drop missing data
regression_data = data.dropna(subset=instrument_vars + [endogenous_var] + exogenous_vars + [outcome_var])

# First-stage (IV) regression data
X_instruments = regression_data[instrument_vars].values
y_endogenous = regression_data[endogenous_var].values.reshape(-1, 1)

# Scale
X_instruments_scaler = MinMaxScaler()
y_endogenous_scaler = MinMaxScaler()
X_instruments_scaled = X_instruments_scaler.fit_transform(X_instruments)
y_endogenous_scaled = y_endogenous_scaler.fit_transform(y_endogenous)

# Convert to tensors (no lookback)
X_instruments_tensor = torch.tensor(X_instruments_scaled, dtype=torch.float32)
y_endogenous_tensor = torch.tensor(y_endogenous_scaled, dtype=torch.float32)

# ---------------------------- Step 5: First-stage SDE ----------------------------
hidden_size = 128
output_size = 1
num_epochs = 10
batch_size = 32

first_stage_sde = NeuralSDE(hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(first_stage_sde.parameters(), lr=1e-3)

train_dataset = TensorDataset(X_instruments_tensor, y_endogenous_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        y0 = inputs[:, 0, 1].unsqueeze(1)
        ts = torch.linspace(0, 1, inputs.size(1)).to(device)
        solutions = first_stage_sde(ts, y0, inputs)
        preds = solutions[-1]
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


# Compute residuals
first_stage_sde.eval()
with torch.no_grad():
    y0_all = X_instruments_tensor
    ts_all = torch.linspace(0, 1, 1)
    sde_outputs = first_stage_sde(ts_all, y0_all)
    y_endogenous_pred = sde_outputs[-1].numpy()
residuals = y_endogenous_scaled.squeeze() - y_endogenous_pred.squeeze()

# ---------------------------- Step 6: Second-stage SDE ----------------------------
X_exog = regression_data[exogenous_vars].values
y_outcome = regression_data[outcome_var].values.reshape(-1, 1)

X_exog_scaler = MinMaxScaler()
y_outcome_scaler = MinMaxScaler()
X_exog_scaled = X_exog_scaler.fit_transform(X_exog)
y_outcome_scaled = y_outcome_scaler.fit_transform(y_outcome)

# Augment exogenous features with first-stage residuals
X_exog_full = np.concatenate([X_exog_scaled, residuals.reshape(-1, 1)], axis=1)
X_exog_tensor = torch.tensor(X_exog_full, dtype=torch.float32)
y_outcome_tensor = torch.tensor(y_outcome_scaled, dtype=torch.float32)

train_dataset2 = TensorDataset(X_exog_tensor, y_outcome_tensor)
train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)

second_stage_sde = NeuralSDE(hidden_size, output_size)
optimizer2 = optim.Adam(second_stage_sde.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    second_stage_sde.train()
    running_loss2 = 0.0
    for inputs2, targets2 in train_loader2:
        optimizer2.zero_grad()
        y0_2 = inputs2
        ts_2 = torch.linspace(0, 1, 1)
        outputs2 = second_stage_sde(ts_2, y0_2)
        predictions2 = outputs2[-1]
        loss2 = criterion(predictions2, targets2)
        loss2.backward()
        optimizer2.step()
        running_loss2 += loss2.item()
    print(f"Second-stage SDE (with residuals) Epoch {epoch+1}/{num_epochs}, Loss: {running_loss2/len(train_loader2):.4f}")

# Train second-stage SDE without residuals (for comparison)
X_exog_tensor_nored = torch.tensor(X_exog_scaled, dtype=torch.float32)
train_dataset2_nored = TensorDataset(X_exog_tensor_nored, y_outcome_tensor)
train_loader2_nored = DataLoader(train_dataset2_nored, batch_size=batch_size, shuffle=True)
second_stage_sde_nored = NeuralSDE(hidden_size, output_size)
optimizer2_nored = optim.Adam(second_stage_sde_nored.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    second_stage_sde_nored.train()
    running_loss2_nored = 0.0
    for inputs2, targets2 in train_loader2_nored:
        optimizer2_nored.zero_grad()
        y0_2 = inputs2
        ts_2 = torch.linspace(0, 1, 1)
        outputs2 = second_stage_sde_nored(ts_2, y0_2)
        predictions2 = outputs2[-1]
        loss2 = criterion(predictions2, targets2)
        loss2.backward()
        optimizer2_nored.step()
        running_loss2_nored += loss2.item()
    print(f"Second-stage SDE (NO residuals) Epoch {epoch+1}/{num_epochs}, Loss: {running_loss2_nored/len(train_loader2_nored):.4f}")

# ---------------------------- Step 7: Model Performance Comparison ----------------------------
second_stage_sde.eval()
second_stage_sde_nored.eval()
with torch.no_grad():
    y0_eval = X_exog_tensor
    ts_eval = torch.linspace(0, 1, 1)
    y_pred_with_resid = second_stage_sde(ts_eval, y0_eval)[-1].numpy()
    y_pred_with_resid = y_outcome_scaler.inverse_transform(y_pred_with_resid)
    mse_with_resid = np.mean((y_pred_with_resid - y_outcome_scaler.inverse_transform(y_outcome_scaled)) ** 2)

    y0_eval_nored = X_exog_tensor_nored
    y_pred_no_resid = second_stage_sde_nored(ts_eval, y0_eval_nored)[-1].numpy()
    y_pred_no_resid = y_outcome_scaler.inverse_transform(y_pred_no_resid)
    mse_no_resid = np.mean((y_pred_no_resid - y_outcome_scaler.inverse_transform(y_outcome_scaled)) ** 2)

print("\nNeural SDE DWH Test Results:")
print(f"Second-stage SDE MSE (with residuals): {mse_with_resid:.4f}")
print(f"Second-stage SDE MSE (NO residuals):   {mse_no_resid:.4f}")

if mse_with_resid + 1e-5 < mse_no_resid:
    print("Endogeneity detected: Including first-stage residuals improves prediction of outcome.")
else:
    print("No strong evidence of endogeneity: Including first-stage residuals does not improve prediction.")

# Plot prediction comparison
plt.figure(figsize=(12,6))
plt.plot(y_outcome_scaler.inverse_transform(y_outcome_scaled), label=f'Actual {outcome_var}')
plt.plot(y_pred_no_resid, label='Predicted (No Residuals)')
plt.plot(y_pred_with_resid, label='Predicted (With Residuals)')
plt.legend()
plt.title(f'{outcome_var} Predictions: With/Without Residuals (Neural SDE DWH)')
plt.xlabel('Time steps')
plt.ylabel(outcome_var)
plt.show()

# ---------------------------- Step 8: Instrument Strength Chart (Neural SDE Analog) ----------------------------
# We estimate instrument strength in neural setting by fitting a separate NeuralSDE for each instrument and comparing MSE reduction.

f_stats = []
labels = instrument_vars
iv_colors = ['dodgerblue', 'limegreen', 'tomato', 'orange', 'purple', 'cyan']

for idx, iv in enumerate(labels):
    # Prepare single-IV data
    X_iv = regression_data[[iv]].values
    y_iv = regression_data[endogenous_var].values.reshape(-1, 1)
    scaler_X_iv = MinMaxScaler()
    scaler_y_iv = MinMaxScaler()
    X_iv_scaled = scaler_X_iv.fit_transform(X_iv)
    y_iv_scaled = scaler_y_iv.fit_transform(y_iv)

    X_iv_tensor = torch.tensor(X_iv_scaled, dtype=torch.float32)
    y_iv_tensor = torch.tensor(y_iv_scaled, dtype=torch.float32)
    train_dataset_iv = TensorDataset(X_iv_tensor, y_iv_tensor)
    train_loader_iv = DataLoader(train_dataset_iv, batch_size=batch_size, shuffle=True)

    # Fit NeuralSDE (single instrument)
    model_iv = NeuralSDE(hidden_size, output_size)
    optimizer_iv = optim.Adam(model_iv.parameters(), lr=1e-3)
    criterion_iv = nn.MSELoss()
    num_epochs_iv = 10
    for epoch in range(num_epochs_iv):
        model_iv.train()
        running_loss_iv = 0.0
        for inputs, targets in train_loader_iv:
            optimizer_iv.zero_grad()
            y0_iv = inputs
            ts_iv = torch.linspace(0, 1, 1)
            outputs_iv = model_iv(ts_iv, y0_iv)
            predictions_iv = outputs_iv[-1]
            loss_iv = criterion_iv(predictions_iv, targets)
            loss_iv.backward()
            optimizer_iv.step()
            running_loss_iv += loss_iv.item()
    # Evaluate instrument's predictive power
    model_iv.eval()
    with torch.no_grad():
        y0_eval_iv = X_iv_tensor
        ts_eval_iv = torch.linspace(0, 1, 1)
        y_pred_iv = model_iv(ts_eval_iv, y0_eval_iv)[-1].numpy()
        mse_iv = np.mean((y_pred_iv - y_iv_scaled) ** 2)
        # For a "strength" analog, we can use the reduction in variance explained by the instrument
        # Or use a pseudo-F-stat: (Variance_explained / MSE)
        var_explained = np.var(y_iv_scaled) - mse_iv
        f_stat_approx = var_explained / mse_iv if mse_iv > 0 else np.nan
        f_stats.append(float(f_stat_approx))

# Plot pseudo-F-statistics for instrument strength
plt.figure(figsize=(10, 5))
plt.bar(labels, f_stats, color=iv_colors[:len(labels)])
plt.axhline(y=1, color='black', linestyle='--', linewidth=1.2, label='Weak IV Threshold (approx. F = 1)')
for i, val in enumerate(f_stats):
    plt.text(i, val + 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
plt.title(f"Instrument Strength for {endogenous_var} (NeuralSDE pseudo-F)", fontsize=14)
plt.ylabel("Pseudo-F-statistic (Variance explained / MSE)")
plt.xlabel("Instrumental Variables")
plt.legend()
plt.tight_layout()
plt.show()
