# ---------------------------- Imports ----------------------------
# Import required libraries for data handling, financial data retrieval, regression, and plotting
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

# ---------------------------- Step 1: Load Financial Data ----------------------------
# Define the time window for analysis
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download daily closing prices for relevant financial instruments from Yahoo Finance
# USD Tether to USD,Russian Ruble to USD,USD to Canadian Dollar,Bitcoin to USD,Crude Oil Futures,Natural Gas Futures,13-week Treasury Bill Rate
usdt_usd = yf.download('USDT-USD', start=start_date, end=end_date)['Close'].squeeze()   
rub_usd = yf.download('RUB=X', start=start_date, end=end_date)['Close'].squeeze()       
usdt_cad = yf.download('USDCAD=X', start=start_date, end=end_date)['Close'].squeeze()   
btc_usd = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].squeeze()     
oil_price = yf.download('CL=F', start=start_date, end=end_date)['Close'].squeeze()       
gas_price = yf.download('NG=F', start=start_date, end=end_date)['Close'].squeeze()       
interest_rate = yf.download('^IRX', start=start_date, end=end_date)['Close'].squeeze()   

# ---------------------------- Step 2: Build DataFrame ----------------------------
# Create a DataFrame consolidating all financial series
data = pd.DataFrame({
    'USDT_USD': usdt_usd,
    'RUB_USD': rub_usd,
    'USDT_CAD_price': usdt_cad,
    'BTC_Price': btc_usd,
    'Oil_Price': oil_price,
    'Gas_Price': gas_price,
    'Interest_Rate': interest_rate
})

# Add derived variables:
# - USDT_RUB_price: conversion of RUB/USD to USDT/USD
# - BTC_USDT_ratio: a proxy measure of crypto market 
data['USDT_RUB_price'] = data['RUB_USD'] / data['USDT_USD']
data['BTC_USDT_ratio'] = data['BTC_Price'] / data['USDT_USD']

# ---------------------------- Step 3: IV Regression ----------------------------
# Apply Instrumental Variables Regression:
# - Exogenous: Oil prices and interest rates (assumed exogenous)
# - Endogenous: USDT/RUB exchange rate (treated as endogenous for model testing)
# - Instruments: BTC/USDT ratio, USDT/CAD price, and gas price
iv_model = IV2SLS(
    dependent=data['USDT_RUB_price'],
    exog=sm.add_constant(data[['Oil_Price', 'Interest_Rate']]),
    endog=data['USDT_RUB_price'],
    instruments=data[['BTC_USDT_ratio', 'USDT_CAD_price', 'Gas_Price']]
).fit()

print("\n IV Regression Results:")
print(iv_model.summary)

# ---------------------------- Step 4: Durbin-Wu-Hausman Test ----------------------------
# Step to test for endogeneity using residuals from first-stage regression

# First-stage regression inputs and outputs- test whether the regressor (USDT_RUB_price) is endogenous.
# If YES then OLS estimates may be biased, and IV regression is preferred.
X_first_stage = sm.add_constant(data[['BTC_USDT_ratio', 'USDT_CAD_price', 'Gas_Price']])
y_first_stage = data['USDT_RUB_price']

# Remove rows with missing data from both predictors and target
common_index = X_first_stage.dropna().index.intersection(y_first_stage.dropna().index)
X_first_stage_clean = X_first_stage.loc[common_index]
y_first_stage_clean = y_first_stage.loc[common_index]

# Run first-stage regression: USDT_RUB_price ~ instruments
first_stage = sm.OLS(y_first_stage_clean, X_first_stage_clean).fit()
# Calculate residuals: these represent variation in USDT_RUB_price not explained by instruments
# If residuals are correlated with the error term in the structural equation, it suggests endogeneity
data['first_stage_resid'] = data['USDT_RUB_price'] - first_stage.fittedvalues.reindex(data.index)

# Build the full model including residuals to test for endogeneity
X_full = sm.add_constant(data[['Oil_Price', 'Interest_Rate', 'first_stage_resid']])
y_full = data['USDT_RUB_price']
X_full_clean = X_full.dropna()
y_full_clean = y_full.loc[X_full_clean.index]
# Fit augmented OLS model with residuals included
model_full = sm.OLS(y_full_clean, X_full_clean).fit()

# Fit reduced model excluding residuals
X_reduced = sm.add_constant(data[['Oil_Price', 'Interest_Rate']])
X_reduced_clean = X_reduced.dropna()
y_reduced_clean = y_full.loc[X_reduced_clean.index]

model_reduced = sm.OLS(y_reduced_clean, X_reduced_clean).fit()

# Extract p-value for residual term to test for endogeneity
p_val = model_full.pvalues.get('first_stage_resid', None)

# Interpret results - print
print("\n Durbin-Wu-Hausman Test Results:")
print(model_full.summary())
print(f"\nEndogeneity test result: p = {p_val:.4f} —", end=" ")
if p_val is not None and p_val < 0.05:
    print("IV regression justified ")
    print("Interpretation: Residual is significant, suggesting endogeneity in the regressor.")
else:
    print("OLS may be sufficient ")
    print("Interpretation: No strong sign of endogeneity — OLS may be valid.")

# ---------------------------- Step 5: Instrument Strength Chart ----------------------------
# Visualize instrument strength using approximate F-statistics from first-stage regressions
f_stats = []
labels = ['BTC_USDT_ratio', 'USDT_CAD_price', 'Gas_Price']
for iv in labels:
    X_iv = sm.add_constant(data[[iv]])
    y_iv = data['USDT_RUB_price']
    common_index = X_iv.dropna().index.intersection(y_iv.dropna().index)
    X_iv_clean = X_iv.loc[common_index]
    y_iv_clean = y_iv.loc[common_index]
    model_iv = sm.OLS(y_iv_clean, X_iv_clean).fit()
    f_val = model_iv.tvalues[iv] ** 2
    f_stats.append(float(f_val))

# Plot F-statistics to evaluate instrument strength
plt.figure(figsize=(10, 5))
plt.bar(labels, f_stats, color=['dodgerblue', 'limegreen', 'tomato'])
plt.axhline(y=10, color='black', linestyle='--', linewidth=1.2, label='Weak IV Threshold (F = 10)')
for i, val in enumerate(f_stats):
    plt.text(i, val + 0.5, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
plt.title("Instrument Strength for USDT_RUB_price (F ≈ t²)", fontsize=14)
plt.ylabel("F-statistic (approx. t²)")
plt.xlabel("Instrumental Variables")
plt.legend()
plt.tight_layout()
plt.show()
