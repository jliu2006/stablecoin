# ---------------------------- Imports ----------------------------
# Import required libraries for data handling, financial data retrieval, regression, and plotting
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to enable interactive plot display

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
from scipy.stats import chi2

# ---------------------------- Step 1: Load Financial Data ----------------------------
# Define the time window for analysis
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download daily closing prices for relevant financial instruments from Yahoo Finance
# USD Tether to USD,Russian Ruble to USD,USD to Canadian Dollar,Bitcoin to USD,Crude Oil Futures,Natural Gas Futures,13-week Treasury Bill Rate, MOEX Index
usdt_usd = yf.download('USDT-USD', start=start_date, end=end_date)['Close'].squeeze()   
rub_usd = yf.download('RUB=X', start=start_date, end=end_date)['Close'].squeeze()       
usdt_cad = yf.download('USDCAD=X', start=start_date, end=end_date)['Close'].squeeze()   
btc_usd = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].squeeze()     
oil_price = yf.download('CL=F', start=start_date, end=end_date)['Close'].squeeze()       
gas_price = yf.download('NG=F', start=start_date, end=end_date)['Close'].squeeze()       
interest_rate = yf.download('^IRX', start=start_date, end=end_date)['Close'].squeeze()  
moex_index = yf.download('IMOEX.ME', start=start_date, end=end_date)['Close'].squeeze() 

# ---------------------------- Step 2: Build DataFrame ----------------------------
# Create a DataFrame consolidating all financial series
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
#USDT_RUB_price ~ BTC_USDT_ratio + USDT_CAD_price + Gas_Price (+ constant) first stage
# MOEX_Index ~ Oil_Price + Interest_Rate + predicted_USDT_RUB_price (+ constant) second stage
iv_model = IV2SLS(
    dependent=data['MOEX_Index'],
    exog=sm.add_constant(data[['Oil_Price', 'Interest_Rate']]),
    endog=data['USDT_RUB_price'],
    instruments=data[['BTC_USDT_ratio', 'USDT_CAD_price', 'Gas_Price']]
).fit()

print("\n IV Regression Results:")
print(iv_model.summary)

# ---------------------------- Step 4: General Hausman Test ----------------------------
#
# Define variables for OLS model
X_ols = sm.add_constant(data[['Oil_Price', 'Interest_Rate', 'USDT_RUB_price']])
y_ols = data['MOEX_Index']

# Combine predictors and target, clean data by removing NaNs and infinite values
combined_ols = pd.concat([X_ols, y_ols], axis=1)
combined_ols = combined_ols.replace([np.inf, -np.inf], np.nan).dropna()

# Split cleaned data into X and y
X_ols_clean = combined_ols.drop(columns='MOEX_Index')
y_ols_clean = combined_ols['MOEX_Index']

# Fit OLS model
ols_model = sm.OLS(y_ols_clean, X_ols_clean).fit()

# Extract coefficient estimates for USDT_RUB_price from both IV and OLS models
beta_iv = iv_model.params[['USDT_RUB_price']]
beta_ols = ols_model.params[['USDT_RUB_price']]

# Extract corresponding variance estimates
var_iv = iv_model.cov.loc[['USDT_RUB_price'], ['USDT_RUB_price']]
var_ols = ols_model.cov_params().loc[['USDT_RUB_price'], ['USDT_RUB_price']]

# Compute difference in coefficients and variances
beta_diff = beta_iv - beta_ols
var_diff = var_iv - var_ols

try:
    # Compute Hausman test statistic using the formula: H = (β_IV - β_OLS)' * [Var_IV - Var_OLS]⁻¹ * (β_IV - β_OLS)
    hausman_stat = float(beta_diff.T @ np.linalg.inv(var_diff) @ beta_diff)
    df = beta_diff.shape[0]  # Degrees of freedom = number of tested coefficients

    # Compute exact p-value using chi-squared survival function (no rounding) - What's the probability that the test statistic is this extreme or more, assuming the null hypothesis is true
    p_value = chi2.sf(hausman_stat, df)

    # Print results with full precision p-value
    print("\nGeneral Hausman Test Results:")
    print(f"Hausman statistic: {hausman_stat:.10e}")
    #endogeneity in only one regressor
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value}")  #  Exact p-value printed without rounding

    # Interpret result based on p-value threshold
    if p_value < 0.05:
        print("Conclusion: IV regression is preferred — significant difference suggests endogeneity.")
    else:
        print("Conclusion: No significant difference — OLS may be valid.")
except np.linalg.LinAlgError:
    # Handle case where variance difference matrix is singular and cannot be inverted
    print("\nGeneral Hausman Test Results:")
    print("Covariance difference matrix is singular — Hausman test not computable.")


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
