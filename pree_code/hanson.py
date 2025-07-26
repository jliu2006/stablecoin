import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------- Step 1: Load Financial Data ----------------------------
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download exchange rate data
usdt_cad = yf.download('USDCAD=X', start=start_date, end=end_date)['Close'].squeeze()
rub_usd = yf.download('RUB=X', start=start_date, end=end_date)['Close'].squeeze()
btc_price = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].squeeze()

# Simulate USDT commission fee (proxy)
np.random.seed(10)
usdt_fee = pd.Series(np.random.normal(0.01, 0.005, len(usdt_cad)), index=usdt_cad.index)

# ---------------------------- Step 1.5: NEW — Load Macroeconomic Instruments ----------------------------
# Added macroeconomic proxies to strengthen instrument set (improves identification and supports Hansen J-Test)
macro_symbols = {
    'Interest_Rate': '^TNX',       # 10Y Treasury yield
    'Inflation_Index': 'TIP',      # Inflation ETF
    'GDP_Index': 'SPY',            # Proxy for economic output
    'FX_Reserves': 'UUP',          # USD ETF (approximate reserve strength)
    'Oil_Price': 'CL=F',           # Crude oil futures (RUB is oil-sensitive)
    'Trade_Balance': 'DX-Y.NYB'    # USD index (proxy for trade balance)
}

macro_data = {}
for name, symbol in macro_symbols.items():
    macro_data[name] = yf.download(symbol, start=start_date, end=end_date)['Close'].squeeze()

macro_df = pd.DataFrame(macro_data).dropna()

# ---------------------------- Step 2: Construct Dataset ----------------------------
data = pd.DataFrame({
    'USDT_CAD_price': usdt_cad,
    'RUB_exchange_rate': rub_usd,
    'USDT_fee': usdt_fee,
    'USDT_CAD_MA': usdt_cad.rolling(window=30).mean(),
    'BTC_Price': btc_price
}).dropna()
#80% weight USDT/CAD
#20% weight USDT fee
#random noise
# Simulate Russian Stablecoin Price
data['RU_stablecoin_price'] = (
    0.8 * data['USDT_CAD_price'] +
    0.2 * data['USDT_fee'] +
    np.random.normal(0, 0.01, len(data))  # Noise added
)

# ---------------------------- Step 2.5: NEW — Merge Macro Instruments ----------------------------
# Merges macroeconomic indicators with main data for modeling and IV construction
data = pd.concat([data, macro_df], axis=1).dropna()

# ---------------------------- Step 3a: First Stage Regression ----------------------------
X_first = sm.add_constant(data[['USDT_CAD_price', 'USDT_CAD_MA']])  # Original instruments
y_first = data['RU_stablecoin_price']
first_stage = sm.OLS(y_first, X_first).fit()
data['RU_stablecoin_fitted'] = first_stage.fittedvalues

print("First Stage Regression:\n", first_stage.summary())

# Instrument relevance check (F-stat > 10 indicates strong instruments)
f_test = first_stage.f_test("USDT_CAD_price = 0")
print("\n F-Test for Instrument Strength:\n", f_test)

# ---------------------------- Step 3b: Second Stage Regression ----------------------------
# Uses predicted stablecoin values instead of raw endogenous variable
X_second = sm.add_constant(data[['RU_stablecoin_fitted', 'USDT_CAD_MA']])
y_second = data['RUB_exchange_rate']
second_stage = sm.OLS(y_second, X_second).fit()
print("\n Second Stage Regression:\n", second_stage.summary())

# ---------------------------- Step 4: UPDATED IV Estimation ----------------------------
# Expanded instrument list includes macro proxies (helps with overidentification and robustness)
data['const'] = 1
iv_model = IV2SLS(
    dependent=data['RUB_exchange_rate'],
    exog=data[['const', 'USDT_CAD_MA', 'BTC_Price']],
    endog=data['RU_stablecoin_price'],
    instruments=data[['USDT_CAD_price', 'USDT_fee',
                      'Interest_Rate', 'Inflation_Index', 'GDP_Index',
                      'FX_Reserves', 'Oil_Price', 'Trade_Balance']]
).fit()
print("\n IV Regression Results:\n", iv_model.summary)

# ---------------------------- Step 4.5: NEW — Hansen J-Test ----------------------------
# Validates overidentification; checks whether instruments are collectively exogenous
if hasattr(iv_model, 'j_statistic'):
    print(f"\n Hansen J-Test:\nJ-stat = {iv_model.j_statistic.stat:.4f}, p-value = {iv_model.j_statistic.pval:.4f}")
else:
    print("\n Hansen J-Test not available — model may not be overidentified.")

# ---------------------------- Step 5: Durbin-Wu-Hausman Test ----------------------------
# Detects endogeneity of the stablecoin variable
data['first_stage_resid'] = data['RU_stablecoin_price'] - data['RU_stablecoin_fitted']
X_dwh = sm.add_constant(data[['RU_stablecoin_price', 'USDT_CAD_MA', 'first_stage_resid']])
dwh_model = sm.OLS(data['RUB_exchange_rate'], X_dwh).fit()

print("\n Durbin-Wu-Hausman Test Results (Manual OLS Augmentation):\n")
print(dwh_model.summary())

# Interpretation — is IV approach needed?
p_val = dwh_model.pvalues['first_stage_resid']
if p_val < 0.05:
    print(f"\n DWH test suggests endogeneity (p = {p_val:.4f}) — IV approach justified.")
else:
    print(f"\n DWH test does NOT find strong endogeneity (p = {p_val:.4f}) — OLS may be sufficient.")

# ---------------------------- Step 6: Visualizations ----------------------------

# First stage: instrument vs endogenous variable
plt.figure(figsize=(10, 6))
sns.regplot(x=data['USDT_CAD_price'], y=data['RU_stablecoin_price'],
            scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.title("First Stage: USDT/CAD vs Russian Stablecoin Price")
plt.xlabel("USDT/CAD Price")
plt.ylabel("USDT/RUB Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# Second stage: fitted values vs dependent variable
plt.figure(figsize=(10, 6))
sns.regplot(x=data['RU_stablecoin_fitted'], y=data['RUB_exchange_rate'],
            scatter_kws={'s': 10}, line_kws={'color': 'blue'})
plt.title("Second Stage: Fitted Stablecoin vs RUB Exchange Rate")
plt.xlabel("Predicted USDT/RUB Price")
plt.ylabel("RUB/USD Exchange Rate")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual distribution from second stage
plt.figure(figsize=(10, 6))
sns.histplot(second_stage.resid, bins=30, kde=True, color='purple')
plt.title("Residuals of Second Stage Regression")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Visualize instrument strength (F-test)
f_val = f_test.statistic
plt.figure(figsize=(6, 4))
plt.bar(["USDT_CAD F-stat"], [f_val], color='green')
plt.axhline(y=10, color='red', linestyle='--', label='Threshold (Weak Instrument)')
plt.title("Instrument Strength (F-Test)")
plt.ylabel("F-statistic Value")
plt.legend()
plt.tight_layout()
plt.show()
