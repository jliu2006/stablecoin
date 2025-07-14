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
# Change later - 0.01 mean avg, 0.005 SD
usdt_fee = pd.Series(np.random.normal(0.01, 0.005, len(usdt_cad)), index=usdt_cad.index)

# Align indexes
data = pd.DataFrame({
    'USDT_CAD_price': usdt_cad,
    'RUB_exchange_rate': rub_usd,
    'USDT_fee': usdt_fee,
    'USDT_CAD_MA': usdt_cad.rolling(window=30).mean(),
    'BTC_Price': btc_price
}).dropna()

# ---------------------------- Step 2: Simulate Russian Stablecoin - NEED DATA----------------------------
#80% weight USDT/CAD
#20% weight USDT fee
#random noise
data['RU_stablecoin_price'] = (
    0.8 * data['USDT_CAD_price'] + 0.2 * data['USDT_fee'] +
    np.random.normal(0, 0.01, len(data))
)

# ---------------------------- Step 3a: First Stage ----------------------------
#CAD is instrument
X_first = sm.add_constant(data[['USDT_CAD_price', 'USDT_CAD_MA']])
y_first = data['RU_stablecoin_price']
#Moving avg CAD
first_stage = sm.OLS(y_first, X_first).fit()
data['RU_stablecoin_fitted'] = first_stage.fittedvalues
print("First Stage Regression:\n", first_stage.summary())
# F-test for instrument strength 
f_test = first_stage.f_test("USDT_CAD_price = 0")
print("\n F-Test for Instrument Strength:\n", f_test)
# ---------------------------- Step 3b: Second Stage ----------------------------
X_second = sm.add_constant(data[['RU_stablecoin_fitted', 'USDT_CAD_MA']])
y_second = data['RUB_exchange_rate']
#Regresses RUB/USD against the fitted values of the stablecoin price - remove endo noise
second_stage = sm.OLS(y_second, X_second).fit()
print("\n Second Stage Regression:\n", second_stage.summary())
# ---------------------------- Step 4: IV Estimation ----------------------------
#Dependent variable: RUB_exchange_rate
#Endogenous variable: RU_stablecoin_price
#Instruments: USDT_CAD_price, USDT_fee
#Exogenous controls: moving average & Bitcoin price
#May help isolate effects more precisely, assuming BTC influences RUB indirectly via sentiment or trading volume?
#Can reduce omitted variable bias if Bitcoin affects both stablecoin pricing and exchange rate outcomes?
iv_data = data.copy()
iv_data['const'] = 1
iv_model = IV2SLS(
    dependent=iv_data['RUB_exchange_rate'],
    exog=iv_data[['const', 'USDT_CAD_MA','BTC_Price']],
    endog=iv_data['RU_stablecoin_price'],
    instruments=iv_data[['USDT_CAD_price', 'USDT_fee']]
).fit()
print("\n IV Regression Results:\n", iv_model.summary)
# ---------------------------- Step 5: Durbin-Wu-Hausman Test ----------------------------
#Ask about during meeting
data['first_stage_resid'] = data['RU_stablecoin_price'] - data['RU_stablecoin_fitted']
X_dwh = sm.add_constant(data[['RU_stablecoin_price', 'USDT_CAD_MA', 'first_stage_resid']])
dwh_model = sm.OLS(data['RUB_exchange_rate'], X_dwh).fit()
# Step 3: Check significance of the residual term
print("\n Durbin-Wu-Hausman Test Results (Manual OLS Augmentation):\n")
print(dwh_model.summary())
# Examine p-value directly?
p_val = dwh_model.pvalues['first_stage_resid']
if p_val < 0.05:
    print(f"\n DWH test suggests endogeneity (p = {p_val:.4f}) — IV approach justified.")
else:
    print(f"\n DWH test does NOT find strong endogeneity (p = {p_val:.4f}) — OLS may be sufficient.")
# ---------------------------- Step 6: Visualizations ----------------------------
#My additions

# First-stage regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=data['USDT_CAD_price'], y=data['RU_stablecoin_price'], scatter_kws={'s': 10}, line_kws={'color': 'red'})
plt.title("First Stage: USDT/CAD vs Russian Stablecoin Price")
plt.xlabel("USDT/CAD Price")
plt.ylabel("USDT/RUB Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# Second-stage regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=data['RU_stablecoin_fitted'], y=data['RUB_exchange_rate'], scatter_kws={'s': 10}, line_kws={'color': 'blue'})
plt.title("Second Stage: Fitted Stablecoin vs RUB Exchange Rate")
plt.xlabel("Predicted USDT/RUB Price")
plt.ylabel("RUB/USD Exchange Rate")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual distribution
plt.figure(figsize=(10, 6))
sns.histplot(second_stage.resid, bins=30, kde=True, color='purple')
plt.title("Residuals of Second Stage Regression")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# F-stat visualization
f_val = f_test.statistic
plt.figure(figsize=(6, 4))
plt.bar(["USDT_CAD F-stat"], [f_val], color='green')
plt.axhline(y=10, color='red', linestyle='--', label='Threshold (Weak Instrument)')
plt.title("Instrument Strength (F-Test)")
plt.ylabel("F-statistic Value")
plt.legend()
plt.tight_layout()
plt.show()
