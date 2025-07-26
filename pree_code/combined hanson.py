import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

# ---------------------------- Step 1: Load Financial & Stablecoin Data ----------------------------
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download exchange rate data (USDT to CAD and RUB to USD)
usdt_cad = yf.download('USDCAD=X', start=start_date, end=end_date)['Close'].squeeze()
rub_usd = yf.download('RUB=X', start=start_date, end=end_date)['Close'].squeeze()
# Bitcoin price (control proxy)
btc_price = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].squeeze()
# US interest rate proxy (13-week T-bill)
interest_rate_proxy = yf.download('^IRX', start=start_date, end=end_date)['Close'].squeeze()
# Oil price (WTI)
oil_price = yf.download('CL=F', start=start_date, end=end_date)['Close'].squeeze()
# European and DAI stablecoin prices
euroe_price = yf.download('EUROE-USD', start=start_date, end=end_date)['Close'].squeeze()
dai_price = yf.download('DAI-USD', start=start_date, end=end_date)['Close'].squeeze()

# ---------------------------- Step 2: Simulate Macroeconomic Data (Placeholders) ----------------------------
# Simulated macroeconomic indicators – replace with official data or API if available
inflation = pd.Series(np.random.normal(2.5, 0.5, len(usdt_cad)), index=usdt_cad.index)        # Russia inflation (%)
gdp_growth = pd.Series(np.random.normal(2.0, 0.3, len(usdt_cad)), index=usdt_cad.index)       # Russia GDP growth (%)
fx_reserves = pd.Series(np.random.normal(35, 2, len(usdt_cad)), index=usdt_cad.index)         # Russia FX reserves
trade_balance = pd.Series(np.random.normal(-50, 10, len(usdt_cad)), index=usdt_cad.index)     # Russia trade balance

# Simulate USDT fee (proxy for transaction costs)
np.random.seed(42)
usdt_fee = pd.Series(np.random.normal(0.01, 0.005, len(usdt_cad)), index=usdt_cad.index)

# ---------------------------- Step 3: Construct Primary DataFrame ----------------------------
data = pd.DataFrame({
    'USDT_CAD_price': usdt_cad,
    'RUB_exchange_rate': rub_usd,
    'USDT_fee': usdt_fee,
    'USDT_CAD_MA': usdt_cad.rolling(window=30).mean(),  # Technical moving average
    'BTC_Price': btc_price,
    'Interest_Rate': interest_rate_proxy,
    'Oil_Price': oil_price,
    'Inflation': inflation,
    'GDP_Growth': gdp_growth,
    'FX_Reserves': fx_reserves,
    'Trade_Balance': trade_balance
})

# Align stablecoin data and forward-fill missing values
data['EUROE_Price'] = euroe_price.reindex(data.index).fillna(method='ffill')
data['DAI_Price'] = dai_price.reindex(data.index).fillna(method='ffill')
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------------------- Step 4: Simulate Russian Stablecoin ----------------------------
# Create synthetic stablecoin price based on USDT/CAD and USDT fee + random noise
#80% weight USDT/CAD
#20% weight USDT fee
#random noise
data['RU_stablecoin_price'] = (
    0.8 * data['USDT_CAD_price'] + 0.2 * data['USDT_fee'] +
    np.random.normal(0, 0.01, len(data))
)

# ---------------------------- Step 5: First-Stage Instrument Regressions ----------------------------
# Stablecoin predicted values and strength of instruments

y_first = data['RU_stablecoin_price']

def run_first_stage(iv_name):
    # Regress stablecoin price on instrument
    X = sm.add_constant(data[[iv_name]])
    model = sm.OLS(y_first, X).fit()
    f_test = model.f_test(f"{iv_name} = 0")
    f_val = f_test.statistic
    p_val = f_test.pvalue
    print(f"\n🔹 First Stage — {iv_name}:\n", model.summary())
    print(f" F-Test: F = {f_val:.2f}, p = {p_val:.4f}")
    return model.fittedvalues

# Fit values using CAD as main instrument
data['RU_stablecoin_fitted'] = run_first_stage('USDT_CAD_price')

# Additional first-stage tests for other candidates
run_first_stage('USDT_fee')
run_first_stage('EUROE_Price')
run_first_stage('DAI_Price')

# ---------------------------- Step 6: Second Stage - Baseline OLS ----------------------------
# RUB/USD exchange rate regressed on predicted stablecoin price and MA
X_second = sm.add_constant(data[['RU_stablecoin_fitted', 'USDT_CAD_MA']])
y_second = data['RUB_exchange_rate']
second_stage = sm.OLS(y_second, X_second).fit()
print("\n Second Stage Regression (OLS):\n", second_stage.summary())

# ---------------------------- Step 7: IV Estimations (EUROE and DAI Instruments) ----------------------------
exog_vars = ['USDT_CAD_MA', 'BTC_Price', 'Interest_Rate', 'Oil_Price',
             'Inflation', 'GDP_Growth', 'FX_Reserves', 'Trade_Balance']

# IV Regression using EUROE
iv_model_euroe = IV2SLS(
    dependent=data['RUB_exchange_rate'],
    exog=sm.add_constant(data[exog_vars]),
    endog=data['RU_stablecoin_price'],
    instruments=data[['EUROE_Price']]
).fit()
print("\n IV Regression (EUROE as Instrument):\n", iv_model_euroe.summary)

# IV Regression using DAI
iv_model_dai = IV2SLS(
    dependent=data['RUB_exchange_rate'],
    exog=sm.add_constant(data[exog_vars]),
    endog=data['RU_stablecoin_price'],
    instruments=data[['DAI_Price']]
).fit()
print("\n IV Regression (DAI as Instrument):\n", iv_model_dai.summary)

# ---------------------------- Step 8: Durbin-Wu-Hausman Test for Endogeneity ----------------------------
# Tests if OLS is biased (i.e., endogenous regressor detected)
data['first_stage_resid'] = data['RU_stablecoin_price'] - data['RU_stablecoin_fitted']
X_dwh = sm.add_constant(data[['RU_stablecoin_price'] + exog_vars + ['first_stage_resid']])
dwh_model = sm.OLS(data['RUB_exchange_rate'], X_dwh).fit()
print("\n Durbin-Wu-Hausman Test:\n", dwh_model.summary())

# Evaluate significance of residual term to detect endogeneity
p_val_dwh = dwh_model.pvalues['first_stage_resid']
if p_val_dwh < 0.05:
    print(f"\n Endogeneity detected (p = {p_val_dwh:.4f}) — IV regression justified.")
else:
    print(f"\n No strong endogeneity (p = {p_val_dwh:.4f}) — OLS may be sufficient.")

# ---------------------------- Step 9: Instrument Strength Visualization ----------------------------
# Visualize F-statistics to detect weak instruments

f_stats = [
    sm.OLS(y_first, sm.add_constant(data[['USDT_CAD_price']])).fit().f_test("USDT_CAD_price = 0").statistic,
    sm.OLS(y_first, sm.add_constant(data[['USDT_fee']])).fit().f_test("USDT_fee = 0").statistic,
    sm.OLS(y_first, sm.add_constant(data[['EUROE_Price']])).fit().f_test("EUROE_Price = 0").statistic,
    sm.OLS(y_first, sm.add_constant(data[['DAI_Price']])).fit().f_test("DAI_Price = 0").statistic
]

labels = ['USDT_CAD', 'USDT_fee', 'EUROE_Price', 'DAI_Price']

plt.figure(figsize=(10, 5))
plt.bar(labels, f_stats, color=['green', 'blue', 'orange', 'purple'])
plt.axhline(y=10, color='red', linestyle='--', label='Weak IV Threshold')
plt.title("Instrument Strength (F-Test Comparison)")
plt.ylabel("F-statistic")
plt.legend()
plt.tight_layout()
plt.show()
