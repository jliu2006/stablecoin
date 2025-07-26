import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

# ---------------------------- Step 1: Load Financial Data ----------------------------
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download exchange rates and macro-financial variables
usdt_cad = yf.download('USDCAD=X', start=start_date, end=end_date)['Close'].squeeze()       # USDT to CAD exchange rate
rub_usd = yf.download('RUB=X', start=start_date, end=end_date)['Close'].squeeze()           # RUB to USD exchange rate
btc_price = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].squeeze()       # Bitcoin price (control)
interest_rate_proxy = yf.download('^IRX', start=start_date, end=end_date)['Close'].squeeze()# US interest rate proxy (13w T-bill)
oil_price = yf.download('CL=F', start=start_date, end=end_date)['Close'].squeeze()          # WTI crude oil price

# ---------------------------- Step 2: Load Stablecoin Instruments ----------------------------
# Download fiat and crypto-backed stablecoin prices
euroe_price = yf.download('EUROE-USD', start=start_date, end=end_date)['Close'].squeeze()   # European stablecoin
dai_price = yf.download('DAI-USD', start=start_date, end=end_date)['Close'].squeeze()       # Crypto-backed stablecoin
busd_price = yf.download('BUSD-USD', start=start_date, end=end_date)['Close'].squeeze()     # Binance USD stablecoin
gyen_price = yf.download('GYEN-USD', start=start_date, end=end_date)['Close'].squeeze()     # Yen-backed stablecoin
eurc_price = yf.download('EURC-USD', start=start_date, end=end_date)['Close'].squeeze()     # Euro Coin by Circle

# ---------------------------- Step 3: Simulate Macroeconomic Controls (placeholders) ----------------------------
# Simulate key macroeconomic indicators for Russia
inflation = pd.Series(np.random.normal(2.5, 0.5, len(usdt_cad)), index=usdt_cad.index)
gdp_growth = pd.Series(np.random.normal(2.0, 0.3, len(usdt_cad)), index=usdt_cad.index)
fx_reserves = pd.Series(np.random.normal(35, 2, len(usdt_cad)), index=usdt_cad.index)
trade_balance = pd.Series(np.random.normal(-50, 10, len(usdt_cad)), index=usdt_cad.index)

# Simulate USDT transaction fee
usdt_fee = pd.Series(np.random.normal(0.01, 0.005, len(usdt_cad)), index=usdt_cad.index)

# ---------------------------- Step 4: Build Primary DataFrame ----------------------------
data = pd.DataFrame({
    'USDT_CAD_price': usdt_cad,
    'RUB_exchange_rate': rub_usd,
    'USDT_fee': usdt_fee,
    'USDT_CAD_MA': usdt_cad.rolling(window=30).mean(),      # Technical trend
    'BTC_Price': btc_price,
    'Interest_Rate': interest_rate_proxy,
    'Oil_Price': oil_price,
    'Inflation': inflation,
    'GDP_Growth': gdp_growth,
    'FX_Reserves': fx_reserves,
    'Trade_Balance': trade_balance
})

# Fill stablecoin data, forward-fill missing values
data['EUROE_Price'] = euroe_price.reindex(data.index).fillna(method='ffill')
data['DAI_Price'] = dai_price.reindex(data.index).fillna(method='ffill')
data['BUSD_Price'] = busd_price.reindex(data.index).fillna(method='ffill')
data['GYEN_Price'] = gyen_price.reindex(data.index).fillna(method='ffill')
data['EURC_Price'] = eurc_price.reindex(data.index).fillna(method='ffill')

# Clean NaNs or infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------------------- Step 5: Simulate Russian Stablecoin Price ----------------------------
# Weighted synthetic stablecoin + random noise
#80% weight USDT/CAD
#20% weight USDT fee
#random noise
data['RU_stablecoin_price'] = (
    0.8 * data['USDT_CAD_price'] + 0.2 * data['USDT_fee'] +
    np.random.normal(0, 0.01, len(data))
)

# ---------------------------- Step 6: First Stage IV - Instrument Relevance ----------------------------
y_first = data['RU_stablecoin_price']

def run_first_stage(iv_name):
    # Regress simulated stablecoin on instrument
    X = sm.add_constant(data[[iv_name]])
    model = sm.OLS(y_first, X).fit()
    f_test = model.f_test(f"{iv_name} = 0")
    f_val = f_test.statistic
    p_val = f_test.pvalue
    print(f"\n🔹 First Stage — {iv_name}:\n", model.summary())
    print(f" F-Test: F = {f_val:.2f}, p = {p_val:.4f}")
    return model.fittedvalues

# Run individual tests for all candidate instruments
data['RU_stablecoin_fitted'] = run_first_stage('USDT_CAD_price')   # Save predicted values
run_first_stage('USDT_fee')
run_first_stage('EUROE_Price')
run_first_stage('DAI_Price')
run_first_stage('BUSD_Price')
run_first_stage('GYEN_Price')
run_first_stage('EURC_Price')

# ---------------------------- Step 7: Second Stage (OLS) ----------------------------
# Use fitted stablecoin to explain RUB exchange rate
X_second = sm.add_constant(data[['RU_stablecoin_fitted', 'USDT_CAD_MA']])
y_second = data['RUB_exchange_rate']
second_stage = sm.OLS(y_second, X_second).fit()
print("\n Second Stage Regression (OLS):\n", second_stage.summary())

# ---------------------------- Step 8: IV Regressions with Individual Instruments ----------------------------
exog_vars = ['USDT_CAD_MA', 'BTC_Price', 'Interest_Rate', 'Oil_Price',
             'Inflation', 'GDP_Growth', 'FX_Reserves', 'Trade_Balance']

# Loop over each instrument
for stablecoin in ['EUROE_Price', 'DAI_Price', 'BUSD_Price', 'GYEN_Price', 'EURC_Price']:
    iv_model = IV2SLS(
        dependent=data['RUB_exchange_rate'],
        exog=sm.add_constant(data[exog_vars]),
        endog=data['RU_stablecoin_price'],
        instruments=data[[stablecoin]]
    ).fit()
    print(f"\n IV Regression ({stablecoin} as Instrument):\n", iv_model.summary)

# ---------------------------- Step 9: Durbin-Wu-Hausman Test ----------------------------
# Diagnostic for endogeneity in stablecoin price
data['first_stage_resid'] = data['RU_stablecoin_price'] - data['RU_stablecoin_fitted']
X_dwh = sm.add_constant(data[['RU_stablecoin_price'] + exog_vars + ['first_stage_resid']])
dwh_model = sm.OLS(data['RUB_exchange_rate'], X_dwh).fit()
p_val_dwh = dwh_model.pvalues['first_stage_resid']
print("\n Durbin-Wu-Hausman Test:\n", dwh_model.summary())
if p_val_dwh < 0.05:
    print(f"\n Endogeneity detected (p = {p_val_dwh:.4f}) — IV regression justified.")
else:
    print(f"\n No strong endogeneity (p = {p_val_dwh:.4f}) — OLS may be sufficient.")

# ---------------------------- Step 10: Instrument Strength Visualization ----------------------------
f_stats = []
labels = []
stablecoins = ['USDT_CAD_price', 'USDT_fee', 'EUROE_Price', 'DAI_Price',
               'BUSD_Price', 'GYEN_Price', 'EURC_Price']

# Compute F-statistics for all instruments
for sc in stablecoins:
    f_stat = sm.OLS(y_first, sm.add_constant(data[[sc]])).fit().f_test(f"{sc} = 0").statistic
    f_stats.append(f_stat)
    labels.append(sc)

# Plot comparison
plt.figure(figsize=(14, 6))
plt.bar(labels, f_stats, color=['green', 'blue', 'orange', 'purple', 'red', 'teal', 'gray'])
plt.axhline(y=10, color='black', linestyle='--', label='Weak IV Threshold')
plt.title("Instrument Strength (F-Test Comparison)")
plt.ylabel("F-statistic")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
