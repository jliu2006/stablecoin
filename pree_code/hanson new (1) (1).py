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

# Download financial data
usdt_cad = yf.download('USDCAD=X', start=start_date, end=end_date)['Close'].squeeze()
rub_usd = yf.download('RUB=X', start=start_date, end=end_date)['Close'].squeeze()
btc_price = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].squeeze()
interest_rate_proxy = yf.download('^IRX', start=start_date, end=end_date)['Close'].squeeze()
oil_price = yf.download('CL=F', start=start_date, end=end_date)['Close'].squeeze()
dai_price = yf.download('DAI-USD', start=start_date, end=end_date)['Close'].squeeze()

# Simulate macroeconomic controls
inflation = pd.Series(np.random.normal(2.5, 0.5, len(usdt_cad)), index=usdt_cad.index)
gdp_growth = pd.Series(np.random.normal(2.0, 0.3, len(usdt_cad)), index=usdt_cad.index)
fx_reserves = pd.Series(np.random.normal(35, 2, len(usdt_cad)), index=usdt_cad.index)
trade_balance = pd.Series(np.random.normal(-50, 10, len(usdt_cad)), index=usdt_cad.index)
usdt_fee = pd.Series(np.random.normal(0.01, 0.005, len(usdt_cad)), index=usdt_cad.index)

# Build data frame
data = pd.DataFrame({
    'USDT_CAD_price': usdt_cad,
    'RUB_exchange_rate': rub_usd,
    'USDT_fee': usdt_fee,
    'USDT_CAD_MA': usdt_cad.rolling(window=30).mean(),
    'BTC_Price': btc_price,
    'Interest_Rate': interest_rate_proxy,
    'Oil_Price': oil_price,
    'Inflation': inflation,
    'GDP_Growth': gdp_growth,
    'FX_Reserves': fx_reserves,
    'Trade_Balance': trade_balance
}).dropna()

#  Add DAI_Price before using it
data['DAI_Price'] = dai_price.reindex(data.index, method='ffill')

# ---------------------------- Step 2: Simulate Stablecoin ----------------------------
data['RU_stablecoin_price'] = (
    0.8 * data['USDT_CAD_price'] + 0.2 * data['USDT_fee'] +
    np.random.normal(0, 0.01, len(data))
)

# ---------------------------- Step 3: First Stage Tests for All IVs ----------------------------
y_first = data['RU_stablecoin_price']

def run_first_stage(iv_name):
    X = sm.add_constant(data[[iv_name]])
    model = sm.OLS(y_first, X).fit()
    f_test = model.f_test(f"{iv_name} = 0")
    f_val = f_test.statistic
    p_val = f_test.pvalue
    print(f"\n🔹 First Stage Regression — {iv_name}:\n", model.summary())
    print(f" F-Test for {iv_name}: F-stat = {f_val:.2f}, p = {p_val:.4f}")
    return model.fittedvalues


data['RU_stablecoin_fitted'] = run_first_stage('USDT_CAD_price')  # Needed for 2nd stage
run_first_stage('USDT_fee')
run_first_stage('DAI_Price')

# ---------------------------- Step 4: Second Stage (OLS) ----------------------------
X_second = sm.add_constant(data[['RU_stablecoin_fitted', 'USDT_CAD_MA']])
y_second = data['RUB_exchange_rate']
second_stage = sm.OLS(y_second, X_second).fit()
print("\n Second Stage Regression (OLS):\n", second_stage.summary())

# ---------------------------- Step 5: IV Regression with DAI as Instrument ----------------------------
exog_vars = ['USDT_CAD_MA', 'BTC_Price', 'Interest_Rate', 'Oil_Price',
             'Inflation', 'GDP_Growth', 'FX_Reserves', 'Trade_Balance']

iv_model_dai = IV2SLS(
    dependent=data['RUB_exchange_rate'],
    exog=sm.add_constant(data[exog_vars]),
    endog=data['RU_stablecoin_price'],
    instruments=data[['DAI_Price']]
).fit()

print("\n IV Regression (DAI as instrument):\n", iv_model_dai.summary)

# ---------------------------- Step 6: Durbin-Wu-Hausman Test ----------------------------
data['first_stage_resid'] = data['RU_stablecoin_price'] - data['RU_stablecoin_fitted']
X_dwh = sm.add_constant(data[['RU_stablecoin_price'] + exog_vars + ['first_stage_resid']])
dwh_model = sm.OLS(data['RUB_exchange_rate'], X_dwh).fit()
p_val = dwh_model.pvalues['first_stage_resid']
print("\n Durbin-Wu-Hausman Test:\n", dwh_model.summary())
if p_val < 0.05:
    print(f"\n Endogeneity detected (p = {p_val:.4f}) — IV regression justified.")
else:
    print(f"\n No strong endogeneity (p = {p_val:.4f}) — OLS may be sufficient.")

# ---------------------------- Step 7: Visualization of IV Strength ----------------------------
f_stats = [
    sm.OLS(y_first, sm.add_constant(data[['USDT_CAD_price']])).fit().f_test("USDT_CAD_price = 0").statistic,
    sm.OLS(y_first, sm.add_constant(data[['USDT_fee']])).fit().f_test("USDT_fee = 0").statistic,
    sm.OLS(y_first, sm.add_constant(data[['DAI_Price']])).fit().f_test("DAI_Price = 0").statistic
]

labels = ['USDT_CAD', 'USDT_fee', 'DAI_Price']

plt.figure(figsize=(8, 5))
plt.bar(labels, f_stats, color=['green', 'blue', 'orange'])
plt.axhline(y=10, color='red', linestyle='--', label='Weak IV Threshold')
plt.title("Instrument Strength (F-Test Comparison)")
plt.ylabel("F-statistic")
plt.legend()
plt.tight_layout()
plt.show()
