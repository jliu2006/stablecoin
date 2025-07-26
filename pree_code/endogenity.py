import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, compare

# Step 1: Download data
df = yf.download(['USDRUB=X', 'USDCAD=X'], start="2020-01-01", end="2025-01-01")['Close']
df.columns = ['USDTRUB', 'USDTCAD']

# Step 2: Feature engineering
df['RUB'] = df['USDTRUB']  # Assume RUB is proxied by USDTRUB
df['X1'] = df['USDTCAD'].pct_change().fillna(0)  # Example control: CAD volatility
df['X2'] = df['USDTRUB'].rolling(window=5).mean().bfill()  # Control: Moving average
df.dropna(inplace=True)

# Step 3: First Stage — Regress USDTRUB on Instrument + Controls
y = df['RUB']
endog = df['USDTRUB']
instrument = df['USDTCAD']
controls = df[['X1', 'X2']]
X_full = sm.add_constant(pd.concat([endog, controls], axis=1))

X_first_stage = sm.add_constant(pd.concat([instrument, controls], axis=1))
first_stage = sm.OLS(endog, X_first_stage).fit()
print("\n First Stage Summary:")
print(first_stage.summary())

# Weak Instrument Check
f_stat = first_stage.fvalue
print(f"\n First Stage F-statistic: {f_stat:.2f} (rule of thumb > 10 for strength)")

# Step 4: Second Stage — Regress RUB on Fitted Values + Controls
endog_hat = first_stage.fittedvalues
X_second_stage = sm.add_constant(pd.concat([endog_hat, controls], axis=1))
second_stage = sm.OLS(y, X_second_stage).fit()
print("\n Second Stage Summary:")
print(second_stage.summary())

# Step 5: Durbin-Wu-Hausman Test — Compare IV vs OLS from linearmodels
# Use linearmodels IV2SLS for both OLS and IV versions
ols_model = IV2SLS.from_formula('RUB ~ 1 + X1 + X2', data=df).fit()
iv_model = IV2SLS.from_formula('RUB ~ 1 + X1 + X2 [USDTRUB ~ USDTCAD]', data=df).fit()

print("\n Durbin-Wu-Hausman Test (IV vs OLS):")
comparison = compare({'OLS': ols_model, 'IV': iv_model})
print(comparison)
