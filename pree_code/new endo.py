import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# Step 1: Download Data
symbols = ['USDRUB=X', 'USDCAD=X', 'CL=F', 'BTC-USD']
df = yf.download(symbols, start="2020-01-01", end="2025-01-01")['Close']
df.columns = ['USDTRUB', 'USDTCAD', 'WTI_Crude', 'BTC']
df['RUB'] = df['USDTRUB']

# Step 2: Feature Engineering
df['X1'] = df['USDTCAD'].pct_change().fillna(0)
df['X2'] = df['USDTRUB'].rolling(window=5).mean().bfill()
df['BTC_vol'] = df['BTC'].pct_change().rolling(window=5).std().fillna(0)
df['Gas_pct_change'] = df['WTI_Crude'].pct_change().fillna(0)
df['Gas_pct_change_smooth'] = df['Gas_pct_change'].rolling(3).mean().bfill()
df.dropna(inplace=True)

# Step 3: IV Regression
iv = IV2SLS(
    dependent=df['RUB'],
    exog=sm.add_constant(df[['X1', 'X2']]),
    endog=df['USDTRUB'],
    instruments=df[['Gas_pct_change_smooth']]
).fit(cov_type='robust')

# Step 4: IV Regression Summary
print("\nIV Regression Summary:")
print(iv.summary)

# Step 5: First-Stage Regression Manually
first_stage_exog = sm.add_constant(df[['Gas_pct_change_smooth', 'X1', 'X2']])
first_stage_ols = sm.OLS(df['USDTRUB'], first_stage_exog).fit()

print("\nFirst-Stage Coefficients:")
print(first_stage_ols.params)

print(f"\nFirst-Stage R-squared: {first_stage_ols.rsquared:.4f}")

print("\nFirst-Stage t-statistics:")
print(first_stage_ols.tvalues)

# Step 6: Wu-Hausman Test for Endogeneity
print("\nWu-Hausman Test:")
print(iv.wu_hausman)
