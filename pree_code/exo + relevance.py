import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, compare

# Step 1: Download data
df = yf.download(['USDRUB=X', 'USDCAD=X', 'BTC-USD'], start="2020-01-01", end="2025-01-01")['Close']
df.columns = ['USDTRUB', 'USDTCAD', 'BTC']
df['RUB'] = df['USDTRUB']

# Step 2: Feature engineering
df['X1'] = df['USDTCAD'].pct_change().fillna(0)
df['X2'] = df['USDTRUB'].rolling(window=5).mean().bfill()
df['BTC_vol'] = df['BTC'].pct_change().rolling(window=5).std().fillna(0)
df.dropna(inplace=True)

# Step 3: First stage regression
X_first_stage = sm.add_constant(pd.concat([df[['USDTCAD', 'BTC_vol']], df[['X1', 'X2']]], axis=1))
first_stage = sm.OLS(df['USDTRUB'], X_first_stage).fit()
print("\nFirst Stage Summary:")
print(first_stage.summary())
print(f"\nFirst Stage F-statistic: {first_stage.fvalue:.2f} (rule of thumb > 10 for strong instrument)")

# Step 4: Second stage regression
endog_hat = first_stage.fittedvalues
X_second_stage = sm.add_constant(pd.concat([endog_hat, df[['X1', 'X2']]], axis=1))
second_stage = sm.OLS(df['RUB'], X_second_stage).fit()
print("\nSecond Stage Summary:")
print(second_stage.summary())

