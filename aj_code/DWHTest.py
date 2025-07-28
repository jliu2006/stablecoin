import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# Define tickers
lira_ticker = 'TRY=X'
usdt_ticker = 'USDT-USD'
# btc_ticker = 'BTC-USD'  # Removed from IVs
oil_ticker = 'CL=F'     # Instrument
irx_ticker = '^IRX'
spx_ticker = '^GSPC'
dxy_ticker = 'DX-Y.NYB'
gold_ticker = 'GC=F'
eurusd_ticker = 'EURUSD=X'
cny_ticker = 'CNY=X'

start_date = '2018-01-01'
end_date = '2025-07-27'

# Download data
tickers = [lira_ticker, usdt_ticker, oil_ticker, irx_ticker, spx_ticker, dxy_ticker, gold_ticker, eurusd_ticker, cny_ticker]
data_raw = yf.download(tickers, start=start_date, end=end_date, interval='1d')['Close']

# Rename columns
data = pd.DataFrame()
data['Lira'] = data_raw[lira_ticker]
data['USDT'] = data_raw[usdt_ticker]
# data['BTC'] = data_raw[btc_ticker]  # Removed
data['OIL'] = data_raw[oil_ticker]
data['IRX'] = data_raw[irx_ticker]
data['SPX'] = data_raw[spx_ticker]
data['DXY'] = data_raw[dxy_ticker]
data['GOLD'] = data_raw[gold_ticker]
data['EURUSD'] = data_raw[eurusd_ticker]
data['CNY'] = data_raw[cny_ticker]
data = data.dropna()

# Define instrumental variables and controls
IVs = ['CNY']  # Removed 'BTC' from IVs
controls = ['OIL']

# First stage: regress Lira on IVs and controls
X = data['Lira']
Z = data[IVs + controls]
Z_const = sm.add_constant(Z)
first_stage = sm.OLS(X, Z_const).fit()
residuals = first_stage.resid

# Second stage: regress USDT on Lira, residuals, and controls
X_ols = pd.DataFrame({'Lira': X, 'Lira_resid': residuals})
for col in controls:
    X_ols[col] = data[col]
X_ols_const = sm.add_constant(X_ols)
y = data['USDT']
ols_model = sm.OLS(y, X_ols_const).fit()

# Output results
print(ols_model.summary())
print('Coefficient for Lira residuals:', ols_model.params['Lira_resid'])
print('P-value for Lira residuals:', ols_model.pvalues['Lira_resid'])
