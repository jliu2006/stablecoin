import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
import seaborn as sns

# Step 1: Download stock data
start_date = "2020-01-01"
end_date = "2025-01-01"
USDT = yf.download("USDT-USD", start=start_date, end=end_date, auto_adjust=True)
RUB = yf.download("RUB=X", start=start_date, end=end_date, auto_adjust=True)

# Step 2: Combine and clean
df = pd.concat([USDT['Close'], RUB['Close']], axis=1)
df.columns = ['USDT-USD', 'RUB=X']
df.dropna(inplace=True)

#  Step 3: Plot raw prices
plt.figure(figsize=(12, 6))
sns.lineplot(data=df)
plt.title("USDT-USD vs RUB=X Closing Prices (2020–2025)")
plt.ylabel("Price (USD)")
plt.xlabel("Date")
plt.grid(True)
plt.show()

#  Step 4: Stationarity check & differencing
def adf_test(series, name):
    result = adfuller(series)
    print(f"{name} ADF test p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print(f"{name} is stationary")
    else:
        print(f"{name} is NOT stationary")

adf_test(df['USDT-USD'], "USDT-USD")
adf_test(df['RUB=X'], "RUB=X")

#  Apply differencing if needed
df_diff = df.diff().dropna()
adf_test(df_diff['USDT-USD'], "USDT-USD (diff)")
adf_test(df_diff['RUB=X'], "RUB=X (diff)")

#  Step 5: Plot differenced data
plt.figure(figsize=(12, 5))
sns.lineplot(data=df_diff)
plt.title("Differenced USDT-USD & RUB=X Prices")
plt.ylabel("Price Change")
plt.xlabel("Date")
plt.grid(True)
plt.show()

#  Step 6: Select optimal lag using AIC
model = VAR(df_diff)
lag_order = model.select_order(maxlags=10)
print("\nLag Selection Summary:")
print(lag_order.summary())
selected_lag = lag_order.aic

#  Step 7: Granger causality tests
print("\nUSDT-USD → RUB=X")
grangercausalitytests(df_diff[['RUB=X', 'USDT-USD']], maxlag=selected_lag, verbose=True)

print("\nRUB=X → USDT-USD")
grangercausalitytests(df_diff[['USDT-USD', 'RUB=X']], maxlag=selected_lag, verbose=True)

#  Step 8: Fit VAR and visualize impulse response
var_model = model.fit(selected_lag)
irf = var_model.irf(10)  # horizon of 10 days

#  Impulse Response Plot
irf.plot(orth=True)
plt.suptitle("Impulse Response Functions (USDT-USD & RUB=X)")
plt.tight_layout()
plt.show()

#  Optional: Forecast Error Variance Decomposition
fevd = var_model.fevd(10)
fevd.plot()
plt.suptitle("Forecast Error Variance Decomposition (FEVD)")
plt.tight_layout()
plt.show()
