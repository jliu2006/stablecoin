# ---------------------------- Imports ----------------------------
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# ---------------------------- Step 1: Load MOEX Data ----------------------------
start_date = '2020-01-01'
end_date = '2025-01-01'

# Download MOEX Index data from Yahoo Finance
moex_index = yf.download('IMOEX.ME', start=start_date, end=end_date)['Close'].squeeze()

# Build DataFrame
data = pd.DataFrame({'MOEX_Index': moex_index})

# Drop missing values for analysis
moex = data['MOEX_Index'].dropna()

# ---------------------------- Step 2: Descriptive Statistics ----------------------------
print("\n Step 1: Descriptive Statistics")
stats = moex.describe()
print(stats)

# Variability check
var_ok = stats['std'] > 0.01  # simple threshold
print("→ Variability Test:", "Pass " if var_ok else "Fail ")

# ---------------------------- Step 3: Distribution Plot ----------------------------
plt.figure(figsize=(8,5))
sns.histplot(moex, kde=True, color='mediumorchid')
plt.title("Distribution of MOEX_Index")
plt.xlabel("MOEX Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ---------------------------- Step 4: Stationarity Test ----------------------------
print("\n Step 2: Augmented Dickey-Fuller Test (ADF)")
adf_result = adfuller(moex)
adf_stat, adf_pval = adf_result[0], adf_result[1]
print(f"ADF Statistic = {adf_stat:.4f}")
print(f"p-value = {adf_pval:.4f}")
stationary = adf_pval < 0.05
print("→ Stationarity Test:", "Pass " if stationary else "Fail ")

# ---------------------------- Step 5: Autocorrelation Plot ----------------------------
plt.figure(figsize=(8,4))
plot_acf(moex, lags=30)
plt.title("Autocorrelation of MOEX_Index")
plt.tight_layout()
plt.show()

# ---------------------------- Step 6: Final Verdict ----------------------------
print("\n Final Evaluation of MOEX_Index as Dependent Variable")
if var_ok and stationary:
    print(" Verdict: YES — MOEX_Index is a statistically valid dependent variable.")
    print("Justification: It shows sufficient variation and passes the stationarity test.")
    print("Distribution and autocorrelation appear reasonable for financial data.")
else:
    print(" Verdict: NO — MOEX_Index may not be suitable without transformation.")
    print("Justification: It either lacks variability or is non-stationary, which can affect model validity.")
    print("Consider differencing the series or applying a log transformation.")
