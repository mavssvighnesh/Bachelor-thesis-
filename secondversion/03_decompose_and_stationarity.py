
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_dataset.csv", index_col='time', parse_dates=True)

# Decomposition
result = seasonal_decompose(df['down'], model='additive', period=48)
result.plot()
plt.tight_layout()
plt.show()

# ADF Test
adf_test = adfuller(df['down'])
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])
print("Stationary:", adf_test[1] < 0.05)
