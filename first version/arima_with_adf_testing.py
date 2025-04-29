import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv('./full_dataset.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

series = df['down']

# Check stationarity
result = adfuller(series.dropna())
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

# If p > 0.05, data is non-stationary → difference it
if result[1] > 0.05:
    series_diff = series.diff().dropna()
    print("Data differenced to achieve stationarity.")
else:
    series_diff = series

# Train-test split
train_size = int(len(series_diff) * 0.8)
train, test = series_diff[:train_size], series_diff[train_size:]

# Auto ARIMA with better constraints
model_auto = auto_arima(train, 
                        start_p=0, start_q=0,
                        max_p=5, max_q=5,
                        d=None,       # let it auto-select d via tests
                        seasonal=False, 
                        stepwise=True,
                        suppress_warnings=True,
                        trace=True,
                        error_action='ignore')

print(model_auto.summary())

# Fit ARIMA with optimized order
model = ARIMA(train, order=model_auto.order)
model_fit = model.fit()

# Forecast
forecast_diff = model_fit.forecast(steps=len(test))

# If differenced, invert differencing
if result[1] > 0.05:
    forecast = forecast_diff.cumsum() + series.iloc[train_size]
    actual = series.iloc[train_size+1:]
else:
    forecast = forecast_diff
    actual = test

# Evaluate
mse = mean_squared_error(actual, forecast)
print(f'Optimized ARIMA Test MSE: {mse:.2f}')

# Plot
plt.figure(figsize=(14,6))
plt.plot(actual.index, actual, label='Actual')
plt.plot(actual.index, forecast, label='Forecast')
plt.title('Optimized ARIMA Forecast vs Actual')
plt.legend()
plt.show()
