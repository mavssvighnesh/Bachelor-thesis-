import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('./full_dataset.csv')

# Convert 'time' to datetime and set as index
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

# Select target variable for ARIMA (e.g. 'down')
series = data['down']

# Train-test split
train_size = int(len(series) * 0.8)
train, test = series[0:train_size], series[train_size:]

# Fit ARIMA model (order needs tuning)
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate
mse = mean_squared_error(test, forecast)
print(f'ARIMA Test MSE: {mse:.2f}')

# Plot forecast vs actual
plt.figure(figsize=(12,6))
plt.plot(test, label='Actual')
plt.plot(test.index, forecast, label='Forecast')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.show()


