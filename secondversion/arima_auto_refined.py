
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Load dataset
df = pd.read_csv("cleaned_dataset.csv", index_col="time", parse_dates=True)
series = df["down"]

# Step 1: Check stationarity
adf_result = adfuller(series)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print("Stationary:", adf_result[1] < 0.05)

# Step 2: Auto ARIMA
model = auto_arima(series,
                   seasonal=True, m=48,
                   start_p=1, start_q=1, max_p=5, max_q=5,
                   d=None, max_d=2,
                   trace=True,
                   error_action='ignore', suppress_warnings=True,
                   stepwise=True)

# Step 3: Forecast 48 steps
n_steps = 48
forecast = model.predict(n_periods=n_steps)
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(minutes=30), periods=n_steps, freq='30T')

# Step 4: Residuals
fitted_values = model.predict_in_sample()
residuals = series.values - fitted_values
residuals_series = pd.Series(residuals, index=series.index)
residuals_series.to_csv("arima_residuals_refined.csv")

# Step 5: Plot
plt.figure(figsize=(14, 6))
plt.plot(series[-200:], label="Observed")
plt.plot(forecast_index, forecast, label="Forecast (48 steps)", color="orange")
plt.title("Auto ARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("Downlink (bits)")
plt.legend()
plt.tight_layout()
plt.show()
