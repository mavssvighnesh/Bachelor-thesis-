
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load modified dataset
df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/full_dataset_modified_only.csv", parse_dates=["time"], index_col="time")
series = df["down"]

# Fit ARIMA model (adjust order as needed)
model = ARIMA(series, order=(2, 0, 5))
model_fit = model.fit()

# Forecast next 48 steps
forecast = model_fit.forecast(steps=48)
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(minutes=30), periods=48, freq='30min')
forecast.index = forecast_index

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(series[-96:], label="Historical")
plt.plot(forecast, label="ARIMA Forecast", linestyle='--')
plt.legend()
plt.title("ARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("Downlink (bits)")
plt.tight_layout()
plt.show()
