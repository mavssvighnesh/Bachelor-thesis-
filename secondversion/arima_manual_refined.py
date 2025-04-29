
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("cleaned_dataset.csv", index_col="time", parse_dates=True)
series = df["down"]

# Fit ARIMA(2, 0, 5)
model = ARIMA(series, order=(2, 0, 5))
model_fit = model.fit()

# Forecast next 48 steps
forecast = model_fit.forecast(steps=48)
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(minutes=30), periods=48, freq='30T')
forecast.index = forecast_index

# Compute residuals
fitted_values = model_fit.fittedvalues
residuals = series - fitted_values
residuals.to_csv("arima_residuals_refined.csv")

# Plot forecast
plt.figure(figsize=(14, 6))
plt.plot(series[-200:], label="Observed")
plt.plot(forecast.index, forecast, label="Forecast (Next 48)", color="orange")
plt.title("ARIMA(2,0,5) Forecast and Residuals")
plt.xlabel("Time")
plt.ylabel("Downlink (bits)")
plt.legend()
plt.tight_layout()
plt.show()
