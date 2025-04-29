
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_dataset.csv", index_col='time', parse_dates=True)

# Fit ARIMA model
model = ARIMA(df['down'], order=(1,1,5))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=48)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=30), periods=48, freq='30T')
forecast.index = forecast_index

plt.figure(figsize=(14, 6))
plt.plot(df['down'].iloc[-200:], label='Historical')
plt.plot(forecast, label='Forecast', color='orange')
plt.legend()
plt.title('ARIMA Forecast')
plt.tight_layout()
plt.show()

# Save residuals
residuals = model_fit.resid
residuals.to_csv("arima_residuals.csv")
