import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# --- CONFIG ---
BASE = r"F:\assign\python practice\Bachelor-thesis-\third version"
DATA_PATH = os.path.join(BASE, "outputs", "processed_data.csv")
RESIDUALS_PATH = os.path.join(BASE, "outputs", "lstm_predicted_residuals.npy")
PLOT_DIR = os.path.join(BASE, "outputs", "plots")

df = pd.read_csv(DATA_PATH)
arima_model = ARIMA(df["n_bytes_sum"], order=(2, 1, 2)).fit()
arima_forecast = arima_model.forecast(steps=48)
lstm_residuals = np.load(RESIDUALS_PATH)

# --- Combine ---
hybrid_forecast = arima_forecast.values + lstm_residuals
forecast_index = np.arange(len(df), len(df) + 48)

# --- Plot ---
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["n_bytes_sum"], label="Observed", color="blue")
plt.plot(forecast_index, arima_forecast, label="ARIMA Forecast", linestyle='--', color="orange")
plt.plot(forecast_index, hybrid_forecast, label="Hybrid Forecast", color="green")
plt.title("Hybrid ARIMA + LSTM Forecast")
plt.xlabel("Time (10-min intervals)")
plt.ylabel("Bytes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "hybrid_forecast.png"))
plt.close()

print("✅ Hybrid forecast plot saved.")
