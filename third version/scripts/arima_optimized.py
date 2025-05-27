import os
import pandas as pd
import numpy as np
import warnings
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# --- CONFIG ---
BASE = r"F:\assign\python practice\Bachelor-thesis-\third version"
DATA_PATH = os.path.join(BASE, "outputs", "processed_data.csv")
PLOT_DIR = os.path.join(BASE, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
series = df["n_bytes_sum"]

# --- STEP 1: Log Transform ---
log_series = np.log1p(series)  # use log(1 + x) to avoid log(0)

# --- STEP 2: ADF test for differencing ---
def get_adf_diff_level(series, max_diff=2):
    for d in range(max_diff + 1):
        diffed = np.diff(series, n=d)
        pval = adfuller(diffed)[1]
        if pval < 0.05:
            return d
    return max_diff

d_opt = get_adf_diff_level(log_series)

# --- STEP 3: Grid Search for Best (p,d,q) using AIC ---
p = q = range(0, 4)
pdq = list(itertools.product(p, [d_opt], q))

best_aic = float("inf")
best_order = None
best_model = None

for order in pdq:
    try:
        model = ARIMA(log_series, order=order).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            best_order = order
            best_model = model
    except:
        continue

print(f"✅ Best ARIMA: {best_order} with AIC: {best_aic:.2f}")

# --- STEP 4: Forecast in Log Space and Convert Back ---
forecast_steps = 48
forecast_log = best_model.forecast(steps=forecast_steps)
forecast = np.expm1(forecast_log)  # invert log1p

# --- Plot ---
forecast_index = np.arange(len(series), len(series) + forecast_steps)
plt.figure(figsize=(14, 6))
plt.plot(series.index, series.values, label="Observed", color="blue")
plt.plot(forecast_index, forecast, label="ARIMA Forecast", linestyle="--", color="orange")
plt.title("Improved ARIMA Forecast (Log-Transformed Series)")
plt.xlabel("Time (10-min intervals)")
plt.ylabel("Total Bytes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "improved_arima_forecast.png"))
plt.close()

# --- Save Forecast ---
forecast_df = pd.DataFrame({
    "time": forecast_index,
    "forecast_bytes": forecast
})
forecast_df.to_csv(os.path.join(BASE, "outputs", "improved_arima_forecast.csv"), index=False)

print("📈 Forecast and plot saved.")
