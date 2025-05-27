import os
import pandas as pd
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# --- CONFIG ---
BASE_PATH = r"F:\assign\python practice\Bachelor-thesis-\third version"
DATA_PATH = os.path.join(BASE_PATH, "outputs", "processed_data.csv")
PLOT_DIR = os.path.join(BASE_PATH, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Load and select series
df = pd.read_csv(DATA_PATH)
series = df["n_bytes_sum"]

# Train/Test Split
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# --- Define ARIMA and SARIMA grid ---
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(1, 1, 1, 144), (0, 1, 1, 144), (1, 0, 1, 144), (1, 1, 0, 144)]  # 1 day = 144 10-min intervals

results = []
best_aic = float("inf")
best_model = None
best_cfg = None

# --- Grid Search ---
for order in pdq:
    for seasonal_order in seasonal_pdq:
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            aic = model.aic
            results.append((order, seasonal_order, aic))

            if aic < best_aic:
                best_aic = aic
                best_model = model
                best_cfg = (order, seasonal_order)
        except:
            continue

# --- Report best config ---
print(f"✅ Best SARIMA config: Order={best_cfg[0]}, Seasonal={best_cfg[1]}, AIC={best_aic:.2f}")

# --- Forecast & Save Plot ---
forecast_steps = len(test)
forecast = best_model.forecast(steps=forecast_steps)

plt.figure(figsize=(14, 6))
plt.plot(series.index[:train_size], train, label="Train")
plt.plot(series.index[train_size:], test, label="Test")
plt.plot(series.index[train_size:], forecast, label="SARIMA Forecast", linestyle='--')
plt.title("Best SARIMA Model Forecast")
plt.xlabel("Time (10-min intervals)")
plt.ylabel("Bytes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "best_sarima_forecast.png"))
plt.close()

# --- Save Top 15 configs by AIC ---
top_df = pd.DataFrame(results, columns=["Order", "Seasonal_Order", "AIC"]).sort_values(by="AIC").head(15)
plt.figure(figsize=(12, 6))
plt.bar([f"{a}-{b}" for a, b, _ in top_df.values], top_df["AIC"])
plt.xticks(rotation=45)
plt.ylabel("AIC")
plt.title("Top 15 SARIMA Configurations by AIC")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "sarima_aic_comparison.png"))
plt.close()

print("📊 Saved forecast and AIC comparison plots.")
