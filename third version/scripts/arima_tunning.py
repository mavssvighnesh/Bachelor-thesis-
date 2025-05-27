import os
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# --- Config ---
BASE_PATH = r"F:\assign\python practice\Bachelor-thesis-\third version"
DATA_PATH = os.path.join(BASE_PATH, "outputs", "processed_data.csv")
PLOT_DIR = os.path.join(BASE_PATH, "outputs", "plots")

df = pd.read_csv(DATA_PATH)
series = df["n_bytes_sum"]

# --- Train/Test Split ---
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# --- Grid Search ARIMA(p,d,q) ---
best_score, best_cfg = float("inf"), None
results = []

for p in range(0, 4):
    for d in range(0, 3):
        for q in range(0, 4):
            try:
                model = ARIMA(train, order=(p, d, q)).fit()
                forecast = model.forecast(steps=len(test))
                rmse = mean_squared_error(test, forecast, squared=False)
                results.append(((p, d, q), rmse))
                if rmse < best_score:
                    best_score, best_cfg = rmse, (p, d, q)
            except:
                continue

# --- Report Best Config ---
print(f"✅ Best ARIMA config: {best_cfg} with RMSE: {best_score:.2f}")

# --- Plot RMSEs for all tried configs ---
results_df = pd.DataFrame(results, columns=["Config", "RMSE"])
results_df.sort_values(by="RMSE", inplace=True)

# Save plot
plt.figure(figsize=(12, 6))
plt.bar([str(cfg) for cfg, _ in results_df.values[:15]], results_df["RMSE"].values[:15])
plt.title("Top 15 ARIMA Configurations by RMSE")
plt.xticks(rotation=45)
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "arima_hyperparameter_rmse.png"))
plt.close()

print("📊 Saved RMSE comparison plot.")
