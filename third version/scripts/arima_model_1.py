import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# --- CONFIG ---
BASE_PATH = r"F:\assign\python practice\Bachelor-thesis-\third version"
DATA_PATH = os.path.join(BASE_PATH, "outputs", "processed_data.csv")
PLOT_DIR = os.path.join(BASE_PATH, "outputs", "plots")
OUTPUT_CSV = os.path.join(BASE_PATH, "outputs", "arima_forecast.csv")

# --- Load aggregated data ---
df = pd.read_csv(DATA_PATH)
series = df["n_bytes_sum"]

# --- Fit ARIMA Model ---
model = ARIMA(series, order=(2, 0, 3))  # You can tune this
model_fit = model.fit()

# --- Forecast next 48 steps (8 hours at 10-min intervals) ---
forecast_steps = 48
forecast = model_fit.forecast(steps=forecast_steps)

# --- Save forecast to CSV ---
forecast_index = range(df["id_time"].max() + 1, df["id_time"].max() + 1 + forecast_steps)
forecast_df = pd.DataFrame({
    "Forecast_Time": forecast_index,
    "ARIMA_Forecast_n_bytes": forecast
})
forecast_df.to_csv(OUTPUT_CSV, index=False)

# --- Plot original series + forecast ---
plt.figure(figsize=(14, 6))
plt.plot(df["id_time"], series, label="Observed", color="blue")
plt.plot(forecast_index, forecast, label="ARIMA Forecast", linestyle="--", color="orange")
plt.title("ARIMA Forecast (Next 8 Hours)")
plt.xlabel("Time (10-min intervals)")
plt.ylabel("Total Bytes (n_bytes_sum)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "arima_forecast_plot.png"))
plt.close()

print("✅ ARIMA forecast saved:")
print(f"- Plot: {os.path.join(PLOT_DIR, 'arima_forecast_plot.png')}")
print(f"- Forecast CSV: {OUTPUT_CSV}")
