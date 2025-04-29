
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from statsmodels.graphics.tsaplots import plot_acf

# Load dataset
df = pd.read_csv("cleaned_dataset.csv", index_col="time", parse_dates=True)
series = df["down"]

# Fit ARIMA model
arima_model = ARIMA(series, order=(1, 1, 5))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=48)
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(minutes=30), periods=48, freq='30min')
arima_forecast.index = forecast_index

# Get residuals
residuals = series - arima_result.fittedvalues.shift(1)
residuals = residuals.dropna()

# Load scaler and LSTM model
scaler = np.load("improved_residual_scaler.npy", allow_pickle=True).item()
res_min = scaler['min']
res_max = scaler['max']
lstm_model = load_model("improved_lstm_model.keras")

# Prepare input sequence for LSTM
def create_last_sequence(data, seq_len=48):
    scaled = (data.values.reshape(-1, 1) - res_min) / (res_max - res_min)
    return scaled[-seq_len:].reshape(1, seq_len, 1)

last_seq = create_last_sequence(residuals)
current_seq = last_seq.copy()
predicted_scaled = []

inference_start = time.time()
for _ in range(48):
    pred = lstm_model.predict(current_seq, verbose=0)[0][0]
    predicted_scaled.append(pred)
    next_input = np.array(pred).reshape(1, 1, 1)
    current_seq = np.concatenate([current_seq[:, 1:, :], next_input], axis=1)
inference_end = time.time()

# Inverse scale
predicted_residuals = np.array(predicted_scaled) * (res_max - res_min) + res_min
predicted_residuals = predicted_residuals.flatten()

# Combine forecasts
hybrid_forecast = arima_forecast.values + predicted_residuals

# Metrics
mae = mean_absolute_error(arima_forecast, hybrid_forecast)
rmse = np.sqrt(mean_squared_error(arima_forecast, hybrid_forecast))
mse = mean_squared_error(arima_forecast, hybrid_forecast)
mape = mean_absolute_percentage_error(arima_forecast, hybrid_forecast) * 100
accuracy = 100 - mape

# Anomaly detection
threshold = np.mean(np.abs(predicted_residuals)) + 3 * np.std(np.abs(predicted_residuals))
anomalies = np.abs(predicted_residuals) > threshold
anomaly_times = forecast_index[anomalies]

# Residual diagnostics
residuals_forecast = hybrid_forecast - arima_forecast.values
ljung_result = acorr_ljungbox(residuals_forecast, lags=[10], return_df=True)
shapiro_stat, shapiro_p = shapiro(residuals_forecast)

# Timing
training_time = time.time()
inference_time = inference_end - inference_start

# Save CSV
results_df = pd.DataFrame({
    "Timestamp": forecast_index,
    "ARIMA_Forecast": arima_forecast.values,
    "Hybrid_Forecast": hybrid_forecast,
    "Predicted_Residual": predicted_residuals,
    "Anomaly": anomalies
})
results_df.to_csv("final_hybrid_forecast_output.csv", index=False)

# Save plot
plt.figure(figsize=(14, 6))
plt.plot(forecast_index, arima_forecast, label="ARIMA Forecast", linestyle='--')
plt.plot(forecast_index, hybrid_forecast, label="Hybrid Forecast", color="green")
plt.scatter(forecast_index[anomalies], hybrid_forecast[anomalies], color='red', label="Anomalies", zorder=5)
plt.title("Final Hybrid Forecast with Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Predicted Downlink (bits)")
plt.legend()
plt.tight_layout()
plt.savefig("final_hybrid_forecast_plot.png")

# ACF plot
plt.figure(figsize=(8, 4))
plot_acf(residuals_forecast, lags=20)
plt.title("Residual Autocorrelation (ACF)")
plt.savefig("residual_acf_plot.png")

# Print metrics
print("\n--- Final Model Evaluation Metrics ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Ljung-Box p-value (lag=10): {ljung_result['lb_pvalue'].values[0]:.4f}")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
print(f"Training Time (approx.): {training_time:.2f} sec")
print(f"Inference Time: {inference_time:.4f} sec")
print(f"Anomalies Detected: {len(anomaly_times)}")
print("Anomaly Timestamps:")
for t in anomaly_times[:5]:
    print(f" - {t}")
