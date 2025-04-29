import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("cleaned_dataset.csv", index_col="time", parse_dates=True)
series = df["down"]

# Fit ARIMA(1,1,5)
arima_model = ARIMA(series, order=(1, 1, 5))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=48)
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(minutes=30), periods=48, freq='30min')
arima_forecast.index = forecast_index

# Get updated residuals
residuals = series - arima_result.fittedvalues.shift(1)
residuals = residuals.dropna()

# Load scaler and model
scaler = np.load("improved_residual_scaler.npy", allow_pickle=True).item()
res_min = scaler['min']
res_max = scaler['max']
lstm_model = load_model("improved_lstm_model.keras")

# Create sequence from residuals
def create_last_sequence(data, seq_len=48):
    scaled = (data.values.reshape(-1, 1) - res_min) / (res_max - res_min)
    return scaled[-seq_len:].reshape(1, seq_len, 1)

last_seq = create_last_sequence(residuals)
current_seq = last_seq.copy()
predicted_scaled = []

for _ in range(48):
    pred = lstm_model.predict(current_seq, verbose=0)[0][0]
    predicted_scaled.append(pred)

    next_input = np.array(pred).reshape(1, 1, 1)
    current_seq = np.concatenate([current_seq[:, 1:, :], next_input], axis=1)

# Inverse scale
predicted_residuals = np.array(predicted_scaled) * (res_max - res_min) + res_min
predicted_residuals = predicted_residuals.flatten()

# Hybrid forecast
hybrid_forecast = arima_forecast.values + predicted_residuals

# Anomaly detection
threshold = np.mean(np.abs(predicted_residuals)) + 3 * np.std(np.abs(predicted_residuals))
anomalies = np.abs(predicted_residuals) > threshold
anomaly_times = forecast_index[anomalies]

# Plot
plt.figure(figsize=(14, 6))
plt.plot(forecast_index, arima_forecast, label="ARIMA Forecast", linestyle='--')
plt.plot(forecast_index, hybrid_forecast, label="Hybrid Forecast", color="green")
plt.scatter(anomaly_times, hybrid_forecast[anomalies], color='red', label="Anomalies", zorder=5)
plt.title("Hybrid Forecast with Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Predicted Downlink (bits)")
plt.legend()
plt.tight_layout()
plt.show()

# Metrics
print("📊 Evaluation Metrics:")
mae = mean_absolute_error(arima_forecast, hybrid_forecast)
rmse = np.sqrt(mean_squared_error(arima_forecast, hybrid_forecast))
acc = 100 - (mae / np.mean(np.abs(arima_forecast))) * 100

print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"Accuracy: {acc:.2f}%")

# Anomaly Summary
print(f"\n🚨 Total Anomalies Detected: {len(anomaly_times)}")
print("Timestamps of first few anomalies:")
for t in anomaly_times[:5]:
    print(f" - {t}")
