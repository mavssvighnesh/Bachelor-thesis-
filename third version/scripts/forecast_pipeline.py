import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy.stats import shapiro

# --- Load data ---
BASE=  r"F:/assign/python practice/Bachelor-thesis-/third version"
df =  pd.read_csv(os.path.join(BASE,"outputs","processed_data.csv"))
series = df["n_bytes_sum"]

# --- Metrics ---
def evaluate_model(true, pred, model_name=""):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred) * 100
    print(f"\n📊 {model_name} Performance:")
    print(f" - RMSE : {rmse:,.2f}")
    print(f" - MAE  : {mae:,.2f}")
    print(f" - MAPE : {mape:.2f}%")
    return rmse, mae, mape

# --- ARIMA ---
arima_model = ARIMA(series, order=(2, 1, 2)).fit()
arima_forecast = arima_model.forecast(steps=48)
forecast_index = np.arange(len(series), len(series) + 48)

plt.figure(figsize=(14, 5))
plt.plot(series.index, series, label="Observed", color="blue", alpha=0.5)
plt.plot(forecast_index, arima_forecast, label="ARIMA Forecast", linestyle='--', color="orange")
plt.title("ARIMA Forecast")
plt.xlabel("Time (10-min intervals)")
plt.ylabel("Bytes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

arima_baseline = np.repeat(series.iloc[-1], 48)
evaluate_model(arima_baseline, arima_forecast.values, "ARIMA")

df["arima_fitted"] = arima_model.fittedvalues
df["residuals"] = df["n_bytes_sum"] - df["arima_fitted"]

plt.figure(figsize=(10, 4))
plt.hist(df["residuals"].dropna(), bins=50, color='purple', alpha=0.7)
plt.title("ARIMA Residuals Histogram")
plt.grid(True)
plt.tight_layout()
plt.show()

_, p_value = shapiro(df["residuals"].dropna())
print(f"Residual Normality (Shapiro-Wilk p={p_value:.4f})")

# --- LSTM ---
scaler = MinMaxScaler()
res_scaled = scaler.fit_transform(df["residuals"].fillna(0).values.reshape(-1, 1))

SEQ_LEN = 48
X, y = [], []
for i in range(SEQ_LEN, len(res_scaled)):
    X.append(res_scaled[i-SEQ_LEN:i])
    y.append(res_scaled[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=30, batch_size=32, verbose=0)

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

last_seq = X[-1:]
preds_scaled = []
for _ in range(48):
    pred = model.predict(last_seq, verbose=0)[0][0]
    preds_scaled.append(pred)
    last_seq = np.append(last_seq[0, 1:, 0], pred).reshape(1, SEQ_LEN, 1)

lstm_forecast_residuals = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
evaluate_model(np.zeros_like(lstm_forecast_residuals), lstm_forecast_residuals, "LSTM (Residual)")

# --- Hybrid Forecast ---
hybrid_forecast = arima_forecast.values + lstm_forecast_residuals

plt.figure(figsize=(14, 6))
plt.plot(df.index, df["n_bytes_sum"], label="Observed", color="blue", alpha=0.5)
plt.plot(forecast_index, arima_forecast, label="ARIMA", linestyle='--', color="orange")
plt.plot(forecast_index, hybrid_forecast, label="Hybrid", color="green")
plt.title("Hybrid ARIMA + LSTM Forecast")
plt.xlabel("Time (10-min intervals)")
plt.ylabel("Bytes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

evaluate_model(arima_forecast.values, hybrid_forecast, "Hybrid Forecast")

# --- Anomaly Detection ---
threshold = np.mean(np.abs(lstm_forecast_residuals)) + 3 * np.std(lstm_forecast_residuals)
anomalies = np.abs(lstm_forecast_residuals) > threshold
anomaly_times = forecast_index[anomalies]

print(f"\n🔍 Anomaly Detection:")
print(f" - Threshold: {threshold:.2f}")
print(f" - Total Detected: {np.sum(anomalies)}")
print(" - Anomaly Time Indexes:", anomaly_times[:5])
