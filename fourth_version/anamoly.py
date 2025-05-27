
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/cleaned_enhanced_downlink.csv", index_col=0, parse_dates=True)
series = df['down'].tail(10000)

start_train_arima = time.time()
model_arima = ARIMA(series, order=(2, 0, 5))
result_arima = model_arima.fit()
end_train_arima = time.time()

start_pred_arima = time.time()
arima_pred = result_arima.predict(start=1, end=len(series)-1)
end_pred_arima = time.time()

residuals = series[1:] - arima_pred
residuals.to_csv("arima_residuals.csv")

residuals = residuals.values.reshape(-1, 1)
scaler = MinMaxScaler()
res_scaled = scaler.fit_transform(residuals)

def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 20
X, y_lstm = create_sequences(res_scaled, seq_len)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start_train_lstm = time.time()
model.fit(X, y_lstm, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)
end_train_lstm = time.time()

start_pred_lstm = time.time()
res_pred_scaled = model.predict(X, verbose=0)
end_pred_lstm = time.time()

res_pred = scaler.inverse_transform(res_pred_scaled)
arima_trimmed = arima_pred[seq_len:].values.reshape(-1, 1)
final_pred = arima_trimmed + res_pred
true_y = series.values[seq_len+1:].reshape(-1, 1)
residuals_total = true_y - final_pred

pd.Series(final_pred.flatten(), index=series.index[seq_len+1:]).to_csv("hybrid_predictions.csv")

mae = mean_absolute_error(true_y, final_pred)
rmse = np.sqrt(mean_squared_error(true_y, final_pred))
mape = mean_absolute_percentage_error(true_y, final_pred) * 100
lb_p = acorr_ljungbox(residuals_total, lags=[10], return_df=True)['lb_pvalue'].values[0]
shapiro_p = shapiro(residuals_total)[1]

print("\n📊 Hybrid ARIMA + LSTM Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Ljung-Box p-value: {lb_p:.4f}")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
print(f"⏱ ARIMA Training Time: {end_train_arima - start_train_arima:.2f} sec")
print(f"⏱ ARIMA Inference Time: {end_pred_arima - start_pred_arima:.2f} sec")
print(f"⏱ LSTM Training Time: {end_train_lstm - start_train_lstm:.2f} sec")
print(f"⏱ LSTM Inference Time: {end_pred_lstm - start_pred_lstm:.2f} sec")

error = np.abs(true_y.flatten() - final_pred.flatten())
mean_err = np.mean(error)
std_err = np.std(error)
threshold = mean_err + 3 * std_err
anomalies = error > threshold
anomaly_indices = np.where(anomalies)[0]
anomaly_times = series.index[seq_len + 1:][anomalies]

print(f"\n🚨 Detected {np.sum(anomalies)} anomalies")

anomaly_df = pd.DataFrame({
    "Time": anomaly_times,
    "Actual": true_y.flatten()[anomalies],
    "Predicted": final_pred.flatten()[anomalies],
    "Error": error[anomalies]
})
anomaly_df.to_csv("detected_anomalies.csv", index=False)

plt.figure(figsize=(14, 6))
plt.plot(true_y, label="Actual Traffic", alpha=0.6)
plt.plot(final_pred, label="Hybrid Forecast", linestyle='--', alpha=0.6)
plt.scatter(anomaly_indices, true_y.flatten()[anomalies], color='red', label="Anomalies", zorder=5)
plt.title("Hybrid ARIMA + LSTM Forecast with Anomaly Detection")
plt.xlabel("Time Step")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hybrid_anomaly_detection.png")
plt.show()
