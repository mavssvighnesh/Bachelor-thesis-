import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings

warnings.filterwarnings("ignore")

# === Load residuals ===
df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/arima_residuals.csv", index_col=0, parse_dates=True)
raw_residuals = df.values.astype(float)

# === Log transform (optional, helps with spikes) ===
residuals = np.sign(raw_residuals) * np.log1p(np.abs(raw_residuals))

# === Normalize ===
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals)

# === Create sequences ===
def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_length = 20
X, y_seq = create_sequences(residuals_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# === Build LSTM model ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# === Train with timing ===
start_train = time.time()
model.fit(X, y_seq, epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)
end_train = time.time()
train_time = end_train - start_train

# === Predict with timing ===
start_pred = time.time()
predicted_scaled = model.predict(X)
end_pred = time.time()
inference_time = end_pred - start_pred

# === Inverse transform prediction ===
predicted_log = scaler.inverse_transform(predicted_scaled)
predicted = np.sign(predicted_log) * (np.expm1(np.abs(predicted_log)))
true_residuals = raw_residuals[seq_length:]

# === Evaluation ===
mae = mean_absolute_error(true_residuals, predicted)
rmse = np.sqrt(mean_squared_error(true_residuals, predicted))
mape = mean_absolute_percentage_error(true_residuals, predicted) * 100
residual_error = true_residuals - predicted
lb_pvalue = acorr_ljungbox(residual_error, lags=[10], return_df=True)['lb_pvalue'].values[0]
shapiro_p = shapiro(residual_error)[1]

print(f"\n📊 LSTM-on-Residuals Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")





accuracy = 100 - mape
print(f"✅ Accuracy: {accuracy:.2f}%")

print(f"⏱ Training Time: {train_time:.2f} seconds")
print(f"⏱ Inference Time: {inference_time:.2f} seconds")


print(f"Ljung-Box p-value: {lb_pvalue:.4f}")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")


# === Plot residual forecast ===
plt.figure(figsize=(12,5))
plt.plot(df.index[seq_length:], true_residuals, label='Actual Residuals')
plt.plot(df.index[seq_length:], predicted, label='LSTM Predicted Residuals', linestyle='--')
plt.title("Improved LSTM on ARIMA Residuals")
plt.xlabel("Time")
plt.ylabel("Residual Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Histogram of prediction error ===
plt.figure(figsize=(10, 4))
plt.hist(residual_error, bins=40, edgecolor='black')
plt.title("Residual Error Distribution (LSTM on ARIMA Residuals)")
plt.grid(True)
plt.tight_layout()
plt.show()
