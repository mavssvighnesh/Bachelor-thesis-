import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings

warnings.filterwarnings("ignore")

# === Load data ===
df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/cleaned_enhanced_downlink.csv", index_col=0, parse_dates=True)
y = df.squeeze()

# === Fit ARIMA ===
arima_order = (2, 0, 5)
arima_model = ARIMA(y, order=arima_order)
arima_result = arima_model.fit()
arima_pred = arima_result.predict(start=1, end=len(y)-1)
residuals = y[1:] - arima_pred

# === LSTM on residuals ===
residuals = residuals.values.reshape(-1, 1)
scaler = MinMaxScaler()
res_scaled = scaler.fit_transform(residuals)

# === Sequence creation ===
def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 20
X_res, y_res = create_sequences(res_scaled, seq_len)
X_res = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))

# === LSTM Model ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_len, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Measure training time ===
start_train = time.time()
model.fit(X_res, y_res, epochs=30, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)
end_train = time.time()
train_time = end_train - start_train

# === Inference time ===
start_pred = time.time()
res_pred_scaled = model.predict(X_res)
end_pred = time.time()
inference_time = end_pred - start_pred

# === Inverse transform and align ===
res_pred = scaler.inverse_transform(res_pred_scaled)
arima_aligned = arima_pred[seq_len:].values.reshape(-1, 1)
final_pred = arima_aligned + res_pred
true_y = y.values[seq_len+1:].reshape(-1, 1)
residuals_hybrid = true_y - final_pred

# === Save predictions ===
pd.Series(final_pred.flatten(), index=df.index[seq_len+1:]).to_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/hybrid_predicted_final.csv")
print("✅ Hybrid predictions saved.")

# === Evaluation ===
mae = mean_absolute_error(true_y, final_pred)
rmse = np.sqrt(mean_squared_error(true_y, final_pred))
mape = mean_absolute_percentage_error(true_y, final_pred) * 100
lb_pvalue = acorr_ljungbox(residuals_hybrid, lags=[10], return_df=True)['lb_pvalue'].values[0]
shapiro_p = shapiro(residuals_hybrid)[1]

print(f"\n📊 Hybrid Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Ljung-Box p-value: {lb_pvalue:.4f}")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
print(f"⏱ Training Time: {train_time:.2f} seconds")
print(f"⏱ Inference Time: {inference_time:.2f} seconds")

# === Plot residual histogram ===
plt.figure(figsize=(10, 4))
plt.hist(residuals_hybrid, bins=40, edgecolor='black')
plt.title("Hybrid Model Residuals Histogram")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Forecast ===
plt.figure(figsize=(12, 5))
plt.plot(true_y, label='Actual Traffic')
plt.plot(final_pred, label='Hybrid Prediction', linestyle='--')
plt.title("Hybrid ARIMA + LSTM Forecast")
plt.xlabel("Time Step")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
