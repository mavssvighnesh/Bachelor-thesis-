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

# === Load dataset ===
df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/cleaned_enhanced_downlink.csv", index_col=0, parse_dates=True)
data = df.values.astype(float)

# === Normalize data ===
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

# === Create sequences ===
def create_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_length = 20
X, y_seq = create_sequences(scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# === Build model ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Training time ===
start_train = time.time()
model.fit(X, y_seq, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)
end_train = time.time()
train_time = end_train - start_train

# === Inference time ===
start_pred = time.time()
predicted_scaled = model.predict(X)
end_pred = time.time()
inference_time = end_pred - start_pred

# === Inverse scale ===
predicted = scaler.inverse_transform(predicted_scaled)
true_y = data[seq_length:]
residuals_lstm = true_y - predicted

# === Save predictions if needed
pd.Series(predicted.flatten(), index=df.index[seq_length:]).to_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/direct_lstm_predicted.csv")
print("✅ LSTM predictions saved.")

# === Evaluation ===
mae = mean_absolute_error(true_y, predicted)
rmse = np.sqrt(mean_squared_error(true_y, predicted))
mape = mean_absolute_percentage_error(true_y, predicted) * 100
lb_pvalue = acorr_ljungbox(residuals_lstm, lags=[10], return_df=True)['lb_pvalue'].values[0]
shapiro_p = shapiro(residuals_lstm)[1]

print(f"\n📊 Direct LSTM Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Ljung-Box p-value: {lb_pvalue:.4f}")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
print(f"⏱ Training Time: {train_time:.2f} seconds")
print(f"⏱ Inference Time: {inference_time:.2f} seconds")

# === Plot predictions ===
plt.figure(figsize=(12, 5))
plt.plot(df.index[seq_length:], true_y, label='Actual Traffic')
plt.plot(df.index[seq_length:], predicted, label='LSTM Predicted Traffic', linestyle='--')
plt.title("Direct LSTM on Full Time Series")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot histogram ===
plt.figure(figsize=(10, 4))
plt.hist(residuals_lstm, bins=40, edgecolor='black')
plt.title("Residual Error Distribution (Direct LSTM)")
plt.grid(True)
plt.tight_layout()
plt.show()
