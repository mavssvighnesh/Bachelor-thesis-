import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
import os
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# --- Load your dataset ---
BASE=  r"F:/assign/python practice/Bachelor-thesis-/third version"
df =  pd.read_csv(os.path.join(BASE,"outputs","processed_data.csv")) # Replace with your real file
series = df["n_bytes_sum"]

# --- Normalize the data ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.values.reshape(-1, 1))

# --- Create input sequences for LSTM ---
SEQ_LEN = 48  # Look-back window
X, y = [], []
for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i - SEQ_LEN:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], SEQ_LEN, 1))

# --- Train-test split ---
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Build LSTM model ---
model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# --- Train the model ---
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# --- Plot training and validation loss ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("LSTM Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Forecast next 48 time steps ---
last_seq = X[-1:]
preds_scaled = []
for _ in range(48):
    pred = model.predict(last_seq, verbose=0)[0][0]
    preds_scaled.append(pred)
    last_seq = np.append(last_seq[0, 1:, 0], pred).reshape(1, SEQ_LEN, 1)

forecast = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
forecast_index = np.arange(len(series), len(series) + 48)

# --- Plot forecast against actual series ---
plt.figure(figsize=(14, 6))
plt.plot(df.index, series, label="Observed", alpha=0.5)
plt.plot(forecast_index, forecast, label="LSTM Forecast", color="green")
plt.title("LSTM-Only Forecast on Raw Traffic Data")
plt.xlabel("Time (10-min intervals)")
plt.ylabel("Bytes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Evaluate the forecast against a naive baseline ---
baseline = np.repeat(series.iloc[-1], 48)
rmse = np.sqrt(mean_squared_error(baseline, forecast))
mae = mean_absolute_error(baseline, forecast)
mape = mean_absolute_percentage_error(baseline, forecast) * 100

print(f"\n📊 LSTM Forecast Performance:")
print(f" - RMSE : {rmse:,.2f}")
print(f" - MAE  : {mae:,.2f}")
print(f" - MAPE : {mape:.2f}%")
