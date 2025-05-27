
import pandas as np
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/full_dataset_modified_only.csv", parse_dates=["time"], index_col="time")
data = df["down"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

seq_len = 96
X, y = create_sequences(data_scaled, seq_len)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_len, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Forecast future values
last_seq = X_test[-1:]
forecast_scaled = []
for _ in range(48):
    pred = model.predict(last_seq)[0][0]
    forecast_scaled.append(pred)
    last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

# Plot forecast
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=30), periods=48, freq='30min')
plt.figure(figsize=(12, 5))
plt.plot(df.index[-96:], df["down"].values[-96:], label="Historical")
plt.plot(forecast_index, forecast, label="LSTM Forecast", linestyle='--')
plt.title("LSTM Forecast")
plt.xlabel("Time")
plt.ylabel("Downlink (bits)")
plt.legend()
plt.tight_layout()
plt.show()
