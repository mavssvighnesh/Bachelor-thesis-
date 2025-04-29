import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("cleaned_dataset.csv", index_col="time", parse_dates=True)
series = df["down"]

# ARIMA model
arima_model = ARIMA(series, order=(1, 1, 5))
arima_result = arima_model.fit()
residuals = series - arima_result.fittedvalues.shift(1)
residuals = residuals.dropna()

# Normalize residuals
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length=48):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(residuals_scaled, 48)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define improved LSTM model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(48, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

# Save model and scaler
model.save("lstm_model_v3.keras")
np.save("residual_scaler_v3.npy", {"min": scaler.data_min_, "max": scaler.data_max_})
