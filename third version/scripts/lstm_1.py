import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- CONFIG ---
BASE = r"F:\assign\python practice\Bachelor-thesis-\third version"
DATA_PATH = os.path.join(BASE, "outputs", "processed_data.csv")
PLOT_DIR = os.path.join(BASE, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Load data and generate ARIMA residuals (assumed available or recalculated here) ---
df = pd.read_csv(DATA_PATH)
from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(df["n_bytes_sum"], order=(2, 0, 3)).fit()
df["arima_fitted"] = arima_model.fittedvalues
df["residuals"] = df["n_bytes_sum"] - df["arima_fitted"]

# --- Normalize residuals ---
scaler = MinMaxScaler()
res_scaled = scaler.fit_transform(df["residuals"].values.reshape(-1, 1))

# --- Create LSTM sequences ---
SEQ_LEN = 24
X, y = [], []
for i in range(SEQ_LEN, len(res_scaled)):
    X.append(res_scaled[i-SEQ_LEN:i])
    y.append(res_scaled[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Build and Train LSTM ---
model = Sequential([
    LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# --- Forecast residuals (next 48 steps) ---
last_seq = X[-1:]
preds_scaled = []
for _ in range(48):
    pred = model.predict(last_seq, verbose=0)[0][0]
    preds_scaled.append(pred)
    new_input = np.append(last_seq[0, 1:, 0], pred).reshape((1, SEQ_LEN, 1))
    last_seq = new_input

# --- Inverse transform ---
preds_residuals = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# --- Save LSTM forecasted residuals ---
np.save(os.path.join(BASE, "outputs", "lstm_predicted_residuals.npy"), preds_residuals)
print("✅ LSTM forecasted residuals saved.")
