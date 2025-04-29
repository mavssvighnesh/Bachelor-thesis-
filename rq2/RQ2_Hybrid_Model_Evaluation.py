
# RQ2 Evaluation: Hybrid ARIMA-LSTM Model with Imperfect Data

import os
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Dataset paths (replace with actual paths if different)
dataset_paths = {
    "missing_5": "cleaned_dataset_missing_5.csv",
    "missing_10": "cleaned_dataset_missing_10.csv",
    "missing_20": "cleaned_dataset_missing_20.csv",
    "irregular_5": "cleaned_dataset_irregular_5.csv",
    "irregular_10": "cleaned_dataset_irregular_10.csv",
    "fluctuations": "cleaned_dataset_fluctuations.csv"
}

results = []

def create_lstm_dataset(series, look_back=96):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

# Main processing loop
for name, path in dataset_paths.items():
    df = pd.read_csv(path, index_col='time', parse_dates=True)

    # Resample and interpolate
    df = df[['down']].resample('30min').mean().interpolate()


    # Scale
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(df[['down']]).flatten()

    # ARIMA
    series = df['down']
    try:
        arima_model = ARIMA(series, order=(2, 0, 5)).fit()
        arima_forecast = arima_model.forecast(steps=48)
        residuals = series - arima_model.predict(start=0, end=len(series)-1)
    except Exception as e:
        results.append({"dataset": name, "error": str(e)})
        continue

    residuals_scaled = scaler.fit_transform(residuals.dropna().values.reshape(-1, 1)).flatten()
    X, y = create_lstm_dataset(residuals_scaled)

    if len(X) == 0:
        results.append({"dataset": name, "error": "Insufficient data for LSTM"})
        continue

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    start_time = time.time()
    model.fit(X, y, epochs=5, verbose=0)
    training_time = time.time() - start_time

    # Forecast residuals with LSTM
    current_seq = X[-1:]
    forecast_residuals_scaled = []
    for _ in range(48):
        pred = model.predict(current_seq, verbose=0)[0][0]
        next_input = np.append(current_seq[0, 1:, 0], pred).reshape(1, X.shape[1], 1)
        current_seq = next_input
        forecast_residuals_scaled.append(pred)

    forecast_residuals_scaled = np.array(forecast_residuals_scaled)
    lstm_forecast_residuals = scaler.inverse_transform(forecast_residuals_scaled.reshape(-1, 1)).flatten()

    hybrid_forecast = arima_forecast.values + lstm_forecast_residuals

    true_values = series[-48:].values
    rmse = mean_squared_error(true_values, hybrid_forecast, squared=False)
    mae = mean_absolute_error(true_values, hybrid_forecast)

    results.append({
        "dataset": name,
        "RMSE": rmse,
        "MAE": mae,
        "Training Time (s)": round(training_time, 2)
    })

# Show Results
results_df = pd.DataFrame(results)
print(results_df)
