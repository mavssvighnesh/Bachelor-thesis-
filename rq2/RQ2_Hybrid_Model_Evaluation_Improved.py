
# RQ2 Improved Evaluation: Hybrid ARIMA-LSTM Model

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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

for name, path in dataset_paths.items():
    df = pd.read_csv(path, index_col='time', parse_dates=True)
    df = df[['down']].resample('30min').mean().interpolate()

    # Scale full down series
    full_scaler = MinMaxScaler()
    series_scaled = full_scaler.fit_transform(df[['down']]).flatten()
    series = df['down']

    # Auto ARIMA
    try:
        stepwise_model = auto_arima(series, seasonal=False, suppress_warnings=True)
        order = stepwise_model.order
        arima_model = stepwise_model.fit(series)
        arima_forecast = arima_model.predict(n_periods=48)
        fitted_values = arima_model.predict_in_sample()
        residuals = series - fitted_values
    except Exception as e:
        results.append({"dataset": name, "error": str(e)})
        continue

    # LSTM part
    residual_scaler = MinMaxScaler()
    residuals_scaled = residual_scaler.fit_transform(residuals.dropna().values.reshape(-1, 1)).flatten()
    X, y = create_lstm_dataset(residuals_scaled)
    if len(X) == 0:
        results.append({"dataset": name, "error": "Not enough data for LSTM"})
        continue

    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(-1, 1)

    model = Sequential()
    model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    start_time = time.time()
    model.fit(X, y, epochs=30, batch_size=32, verbose=0, callbacks=[es])
    training_time = time.time() - start_time

    # Forecast residuals
    current_seq = X[-1:]
    lstm_forecast_scaled = []
    for _ in range(48):
        pred = model.predict(current_seq, verbose=0)[0][0]
        next_input = np.append(current_seq[0, 1:, 0], pred).reshape(1, X.shape[1], 1)
        current_seq = next_input
        lstm_forecast_scaled.append(pred)

    lstm_forecast_residuals = residual_scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()
    hybrid_forecast = arima_forecast + lstm_forecast_residuals
    true_values = series[-48:].values
    rmse = mean_squared_error(true_values, hybrid_forecast, squared=False)
    mae = mean_absolute_error(true_values, hybrid_forecast)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label="True", linewidth=2)
    plt.plot(arima_forecast, '--', label="ARIMA")
    plt.plot(hybrid_forecast, label="Hybrid", color="green")
    plt.title(f"Forecast Comparison - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"forecast_plot_{name}.png")
    plt.close()

    results.append({
        "dataset": name,
        "ARIMA Order": order,
        "RMSE": rmse,
        "MAE": mae,
        "Training Time (s)": round(training_time, 2)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("rq2_hybrid_evaluation_results.csv", index=False)
print(results_df)
