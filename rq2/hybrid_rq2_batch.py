
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

datasets = {
    "missing_5": "F:/assign/python practice/Bachelor-thesis-/rq2/imputed_missing_5.csv",
    "missing_10": "F:/assign/python practice/Bachelor-thesis-/rq2/imputed_missing_10.csv",
    "missing_20": "F:/assign/python practice/Bachelor-thesis-/rq2/imputed_missing_20.csv",
    "irregular_resampled": "F:/assign/python practice/Bachelor-thesis-/rq2/resampled_irregular_intervals.csv",
    "fluctuated": "F:/assign/python practice/Bachelor-thesis-/rq2/with_fluctuations.csv"
}

def run_hybrid_model(name, path, seq_len=20):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = df['down'].tail(10000)

    start_train_arima = time.time()
    model_arima = ARIMA(series, order=(2, 1, 3))
    result_arima = model_arima.fit()
    end_train_arima = time.time()

    start_pred_arima = time.time()
    arima_pred = result_arima.predict(start=1, end=len(series)-1)
    end_pred_arima = time.time()

    residuals = series[1:] - arima_pred
    residuals = residuals.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    res_scaled = scaler.fit_transform(residuals)

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

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
    model.fit(X, y_lstm, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
    end_train_lstm = time.time()

    start_pred_lstm = time.time()
    res_pred_scaled = model.predict(X, verbose=0)
    end_pred_lstm = time.time()

    res_pred = scaler.inverse_transform(res_pred_scaled)
    arima_trimmed = arima_pred[seq_len:].values.reshape(-1, 1)
    final_pred = arima_trimmed + res_pred
    true_y = series.values[seq_len+1:].reshape(-1, 1)
    residuals_total = true_y - final_pred

    mae = mean_absolute_error(true_y, final_pred)
    rmse = np.sqrt(mean_squared_error(true_y, final_pred))
    mape = mean_absolute_percentage_error(true_y, final_pred) * 100
    lb_p = acorr_ljungbox(residuals_total, lags=[10], return_df=True)['lb_pvalue'].values[0]
    shapiro_p = shapiro(residuals_total)[1]
    train_time = (end_train_arima - start_train_arima) + (end_train_lstm - start_train_lstm)
    inference_time = (end_pred_arima - start_pred_arima) + (end_pred_lstm - start_pred_lstm)

    plt.figure(figsize=(12, 5))
    plt.plot(true_y, label='Actual Traffic')
    plt.plot(final_pred, label='Hybrid Forecast', linestyle='--')
    plt.title(f"Hybrid ARIMA + LSTM Forecast - {name}")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hybrid_forecast_{name}.png")
    plt.close()

    return {
        "Dataset": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "LjungBox_p": lb_p,
        "Shapiro_p": shapiro_p,
        "Training_Time_sec": train_time,
        "Inference_Time_sec": inference_time
    }

results = []
for name, path in datasets.items():
    print(f"Processing: {name}")
    metrics = run_hybrid_model(name, path)
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv("hybrid_model_evaluation_summary.csv", index=False)
