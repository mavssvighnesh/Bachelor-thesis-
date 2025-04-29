import pandas as pd
import numpy as np
import time
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load your dataset
data = pd.read_csv('./full_dataset.csv')
series = data['down'].astype(float)

# Log transform to stabilize variance
series_log = np.log1p(series)

# Check stationarity and apply differencing
def adf_test(series):
    result = adfuller(series)
    return result[1]

diff_count = 0
p_value = adf_test(series_log)
while p_value > 0.05:
    series_log = series_log.diff().dropna()
    p_value = adf_test(series_log)
    diff_count += 1

# Split data
train_size = int(len(series_log) * 0.8)
train, test = series_log[:train_size], series_log[train_size:]

# ARIMA model
arima_order = (1,1,5)
start_time = time.time()
arima_model = ARIMA(train, order=arima_order)
arima_model_fit = arima_model.fit()
arima_time = time.time() - start_time

# Forecast residuals
arima_forecast = arima_model_fit.predict(start=0, end=len(series_log)-1)
residuals = series_log.values - arima_forecast

# Scale residuals
scaler = MinMaxScaler(feature_range=(0, 1))
residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

# Prepare LSTM dataset
def create_dataset(dataset, look_back=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30
X, y = create_dataset(residuals_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1)
])

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

start_time = time.time()
history = model.fit(X, y, epochs=25, batch_size=64, verbose=1)
lstm_time = time.time() - start_time

# Forecast residuals with LSTM
lstm_forecast = model.predict(X)
lstm_forecast_inversed = scaler.inverse_transform(lstm_forecast)

# Combine ARIMA + LSTM
hybrid_forecast = arima_forecast[look_back:] + lstm_forecast_inversed.flatten()

# Inverse differencing
for _ in range(diff_count):
    hybrid_forecast = np.r_[series_log.values[look_back-1], hybrid_forecast].cumsum()

# Inverse log
final_forecast = np.expm1(hybrid_forecast)

# Actual values
true_values = series.values[look_back:]

# Metrics
mse = mean_squared_error(true_values, final_forecast)
mae = mean_absolute_error(true_values, final_forecast)
r2 = r2_score(true_values, final_forecast)
non_zero_mask = true_values != 0
mape = np.mean(np.abs((true_values[non_zero_mask] - final_forecast[non_zero_mask]) / true_values[non_zero_mask])) * 100
accuracy = 100 - mape
total_time = arima_time + lstm_time

# Results
print(f"\n📊 Hybrid ARIMA+LSTM Final Results:")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total Training Time: {total_time:.2f} seconds")

# Plot results
plt.figure(figsize=(12,5))
plt.plot(true_values, label='Actual', color='blue')
plt.plot(final_forecast, label='Hybrid Forecast', color='orange')
plt.title('Hybrid ARIMA+LSTM Forecast vs Actual')
plt.legend()
plt.tight_layout()
plt.show()
