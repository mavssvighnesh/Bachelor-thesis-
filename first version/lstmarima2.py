import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import time
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("./full_dataset.csv")

# Focus on 'down' as target variable (replaceable as needed)
series = df['down'].values.reshape(-1,1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

# Fit ARIMA(1,1,5)
arima_model = ARIMA(series_scaled.flatten(), order=(1,1,5))
arima_result = arima_model.fit()

# ARIMA fitted values and residuals
arima_forecast = arima_result.fittedvalues  # length N-1
actual_series = series_scaled.flatten()[1:]  # length N-1

# Align to same length
min_len = min(len(arima_forecast), len(actual_series))
residuals = actual_series[:min_len] - arima_forecast[:min_len]

# Prepare LSTM dataset
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 5
trainX, trainY = create_dataset(residuals, look_back)

# Reshape for LSTM [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(Input(shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train LSTM with timing
start_time = time.time()
history = model.fit(trainX, trainY, epochs=15, batch_size=64, verbose=1)
end_time = time.time()

# LSTM predictions
lstm_forecast = model.predict(trainX).flatten()

# Hybrid forecast: ARIMA + LSTM (aligned)
min_len2 = min(len(arima_forecast[look_back:]), len(lstm_forecast))
hybrid_forecast = arima_forecast[look_back:look_back+min_len2] + lstm_forecast[:min_len2]

# Actual values for comparison
actual_values = series_scaled.flatten()[look_back+1:look_back+1+min_len2]

# Inverse scaling to original
hybrid_forecast_inv = scaler.inverse_transform(hybrid_forecast.reshape(-1,1)).flatten()
actual_values_inv = scaler.inverse_transform(actual_values.reshape(-1,1)).flatten()

# Evaluation metrics
mse = mean_squared_error(actual_values_inv, hybrid_forecast_inv)
mae = mean_absolute_error(actual_values_inv, hybrid_forecast_inv)
r2 = r2_score(actual_values_inv, hybrid_forecast_inv)
training_time = end_time - start_time

# Print results
print(f"\nHybrid ARIMA+LSTM Results:")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

# Plot result
plt.figure(figsize=(14,6))
plt.plot(actual_values_inv, label='Actual')
plt.plot(hybrid_forecast_inv, label='Hybrid Forecast')
plt.title("Hybrid ARIMA+LSTM Network Traffic Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Down Traffic")
plt.legend()
plt.show()