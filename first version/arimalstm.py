import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Load your time series
series = pd.read_csv('./full_dataset.csv')  # replace with your actual data source
series = series['down']  # replace with your actual column name

# ARIMA model
arima_order = (1, 1, 5)
arima_model = ARIMA(series, order=arima_order)
arima_result = arima_model.fit()

# ARIMA Forecast
arima_forecast = arima_result.predict(start=0, end=len(series)-1, typ='levels')

# Residuals (errors)
residuals = series.values - arima_forecast.values

# Scale residuals for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_residuals = scaler.fit_transform(residuals.reshape(-1, 1))

# Create sequences
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_residuals, look_back)

# Train-Test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# LSTM model for residuals
lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, input_shape=(look_back, 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(64))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

# Compile
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit LSTM model
history = lstm_model.fit(X_train, y_train, epochs=100, batch_size=64,
                         validation_split=0.2, callbacks=[early_stop], verbose=1)

# Predict residuals
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Final hybrid forecast = ARIMA forecast + LSTM residual forecast
hybrid_forecast = arima_forecast.values[look_back + train_size:] + lstm_predictions.flatten()

# Actual values for comparison
actual_values = series.values[look_back + train_size:]

# Plot results
plt.figure(figsize=(14,6))
plt.plot(actual_values, label='Actual')
plt.plot(hybrid_forecast, label='Hybrid Forecast', color='orange')
plt.title('Hybrid ARIMA + LSTM Forecast vs Actual')
plt.legend()
plt.show()

# Loss plot (optional)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Loss Curve for Residual Modeling')
plt.legend()
plt.show()
