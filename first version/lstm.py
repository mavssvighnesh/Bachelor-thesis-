import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('full_dataset.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

series = df['down'].values.reshape(-1, 1)

# Scaling
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series)

# Train-test split
train_size = int(len(scaled_series) * 0.8)
train, test = scaled_series[0:train_size], scaled_series[train_size:]

# Create sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predict
predictions = model.predict(X_test)

# Inverse transform predictions
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Evaluate
mse = mean_squared_error(y_test_actual, predictions_actual)
print(f'LSTM Test MSE: {mse:.2f}')

# Plot
plt.figure(figsize=(14,6))
plt.plot(y_test_actual, label='Actual')
plt.plot(predictions_actual, label='Predicted')
plt.title('LSTM Forecast vs Actual')
plt.legend()
plt.show()
