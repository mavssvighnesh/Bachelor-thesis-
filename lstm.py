from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib as plt 

# Scale data
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.values.reshape(-1,1))

# Train-test split
train_size = int(len(scaled_series) * 0.8)
train, test = scaled_series[0:train_size], scaled_series[train_size:]

# Prepare sequences
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit model
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

# Predict
lstm_predictions = model.predict(X_test)

# Inverse scaling
lstm_predictions = scaler.inverse_transform(lstm_predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Plot LSTM results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual')
plt.plot(lstm_predictions, label='Predicted')
plt.title('LSTM Forecast vs Actual')
plt.legend()
plt.show()
