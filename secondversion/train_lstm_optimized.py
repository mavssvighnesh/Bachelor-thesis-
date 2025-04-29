
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
data = np.load("lstm_data_v2.npz")
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
res_min = data['res_min']
res_max = data['res_max']

# Build model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Predict and inverse scale
y_pred = model.predict(X_test)
y_pred_rescaled = y_pred * (res_max - res_min) + res_min
y_test_rescaled = y_test * (res_max - res_min) + res_min

# Evaluate
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save model
model.save("lstm_model_v2.keras")
