
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the trained model
model = load_model("lstm_model.keras")

# Load data
data = np.load("lstm_data.npz")
X_test = data['X_test']
y_test = data['y_test']
scaler_min = data['scaler_min']
scaler_max = data['scaler_max']

# Predict
y_pred = model.predict(X_test)
y_pred_rescaled = y_pred * (scaler_max - scaler_min) + scaler_min
y_test_rescaled = y_test * (scaler_max - scaler_min) + scaler_min

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled[:300], label="Actual", linewidth=2)
plt.plot(y_pred_rescaled[:300], label="Predicted", linewidth=2)
plt.title("LSTM Residual Forecasting: Actual vs Predicted")
plt.xlabel("Time Steps")
plt.ylabel("Residual Value (bits)")
plt.legend()
plt.tight_layout()
plt.show()
