import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load ARIMA residuals
residuals = pd.read_csv("arima_residuals.csv", index_col=0).squeeze("columns")

# Define the sequence length (e.g., 48 time steps)
SEQ_LEN = 48

# Scale residuals
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# Create sequences
X, y = [], []
for i in range(len(residuals_scaled) - SEQ_LEN):
    X.append(residuals_scaled[i:i + SEQ_LEN])
    y.append(residuals_scaled[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Save everything to an .npz file for later use in LSTM training
np.savez("lstm_data.npz",
         X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test,
         scaler_min=scaler.data_min_, scaler_max=scaler.data_max_)
