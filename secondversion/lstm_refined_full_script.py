
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === PREPROCESSING ===

SEQ_LEN = 96

# Load residuals
residuals = pd.read_csv("arima_residuals_refined.csv", index_col=0, parse_dates=True).squeeze("columns")

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv", index_col='time')
df.index = pd.to_datetime(df.index)
df = df.reindex(residuals.index)

# Time and network features
df['hour'] = df.index.map(lambda x: x.hour)
df['day_of_week'] = df.index.map(lambda x: x.dayofweek)
df['rnti_count'] = df['rnti_count'].fillna(method='ffill').fillna(method='bfill')

# Drop NaNs and realign residuals
df.dropna(inplace=True)
residuals = residuals.loc[df.index]

# Scale features
scalers = {}
scaled_features = []

for col in ['rnti_count', 'hour', 'day_of_week']:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[col].values.reshape(-1, 1))
    scaled_features.append(scaled)
    scalers[col] = (scaler.data_min_, scaler.data_max_)

# Scale residuals
res_scaler = MinMaxScaler()
res_scaled = res_scaler.fit_transform(residuals.values.reshape(-1, 1))
scalers['residuals'] = (res_scaler.data_min_, res_scaler.data_max_)

# Stack features
X_all = np.hstack([res_scaled] + scaled_features)

# Sequence creation
X, y = [], []
for i in range(len(X_all) - SEQ_LEN):
    X.append(X_all[i:i + SEQ_LEN])
    y.append(X_all[i + SEQ_LEN, 0])

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === TRAINING ===

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=60,
                    batch_size=64,
                    callbacks=[early_stop],
                    verbose=1)

# === EVALUATION ===

y_pred = model.predict(X_test)
res_min, res_max = scalers['residuals']
y_pred_rescaled = y_pred * (res_max - res_min) + res_min
y_test_rescaled = y_test * (res_max - res_min) + res_min

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"Final Refined LSTM - MAE: {mae:.2f}")
print(f"Final Refined LSTM - RMSE: {rmse:.2f}")

# Save model
model.save("lstm_model_refined_full.keras")
