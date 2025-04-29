
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load ARIMA residuals
residuals = pd.read_csv("arima_residuals.csv", index_col=0, parse_dates=True).squeeze("columns")


# Extend to 96 steps
SEQ_LEN = 96

# Load cleaned dataset and ensure datetime index
df = pd.read_csv("cleaned_dataset.csv", index_col='time')
df.index = pd.to_datetime(df.index)

# Align dataset to residuals index and extract hour
df = df.reindex(residuals.index)
df.index = pd.to_datetime(df.index)  # re-ensure datetime index
df['hour'] = df.index.map(lambda x: x.hour)

# Drop NaNs after alignment
df.dropna(inplace=True)
residuals = residuals.loc[df.index]

# Scale hour
hour_scaler = MinMaxScaler()
hour_scaled = hour_scaler.fit_transform(df['hour'].values.reshape(-1, 1))

# Scale residuals
res_scaler = MinMaxScaler()
res_scaled = res_scaler.fit_transform(residuals.values.reshape(-1, 1))

# Combine residual and hour
X_all = np.hstack((res_scaled, hour_scaled))

# Create sequences
X, y = [], []
for i in range(len(X_all) - SEQ_LEN):
    X.append(X_all[i:i + SEQ_LEN])
    y.append(X_all[i + SEQ_LEN, 0])

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Save everything
np.savez("lstm_data_v2.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
         res_min=res_scaler.data_min_, res_max=res_scaler.data_max_)
