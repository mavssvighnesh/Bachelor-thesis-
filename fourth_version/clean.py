
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# === Step 1: Load original dataset ===
df = pd.read_csv('F:/assign/python practice/Bachelor-thesis-/fourth_version/full_dataset.csv')

# === Step 2: Convert 'time' to datetime and set as index ===
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df.sort_index(inplace=True)

# === Step 3: Interpolate missing values in all numeric columns ===
df.interpolate(method='linear', inplace=True)

# === Step 4: Resample a key feature ('down') to 2-minute intervals ===
y = df['down'].resample('2T').mean().interpolate()

# === Step 5: Add linearity to help ARIMA ===
time_numeric = np.arange(len(y)).reshape(-1, 1)
model = LinearRegression()
model.fit(time_numeric, y.values)
linear_trend = model.predict(time_numeric)

# Add a gentle linear component
y_linearized = y + 0.00001 * linear_trend

# === Step 6: Plot the enhanced series (optional) ===
plt.figure(figsize=(12, 5))
plt.plot(y_linearized, label="Enhanced Downlink Traffic with Linearity", color='orange')
plt.title("Enhanced Network Traffic (with linearity)")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Step 7: Save the cleaned and enhanced dataset locally ===
y_linearized.to_csv("cleaned_enhanced_downlink.csv")
print("Saved cleaned dataset as 'cleaned_enhanced_downlink.csv'")
