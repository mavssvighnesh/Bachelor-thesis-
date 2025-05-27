
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/secondversion/full_dataset.csv", parse_dates=["time"], index_col="time")

# Backup the original series
original_series = df["down"].copy()

# Add a gentle linear trend
time_index = np.arange(len(original_series))
trend = 0.0005 * time_index  # Adjust coefficient as needed
df["down_linear"] = original_series + trend

# Apply light smoothing using Savitzky-Golay filter
df["down_modified"] = savgol_filter(df["down_linear"], window_length=11, polyorder=2)

# Plot the original and modified series
plt.figure(figsize=(15, 5))
plt.plot(df.index, original_series, label="Original", alpha=0.5)
plt.plot(df.index, df["down_modified"], label="Modified (Linear + Smoothing)", linewidth=2)
plt.title("Original vs Modified 'Down' Series")
plt.xlabel("Time")
plt.ylabel("Downlink Traffic")
plt.legend()
plt.tight_layout()
plt.show()

# Save three versions of the dataset
df[["down"]].to_csv("full_dataset_original_only.csv")  # Original only
df[["down_modified"]].rename(columns={"down_modified": "down"}).to_csv("full_dataset_modified_only.csv")  # Modified only
df[["down", "down_modified"]].to_csv("full_dataset_combined.csv")  # Both versions
