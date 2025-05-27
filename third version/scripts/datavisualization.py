import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
BASE_PATH = r"F:/assign/python practice/Bachelor-thesis-/third version"
DATA_PATH = os.path.join(BASE_PATH, "outputs", "processed_data.csv")
PLOT_DIR = os.path.join(BASE_PATH, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_series():
    df = pd.read_csv(DATA_PATH)

    # Line plot of total bytes
    plt.figure(figsize=(14, 6))
    plt.plot(df["id_time"], df["n_bytes_sum"], label="Total Traffic")
    plt.title("Aggregated Network Traffic (n_bytes)")
    plt.xlabel("Time (10-min intervals)")
    plt.ylabel("Total Bytes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "traffic_over_time.png"))
    plt.close()
    print("✅ Time series plot saved.")

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr = df.drop(columns=["id_time"]).corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation between IPs (n_bytes)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"))
    plt.close()
    print("✅ Correlation heatmap saved.")

if __name__ == "__main__":
    plot_series()
