import os
import pandas as pd

# Path to 10-minute IP data
BASE_PATH = r"F:/assign/python practice/Bachelor-thesis-/third version"
DATA_DIR = os.path.join(BASE_PATH, "ip_addresses_sample", "agg_10_minutes")
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs", "processed_data.csv")

# Select and combine 10 IPs
FILES = os.listdir(DATA_DIR)[:10]

def preprocess():
    dfs = []
    for f in FILES:
        path = os.path.join(DATA_DIR, f)
        df = pd.read_csv(path)[["id_time", "n_bytes"]]
        df = df.rename(columns={"n_bytes": f"n_bytes_{f.split('.')[0]}"})
        dfs.append(df)

    agg_df = dfs[0]
    for df in dfs[1:]:
        agg_df = pd.merge(agg_df, df, on="id_time", how="outer")

    agg_df.fillna(0, inplace=True)
    agg_df["n_bytes_sum"] = agg_df.drop(columns=["id_time"]).sum(axis=1)
    agg_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Processed data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
