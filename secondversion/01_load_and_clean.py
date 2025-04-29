
import pandas as pd

# Load dataset
df = pd.read_csv("F:/assign/python practice/Bachelor-thesis-/secondversion/full_dataset.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Impute missing values
df = df.fillna(method='ffill').fillna(method='bfill')
df.to_csv("cleaned_dataset.csv")
