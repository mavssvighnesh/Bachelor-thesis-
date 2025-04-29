
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_dataset.csv", index_col='time', parse_dates=True)

cols_to_plot = ['down', 'up', 'rnti_count', 'mcs_down', 'mcs_up']
plt.figure(figsize=(15, 10))
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(len(cols_to_plot), 1, i)
    plt.plot(df.index, df[col])
    plt.title(f'Time Series Plot of {col}')
plt.tight_layout()
plt.show()
