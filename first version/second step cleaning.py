# 📦 Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import skew

# 📄 Load the dataset
df = pd.read_csv('./full_dataset.csv')  # change path if needed

# Convert time column to datetime if not already
df['time'] = pd.to_datetime(df['time'])

# 📊 1️⃣ Check Missing Values
print("\n🕳️ Missing values per column:")
print(df.isnull().sum())

msno.bar(df)
plt.title('Missing Values per Column')
plt.show()

# If you want to check missing values over time:
missing_counts_over_time = df.set_index('time').isnull().sum(axis=1)

plt.figure(figsize=(16, 5))
sns.lineplot(x=missing_counts_over_time.index, y=missing_counts_over_time.values)
plt.title('Missing Value Count Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Missing Values')
plt.grid(True)
plt.show()
# 📊 2️⃣ Impute Missing Values
# Impute numerical columns with median (robust to outliers)
num_cols = ['mcs_down', 'mcs_down_var', 'mcs_up', 'mcs_up_var', 
            'rb_down', 'rb_down_var', 'rb_up', 'rb_up_var']

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\n✅ Missing values after imputation:")
print(df.isnull().sum())

# 📊 3️⃣ Outlier Detection with Boxplots
plt.figure(figsize=(16, 8))
sns.boxplot(data=df[['down', 'up', 'mcs_down', 'mcs_up']])
plt.title('Boxplot of Key Features')
plt.show()

# 📊 4️⃣ Feature Distribution Histograms
df.hist(figsize=(18, 14), bins=50, color='skyblue', edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=22)
plt.show()

# 📊 5️⃣ Time-Series Trend of Traffic
plt.figure(figsize=(16, 6))
plt.plot(df['time'], df['down'], label='Downlink Traffic', alpha=0.7)
plt.plot(df['time'], df['up'], label='Uplink Traffic', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Traffic')
plt.title('Traffic Over Time')
plt.legend()
plt.grid(True)
plt.show()

# 📊 6️⃣ District-wise Traffic Analysis
plt.figure(figsize=(14, 6))
sns.boxplot(x='District', y='down', data=df)
plt.title('Downlink Traffic Distribution by District')
plt.xticks(rotation=45)
plt.show()

# 📊 7️⃣ Skewness Check and Log-Transform Skewed Columns
skewed_cols = ['down', 'up', 'rb_down', 'rb_up']

for col in skewed_cols:
    print(f"{col} skewness before: {skew(df[col]):.2f}")

    # Apply log1p (log(1+x)) transform
    df[col] = np.log1p(df[col])

    print(f"{col} skewness after: {skew(df[col]):.2f}")

# 📊 8️⃣ Replot Distributions After Log-Transform
df[skewed_cols].hist(figsize=(12, 8), bins=50, color='lightgreen', edgecolor='black')
plt.suptitle('Distributions After Log-Transform', fontsize=18)
plt.show()
