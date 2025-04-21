# EDA for Network Traffic Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load the dataset
df = pd.read_csv('./full_dataset.csv')

# Display first 5 rows
print("🔍 First 5 records:")
print(df.head())

# Dataset shape
print(f"\n📏 Dataset shape: {df.shape}")

# Columns overview
print(f"\n📝 Columns:\n{df.columns.tolist()}")

# Check for missing values
print("\n🕳️ Missing values per column:")
print(df.isnull().sum())

# Visualize missing values heatmap
msno.matrix(df)
plt.title('Missing Values Visualization')
plt.show()

# Basic descriptive statistics
print("\n📊 Descriptive statistics:")
print(df.describe())


print('vighnesh')
# Plot distribution of each feature
df.hist(bins=30, figsize=(16, 12), color='skyblue', edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=18)
plt.tight_layout()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()

# Time series plots for key features
key_features = ['RB_d1', 'TB_d1', 'RB_u1', 'TB_u1', 'MCS_d1', 'MCS_u1']

plt.figure(figsize=(14, 8))
for i, feature in enumerate(key_features, 1):
    plt.subplot(3, 2, i)
    plt.plot(df[feature], color='navy')
    plt.title(f'{feature} over Time')
    plt.xlabel('Time Index')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
