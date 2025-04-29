
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import shapiro

residuals = pd.read_csv("arima_residuals.csv", index_col=0).squeeze("columns")

# Time plot
plt.figure(figsize=(12, 4))
plt.plot(residuals)
plt.title("Residuals Over Time")
plt.show()

# Histogram
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

# ACF & PACF
plot_acf(residuals[-1000:], lags=40)
plt.title("ACF of Residuals")
plt.show()

plot_pacf(residuals[-1000:], lags=40)
plt.title("PACF of Residuals")
plt.show()

# Shapiro test
stat, p = shapiro(residuals.sample(500))
print("Shapiro-Wilk Test: W =", stat, ", p =", p)
