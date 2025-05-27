import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time
import warnings

# === Settings ===
warnings.filterwarnings("ignore")
path = "F:/assign/python practice/Bachelor-thesis-/fourth_version/cleaned_enhanced_downlink.csv"

# === Load dataset ===
df = pd.read_csv(path, index_col=0, parse_dates=True)
y = df.squeeze()  # Convert DataFrame to Series

# === Measure training time ===
start_train = time.time()
arima_order = (2, 1, 3)
model = ARIMA(y, order=arima_order)
results = model.fit()
end_train = time.time()
train_time = end_train - start_train

# === Measure inference time ===
start_pred = time.time()
predictions = results.predict(start=1, end=len(y)-1)
end_pred = time.time()
inference_time = end_pred - start_pred

# === Compute residuals ===
residuals = y[1:] - predictions

# === Save residuals and predictions ===
residuals.to_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/arima_residuals.csv")
predictions.to_csv("F:/assign/python practice/Bachelor-thesis-/fourth_version/arima_predicted.csv")
print("✅ Residuals and predictions saved.")

# === Evaluation Metrics ===
def evaluate_model(y_true, y_pred, residuals, label="ARIMA"):
    mae = mean_absolute_error(y_true[1:], y_pred)
    rmse = np.sqrt(mean_squared_error(y_true[1:], y_pred))
    mape = mean_absolute_percentage_error(y_true[1:], y_pred) * 100
    lb_pvalue = acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].values[0]
    shapiro_p = shapiro(residuals)[1]

    print(f"\n📊 {label} Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    print(f"⏱ Training Time: {train_time:.2f} seconds")
    print(f"⏱ Inference Time: {inference_time:.2f} seconds")

    print(f"Ljung-Box p-value: {lb_pvalue:.4f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")

    # Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(residuals, bins=40, edgecolor='black')
    plt.title(f"{label} Residuals Histogram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Plot Actual vs Predicted ===
plt.figure(figsize=(12, 5))
plt.plot(y, label='Actual', alpha=0.6)
plt.plot(predictions, label='ARIMA Predicted', linestyle='--')
plt.title("ARIMA(2,1,3) Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Residuals ===
plt.figure(figsize=(12, 4))
plt.plot(residuals, color='red', label='Residuals')
plt.title("ARIMA Residuals (To Feed LSTM)")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Run Evaluation ===
evaluate_model(y, predictions, residuals, label="ARIMA")
