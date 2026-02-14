import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 1. DATA COLLECTION
# ===============================
# Download stock data (Example: Apple)
df = yf.download("AAPL", start="2018-01-01", end="2023-12-31")
df = df[['Close']]  # Use only close price

# ===============================
# 2. FEATURE ENGINEERING
# ===============================

# Simple Moving Average
df['SMA'] = df['Close'].rolling(window=20).mean()

# Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close'], 14)

# Drop NaN values created by indicators
df = df.dropna()

# ===============================
# 3. SCALING DATA
# ===============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close', 'SMA', 'RSI']])

# Features and target
X = scaled_data[:-1]        # Features (all but last row)
y = scaled_data[1:, 0]      # Target (next day's close price)

# ===============================
# 4. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ===============================
# 5. TRAINING THE MODEL
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# 6. EVALUATION
# ===============================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(" Model Evaluation:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Root Mean Squared Error: {rmse:.6f}")
print(f"R-squared: {r2:.6f}")

# ===============================
# 7. VISUALIZATION
# ===============================
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test, label="Actual")
plt.plot(df.index[-len(y_test):], y_pred, label="Predicted")
plt.title("Stock Price Prediction (Normalized)")
plt.xlabel("Date")
plt.ylabel("Scaled Price")
plt.legend()
plt.show()
