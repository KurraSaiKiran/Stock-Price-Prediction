import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from stock_data import stock_data  # Import stock data from stock_data.py
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 3: Data Preprocessing
stock_data.dropna(inplace=True)  # Remove missing values
stock_prices = stock_data[['Close']]  # Extract Close prices

# Scale the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(stock_prices)
scaled_prices = pd.DataFrame(scaled_prices, index=stock_prices.index, columns=['Close'])

# Split Data into Training & Testing Sets
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

# Step 4: Convert Time-Series Data into Supervised Learning Format
N_DAYS = 60  # Use past 60 days for prediction

def create_features_labels(data, n_days):
    X, y = [], []
    for i in range(n_days, len(data)):
        X.append(data[i - n_days:i, 0])  # Past N days as features
        y.append(data[i, 0])  # Next day's stock price as label
    return np.array(X), np.array(y)

# Convert DataFrames to NumPy Arrays
train_array = train_data.values
test_array = test_data.values

# Create Features & Labels for Training & Testing
X_train, y_train = create_features_labels(train_array, N_DAYS)
X_test, y_test = create_features_labels(test_array, N_DAYS)

# Reshape Data for LSTM (LSTM needs 3D input: samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 5: Build the LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),  # First LSTM layer
    Dropout(0.2),  # Regularization to prevent overfitting
    
    LSTM(units=50, return_sequences=True),  # Second LSTM layer
    Dropout(0.2),

    LSTM(units=50),  # Third LSTM layer
    Dropout(0.2),

    Dense(units=1)  # Output layer (Predicting next stock price)
])

# Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display Model Summary
model.summary()

# Step 5.1: Train the Model
EPOCHS = 50  # Number of times the model sees the full dataset
BATCH_SIZE = 32  # Number of samples per training batch

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Step 5.2: Save the Trained Model
model.save("stock_price_model.h5")
print("Model saved successfully!")

# Step 6.1: Load the Trained Model
model = load_model("stock_price_model.h5")
print("Model loaded successfully!")

# Step 6.2: Make Predictions on Test Data
predicted_prices = model.predict(X_test)

# Inverse Transform Predictions (Convert Back to Original Price Scale)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Inverse Transform Actual Prices (Convert Test Data Back to Original Scale)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6.3: Plot Actual vs. Predicted Prices
plt.figure(figsize=(10, 5))
plt.plot(actual_prices, color="blue", label="Actual Stock Price")
plt.plot(predicted_prices, color="red", label="Predicted Stock Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# Step 6.4: Compute RMSE (Root Mean Squared Error)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 7: Predict Future Stock Prices
# Step 7.1: Prepare Input Data for Future Prediction
last_60_days = scaled_prices[-60:].values  # Get the last 60 days of stock prices
last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))  # Reshape for LSTM model

# Step 7.2: Predict the Next Day's Stock Price
future_price = model.predict(last_60_days)

# Convert Back to Original Scale
future_price = scaler.inverse_transform(future_price)

print(f"Predicted Stock Price for the Next Day: {future_price[0][0]}")

# Step 7.3: Predict the Next 7 Days of Stock Prices
future_predictions = []

input_data = last_60_days.copy()

for _ in range(7):  # Predict for 7 days
    next_price = model.predict(input_data)  # Predict next day's price
    future_predictions.append(next_price[0][0])  # Store the prediction
    
    # Update input data (Remove first value, add predicted price)
    next_input = np.append(input_data[:, 1:, :], [[next_price]], axis=1)
    input_data = np.reshape(next_input, (1, 60, 1))

# Convert Back to Original Scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

print("Predicted Stock Prices for the Next 7 Days:")
print(future_predictions)

# Step 7.4: Plot Future Stock Price Predictions
plt.figure(figsize=(8, 4))
plt.plot(future_predictions, color="green", marker="o", linestyle="dashed", label="Predicted Future Prices")
plt.title("Predicted Stock Prices for Next 7 Days")
plt.xlabel("Day")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
