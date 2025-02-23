import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Price Prediction")

# Upload trained model
uploaded_model = st.file_uploader("Upload trained model (.h5)", type=["h5"])

# Upload dataset
uploaded_dataset = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_model is not None and uploaded_dataset is not None:
    # Save the uploaded model file
    with open("stock_price_model.h5", "wb") as f:
        f.write(uploaded_model.read())

    # Load the trained model
    model = load_model("stock_price_model.h5")
    st.success("✅ Model loaded successfully!")

    # Load dataset from uploaded file
    df = pd.read_csv(uploaded_dataset)

    st.write("📊 **Dataset Preview:**")
    st.dataframe(df.head())

    # Plot the stock price history
    st.subheader("📉 Stock Price History")
    fig = px.line(df, x=df.index, y='Close', title="Stock Price Over Time")
    st.plotly_chart(fig)

    # Process data
    stock_prices = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(stock_prices)

    # Prepare last 60 days for prediction
    last_60_days = scaled_prices[-60:].reshape(1, 60, 1)

    # Predict stock price
    prediction = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    st.subheader("📉 Predicted Stock Price:")
    st.write(f"💰 {predicted_price:.2f} USD")

    # 📌 Visualize Prediction
    st.subheader("📊 Prediction vs Actual Prices")

    # Append prediction to dataset for visualization
    df['Predicted'] = np.nan  # Create a new column for predicted values
    df.loc[len(df)] = [None] * (len(df.columns) - 1) + [predicted_price]  # Add prediction at the end

    # Plot actual and predicted prices
    fig2 = px.line(df, x=df.index, y=['Close', 'Predicted'], title="Actual vs Predicted Price")
    fig2.update_traces(mode='lines+markers')  # Show points on graph
    st.plotly_chart(fig2)
