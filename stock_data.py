import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the stock symbol and time period
stock_symbol = 'AAPL'  # Change this to any stock symbol (e.g., 'TSLA', 'GOOGL')
start_date = '2020-01-01'
end_date = '2024-01-01'

# Fetch stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Display the first few rows
print(stock_data.head())
