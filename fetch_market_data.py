"""
Fetch adjusted close prices from Yahoo Finance for multiple tickers.

Downloads historical data for indices, ETFs, and stocks from 2000-01-01 to 2023-02-28
and saves as a single CSV file with one column per ticker.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Configuration
TICKERS = ['^GSPC', 'GLD', 'SHY', 'TLT', 'VTI', '^VIX', 'AMZN', 'MSFT', 'GOOGL', 'V', 'MA']
START_DATE = '2000-01-01'
END_DATE = '2023-02-28'
OUTPUT_DIR = 'data'
OUTPUT_FILE = 'market_data_adj_close.csv'

print("=" * 80)
print("FETCHING MARKET DATA FROM YAHOO FINANCE")
print("=" * 80)
print(f"Tickers: {', '.join(TICKERS)}")
print(f"Date range: {START_DATE} to {END_DATE}")
print()

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch data
print("Downloading data from Yahoo Finance...")
try:
    data = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,  # Keep Adj Close as separate column
        progress=True
    )

    # Extract adjusted close prices
    if len(TICKERS) == 1:
        # Single ticker returns different structure
        adj_close = pd.DataFrame(data['Adj Close'])
        adj_close.columns = TICKERS
    else:
        # Multiple tickers
        adj_close = data['Adj Close']

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    adj_close.to_csv(output_path)

    print()
    print("=" * 80)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Date range: {adj_close.index[0]} to {adj_close.index[-1]}")
    print(f"Number of rows: {len(adj_close)}")
    print(f"Number of tickers: {len(adj_close.columns)}")
    print()
    print("Ticker summary:")
    print("-" * 40)
    for ticker in adj_close.columns:
        non_null = adj_close[ticker].notna().sum()
        null_count = adj_close[ticker].isna().sum()
        print(f"  {ticker:8s}: {non_null:5d} valid days, {null_count:4d} missing")
    print()
    print("First few rows:")
    print(adj_close.head())
    print()
    print("Last few rows:")
    print(adj_close.tail())

except Exception as e:
    print(f"ERROR: Failed to download data: {e}")
    raise
