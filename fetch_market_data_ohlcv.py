"""
Fetch OHLCV data from Yahoo Finance for multi-stock backfilling model.

Downloads Open, High, Low, Close, Adj Close, and Volume data for AMZN, MSFT, and S&P 500.
Used to compute multi-feature time series (returns, volatility, volume change, intraday range).
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Configuration
TICKERS = ['AMZN', 'MSFT', '^GSPC']  # Amazon, Microsoft, S&P 500
START_DATE = '2000-01-01'
END_DATE = '2023-02-28'
OUTPUT_DIR = 'data'
OUTPUT_FILE = 'market_data_ohlcv.csv'

print("=" * 80)
print("FETCHING OHLCV DATA FROM YAHOO FINANCE")
print("=" * 80)
print(f"Tickers: {', '.join(TICKERS)}")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Data: Open, High, Low, Close, Adj Close, Volume")
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

    # Flatten multi-level columns to single level
    # yfinance returns: MultiIndex columns with (Price type, Ticker)
    # We want: Flat columns like 'AMZN_Open', 'AMZN_High', etc.

    if len(TICKERS) == 1:
        # Single ticker case - simpler structure
        data.columns = [f"{TICKERS[0]}_{col}" for col in data.columns]
    else:
        # Multiple tickers - need to flatten MultiIndex
        # Original: ('Open', 'AMZN'), ('Open', 'MSFT'), ...
        # Target: 'AMZN_Open', 'MSFT_Open', ...
        flattened_cols = []
        for col in data.columns:
            # col is tuple like ('Open', 'AMZN')
            price_type, ticker = col
            flattened_cols.append(f"{ticker}_{price_type}")
        data.columns = flattened_cols

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    data.to_csv(output_path)

    print()
    print("=" * 80)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Number of rows: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")
    print()

    # Check for missing values per ticker
    print("Missing data summary:")
    print("-" * 60)
    for ticker in TICKERS:
        ticker_cols = [col for col in data.columns if col.startswith(ticker)]
        missing_per_col = {}
        for col in ticker_cols:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_per_col[col] = missing_count

        if missing_per_col:
            print(f"{ticker}:")
            for col, count in missing_per_col.items():
                print(f"  {col}: {count} missing values")
        else:
            print(f"{ticker}: No missing values ✓")

    print()
    print("First few rows:")
    print(data.head())
    print()
    print("Last few rows:")
    print(data.tail())

except Exception as e:
    print(f"ERROR: Failed to download data: {e}")
    raise
