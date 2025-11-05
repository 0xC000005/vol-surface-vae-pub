"""
Prepare stock return data for 1D VAE training.

Converts market_data_adj_close.csv to stock_returns.npz with:
- Amazon returns as target
- SP500/MSFT returns as conditioning features

Output:
    data/stock_returns.npz with keys:
        - amzn_returns: (N,) - Amazon log returns
        - sp500_returns: (N,) - SP500 log returns
        - msft_returns: (N,) - Microsoft log returns
        - dates: (N,) - Trading dates
        - cond_sp500: (N, 1) - SP500 for conditioning
        - cond_msft: (N, 1) - MSFT for conditioning
        - cond_both: (N, 2) - [SP500, MSFT] for conditioning
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = "data/market_data_adj_close.csv"
OUTPUT_FILE = "data/stock_returns.npz"

print("=" * 80)
print("PREPARING STOCK RETURN DATA FOR 1D VAE")
print("=" * 80)
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

# Load data
print("Loading market data...")
df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

print(f"  Loaded {len(df)} rows")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Columns: {list(df.columns)}")
print()

# Compute log returns
print("Computing log returns...")
amzn_prices = df["AMZN"]
sp500_prices = df["^GSPC"]
msft_prices = df["MSFT"]

amzn_returns = np.log(amzn_prices / amzn_prices.shift(1))
sp500_returns = np.log(sp500_prices / sp500_prices.shift(1))
msft_returns = np.log(msft_prices / msft_prices.shift(1))

# Create aligned dataframe (drop NaN from log returns)
returns_df = pd.DataFrame({
    "amzn": amzn_returns,
    "sp500": sp500_returns,
    "msft": msft_returns,
}).dropna()

print(f"  After computing returns and dropping NaN: {len(returns_df)} rows")
print(f"  Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
print()

# Extract arrays
amzn_ret = returns_df["amzn"].values
sp500_ret = returns_df["sp500"].values
msft_ret = returns_df["msft"].values
dates = returns_df.index.values

# Create conditioning feature arrays
cond_sp500 = sp500_ret[:, np.newaxis]  # (N, 1)
cond_msft = msft_ret[:, np.newaxis]  # (N, 1)
cond_both = np.stack([sp500_ret, msft_ret], axis=1)  # (N, 2)

# Print statistics
print("Return statistics:")
print("-" * 80)
print(f"Amazon:")
print(f"  Mean: {amzn_ret.mean():.6f}")
print(f"  Std:  {amzn_ret.std():.6f}")
print(f"  Min:  {amzn_ret.min():.6f}")
print(f"  Max:  {amzn_ret.max():.6f}")
print()
print(f"SP500:")
print(f"  Mean: {sp500_ret.mean():.6f}")
print(f"  Std:  {sp500_ret.std():.6f}")
print(f"  Min:  {sp500_ret.min():.6f}")
print(f"  Max:  {sp500_ret.max():.6f}")
print()
print(f"MSFT:")
print(f"  Mean: {msft_ret.mean():.6f}")
print(f"  Std:  {msft_ret.std():.6f}")
print(f"  Min:  {msft_ret.min():.6f}")
print(f"  Max:  {msft_ret.max():.6f}")
print()

# Save to NPZ
print("Saving to NPZ file...")
np.savez(
    OUTPUT_FILE,
    amzn_returns=amzn_ret,
    sp500_returns=sp500_ret,
    msft_returns=msft_ret,
    dates=dates,
    cond_sp500=cond_sp500,
    cond_msft=cond_msft,
    cond_both=cond_both,
)

print(f"Saved: {OUTPUT_FILE}")
print()

# Verify saved data
print("Verifying saved data...")
loaded = np.load(OUTPUT_FILE)
print("  Keys:", list(loaded.keys()))
print("  amzn_returns shape:", loaded["amzn_returns"].shape)
print("  sp500_returns shape:", loaded["sp500_returns"].shape)
print("  msft_returns shape:", loaded["msft_returns"].shape)
print("  dates shape:", loaded["dates"].shape)
print("  cond_sp500 shape:", loaded["cond_sp500"].shape)
print("  cond_msft shape:", loaded["cond_msft"].shape)
print("  cond_both shape:", loaded["cond_both"].shape)
print()

print("=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)
print()
print("Summary:")
print(f"  - Total samples: {len(amzn_ret)}")
print(f"  - Date range: {returns_df.index[0].strftime('%Y-%m-%d')} to {returns_df.index[-1].strftime('%Y-%m-%d')}")
print(f"  - Target: Amazon returns")
print(f"  - Conditioning features: SP500, MSFT returns")
print()
print("Next steps:")
print("  1. Run: python train_1d_models.py")
print("  2. Train 4 model variants:")
print("     - Amazon only (baseline)")
print("     - Amazon + SP500 (no loss)")
print("     - Amazon + MSFT (no loss)")
print("     - Amazon + SP500 + MSFT (no loss)")
